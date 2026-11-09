#!/bin/bash
# 三资源共用的一条"部署口径"验证集解码链（11-06 裁定：一切实验读数都要按最后结果怎么产出来量，
# 把带标准答案的验证集当成一份有答案的测试集）。
#
# 一条链按级联的真实顺序走三环：
#   第 1 环  任务一用自己的权重给验证集每条记录判框架名称   -> 框架名称准确率 FI
#   第 2 环  任务二用自己的权重给验证集每条记录发跨度        -> 跨度层面 F1 AI
#   第 3 环  任务三吃第 2 环发出来的跨度；只有"框架名称进输入"那一档还吃第 1 环判出的框架名称
#            -> 跨度加角色名称严格匹配的 F1 RI，角色打分的 argmax 落在空类就不交这一行
#   最后    score_official.py 用该资源自己的验证集金标算三项；赛题那一格额外给综合加权分。
#
# 三个资源用同一脚本、同一条任务二及格线（-0.5，与部署匹配负例清单挖掘时用的那条相同）、
#   同一条弃权规则（不传 --null_threshold 即"空类拿第一就不交"）。
#   这样"金标跨度对预测跨度""金标框架名对预测框架名"不再是某套资源特有的口径。
#
# 不覆盖任何既有产物：任务三的预测落在本脚本自己新建的 dataset/devladder_<资源>/ 目录；
#   解码器本身只能写 dataset/dev/dev_task3_test.json 这个固定路径（params.py:275 按验证集文件名
#   自动定输出目录，没有开关），所以那一格先原件备份到本脚本目录、跑完再原样还原。
#
# 用法：bash decode_ladder_res_1106.sh c10        # 或 en / c22
set -e
RES="${1:?用法: bash decode_ladder_res_1106.sh <c10|en|c22>}"
W=/work/workspace/ccl25

case "$RES" in
  c10)
    DIR=$W/baseline_c10;  ENC=$W/baseline/chinese-macbert-large
    INIT=$ENC/pytorch_model.bin
    T1=./saves/t1_c10_best.bin;  T2=./saves/t2_c10_best.bin
    ARMS="t3_c10_L0 t3_c10_L1 t3_c10_FN t3_c10_L3";  FRAME_ARMS="t3_c10_FN"
    GOLD=./dataset/task3_dev.json;  WCOMB=0
    ;;
  en)
    DIR=$W/baseline_en;   ENC=$W/baseline/bge-large-en-v1.5
    INIT=$ENC/pytorch_model_bertprefixed.bin
    T1=./saves/t1_en_best.bin;  T2=./saves/t2_en_best.bin
    ARMS="t3_en_L0 t3_en_L1 t3_en_FN t3_en_L3";  FRAME_ARMS="t3_en_FN"
    GOLD=./dataset/task3_dev.json;  WCOMB=0
    ;;
  c22)
    DIR=$W/baseline;      ENC=$W/baseline/chinese-macbert-large
    INIT=$ENC/pytorch_model.bin
    T1=./saves/model_task1_best.bin;  T2=./saves/model_task2_best.bin
    ARMS="t3_c22_L0";  FRAME_ARMS=""
    GOLD=./dataset/cfn-dev.json;  WCOMB=1
    ;;
  *) echo "[FAIL] 资源代号只认 c10 / en / c22"; exit 1 ;;
esac

# 解码必须显式带本资源的底座目录：params.py:32 的默认值是 chinese-roberta-wwm-ext-large，
# 三个工作目录里都没有那份配置，不传会在 BertConfig.from_json_file 直接报错（英文格还会拿错词表）。
COMMON="--seed 42 --config_file $ENC/config.json --vocab_file $ENC/vocab.txt --init_checkpoint $INIT"

# 显存占用方在 nvidia-smi 只显示可执行文件路径，用 python3 跑就是一列 "python3"，
#   同机的人分不出哪格是谁的、也看不出卡在级联的第几环。做法：在解释器同目录加一个带作业名的
#   硬链接再用它启动（必须同目录，Python 按可执行文件位置找标准库），链接不成退回拷贝。
mkpy() {
    local N="/opt/conda/bin/$1"
    ln -f /opt/conda/bin/python3.11 "$N" 2>/dev/null || cp -f /opt/conda/bin/python3.11 "$N"
    [ -x "$N" ] || { echo "[FAIL] 造不出带作业名的解释器 $N"; exit 1; }
    echo "$N"
}
PY1="$(mkpy "ccl25_dec1_${RES}")"   # 第 1 环 任务一
PY2="$(mkpy "ccl25_dec2_${RES}")"   # 第 2 环 任务二
PY3="$(mkpy "ccl25_dec3_${RES}")"   # 第 3 环 任务三（各档位串行，共用一个名字）
echo "[进程命名] 三环分别记为 ccl25_dec1_$RES / ccl25_dec2_$RES / ccl25_dec3_$RES"

cd "$DIR"
OUT=./dataset/devladder_$RES
T3SLOT=./dataset/dev/dev_task3_test.json
mkdir -p "$OUT" ./dataset/dev logfile
LOG=logfile/decode_ladder_$RES.log

echo "=== 部署口径解码链 资源=$RES  工作目录=$DIR  $(date '+%F %T') ===" | tee "$LOG"

# ---- 发车前自检 ----
[ -f "$INIT" ] || { echo "[FAIL] 缺底座权重 $INIT"; exit 1; }
[ -f "$T1" ] || { echo "[FAIL] 缺任务一权重 $T1 —— 先跑 chain_t1_${RES}_1106.sh"; exit 1; }
[ -f "$T2" ] || { echo "[FAIL] 缺任务二权重 $T2 —— 该资源链的第 2 步还没跑完"; exit 1; }
NREADY=0
for A in $ARMS; do
    if [ -f "saves/${A}_best.bin" ]; then NREADY=$((NREADY + 1)); else echo "[提示] 任务三档位 $A 权重未落盘，本轮跳过"; fi
done
[ "$NREADY" -gt 0 ] || { echo "[FAIL] 本轮一个任务三档位都没有，不跑空链"; exit 1; }
# 固定输出格里若已有别的任务留下的文件，先按字节备份，链尾还原
if [ -f "$T3SLOT" ]; then
    cp -p "$T3SLOT" "$OUT/preattempt__task3_test.json.bak"
    echo "[备份] $T3SLOT 已存在（$(stat -c %s "$T3SLOT") 字节）-> $OUT/preattempt__task3_test.json.bak，链尾原样还原"
fi

echo "--- 第 1 环 任务一：给验证集判框架名称（后面用的是这份预测，不是标准答案） ---" | tee -a "$LOG"
"$PY1" predict_task1.py $COMMON --test_file "$GOLD" --model_path "$T1" \
    --out_name "$OUT/dev_task1_test.json" > logfile/dec_task1_$RES.log 2>&1
[ -s "$OUT/dev_task1_test.json" ] || { echo "[FAIL] 任务一预测文件为空，看 logfile/dec_task1_$RES.log"; exit 1; }
grep -E "加载权重|加载模型" logfile/dec_task1_$RES.log | tail -1 | tee -a "$LOG"
python3 - "$OUT/dev_task1_test.json" "$GOLD" <<'PY' | tee -a "$LOG"
import json, sys
pred = json.load(open(sys.argv[1], encoding="utf8"))
gold = json.load(open(sys.argv[2], encoding="utf8"))
n = len(gold)
assert len(pred) == n, "任务一预测行数 %d 与验证记录数 %d 不一致，FI 的位置对位会错" % (len(pred), n)
bad = sum(1 for i, r in enumerate(gold) if pred[i][0] != r["sentence_id"])
hit = sum(1 for i, r in enumerate(gold) if pred[i][1] == r.get("frame"))
print("[读数] 框架名称：命中 %d / %d 行 = %.4f   行序错位 %d 处（0 才对）" % (hit, n, 100.0 * hit / n, bad))
PY

echo "--- 第 2 环 任务二：给验证集发跨度（及格线 -0.5） ---" | tee -a "$LOG"
"$PY2" predict_task2.py $COMMON --test_file "$GOLD" --model_path "$T2" \
    --logit_threshold -0.5 --out_name "$OUT/dev_task2_test.json" > logfile/dec_task2_$RES.log 2>&1
[ -s "$OUT/dev_task2_test.json" ] || { echo "[FAIL] 任务二预测文件为空，看 logfile/dec_task2_$RES.log"; exit 1; }
grep -E "加载权重|结构开关" logfile/dec_task2_$RES.log | tail -2 | tee -a "$LOG"

echo "--- 第 3 环 任务三：逐个档位吃同一份第 1/2 环输出 ---" | tee -a "$LOG"
for A in $ARMS; do
    [ -f "saves/${A}_best.bin" ] || continue
    EXTRA=""
    case " $FRAME_ARMS " in *" $A "*) EXTRA="--with_frame_name --task1_result $OUT/dev_task1_test.json" ;; esac
    echo "=== 档位 $A  $(date '+%F %T')  $EXTRA ===" | tee -a "$LOG"
    START=$(date +%s)
    "$PY3" predict_task3.py $COMMON --test_file "$GOLD" --model_path "saves/${A}_best.bin" \
        --task2_result "$OUT/dev_task2_test.json" $EXTRA > "logfile/dec_$A.log" 2>&1
    [ -f "$T3SLOT" ] || { echo "[FAIL] $A 没写出任务三预测，看 logfile/dec_$A.log"; exit 1; }
    NEW=$(stat -c %Y "$T3SLOT")
    [ "$NEW" -ge "$START" ] || { echo "[FAIL] $A 这一格没重写预测文件（时间戳 $NEW < 起算 $START），拒绝拿旧文件算分"; exit 1; }
    cp -p "$T3SLOT" "$OUT/${A}__task3_test.json"
    grep -E "结构开关|丢弃|NULL 弃权" "logfile/dec_$A.log" | tail -3 | tee -a "$LOG"
    if [ "$WCOMB" = 1 ]; then
        python3 score_official.py --gold "$GOLD" --task1 "$OUT/dev_task1_test.json" \
            --task2 "$OUT/dev_task2_test.json" --task3 "$OUT/${A}__task3_test.json" | tee -a "$LOG"
    else
        # 新资源不构造综合加权分：0.3/0.3/0.4 那三个权重是赛题平台给的，别的资源没有这笔账。
        python3 score_official.py --gold "$GOLD" --task1 "$OUT/dev_task1_test.json" \
            --task2 "$OUT/dev_task2_test.json" --task3 "$OUT/${A}__task3_test.json" \
            | grep -E "FI acc|AI f1|RI f1|FI 命中" | tee -a "$LOG"
    fi
done

# ---- 还原固定格里原本属于别的任务的文件 ----
if [ -f "$OUT/preattempt__task3_test.json.bak" ]; then
    cp -p "$OUT/preattempt__task3_test.json.bak" "$T3SLOT"
    echo "[还原] $T3SLOT 已按备份原样放回，md5 核对："
    md5sum "$T3SLOT" "$OUT/preattempt__task3_test.json.bak" | tee -a "$LOG"
fi

echo "=== 解码链 $RES 完成 $(date '+%F %T')  产物在 $DIR/$OUT/ ===" | tee -a "$LOG"
