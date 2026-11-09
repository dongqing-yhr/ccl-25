#!/bin/bash
# 导出两套对照资源的任务三成员概率表（R48 加权平均、R37 票数与弃权、R50 先验注入位置三项的前置）。
# 纯推理，不训练。做法：拿 11-06 部署解码链第三环的同一行命令重跑一遍，只多挂一个 --save_logits
# 侧车开关（生产解码器自带，写满每条候选跨度的全表 softmax）。
# 锚点：本次写出的任务三预测文件必须与 11-06 部署解码链的产物逐字节相同（md5）。对不上就立刻中断，
# 不在一张复现不了答卷的概率表上算差分。
set -e
RES="${1:?用法: bash t3probs_dump_1108.sh <c10|en>}"
W=/work/workspace/ccl25
case "$RES" in
  c10) DIR=$W/baseline_c10; ENC=$W/baseline/chinese-macbert-large; INIT=$ENC/pytorch_model.bin
       ARMS="t3_c10_L0 t3_c10_L1 t3_c10_FN t3_c10_L3"; FRAME_ARMS="t3_c10_FN" ;;
  en)  DIR=$W/baseline_en;  ENC=$W/baseline/bge-large-en-v1.5;  INIT=$ENC/pytorch_model_bertprefixed.bin
       ARMS="t3_en_L0 t3_en_L1 t3_en_FN t3_en_L3"; FRAME_ARMS="t3_en_FN" ;;
  *) echo "[FAIL] 资源代号只认 c10 / en"; exit 1 ;;
esac
COMMON="--seed 42 --config_file $ENC/config.json --vocab_file $ENC/vocab.txt --init_checkpoint $INIT"

N="/opt/conda/bin/ccl25_t3probs_${RES}"
ln -f /opt/conda/bin/python3.11 "$N" 2>/dev/null || cp -f /opt/conda/bin/python3.11 "$N"
[ -x "$N" ] || { echo "[FAIL] 造不出带作业名的解释器 $N"; exit 1; }

cd "$DIR"
OUT=./dataset/devladder_$RES
PB=$OUT/probs
T3SLOT=./dataset/dev/dev_task3_test.json
GOLD=./dataset/task3_dev.json
LOG=logfile/t3probs_dump_$RES.log
mkdir -p "$PB" ./dataset/dev

echo "=== 任务三成员概率表导出 资源=$RES 工作目录=$DIR 进程名=ccl25_t3probs_$RES  $(date '+%F %T') ===" | tee "$LOG"
[ -f "$INIT" ] || { echo "[FAIL] 缺底座权重 $INIT"; exit 1; }
[ -s "$OUT/dev_task2_test.json" ] || { echo "[FAIL] 缺第 2 环跨度文件 $OUT/dev_task2_test.json"; exit 1; }
[ -s "$OUT/dev_task1_test.json" ] || { echo "[FAIL] 缺第 1 环框架名称文件 $OUT/dev_task1_test.json"; exit 1; }
if [ -f "$T3SLOT" ]; then
  cp -p "$T3SLOT" "$PB/preattempt__task3_test.json.bak"
  echo "[备份] $T3SLOT 已存在 -> $PB/preattempt__task3_test.json.bak，跑完原样还原" | tee -a "$LOG"
fi

for A in $ARMS; do
  [ -f "saves/${A}_best.bin" ] || { echo "[FAIL] 缺档位权重 saves/${A}_best.bin"; exit 1; }
  [ -f "$OUT/${A}__task3_test.json" ] || { echo "[FAIL] 缺 11-06 部署解码产物 $OUT/${A}__task3_test.json，锚点没有底本"; exit 1; }
  EXTRA=""
  case " $FRAME_ARMS " in *" $A "*) EXTRA="--with_frame_name --task1_result $OUT/dev_task1_test.json" ;; esac
  echo "--- 档位 $A  $(date '+%F %T')  额外参数: $EXTRA ---" | tee -a "$LOG"
  START=$(date +%s)
  "$N" predict_task3.py $COMMON --test_file "$GOLD" --model_path "saves/${A}_best.bin" \
      --task2_result "$OUT/dev_task2_test.json" $EXTRA \
      --save_logits --logits_out "$PB/probs_${A}.json" > "logfile/t3probs_${A}.log" 2>&1
  [ -f "$T3SLOT" ] || { echo "[FAIL] $A 没写出任务三预测，看 logfile/t3probs_${A}.log"; exit 1; }
  [ "$(stat -c %Y "$T3SLOT")" -ge "$START" ] || { echo "[FAIL] $A 没重写预测文件（时间戳早于起算），拒绝拿旧文件对锚点"; exit 1; }
  [ -s "$PB/probs_${A}.json" ] || { echo "[FAIL] $A 侧车概率表为空"; exit 1; }
  M1=$(md5sum "$T3SLOT" | cut -d' ' -f1)
  M2=$(md5sum "$OUT/${A}__task3_test.json" | cut -d' ' -f1)
  [ "$M1" = "$M2" ] || { echo "[FAIL] $A 锚点漂移：本次解码 md5=$M1 对 11-06 部署解码 md5=$M2 ⇒ 停，不在这张表上算差分"; exit 1; }
  echo "[锚点OK] $A 本次解码与 11-06 部署解码逐字节相同 md5=$M1；侧车 $(stat -c %s "$PB/probs_${A}.json") 字节" | tee -a "$LOG"
  mv "$T3SLOT" "$PB/dec_${A}.json"
done

if [ -f "$PB/preattempt__task3_test.json.bak" ]; then
  cp -p "$PB/preattempt__task3_test.json.bak" "$T3SLOT"
  echo "[还原] $T3SLOT 已按备份原样放回" | tee -a "$LOG"
fi
echo "=== 概率表导出完成 资源=$RES  $(date '+%F %T') ===" | tee -a "$LOG"
echo "T3PROBS_DUMP_DONE"
