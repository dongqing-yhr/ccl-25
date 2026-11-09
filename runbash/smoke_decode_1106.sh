#!/bin/bash
# 部署口径解码链的 20 行空跑（CPU，不占卡），只验管道通不通，不出论文读数。
# 验四件事：底座参数路径对不对（params.py:32 默认指向本目录不存在的 roberta）、FI 的行数与行序断言、
#           任务三固定输出格的备份与还原、score_official 能不能吃这三份文件。
# 任务三这里临时用现役权重 model_task3_best.bin 当道具（R32 那一格还在训，它的权重每轮都在重写，不能拿来空跑）。
set -e
W=/work/workspace/ccl25
cd "$W/baseline"
S=./dataset/devladder_smoke
mkdir -p "$S" ./dataset/dev logfile
export CUDA_VISIBLE_DEVICES=""
CAP=20

echo "=== 空跑开始 $(date '+%F %T')  CPU  每环前 $CAP 行 ==="
python3 - "$CAP" <<'PY'
import json, sys
cap = int(sys.argv[1])
dv = json.load(open("./dataset/cfn-dev.json", encoding="utf8"))
mini = dv[:cap]
json.dump(mini, open("./dataset/devmini.json", "w", encoding="utf8"), ensure_ascii=False)
print("[构造] devmini.json %d 行  框架名称非空 %d 行" % (len(mini), sum(1 for r in mini if r.get("frame"))))
PY

BK=./dataset/dev/dev_task3_test.json
[ -f "$BK" ] && cp -p "$BK" "$S/preattempt.bak" && echo "[备份] $BK（$(stat -c %s "$BK") 字节）-> $S/preattempt.bak"

COMMON="--config_file ./chinese-macbert-large/config.json --vocab_file ./chinese-macbert-large/vocab.txt"

echo "--- 环1 任务一 ---"
python3 predict_task1.py $COMMON --test_file ./dataset/devmini.json \
    --model_path ./saves/model_task1_best.bin --out_name "$S/dev_task1_test.json" 2>&1 | tail -3
[ -s "$S/dev_task1_test.json" ] || { echo "[FAIL] 环1 没产出"; exit 1; }

echo "--- 环2 任务二 ---"
python3 predict_task2.py $COMMON --test_file ./dataset/devmini.json \
    --model_path ./saves/model_task2_best.bin --logit_threshold -0.5 \
    --out_name "$S/dev_task2_test.json" 2>&1 | tail -3
[ -s "$S/dev_task2_test.json" ] || { echo "[FAIL] 环2 没产出"; exit 1; }

echo "--- 环3 任务三（吃环2 的跨度）---"
START=$(date +%s)
python3 predict_task3.py $COMMON --test_file ./dataset/devmini.json \
    --model_path ./saves/model_task3_best.bin --task2_result "$S/dev_task2_test.json" 2>&1 | tail -3
[ -f "$BK" ] || { echo "[FAIL] 环3 没写固定输出格"; exit 1; }
[ "$(stat -c %Y "$BK")" -ge "$START" ] || { echo "[FAIL] 环3 输出格时间戳比起算早，说明用的是旧文件"; exit 1; }
cp -p "$BK" "$S/smoke__task3_test.json"

echo "--- 打分 ---"
python3 score_official.py --gold ./dataset/devmini.json --task1 "$S/dev_task1_test.json" \
    --task2 "$S/dev_task2_test.json" --task3 "$S/smoke__task3_test.json"

if [ -f "$S/preattempt.bak" ]; then
    cp -p "$S/preattempt.bak" "$BK"
    echo "[还原] $BK 原样放回，md5 核对："
    md5sum "$BK" "$S/preattempt.bak"
fi
echo "=== 空跑结束 $(date '+%F %T')  上面三行分数只证明管道通，20 行样本不作任何读数 ==="
