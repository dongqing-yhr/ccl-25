#!/bin/bash
# 任务一（框架名称）在英文 FrameNet 1.7 自己的三份公开切分上训练一次，产出部署口径链路的第一环。
# 与链 B 的 1.0 任务一同构，只差底座与数据：底座 = ../baseline/bge-large-en-v1.5（BERT 结构、24 层）。
# 英文底座权重必须先经 baseline/fix_bge_keys_1106.py 生成键名带 bert. 前缀的那一份，
#   否则 train_task1.py:252 的 strict=False 会把 392 个键全当成多余键，等于编码器随机初始化。
# 已核过（check_task1_ready_1106.py）：转换后的英文记录里 text / target[-1] / frame / sentence_id 齐全，
#   跨度与目标词都是原文拼接串的字符下标，与任务三链用的是同一套坐标，不需要再转格式。
set -e
W=/work/workspace/ccl25
cd "$W/baseline_en"
mkdir -p logfile

ENC=../baseline/bge-large-en-v1.5
INIT=$ENC/pytorch_model_bertprefixed.bin
COMMON="--seed 42 --config_file $ENC/config.json --vocab_file $ENC/vocab.txt --init_checkpoint $INIT"

# 显存占用方在 nvidia-smi 里只显示可执行文件路径，用 python3 跑就只有一列 "python3"，
#   管理员和同机的人分不出哪一格是谁的。做法：在解释器同目录给它加一个带作业名的硬链接，
#   用那个名字启动，宿主机侧就会显示 /opt/conda/bin/ccl25_t1_en。
#   必须同目录（Python 按可执行文件位置找标准库），链接失败退回拷贝。
PYBIN=/opt/conda/bin/ccl25_t1_en
ln -f /opt/conda/bin/python3.11 "$PYBIN" 2>/dev/null || cp -f /opt/conda/bin/python3.11 "$PYBIN"
[ -x "$PYBIN" ] || { echo "[FAIL] 造不出带作业名的解释器 $PYBIN"; exit 1; }
echo "[进程命名] 本格显存将记为 $PYBIN"

echo "=== 发车前自检 $(date '+%F %T') ==="
[ -f "$INIT" ] || { echo "[FAIL] 缺加过前缀的权重 $INIT —— 先在容器里跑 python3 ../baseline/fix_bge_keys_1106.py"; exit 1; }
[ -f ./dataset/frame_info.json ] || { echo "[FAIL] 缺英文类别表 ./dataset/frame_info.json"; exit 1; }
python3 ../datasets_paper/check_task1_ready_1106.py | tail -6
nvidia-smi -L | head -2

echo "=== [1/1] 任务一 t1_en  $(date '+%F %T') ==="
"$PYBIN" train_task1.py $COMMON \
    --train_file ./dataset/task3_train.json --val_file ./dataset/task3_dev.json \
    --model_prefix t1_en \
    --num_train_epochs 10 --batch_size 8 --accumulate_gradients 2 --lr 2e-5 \
    > logfile/t1_en.log 2>&1
[ -f saves/t1_en_best.bin ] || { echo "[FAIL] 任务一没落权重，看 logfile/t1_en.log"; exit 1; }
grep -E "saved! new best acc|current acc" logfile/t1_en.log | tail -4
echo "=== 任务一（英文）完成 $(date '+%F %T')  读数：日志里 new best acc 那行的数字就是 2278 条记录上的框架名称准确率 ==="
