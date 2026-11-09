#!/bin/bash
# 任务一（框架名称）在 CFN 1.0 自己的切分上训练一次，产出部署口径链路的第一环。
# 为什么必须补这一次：按 11-06 的裁定，一切实验读数都要走"最后结果怎么产出"那条路——
#   任务三拿到的框架名称必须是上一环预测出来的，不能是标准答案；而预测框架名称的模型
#   在 1.0 上还不存在（正在跑的链 B 只训任务三与任务二）。
# 配方照赛题现役任务一：10 轮 / 批 8 / 累积 2 / 学习率 2e-5 / seed 42，底座 MacBERT-large。
# 数据不用转格式：datasets_paper/check_task1_ready_1106.py 已核过 1.0 的记录里
#   text / target[-1] / frame / sentence_id 四个字段条条齐全，dataset_task1 能直接装载。
set -e
W=/work/workspace/ccl25
cd "$W/baseline_c10"
mkdir -p logfile

ENC=../baseline/chinese-macbert-large
COMMON="--seed 42 --config_file $ENC/config.json --vocab_file $ENC/vocab.txt --init_checkpoint $ENC/pytorch_model.bin"

echo "=== 发车前自检 $(date '+%F %T') ==="
[ -f "$ENC/pytorch_model.bin" ] || { echo "[FAIL] 缺底座权重 $ENC/pytorch_model.bin"; exit 1; }
[ -f ./dataset/frame_info.json ] || { echo "[FAIL] 缺 1.0 的类别表 ./dataset/frame_info.json"; exit 1; }
python3 ../datasets_paper/check_task1_ready_1106.py | tail -6
nvidia-smi -L | head -2

echo "=== [1/1] 任务一 t1_c10  $(date '+%F %T') ==="
python3 train_task1.py $COMMON \
    --train_file ./dataset/task3_train.json --val_file ./dataset/task3_dev.json \
    --model_prefix t1_c10 \
    --num_train_epochs 10 --batch_size 8 --accumulate_gradients 2 --lr 2e-5 \
    > logfile/t1_c10.log 2>&1
[ -f saves/t1_c10_best.bin ] || { echo "[FAIL] 任务一没落权重，看 logfile/t1_c10.log"; exit 1; }
grep -E "saved! new best acc|current acc" logfile/t1_c10.log | tail -4
echo "=== 任务一（1.0）完成 $(date '+%F %T')  读数：日志里 new best acc 那行的数字就是 2000 条记录上的框架名称准确率 ==="
