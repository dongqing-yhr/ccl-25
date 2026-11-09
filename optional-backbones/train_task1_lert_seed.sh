#!/bin/bash
# 任务一：LERT-large 底座，按我方配方重训（不含验证集），供 γ 算子做 ±对照
#
# 与 runbash/train_task1_macbert_seed.sh 的唯一差别是三个预训练模型路径换成 LERT-large，
# 其余超参数逐字照抄，这样"换预训练模型"与"改配方"两件事不会混在一起。
#
# 装配已实测（check_lert_assembly_task1.py）：编码器可学习参数覆盖率 391/391 = 100%，
# 形状不一致 0 个，与 MacBERT-large 的读数完全相同，两者 config 同构（1024 维 / 24 层）。
#
# 用法: SEED=42 GPU=1 bash runbash/train_task1_lert_seed.sh   （在容器 baseline 目录）
set -e
cd /work/workspace/ccl25/baseline
SEED=${SEED:?必须指定 SEED，例如 SEED=42 GPU=1 bash runbash/train_task1_lert_seed.sh}
GPU=${GPU:-1}
PREFIX="model_task1_lert_s${SEED}"
export CUDA_VISIBLE_DEVICES=${GPU}
BERT=./models/hfl--LERT-large

echo "=========================================="
echo "任务一 LERT-large 重训（我方配方，训练数据不含验证集）"
echo "  seed=${SEED}  model_prefix=${PREFIX}  容器卡号=${GPU}"
echo "  预训练模型目录: ${BERT}"
echo "  训练集: ./dataset/cfn-train.json（默认值，不含验证集）"
echo "  验证集: ./dataset/cfn-dev.json（默认值，只用于挑检查点，不进训练数据）"
echo "  10 轮 / bs 8 / 梯度累积 2（有效 16）/ lr 2e-5 / FGM 代码固定开 / 不存 EMA"
echo "  产物: saves/${PREFIX}_best.bin"
echo "  启动: $(date)"
echo "=========================================="

test -f "${BERT}/config.json" || { echo "!! 缺 ${BERT}/config.json"; exit 1; }
test -f "${BERT}/vocab.txt" || { echo "!! 缺 ${BERT}/vocab.txt"; exit 1; }
test -f "${BERT}/pytorch_model.bin" || { echo "!! 缺 ${BERT}/pytorch_model.bin"; exit 1; }

python3 train_task1.py \
    --seed ${SEED} \
    --model_prefix ${PREFIX} \
    --config_file ${BERT}/config.json \
    --vocab_file ${BERT}/vocab.txt \
    --init_checkpoint ${BERT}/pytorch_model.bin \
    --num_train_epochs 10 \
    --batch_size 8 \
    --accumulate_gradients 2 \
    --lr 2e-5

# fail loudly：产物必须存在，否则中断，不静默复用旧文件
test -f "saves/${PREFIX}_best.bin" || { echo "!! 缺 saves/${PREFIX}_best.bin，本次训练未产出"; exit 1; }
echo ">>> OK  seed=${SEED} -> saves/${PREFIX}_best.bin  完成: $(date)"
