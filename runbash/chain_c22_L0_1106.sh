#!/bin/bash
# R32（${CCL25_HOST} 卡 4）：赛题 CFN 2.2 的裸基线一次训练，作跨资源阶梯的赛题那一格
#
# 与链 A／链 B 的第 1 步完全同构：去掉现役配方里的两份外部文件
#   （--oof_frame_file 上一节点预测的框架名称、--deploy_negatives_file 部署匹配负例），
#   不开输入端框架名称注入，训练期先验 la_tau 取 0。其余超参照搬 runbash/run_task3_oof.sh：
#   seed 42、chinese-macbert-large、10 轮、批 8、梯度累积 2、学习率 2e-5。
# 权重写到新前缀 t3_c22_L0，不动 saves/ 里任何现役文件。
set -e
W=/work/workspace/ccl25
cd "$W/baseline"
mkdir -p logfile

ENC=./chinese-macbert-large
COMMON="--seed 42 --config_file $ENC/config.json --vocab_file $ENC/vocab.txt --init_checkpoint $ENC/pytorch_model.bin"

echo "=== 发车前自检 $(date '+%F %T') ==="
[ -f "$ENC/pytorch_model.bin" ] || { echo "[FAIL] 缺底座权重"; exit 1; }
python3 - <<'PY'
import json
info = json.load(open("./dataset/frame_info.json", encoding="utf8"))
roles = []
for it in info:
    for fe in it["fes"]:
        if fe["fe_name"] not in roles:
            roles.append(fe["fe_name"])
dv = json.load(open("./dataset/cfn-dev.json", encoding="utf8"))
print("[自检] 赛题 框架名称 %d 个  角色名称 %d 个（加空类共 %d 列）  验证记录 %d 条"
      % (len(info), len(roles), len(roles) + 1, len(dv)))
assert len(roles) + 1 == 1010, "角色列数不是 1010，类别表放错了"
PY
[ -f saves/t3_c22_L0_best.bin ] && { echo "[FAIL] saves/t3_c22_L0_best.bin 已存在，先确认是不是要重跑"; exit 1; }
nvidia-smi -L | head -2

echo "=== R32 t3_c22_L0 裸基线 $(date '+%F %T') ==="
python3 train_task3.py $COMMON \
    --model_prefix t3_c22_L0 \
    --num_train_epochs 10 --batch_size 8 --accumulate_gradients 2 --lr 2e-5 \
    > logfile/t3_c22_L0_1106.log 2>&1
[ -f saves/t3_c22_L0_best.bin ] || { echo "[FAIL] R32 没落权重，看 logfile/t3_c22_L0_1106.log"; exit 1; }
grep -E "训练样本数|验证样本数|加载预训练权重|最佳" logfile/t3_c22_L0_1106.log | tail -6
echo "=== R32 完成 $(date '+%F %T') ==="
