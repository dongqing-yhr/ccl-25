#!/bin/bash
# 链 A（${CCL25_HOST} 卡 2）：英文 FrameNet 1.7 一套完整阶梯，共 5 次训练（2026-11-06 裁定 A 选项 1）
#
# 顺序与链 B 逐条同构，只差底座与数据：
#   1  R35-1 裸基线        t3_en_L0    （无负例、无输入注入、训练期先验关）
#   2  裁定 A 新增前置      t2_en       （该资源自己的任务二成员，只用来给负例清单提供假阳性出处）
#   3  该资源自己的负例     dataset/train_deploy_negatives.json（规则与赛题同：每句预算 4 条）
#   4  R35-2 加负例         t3_en_L1
#   5  R44-2 输入注入       t3_en_FN    （与 R35-2 逐字同配方，只差 --with_frame_name 一个开关，紧跟其后启动）
#   6  R35-3 训练期先验     t3_en_L3    （R35-2 加 la_tau 0.5 / mode frame / alpha 1.0 / clamp 2.0，取值照搬不改）
#
# 预训练模型=../baseline/bge-large-en-v1.5（BERT 结构、24 层、位置上限 512）。
# model_task3.py:27 是 self.bert = BertModel(config)，train_task3.py:421 用 strict=False 装载，
# 键名对不上不报错。冒烟实测这份权重发布的 392 个键全不带 bert. 前缀，直接装等于装了个随机初始化的编码器，
# 所以先跑 baseline/fix_bge_keys_1106.py 生成 pytorch_model_bertprefixed.bin（键加 bert. 前缀，原下载件不动），
# 本脚本的 --init_checkpoint 指那一份；验收看冒烟日志里"装配缺失 bert. 键 0 个"。
#
# 数据已在 datasets_paper/prep_en_task3_1106.py 里做了两条装载级处理：
#   原文超过 470 字符的记录剔除（470 = 512 - 4 个固定词元 - 1 个 [SEP] - 全表最长框架名 35 - 2 余量），
#   类别表取三份切分并集（796 个框架名称 / 822 个角色名称，加空类 823 列）。
set -e
W=/work/workspace/ccl25
cd "$W/baseline_en"
mkdir -p logfile

ENC=../baseline/bge-large-en-v1.5
INIT=$ENC/pytorch_model_bertprefixed.bin
COMMON="--seed 42 --config_file $ENC/config.json --vocab_file $ENC/vocab.txt --init_checkpoint $INIT"
NEG=./dataset/train_deploy_negatives.json

echo "=== 发车前自检 $(date '+%F %T') ==="
[ -f "$INIT" ] || { echo "[FAIL] 缺加过前缀的权重 $INIT —— 先在容器里跑 python3 ../baseline/fix_bge_keys_1106.py"; exit 1; }
python3 - <<'PY'
import json
info = json.load(open("./dataset/frame_info.json", encoding="utf8"))
roles = []
for it in info:
    for fe in it["fes"]:
        if fe["fe_name"] not in roles:
            roles.append(fe["fe_name"])
tr = json.load(open("./dataset/task3_train.json", encoding="utf8"))
dv = json.load(open("./dataset/task3_dev.json", encoding="utf8"))
rows_tr = sum(len(l["cfn_spans"]) for l in tr)
rows_dv = sum(len(l["cfn_spans"]) for l in dv)
longest = max(len(x["frame_name"]) for x in info)
mx = max(len(l["text"]) for l in tr + dv) + 5 + longest
print("[自检] 框架名称 %d 个  角色名称 %d 个（加空类共 %d 列）" % (len(info), len(roles), len(roles) + 1))
print("[自检] 训练记录 %d 条 / 标注行合计 %d 条；验证记录 %d 条 / 标注行合计 %d 条"
      % (len(tr), rows_tr, len(dv), rows_dv))
print("[自检] 最长原文 %d 字符 + 5 个固定词元 + 全表最长框架名 %d = 序列 %d，编码器位置上限 512"
      % (max(len(l["text"]) for l in tr + dv), longest, mx))
assert len(roles) + 1 == 823, "角色列数不是 823，类别表放错了"
assert len(dv) == 2278, "验证记录数不是 2278"
assert mx <= 512, "帧名注入后会越出位置表"
PY
nvidia-smi -L | head -2

# ---- 1. R35-1 裸基线 ----
echo "=== [1/6] R35-1 裸基线 t3_en_L0  $(date '+%F %T') ==="
python3 train_task3.py $COMMON \
    --train_file ./dataset/task3_train.json --val_file ./dataset/task3_dev.json \
    --model_prefix t3_en_L0 \
    --num_train_epochs 10 --batch_size 8 --accumulate_gradients 2 --lr 2e-5 \
    > logfile/t3_en_L0.log 2>&1
[ -f saves/t3_en_L0_best.bin ] || { echo "[FAIL] R35-1 没落权重，看 logfile/t3_en_L0.log"; exit 1; }
grep -E "训练样本数|验证样本数|最佳" logfile/t3_en_L0.log | tail -4

# ---- 2. 该资源的任务二成员（只为负例出处，不参与任何主张的对照）----
echo "=== [2/6] 任务二成员 t2_en  $(date '+%F %T') ==="
python3 train_task2.py $COMMON \
    --train_file ./dataset/task3_train.json --val_file ./dataset/task3_dev.json \
    --model_prefix t2_en \
    > logfile/t2_en.log 2>&1
[ -f saves/t2_en_best.bin ] || { echo "[FAIL] 任务二没落权重，看 logfile/t2_en.log"; exit 1; }

# ---- 3. 该资源自己的部署匹配负例 ----
echo "=== [3/6] 负例清单：任务二在训练集上的假阳性  $(date '+%F %T') ==="
python3 predict_task2.py $COMMON \
    --model_path ./saves/t2_en_best.bin \
    --test_file ./dataset/task3_train.json \
    --logit_threshold -0.5 \
    --out_name ./dataset/train_task2_pred.json \
    > logfile/predict_t2_train_en.log 2>&1
grep -i "加载权重\|加载模型" logfile/predict_t2_train_en.log | tail -2
python3 mine_deploy_negatives.py \
    --gold_file ./dataset/task3_train.json \
    --task2_pred ./dataset/train_task2_pred.json \
    --output "$NEG" \
    --budget_per_sentence 4 | tee logfile/mine_negatives_en.log
[ -s "$NEG" ] || { echo "[FAIL] 负例清单为空"; exit 1; }

# ---- 4. R35-2 加负例 ----
echo "=== [4/6] R35-2 t3_en_L1（只比 R35-1 多一份负例）  $(date '+%F %T') ==="
python3 train_task3.py $COMMON \
    --train_file ./dataset/task3_train.json --val_file ./dataset/task3_dev.json \
    --model_prefix t3_en_L1 --deploy_negatives_file "$NEG" \
    --num_train_epochs 10 --batch_size 8 --accumulate_gradients 2 --lr 2e-5 \
    > logfile/t3_en_L1.log 2>&1
[ -f saves/t3_en_L1_best.bin ] || { echo "[FAIL] R35-2 没落权重"; exit 1; }

# ---- 5. R44-2 输入注入（紧跟 R35-2，中间不许插别的改动）----
echo "=== [5/6] R44-2 t3_en_FN（与 R35-2 只差 --with_frame_name）  $(date '+%F %T') ==="
python3 train_task3.py $COMMON \
    --train_file ./dataset/task3_train.json --val_file ./dataset/task3_dev.json \
    --model_prefix t3_en_FN --deploy_negatives_file "$NEG" --with_frame_name \
    --num_train_epochs 10 --batch_size 8 --accumulate_gradients 2 --lr 2e-5 \
    > logfile/t3_en_FN.log 2>&1
[ -f saves/t3_en_FN_best.bin ] || { echo "[FAIL] R44-2 没落权重"; exit 1; }

# ---- 6. R35-3 训练期先验 ----
echo "=== [6/6] R35-3 t3_en_L3（R35-2 加训练期帧条件先验）  $(date '+%F %T') ==="
python3 train_task3.py $COMMON \
    --train_file ./dataset/task3_train.json --val_file ./dataset/task3_dev.json \
    --model_prefix t3_en_L3 --deploy_negatives_file "$NEG" \
    --la_tau 0.5 --la_mode frame --la_alpha 1.0 --la_clamp 2.0 \
    --num_train_epochs 10 --batch_size 8 --accumulate_gradients 2 --lr 2e-5 \
    > logfile/t3_en_L3.log 2>&1
[ -f saves/t3_en_L3_best.bin ] || { echo "[FAIL] R35-3 没落权重"; exit 1; }
grep "logit_adj" logfile/t3_en_L3.log | tail -2

echo "=== 链 A 全部完成 $(date '+%F %T') ==="
ls -l saves/*.bin
echo "读数收集：每份日志里的『最佳验证』行 + 类别表行数；差分只看同一资源内部开/关，跨资源不比绝对分。"
