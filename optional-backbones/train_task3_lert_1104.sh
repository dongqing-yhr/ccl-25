#!/usr/bin/env bash
# 任务三 LERT-large 底座重训（论文"先验算子换预训练模型还成不成立"那一格）
#
# 两臂只差一个 LA 开关，其余逐字照抄现役 MacBERT 两臂的生产配方：
#   ARM=fn  ->  框架名称注入 + OOF 帧 + 部署匹配负例, la_tau=0            （= 现役 fn 成员的配方）
#   ARM=la  ->  上一行 + --la_tau 0.5 --la_mode frame --la_alpha 1.0 --la_clamp 2.0 （= 现役 la 成员的配方）
# 现役 la 那两枚超参不是凭记忆填的，是从 logfile/train_task3_la_frame_0.5.log 的
#   [logit_adj] mode=frame tau=0.5 alpha=1.0 clamp=2.0 帧数=515 类别数=1010
#   [logit_adj] 角色先验范围 log pi ∈ [-2.00, -0.41] (NULL 列已置 0)
# 两行原样读回来的；fn 那臂的配方读自 runbash/logs/train_task3_fn_1020.log 的配置回显。
#
# 与 MacBERT 两臂的唯一差别 = 三个预训练模型路径（config.json / vocab.txt / pytorch_model.bin），
# 这样"换预训练模型"与"改配方"两件事不会混在一起。
# 选权重口径也不传 --dev_cand_file，跟 MacBERT 那两枚（10-20 产出）一致：按金标跨度准确率存 best.bin。
#
# 用法（在容器 baseline 目录）:  ARM=fn GPU=0 SEED=42 bash runbash/train_task3_lert_1104.sh
#   GPU 是容器内可见卡号；本容器 NVIDIA_VISIBLE_DEVICES=2,4,6，即 GPU=0→宿主卡2、1→卡4、2→卡6。
set -euo pipefail
cd /work/workspace/ccl25/baseline

ARM=${ARM:?必须指定 ARM=fn 或 ARM=la}
GPU=${GPU:?必须指定 GPU（容器内可见卡号），发车人定卡，脚本不猜}
SEED=${SEED:-42}
BERT=./models/hfl--LERT-large
LF=logfile
mkdir -p "$LF" saves

log() { echo "[$(date -u +%m-%d_%H:%M:%S)Z / $(date -u -d '+8 hours' +%m-%d_%H:%M:%S)BJ] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

case "$ARM" in
    fn) PREFIX=model_task3_lertfn_s${SEED}; LAARGS=() ;;
    la) PREFIX=model_task3_lertla_s${SEED}
        LAARGS=(--la_tau 0.5 --la_mode frame --la_alpha 1.0 --la_clamp 2.0) ;;
    *) die "ARM 只认 fn / la，实得 $ARM" ;;
esac
CKPT=saves/${PREFIX}_best.bin
LOGF=$LF/train_${PREFIX}_$(date -u +%m%d_%H%M).log

# ---- 发车前的门：文件齐备 / 不覆盖已有权重 / 底座装配自检已过 ----
for f in config.json vocab.txt pytorch_model.bin; do
    [ -s "$BERT/$f" ] || die "LERT 底座 $BERT 缺 $f"
done
for f in ./dataset/train_oof_frames.json ./dataset/train_deploy_negatives.json ./dataset/frame_info.json; do
    [ -s "$f" ] || die "生产配方要用的数据文件缺 $f"
done
[ -e "$CKPT" ] && die "$CKPT 已存在，换 SEED 或先归档，不覆盖已有权重" || true
export CUDA_VISIBLE_DEVICES=$GPU
nvidia-smi -L >/dev/null 2>&1 || die "容器里看不到 GPU（CUDA_VISIBLE_DEVICES=$GPU）"
case "$GPU" in
    0) HOSTCARD=2 ;;
    1) HOSTCARD=4 ;;
    2) HOSTCARD=6 ;;
    *) HOSTCARD="未知（容器卡号 $GPU 不在 2/4/6 映射里）" ;;
esac
LADESC="关闭（不传 --la_tau，默认 la_tau=0）"
[ "$ARM" = la ] && LADESC="--la_tau 0.5 --la_mode frame --la_alpha 1.0 --la_clamp 2.0"

log "===== 任务三 LERT 重训  臂=$ARM  宿主卡=$HOSTCARD  容器卡号=$GPU"
log "  预训练模型目录: $BERT   （原始权重，不含我方角色头）"
log "  底座=24层/1024维  seed=$SEED  epochs=10  bs=8  accum=2  lr=2e-5"
log "  框架名称注入=启用  OOF 帧=./dataset/train_oof_frames.json  部署负例=./dataset/train_deploy_negatives.json"
log "  LA 帧条件先验: $LADESC"
log "  训练集=./dataset/cfn-train.json（默认，不含验证集）  验证集=./dataset/cfn-dev.json（只挑检查点）"
log "  选权重口径=金标跨度准确率（不传 --dev_cand_file，与 10-20 的 MacBERT fn/la 两枚一致）"
log "  model_prefix=$PREFIX  产物=saves/${PREFIX}_best.bin  日志=$LOGF"

python3 -u train_task3.py \
    --seed "$SEED" \
    --model_prefix "$PREFIX" \
    --config_file "$BERT/config.json" \
    --vocab_file "$BERT/vocab.txt" \
    --init_checkpoint "$BERT/pytorch_model.bin" \
    --with_frame_name \
    --oof_frame_file ./dataset/train_oof_frames.json \
    --deploy_negatives_file ./dataset/train_deploy_negatives.json \
    --num_train_epochs 10 \
    --batch_size 8 \
    --accumulate_gradients 2 \
    --lr 2e-5 \
    ${LAARGS[@]+"${LAARGS[@]}"} > "$LOGF" 2>&1 \
    || { tail -n 30 "$LOGF" >&2 || true; die "训练器退出码非 0（臂 $ARM），上面是 $LOGF 末尾 30 行"; }

# ---- 跑完的验收：只认产物文件 + 日志里的关键行，不认退出码 ----
[ -s "$CKPT" ] || die "退出码 0 但没有 $CKPT"
grep -a '加载预训练权重' "$LOGF" >/dev/null || die "$LOGF 里没有\"加载预训练权重\"行，底座权重没装上"
grep -a 'Logit adjustment' "$LOGF" >/dev/null || die "$LOGF 里没有配置回显行"
if [ "$ARM" = la ]; then
    grep -a '\[logit_adj\]' "$LOGF" >/dev/null || die "la 臂的日志里没有 [logit_adj] 行，先验矩阵没生效"
    grep -a 'clamp=2.0' "$LOGF" >/dev/null || die "la 臂的 [logit_adj] 行里 clamp 不是 2.0，超参没按现役配方生效"
fi
log "  [OK] $ARM -> $CKPT  完成: $(date -u +%m-%d_%H:%M:%S)Z"
echo "T3LERT_DONE arm=$ARM prefix=$PREFIX"
