#!/usr/bin/env bash
# 并行发车：任务三 LERT-large 底座的两臂（fn / la），各占一张空卡。
# 在容器 baseline 目录跑：  bash runbash/launch_t3lert2_1104.sh
# 本容器 NVIDIA_VISIBLE_DEVICES=2,4,6 → 容器卡号 0/1/2 = 宿主卡 2/4/6。
# 门（任一不过就中断，不起任何训练进程）：
#   1) 容器里看不到 3 张卡；
#   2) 容器卡 0 或 1 已被占用（>1000 MiB）；
#   3) 已有 train_task*/predict_task 进程在跑（共享卡不并发）；
#   4) LERT 装配自检不过（编码器覆盖率必须 100%、形状不一致必须 0 个）；
#   5) 两臂的产物权重已存在（不覆盖）。
set -euo pipefail
cd /work/workspace/ccl25/baseline

LF=logfile
BERT=./models/hfl--LERT-large
SEED=${SEED:-42}
mkdir -p "$LF"

log() { echo "[$(date -u +%m-%d_%H:%M:%S)Z / $(date -u -d '+8 hours' +%m-%d_%H:%M:%S)BJ] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

NGPU=$(nvidia-smi -L | wc -l)
[ "$NGPU" -ge 2 ] || die "容器里只看到 $NGPU 张卡，不够两臂并行"
log "门1 容器可见卡数 = $NGPU（宿主映射 0→2, 1→4, 2→6）"

for g in 0 1; do
    used=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$used" -lt 1000 ] || die "门2 容器卡号 $g 已用 ${used} MiB，不是空卡"
    log "门2 容器卡号 $g 已用 ${used} MiB（空）"
done

pgrep -f '[t]rain_task1.py|[t]rain_task2.py|[t]rain_task3.py|[p]redict_task' >/dev/null \
    && die "门3 已有训练/预测进程在跑，拒绝撞卡（先确认那些进程是不是别人的）" || log "门3 无其他训练/预测进程"

for p in "model_task3_lertfn_s${SEED}" "model_task3_lertla_s${SEED}"; do
    [ -e "saves/${p}_best.bin" ] && die "门5 saves/${p}_best.bin 已存在，换 SEED 或先归档，不覆盖已有权重" || true
done
log "门5 两只产物名尚未被占用：model_task3_lertfn_s${SEED} / model_task3_lertla_s${SEED}"

log "门4 LERT 装配自检（CPU，逐字照抄 train_task3.py:419-423 的 strict=False 装配路径）"
ASMF=$LF/lert_asm_t3_$(date -u +%m%d_%H%M).log
python3 -u check_lert_assembly_task3.py "$BERT/config.json" "$BERT/pytorch_model.bin" ./dataset/frame_info.json \
    2>&1 | tee "$ASMF" || true
grep -q '可以按我方配方训练 LERT' "$ASMF" || die "门4 装配自检未通过，见 $ASMF"

log "门6 底座冒烟（分词对齐／装配／前向数值／LA 先验进损失，四道闸）"
SMKF=$LF/lert_smoke_t3_$(date -u +%m%d_%H%M).log
python3 -u smoke_lert_task3.py 0 > "$SMKF" 2>&1 || die "门6 冒烟未通过（退出码非 0），见 $SMKF"
grep -q '冒烟通过' "$SMKF" || die "门6 冒烟日志里没有结论行，见 $SMKF"
grep -a '闸' "$SMKF" || true

log "==== 发车：两臂并行（fn→容器卡0=宿主卡2／la→容器卡1=宿主卡4）===="
nohup env ARM=fn GPU=0 SEED=$SEED bash runbash/train_task3_lert_1104.sh \
    > "$LF/launch_t3lert_fn_$(date -u +%m%d_%H%M).out" 2>&1 < /dev/null &
PID_FN=$!
disown $PID_FN
nohup env ARM=la GPU=1 SEED=$SEED bash runbash/train_task3_lert_1104.sh \
    > "$LF/launch_t3lert_la_$(date -u +%m%d_%H%M).out" 2>&1 < /dev/null &
PID_LA=$!
disown $PID_LA
log "  驱动 PID: fn=$PID_FN  la=$PID_LA"

sleep 240
ALIVE=0
for pid in $PID_FN $PID_LA; do
    kill -0 "$pid" 2>/dev/null && ALIVE=$((ALIVE + 1)) || true
done
[ "$ALIVE" -eq 2 ] || die "240 秒后只有 $ALIVE/2 个驱动还活着，看 $LF/launch_t3lert_*.out"
NTRAIN=$(pgrep -cf '[t]rain_task3.py' || true)
log "  在跑的 train_task3.py 进程数 = $NTRAIN"
[ "$NTRAIN" -ge 2 ] || die "没有看到两个训练进程"

for a in fn la; do
    L=$(ls -t "$LF"/train_model_task3_lert${a}_s${SEED}_*.log 2>/dev/null | head -1)
    [ -n "$L" ] || die "找不到 $a 臂的训练日志"
    log "  ---- $a 臂日志 $L 的配置回显（未打印则说明还在装权重）----"
    awk '/^Task3 训练配置/{p=1} p{print; c++} c>32{exit}' "$L" || true
done

# 两臂只差 LA 那一行：回显里 fn 必须是 tau=0.0、la 必须是 tau=0.5 clamp=2.0
L_FN=$(ls -t "$LF"/train_model_task3_lertfn_s${SEED}_*.log | head -1)
L_LA=$(ls -t "$LF"/train_model_task3_lertla_s${SEED}_*.log | head -1)
if grep -aq 'Logit adjustment' "$L_FN"; then
    grep -aq 'Logit adjustment: tau=0.0' "$L_FN" || die "fn 臂的回显里 la_tau 不是 0.0，两臂没拉开单变量"
    log "  [核对] fn 臂回显 la_tau=0.0"
else
    log "  [提示] fn 臂配置回显还没打出来，发车回执里再核"
fi
if grep -aq 'Logit adjustment' "$L_LA"; then
    grep -aq 'tau=0.5 mode=frame alpha=1.0' "$L_LA" || die "la 臂的回显 LA 超参不是 0.5/frame/1.0"
    log "  [核对] la 臂回显 tau=0.5 mode=frame alpha=1.0"
    grep -aq '\[logit_adj\].*clamp=2.0' "$L_LA" || die "la 臂没有 [logit_adj] clamp=2.0 行，先验矩阵没按现役配方生效"
    grep -a '\[logit_adj\]' "$L_LA" || true
else
    log "  [提示] la 臂配置回显还没打出来，发车回执里再核"
fi

log "  两臂显存占用（容器卡号）："
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

log "==== 已发车 ===="
echo "T3LERT_LAUNCH_DONE seed=$SEED pid_fn=$PID_FN pid_la=$PID_LA log_fn=$L_FN log_la=$L_LA"
