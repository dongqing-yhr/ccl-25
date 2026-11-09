#!/bin/bash
# 等齐依赖标记后，用指定卡跑某个资源的"部署口径"解码链（三环：任务一判框架名称 -> 任务二发跨度 -> 任务三吃前两环的产物）。
# 为什么排队而不是立刻跑：白天三张登记卡都排了训练，解码要占 GPU，抢在同一张卡上会拖慢正在跑的那一格，
#   也可能显存不够直接崩。判"上一格跑完没有"只认链日志末尾的完成标记，不认权重文件存在
#   （训练每轮验证通过都会重写同一份权重，文件在不代表那一格跑完）。
#
# 用法：bash queue_decode_1106.sh <资源> <容器可见卡号> <标记1>=<日志1> [<标记2>=<日志2> ...]
#   资源        c10 / en / c22
#   容器可见卡号 0=宿主机卡2  1=宿主机卡4  2=宿主机卡6（单卡机器上容器只见卡 0，写 0）
#   标记与日志   用 = 号分成两半，日志路径给绝对路径；要给多个依赖就写多对，全部出现才放行
set -u
RES="${1:?用法: queue_decode_1106.sh <资源> <卡号> <标记>=<日志> ...}"
IDX="${2:?缺容器可见卡号}"
shift 2
[ "$#" -ge 1 ] || { echo "[FAIL] 至少要一个依赖标记"; exit 1; }

W=/work/workspace/ccl25
LOG=$W/queue_decode_${RES}_1106.log
DEADLINE=$(( $(date +%s) + 20 * 3600 ))   # 20 小时还没等齐就放弃，不无限挂着

echo "[排队开始] $(date '+%F %T')  资源=$RES  预定卡号=$IDX  依赖：$*" >> "$LOG"

while : ; do
    MISSING=""
    for PAIR in "$@"; do
        MARKER="${PAIR%%=*}"
        MARKLOG="${PAIR#*=}"
        if ! grep -q "$MARKER" "$MARKLOG" 2>/dev/null; then
            MISSING="$MISSING [$MARKER @ $MARKLOG]"
        fi
    done
    [ -z "$MISSING" ] && break
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "[超时退出] $(date '+%F %T')  等了 20 小时仍缺：$MISSING  —— 解码不发车，先看对应链日志" >> "$LOG"
        exit 3
    fi
    sleep 60
done

echo "[放行] $(date '+%F %T')  依赖已齐，改用容器可见编号 $IDX 跑 $RES 的解码链" >> "$LOG"
sleep 20
cd "$W"
CUDA_VISIBLE_DEVICES="$IDX" bash decode_ladder_res_1106.sh "$RES" >> "$LOG" 2>&1
RC=$?
echo "[解码链退出] $(date '+%F %T')  退出码 $RC（0 才算跑通，非 0 看本日志末尾）" >> "$LOG"
exit "$RC"
