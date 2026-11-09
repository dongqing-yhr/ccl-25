#!/bin/bash
# 把英文任务一的训练排在赛题那一格（R32）之后，用同一张卡串起来，不并发抢显存。
# 11-06 白天只有登记过的卡可用，${CCL25_HOST} 容器可见的三张 A100 里
#   编号 1 = 宿主机卡 4 在跑英文任务三链，编号 2 = 宿主机卡 6 在跑赛题 R32，
#   编号 0 = 宿主机卡 2 刚被用来发 1.0 的任务一。所以英文任务一只能等编号 2 上那一格跑完。
# 判"跑完"只认链日志里的完成标记，不认权重文件存在（训练每轮验证通过都会重写同一份权重）。
W=/work/workspace/ccl25
LOG=$W/queue_t1_en_1106.log
MARKER="R32 完成"
DEADLINE=$(( $(date +%s) + 12 * 3600 ))

echo "[排队开始] $(date '+%F %T')  等 $W/chain_c22_L0.log 里出现『$MARKER』" | tee -a "$LOG"
while ! grep -q "$MARKER" "$W/chain_c22_L0.log" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "[超时退出] 等了 12 小时没等到完成标记，英文任务一不发车；先看 chain_c22_L0.log" | tee -a "$LOG"
        exit 3
    fi
    sleep 60
done
echo "[放行] $(date '+%F %T')  赛题那一格已跑完，改用容器可见编号 2（宿主机卡 6）发英文任务一" | tee -a "$LOG"
sleep 30
cd "$W"
CUDA_VISIBLE_DEVICES=2 bash chain_t1_en_1106.sh >> "$LOG" 2>&1
echo "[队列结束] $(date '+%F %T')  退出码 $?" | tee -a "$LOG"
