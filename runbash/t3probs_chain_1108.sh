#!/bin/bash
# 两套对照资源的任务三成员概率表：导出（生产解码器第三环 + --save_logits 侧车）后立刻接侧车转表与
# 两道复核（复原解码逐条等于 11-06 部署解码产物、角色侧 F1 复算等于台账记录的读数）。
# 任一档复核不上，脚本当场中断，后面的资源不再跑。
set -e
W=/work/workspace/ccl25
echo "=== 概率表导出与转表链 起 $(date '+%F %T') ==="
bash $W/t3probs_dump_1108.sh c10
cd $W/baseline_c10 && PYTHONPATH=. /opt/conda/bin/python3.11 runbash/probs_t3_to_npy_1108.py \
    --probs_dir dataset/devladder_c10/probs \
    --arms t3_c10_L0,t3_c10_L1,t3_c10_FN,t3_c10_L3 \
    --expect t3_c10_L0=62.3051,t3_c10_L1=62.9820,t3_c10_FN=62.8143,t3_c10_L3=62.8614 \
    2>&1 | tee logfile/probs_t3_to_npy_c10_1108.log
bash $W/t3probs_dump_1108.sh en
cd $W/baseline_en && PYTHONPATH=. /opt/conda/bin/python3.11 runbash/probs_t3_to_npy_1108.py \
    --probs_dir dataset/devladder_en/probs \
    --arms t3_en_L0,t3_en_L1,t3_en_FN,t3_en_L3 \
    --expect t3_en_L0=34.4602,t3_en_L1=36.0599,t3_en_FN=36.7977,t3_en_L3=32.3513 \
    2>&1 | tee logfile/probs_t3_to_npy_en_1108.log
echo "=== 概率表导出与转表链 全部完成 $(date '+%F %T') ==="
echo "T3PROBS_CHAIN_DONE"
