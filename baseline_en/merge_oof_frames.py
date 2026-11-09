#!/usr/bin/python3
# 合并 5 折 OOF 预测到一个 dataset/train_oof_frames.json
# 用法: python3 merge_oof_frames.py

import json
import os

K = 5
FOLD_DIR = './dataset/folds'
OUT = './dataset/train_oof_frames.json'

frames = []
seen = set()
for k in range(K):
    path = '{}/oof_f{}.json'.format(FOLD_DIR, k)
    if not os.path.exists(path):
        print('missing: {}'.format(path))
        continue
    with open(path, 'r', encoding='utf8') as f:
        part = json.load(f)
    for item in part:
        # item = [sentence_id, frame_name]
        sid = item[0]
        if sid in seen:
            continue
        frames.append(item)
        seen.add(sid)

with open(OUT, 'w', encoding='utf8') as f:
    json.dump(frames, f, ensure_ascii=False, indent=1)

print('merged OOF frames: {} sids -> {}'.format(len(frames), OUT))
