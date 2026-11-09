#!/usr/bin/python3
# 把 cfn-train.json 切成 5 折 (每折 1/5 样本量), 供 task1 OOF 训练使用
# 用法: python3 split_train_folds.py

import json
import os
import random

random.seed(42)

SRC = './dataset/cfn-train.json'
DST_DIR = './dataset/folds'
K = 5

os.makedirs(DST_DIR, exist_ok=True)

with open(SRC, 'r', encoding='utf8') as f:
    data = json.load(f)

# 按 sentence_id 切, 同一 sentence_id 归同一折, 避免泄漏
from collections import defaultdict
sid2items = defaultdict(list)
for item in data:
    sid2items[item['sentence_id']].append(item)

sids = list(sid2items.keys())
random.shuffle(sids)

fold_size = (len(sids) + K - 1) // K
for k in range(K):
    fold_sids = set(sids[k * fold_size : (k + 1) * fold_size])
    fold_items = []
    for sid in fold_sids:
        fold_items.extend(sid2items[sid])
    out = '{}/cfn-train-f{}.json'.format(DST_DIR, k)
    with open(out, 'w', encoding='utf8') as f:
        json.dump(fold_items, f, ensure_ascii=False, indent=1)
    print('fold {} : {} sids, {} items -> {}'.format(k, len(fold_sids), len(fold_items), out))

    # 同时输出 leave-one-out 训练文件 (除了第 k 折之外的所有数据)
    loo_items = [it for it in data if it['sentence_id'] not in fold_sids]
    loo_out = '{}/cfn-train-except-f{}.json'.format(DST_DIR, k)
    with open(loo_out, 'w', encoding='utf8') as f:
        json.dump(loo_items, f, ensure_ascii=False, indent=1)
    print('  leave-out {} : {} items -> {}'.format(k, len(loo_items), loo_out))

print('total sids: {}, total items: {}'.format(len(sids), len(data)))
