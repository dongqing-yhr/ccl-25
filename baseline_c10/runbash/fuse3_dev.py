#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dev: 用三份已导出的角色概率侧车做分数级集成对照(不需要再跑模型)。

侧车来自 runbash/dev_fn_compare.py 的 --save_logits, 每行 [句子号, 跨度起, 跨度止, 1009 个角色概率 + NULL 概率],
三个模型行序一致, 逐行对齐后按权重平均, 再按「NULL 概率超阈值就弃权、否则取概率最大角色」解码。
门禁 = 只保留预测框架定义过的角色。

用途: 判断框架名注入(FN)能不能和帧条件先验(LA)、OOF 底模的融合叠加。
"""
import csv
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

BASE = '/work/workspace/ccl25/baseline'
RES = os.path.join(BASE, 'dataset/dev/fncmp')
D1 = os.path.join(BASE, 'dataset/dev/dev_task1_test.json')
D2 = os.path.join(BASE, 'dataset/dev/dev_task2_test.json')
OUT = os.path.join(BASE, 'logfile/dev_fuse3_grid.csv')

TAUS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]
# 每个配置 = 三个模型的权重(0 表示不参与)
CONFIGS = [
    ('fn_only',      {'oof': 0.0, 'la': 0.0, 'fn': 1.0}),
    ('la_only',      {'oof': 0.0, 'la': 1.0, 'fn': 0.0}),
    ('oof_la',       {'oof': 0.5, 'la': 0.5, 'fn': 0.0}),
    ('fn_la',        {'oof': 0.0, 'la': 0.5, 'fn': 0.5}),
    ('oof_fn',       {'oof': 0.5, 'la': 0.0, 'fn': 0.5}),
    ('three_eq',     {'oof': 1 / 3, 'la': 1 / 3, 'fn': 1 / 3}),
    ('three_fnhalf', {'oof': 0.25, 'la': 0.25, 'fn': 0.5}),
]

RE_RI = re.compile(r'RI f1\s*:\s*([0-9.]+)\s*\(p\s*([0-9.]+)\s*/\s*r\s*([0-9.]+)\)')
RE_SC = re.compile(r'([0-9.]+)\s*$')


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def load(name):
    rows = json.load(open(os.path.join(RES, 'logits_%s.json' % name), encoding='utf8'))
    keys = [(r[0], r[1], r[2]) for r in rows]
    P = np.array([r[3:] for r in rows], dtype=np.float64)
    return keys, P


log('读取三份概率侧车 ...')
K, P = {}, {}
for n in ('oof', 'la', 'fn'):
    K[n], P[n] = load(n)
    log('  %s: %d 行, 每行 %d 个概率' % (n, len(K[n]), P[n].shape[1]))
assert K['oof'] == K['la'] == K['fn'], '三个模型的候选跨度行序不一致, 不能逐行平均'
NC = P['oof'].shape[1]
NL = NC - 1  # 最后一个下标是 NULL(弃权)类

fr = json.load(open(os.path.join(BASE, 'dataset/frame_roles.json'), encoding='utf8'))
allowed = {}
for sid, frame in json.load(open(D1, encoding='utf8')):
    allowed.setdefault(sid, set()).update(fr.get(frame, []))
log('门禁角色集: %d 个句子号有允许角色表' % len(allowed))

# 角色名表与训练/预测一致: 按 frame_info.json 里 fes 出现顺序去重
idx2label = []
for line in json.load(open(os.path.join(BASE, 'dataset/frame_info.json'), encoding='utf8')):
    for fes in line['fes']:
        if fes['fe_name'] not in idx2label:
            idx2label.append(fes['fe_name'])
assert len(idx2label) == NL, (len(idx2label), NL)


def decode(Pm, tau, veto):
    out = []
    for i, key in enumerate(K['oof']):
        if Pm[i, NL] > tau:
            continue
        j = int(np.argmax(Pm[i, :NL]))
        name = idx2label[j]
        if veto and (key[0] in allowed) and (name not in allowed[key[0]]):
            continue
        out.append([key[0], key[1], key[2], name])
    return out


def score(triples, tag):
    p = os.path.join(RES, '_fuse3_%s.json' % tag)
    json.dump(triples, open(p, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    r = subprocess.run([sys.executable, 'score_official.py', '--gold', 'dataset/cfn-dev.json',
                        '--task1', os.path.relpath(D1, BASE), '--task2', os.path.relpath(D2, BASE),
                        '--task3', os.path.relpath(p, BASE)], cwd=BASE, capture_output=True, text=True)
    m = RE_RI.search(r.stdout)
    if not m:
        log('FAIL 打分无输出:\n' + (r.stdout[-500:] + r.stderr[-800:]))
        raise SystemExit(1)
    sc = RE_SC.search([l for l in r.stdout.splitlines() if '0.3*FI' in l][0]).group(1)
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), float(sc)


rows = []
for cname, w in CONFIGS:
    Pm = sum(w[n] * P[n] for n in w) / sum(w.values())
    for gate in ('raw', 'veto'):
        best = None
        for tau in TAUS:
            ri, p_, r_, sc = score(decode(Pm, tau, gate == 'veto'), '%s_%s_t%.2f' % (cname, gate, tau))
            rows.append([cname, gate, tau, ri, p_, r_, sc])
            log('%-13s %-4s t=%.2f  RI=%.4f (p %.4f / r %.4f) 综合 %.4f' % (cname, gate, tau, ri, p_, r_, sc))
            if best is None or ri > best[1]:
                best = (tau, ri)
        log('  -> %s %s 最优阈值 %.2f, RI %.4f' % (cname, gate, best[0], best[1]))

with open(OUT, 'w', encoding='utf8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['config', 'gate', 'tau', 'RI', 't3p', 't3r', 'score'])
    w.writerows(rows)
log('汇总写出 -> ' + OUT)

print()
print('=== 每列最优 (RI) ===')
for cname, _ in CONFIGS:
    for gate in ('raw', 'veto'):
        sub = [r for r in rows if r[0] == cname and r[1] == gate]
        b = max(sub, key=lambda r: r[3])
        print('%-13s %-4s best tau=%.2f  RI %.4f  综合 %.4f' % (cname, gate, b[2], b[3], b[6]))
log('FUSE3_DONE')
