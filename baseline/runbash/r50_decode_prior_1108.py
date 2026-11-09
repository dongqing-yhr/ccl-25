#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""R50 (CPU, zero training): the SAME prior table added at train time vs added at decode time.

Three readings on one resource, all from that resource's own validation split:
  no prior         the member trained without the frame-conditional role prior  (--arm table)
  decode-time      that same member's dumped probabilities, with tau*log pi(frame) added to the role
                   columns right before the argmax                                      (--arm table)
  train-time       the member that had the table folded into its loss during training  (--train_arm)

Why the table is rebuilt by importing the training data loader instead of counting by hand:
train_task3.py:429-453 builds it from train_dataset.data, and which 框架名称 a training span is
keyed by depends on that loader's own switches (with --oof_frame_file the key is an out-of-fold
prediction, without it the gold 框架名称; the deployment-matched NULL negatives also feed the row
and column totals that the Dirichlet smoothing uses). Re-using the loader means the numbers cannot
drift from what the member actually saw, and the printed fingerprint (帧数 / 类别数 / log pi range)
is compared to the [logit_adj] line of that member's own training log - a mismatch stops the run.

The decode-time key is the 框架名称 that 任务一 predicted for this sentence, read the same way
predict_task3.py reads it (file order, later row wins). A sentence whose predicted 框架名称 is not a
row of the table gets no adjustment at all; how many such rows there are is printed.

Adjustment happens on the log scale and the row is softmax-renormalised afterwards, so the NULL
column stays a probability and the abstention rule of --rule is still meaningful. log p and the
model's own logits differ only by a per-row constant, so this is the same arithmetic the training
loss did, not an approximation of it. The dumped sidecar is rounded to 6 decimals, which perturbs
log p by ~1e-6 against an adjustment of order tau*clamp = 1.0.

  PYTHONPATH=. python3 runbash/r50_decode_prior_1108.py --dump dataset/devladder_c10/probs \
      --arm t3_c10_L1 --train_arm t3_c10_L3 --key_file dataset/devladder_c10/dev_task1_test.json \
      --train_file ./dataset/task3_train.json --frame_info ./dataset/frame_info.json \
      --deploy_negatives_file ./dataset/train_deploy_negatives.json \
      --expect_nf 629 --expect_nl 991 --expect_lo -2.00 --expect_hi -0.18 \
      --expect_arm_ri 62.9820 --expect_train_ri 62.8614
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import score_official as so  # noqa: E402
from score_official import build_gold, load_json, score_ri  # noqa: E402
from dataset_task3 import Dataset  # noqa: E402

so.ARGS = argparse.Namespace(expand_words=False)

ap = argparse.ArgumentParser()
ap.add_argument('--dump', required=True)
ap.add_argument('--prefix', default='P_')
ap.add_argument('--arm', required=True, help='member trained WITHOUT the prior')
ap.add_argument('--train_arm', required=True, help='member trained WITH it (train-time arm)')
ap.add_argument('--train_file', required=True)
ap.add_argument('--frame_info', default='./dataset/frame_info.json')
ap.add_argument('--oof_frame_file', default='')
ap.add_argument('--deploy_negatives_file', default='')
ap.add_argument('--key_file', required=True, help='任务一 预测框架名称 [sid, frame, ...]')
ap.add_argument('--tau', type=float, default=0.5)
ap.add_argument('--alpha', type=float, default=1.0)
ap.add_argument('--clamp', type=float, default=2.0)
ap.add_argument('--rule', default='prod', choices=['prod', 'incumbent'])
ap.add_argument('--tau_null', type=float, default=0.10)
ap.add_argument('--gate', default='veto', choices=['plain', 'veto'])
ap.add_argument('--task1', default='', help='frame-legal list source (incumbent rule)')
ap.add_argument('--frame_roles', default='dataset/frame_roles.json')
ap.add_argument('--gold', default=None)
ap.add_argument('--expect_nf', type=int, default=0)
ap.add_argument('--expect_nl', type=int, default=0)
ap.add_argument('--expect_lo', type=float, default=0.0)
ap.add_argument('--expect_hi', type=float, default=0.0)
ap.add_argument('--expect_arm_ri', type=float, default=0.0)
ap.add_argument('--expect_train_ri', type=float, default=0.0)
A = ap.parse_args()


def log(m):
    print('[%s] %s' % (time.strftime('%FT%TZ', time.gmtime()), m), flush=True)


def die(m):
    print('[FATAL] %s' % m, flush=True)
    sys.exit(1)


def table_path(name):
    for cand in (os.path.join(A.dump, A.prefix + name + '.npy'),
                 os.path.join(A.dump, name + '.npy')):
        if os.path.exists(cand):
            return cand
    die('no probability table for member %s in %s' % (name, A.dump))


for f in (os.path.join(A.dump, 'meta.json'), os.path.join(A.dump, 'labels.json'),
          A.train_file, A.frame_info, A.key_file):
    if not os.path.exists(f):
        die('input file missing: %s' % f)
gold = A.gold or ('dataset/cfn-dev.json' if A.rule == 'incumbent' else './dataset/task3_dev.json')
if not os.path.exists(gold):
    die('input file missing: %s' % gold)

meta = load_json(os.path.join(A.dump, 'meta.json'))
idx2label = load_json(os.path.join(A.dump, 'labels.json'))
R = len(idx2label)                    # role columns; NULL sits at index R
W = R + 1

# ---- 1. rebuild the table exactly the way the trainer built it ------------------------------
# the loader only needs the tokenizer inside __getitem__, which this script never calls - building
# the prior table reads ds.data, so no vocabulary is required here.
ds = Dataset(A.train_file, A.frame_info, None,
             oof_frame_file=(A.oof_frame_file or None),
             deploy_negatives_file=(A.deploy_negatives_file or None))
NF = len(ds.frame2idx)
cnt = np.zeros((NF, W))
for d in ds.data:
    f = ds.frame2idx.get(d.get('frame'), -1)
    c = d['label_class']
    if 0 <= f < NF and 0 <= c < W:
        cnt[f, c] += 1
tot = cnt.sum()
glob = cnt.sum(0) / max(tot, 1.0)
row = cnt.sum(1, keepdims=True)
p = (cnt + A.alpha * glob) / (row + A.alpha)
logpi = np.log(p + 1e-12)
if A.clamp > 0:
    logpi = np.clip(logpi, -A.clamp, 0.0)
logpi[:, R] = 0.0
log('先验表：帧数=%d 类别数=%d（含 NULL）配对计数总量=%d 有角色的计数=%d'
    % (NF, W, int(tot), int(cnt[:, :R].sum())))
log('log pi 范围 [%.2f, %.2f]（NULL 列已置 0），tau=%s alpha=%s clamp=%s'
    % (logpi[:, :R].min(), logpi[:, :R].max(), A.tau, A.alpha, A.clamp))
if A.expect_nf and NF != A.expect_nf:
    die('帧数=%d but the training log says %d - this is not the table that member was trained with'
        % (NF, A.expect_nf))
if A.expect_nl and W != A.expect_nl:
    die('类别数=%d but the training log says %d - mixed label space' % (W, A.expect_nl))
if A.expect_lo and abs(round(logpi[:, :R].min(), 2) - A.expect_lo) > 0.011:
    die('log pi min %.2f vs training log %.2f - the smoothing/count path differs, stop'
        % (round(logpi[:, :R].min(), 2), A.expect_lo))
if A.expect_hi and abs(round(logpi[:, :R].max(), 2) - A.expect_hi) > 0.011:
    die('log pi max %.2f vs training log %.2f - the smoothing/count path differs, stop'
        % (round(logpi[:, :R].max(), 2), A.expect_hi))
log('指纹与训练日志的 [logit_adj] 行一致')

# ---- 2. decode-time key: the 框架名称 任务一 predicted for this sentence ---------------------
sid2frame = {}
for row_ in load_json(A.key_file):
    sid2frame[int(row_[0])] = row_[1]
fidx = np.full(len(meta), -1, dtype=np.int64)
unknown_sid = 0
for i, m in enumerate(meta):
    fr = sid2frame.get(int(m[0]))
    if fr is None:
        unknown_sid += 1
        continue
    fidx[i] = ds.frame2idx.get(fr, -1)
cov = int((fidx >= 0).sum())
log('解码期键来自 %s：%d 个候选里 %d 拿到了表中的框架名称（%.2f%%），'
    '句子上没有任务一记录 %d 行，键在表外（训练里没见过这个框架名称）%d 行'
    % (A.key_file, len(meta), cov, 100.0 * cov / len(meta), unknown_sid,
       int(((fidx < 0) & np.array([int(m[0]) in sid2frame for m in meta])).sum())))

# ---- 3. the three readings -------------------------------------------------------------------
allowed = {}
if A.rule == 'incumbent':
    for f in (A.task1, A.frame_roles):
        if not os.path.exists(f):
            die('the incumbent rule needs %s' % f)
    fr = load_json(A.frame_roles)
    for sid, frame in load_json(A.task1):
        allowed.setdefault(int(sid), set()).update(fr.get(frame, []))


def decide(mat):
    n = mat.shape[0]
    if A.rule == 'prod':
        top = mat.argmax(axis=1)
        name = np.empty(n, dtype=object)
        name[:] = None
        for i in np.nonzero(top < R)[0]:
            name[i] = idx2label[int(top[i])]
        return name
    top = mat[:, :R].argmax(axis=1)
    name = np.empty(n, dtype=object)
    name[:] = None
    for i in range(n):
        if mat[i, R] > A.tau_null:
            continue
        cand = idx2label[int(top[i])]
        sid = int(meta[i][0])
        if A.gate == 'veto' and sid in allowed and cand not in allowed[sid]:
            continue
        name[i] = cand
    return name


def rows_of(nm):
    return [[meta[i][0], meta[i][1], meta[i][2], nm[i]] for i in range(len(meta))
            if nm[i] is not None]


def correct_idx(nm):
    return set(i for i in range(len(meta)) if nm[i] is not None
               and (meta[i][1], meta[i][2], nm[i]) in gset.get(int(meta[i][0]), set()))


def report(nm, tag, expect):
    rows = rows_of(nm)
    ri, rp, rr = score_ri(gt, rows)
    ok = sum(1 for q in rows if (q[1], q[2], q[3]) in gset.get(q[0], set()))
    log('%-24s 提交 %5d 行  判对 %5d 行  严格匹配 F1 %8.4f  (查准 %.4f 查全 %.4f)'
        % (tag, len(rows), ok, ri, rp, rr))
    if expect and abs(ri - expect) > 5e-4:
        die('%s gives F1 %.4f but the recorded reading is %.4f - the table or the rule is not the '
            'one that reading came from, stop' % (tag, ri, expect))
    return ri, correct_idx(nm)


gt = build_gold(load_json(gold))
gset = {sid: set(g['triples']) for sid, g in gt.items()}
log('金标：%d 句、%d 个 (起,止,角色名称) 三元组（下面每个查准／查全的分母）'
    % (len(gset), sum(len(v) for v in gset.values())))

Pa = np.load(table_path(A.arm))
if Pa.shape != (len(meta), W):
    die('%s table is %s, expected %s' % (A.arm, Pa.shape, (len(meta), W)))
base = np.log(Pa.astype(np.float64) + 1e-12)
S = base.copy()
S[:, :R] += A.tau * logpi[fidx.clip(min=0), :R]
S[fidx < 0, :R] = base[fidx < 0, :R]
mx = S.max(axis=1, keepdims=True)
ex = np.exp(S - mx)
Pdec = (ex / ex.sum(axis=1, keepdims=True)).astype(np.float32)

nmA = decide(Pa)
nmD = decide(Pdec)
ri0, c0 = report(nmA, 'no prior (%s)' % A.arm, A.expect_arm_ri)
ri1, c1 = report(nmD, 'decode-time prior', 0)
Pb = np.load(table_path(A.train_arm))
if Pb.shape != (len(meta), W):
    die('%s table is %s, expected %s' % (A.train_arm, Pb.shape, (len(meta), W)))
nmB = decide(Pb)
ri2, c2 = report(nmB, 'train-time (%s)' % A.train_arm, A.expect_train_ri)
log('差分：解码期加 对 不加 %+0.4f；训练期加 对 不加 %+0.4f；解码期加 对 训练期加 %+0.4f'
    % (ri1 - ri0, ri2 - ri0, ri1 - ri2))

chg = [i for i in range(len(meta)) if nmA[i] != nmD[i]]
log('解码期加法动了多少行：提交的角色名称或提交与否变了的 %d 个候选里占 %d（%.2f%%）'
    % (len(meta), len(chg), 100.0 * len(chg) / len(meta)))
g = c1 - c0
l = c0 - c1
log('这一批里判对多出来的行 = %d，判对丢掉的行 = %d，净 %+d（分母同一批 %d 个候选）'
    % (len(g), len(l), len(g) - len(l), len(meta)))
log('不加与训练期加都判对的行里解码期加丢掉的 = %d；两档都判错、解码期加新判对的 = %d'
    % (len((c0 & c2) - c1), len(c1 - c0 - c2)))
print('R50_DONE', flush=True)
