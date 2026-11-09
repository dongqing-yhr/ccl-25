#!/usr/bin/env python3
"""R36 equal-strength prior-table control, resource-agnostic (CFN 1.0 / English).

Same arithmetic as baseline/runbash/probe_gamma_control_1027.py, which byte-for-byte
reproduced its 10-27 product byte-for-byte on 2026-11-08.
Three arms, identical strength grid and identical shrinkage:

  C  conditional   log P(frame | this row's target word)   <- what the method uses
  U  unconditional log P(frame)                            <- a class prior, no conditioning
  R  shuffled      log P(frame | some other row's word)    <- table shape kept, link broken

Anchors (a run whose anchor fails is not a reading):
  gamma 0.00 argmax accuracy == the decode chain's ring-1 frame accuracy, and
  gamma 0.00 fired == 0 (nothing re-decided without the prior).
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--probs', required=True, help='npz from predict_task1.py --prob_dump')
ap.add_argument('--dev', required=True, help="this resource's validation file")
ap.add_argument('--train', required=True, help="this resource's training file (prior source)")
ap.add_argument('--pred', required=True, help='ring-1 prediction json, for the anchor')
ap.add_argument('--gold', default=None, help='defaults to --dev')
ap.add_argument('--alpha', type=float, default=0.5)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--step', type=float, default=0.05)
A = ap.parse_args()

ALPHA, SEED = A.alpha, A.seed
GRID = [round(A.step * k, 2) for k in range(int(round(1.0 / A.step)) + 1)]


def load(path):
    with open(path, encoding='utf8') as f:
        return json.load(f)


def surface(line):
    tg = line.get('target') or []
    if not tg:
        return None
    t = tg[0]
    return line['text'][int(t['start']):int(t['end']) + 1]


d = np.load(A.probs, allow_pickle=True)
P = d['probs'].astype(np.float64)
sids = [int(s) for s in d['sentence_id']]
NP = P.shape[0]
idx2label = load(A.probs + '.labels.json')
if isinstance(idx2label, dict):
    idx2label = [idx2label[str(i)] if str(i) in idx2label else idx2label[i] for i in range(len(idx2label))]
Vn = len(idx2label)
assert P.shape[1] == Vn, 'label table width %d != probs width %d' % (Vn, P.shape[1])

dev = load(A.dev)
gold_rows = load(A.gold) if A.gold else dev
assert [int(x['sentence_id']) for x in dev] == sids, 'dev row order differs from the prob table'
gold = [x.get('frame') for x in dev]
dev_word = [surface(l) for l in dev]

pred = load(A.pred)
assert [p[0] for p in pred] == sids, 'ring-1 prediction order differs from the prob table'
inc_file = sum(1 for p, g in zip(pred, gold) if p[1] == g)

LPeq = np.log(np.maximum(P, 1e-12))
arg0 = [idx2label[int(t)] for t in P.argmax(axis=1)]
acc = lambda out: 100.0 * sum(1 for a, b in zip(out, gold) if a == b) / NP
INC = acc(arg0)
acc_file = 100.0 * inc_file / NP
print('[load] rows=%d labels=%d  prob-table argmax acc %.4f  ring-1 file acc %.4f  hits %d/%d'
      % (NP, Vn, INC, acc_file, inc_file, NP), flush=True)
assert arg0 == [p[1] for p in pred], 'argmax of the dumped probabilities != ring-1 prediction'
assert abs(INC - acc_file) < 1e-9, 'anchor accuracy drifted'

cnt = defaultdict(Counter)
glob = Counter()
for line in load(A.train):
    w, fr = surface(line), line.get('frame')
    if w is None or not fr:
        continue
    cnt[w][fr] += 1
    glob[fr] += 1
gsum = float(sum(glob.values()))
V = len(glob)
print('[prior] train word-frame pairs=%d distinct words=%d distinct frames=%d'
      % (int(gsum), len(cnt), V), flush=True)

LPc = np.full((NP, Vn), -1e9, dtype=np.float64)
for i, w in enumerate(dev_word):
    c = cnt.get(w)
    if not c:
        continue
    tot = float(sum(c.values()))
    row = np.zeros(Vn)
    for j, f in enumerate(idx2label):
        row[j] = (c.get(f, 0.0) + ALPHA * (1.0 / V)) / (tot + ALPHA)
    LPc[i] = np.log(row)
seen = LPc[:, 0] > -1e8

LPu = np.array([np.log((glob.get(f, 0.0) + ALPHA * (1.0 / V)) / (gsum + ALPHA)) for f in idx2label])

rng = np.random.default_rng(SEED)
perm = rng.permutation(NP)
LPr = LPc[perm]


def run(LP, masked):
    out = []
    for g in GRID:
        sc = LPeq + g * LP
        if masked:
            sc = np.where(LP < -1e8, LPeq, sc)
        lab = [idx2label[int(t)] for t in sc.argmax(axis=1)]
        fx = [i for i in range(NP) if lab[i] != arg0[i]]
        out.append((g, len(fx),
                    sum(1 for i in fx if lab[i] == gold[i]),
                    sum(1 for i in fx if arg0[i] == gold[i]),
                    acc(lab)))
    return out


ARMS = {'C conditional  P(frame|word)': run(LPc, True),
        'U unconditional P(frame)    ': run(np.tile(LPu, (NP, 1)), False),
        'R shuffled     P(frame|other)': run(LPr, True)}

z = ARMS['C conditional  P(frame|word)'][0]
print('\n[anchor] arm C gamma 0.00 must re-decide nothing: fired %d fix %d break %d acc %.4f'
      % (z[1], z[2], z[3], z[4]))
assert z[1] == 0 and abs(z[4] - INC) < 1e-9, 'gamma 0 anchor failed'

print('\n[coverage] rows the prior is defined on (a row it cannot see can never be re-decided)')
print('  C conditional  : %d/%d = %.2f%%' % (seen.sum(), NP, 100.0 * seen.mean()))
print('  U unconditional: %d/%d = 100.00%% (one global frame-frequency row)' % (NP, NP))
print('  R shuffled     : %d/%d = %.2f%% (same count as C, wrong sentence)'
      % (seen[perm].sum(), NP, 100.0 * seen[perm].mean()))

best = {}
for nm, res in ARMS.items():
    print('\n=== %s ===' % nm)
    print('  gamma fired  fix break   net     acc   d_acc_vs_incumbent')
    for g, f, fx, br, a in res:
        print('   %.2f  %5d %5d %6d  %+5d  %8.4f  %+8.4f' % (g, f, fx, br, fx - br, a, a - INC))
    b = max(res, key=lambda r: r[2] - r[3])
    best[nm] = b
    print('  BEST net %+d at gamma %.2f (acc %.4f, %+0.4f vs incumbent %.4f)'
          % (b[2] - b[3], b[0], b[4], b[4] - INC, INC))

print('\n[summary] net rows gained per arm at its own best gamma')
for nm in ARMS:
    b = best[nm]
    print('  %s  gamma %.2f  net %+d  acc %.4f' % (nm, b[0], b[2] - b[3], b[4]))
print('CONTROL_PROBE_DONE')
