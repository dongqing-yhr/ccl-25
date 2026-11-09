#!/usr/bin/env python3
"""Read-only control probe: is gamma's gain from conditioning on the target word?

Same arithmetic and same inputs as runbash/probe_gamma_effect_1027.py (four task-1 members
averaged in probability space, log domain, ALPHA=0.5 Dirichlet shrinkage toward the uniform
frame distribution). Only the prior table changes across three arms:

  C  conditional  log P(frame | this row's target word)   <- what gamma actually uses
  U  unconditional log P(frame)                           <- naive transfer: a class prior
  R  shuffled     log P(frame | some other row's word)    <- keeps the table's shape, breaks the link

Self-check anchors (must match the gamma probe or this file is wrong):
  incumbent accuracy 74.0000, arm C at gamma 0.30 net +16 rows.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(BASE, *a)
DEV_PROBS = P('dataset', 'dev', 'cmp1023', 't1probs')
ALPHA = 0.5
GRID = [round(0.05 * k, 2) for k in range(0, 21)]
SEED = 42


def load(path):
    with open(path, encoding='utf8') as f:
        return json.load(f)


man = load(P(DEV_PROBS, 'members.json'))
idx2label = man['idx2label']
sids = [int(s) for s in man['sids']]
gold = man['gold_frames']
NP, Vn = len(sids), len(idx2label)


def member_path(nm):
    for cand in (nm, nm + '.npy', 'model_task1_' + nm + '.npy'):
        p = os.path.join(DEV_PROBS, cand)
        if os.path.exists(p):
            return p
    raise SystemExit('no table for member ' + nm)


print('[load] members=%s rows=%d labels=%d' % (man['members'], NP, Vn), flush=True)
PF = sum(np.load(member_path(nm)) for nm in man['members']) / float(len(man['members']))
LPeq = np.log(np.maximum(PF, 1e-12))
pred = [idx2label[int(t)] for t in PF.argmax(axis=1)]
acc = lambda out: 100.0 * sum(1 for a, b in zip(out, gold) if a == b) / NP
INC = acc(pred)
print('[load] incumbent frame accuracy %.4f' % INC, flush=True)


def surface(line):
    tg = line.get('target') or []
    if not tg:
        return None
    t = tg[0]
    return line['text'][int(t['start']):int(t['end']) + 1]


dev_lines = load(P('dataset', 'cfn-dev.json'))
assert [int(x['sentence_id']) for x in dev_lines] == sids, 'dev row order differs from members.json'
dev_word = [surface(l) for l in dev_lines]

cnt = defaultdict(Counter)
glob = Counter()
for line in load(P('dataset', 'cfn-train.json')):
    w, fr = surface(line), line.get('frame')
    if w is None or not fr:
        continue
    cnt[w][fr] += 1
    glob[fr] += 1
gsum = float(sum(glob.values()))
V = len(glob)
print('[prior] train word-frame pairs=%d distinct words=%d distinct frames=%d'
      % (int(gsum), len(cnt), V), flush=True)

# ---- arm C: log P(frame | word), shrunk toward uniform over frames (identical to gamma probe)
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

# ---- arm U: log P(frame), same shrinkage target, one row broadcast to every sentence
LPu = np.array([np.log((glob.get(f, 0.0) + ALPHA * (1.0 / V)) / (gsum + ALPHA)) for f in idx2label])

# ---- arm R: arm C with the sentence-to-word link broken (fixed permutation).
# LPc already holds -1e9 on unseen-word rows, so permuting rows carries the mask along: the
# permuted row is still a real word's distribution, just attached to the wrong sentence.
rng = np.random.default_rng(SEED)
perm = rng.permutation(NP)
LPr = LPc[perm]


def run(LP, masked):
    """Return gamma -> (fired, fix, break, acc) for this prior table."""
    out = []
    for g in GRID:
        sc = LPeq + g * LP
        if masked:
            sc = np.where(LP < -1e8, LPeq, sc)
        lab = [idx2label[int(t)] for t in sc.argmax(axis=1)]
        fx = [i for i in range(NP) if lab[i] != pred[i]]
        out.append((g, len(fx),
                    sum(1 for i in fx if lab[i] == gold[i]),
                    sum(1 for i in fx if pred[i] == gold[i]),
                    acc(lab)))
    return out


ARMS = {'C conditional  P(frame|word)': run(LPc, True),
        'U unconditional P(frame)    ': run(np.tile(LPu, (NP, 1)), False),
        'R shuffled     P(frame|other)': run(LPr, True)}

print('\n[anchor] arm C gamma 0.30 should be net +16 (fired 80 / fix 29 / break 13)')
row = [r for r in ARMS['C conditional  P(frame|word)'] if r[0] == 0.30][0]
print('         got: fired %d fix %d break %d net %+d acc %.4f' % (row[1], row[2], row[3], row[2] - row[3], row[4]))
assert abs(row[4] - 74.70) < 0.05, 'arm C anchor mismatch -> table or arithmetic drifted'

print('\n[coverage] rows the prior is defined on (a row it cannot see can never be re-decided)')
print('  C conditional  : %d/%d = %.2f%%' % (seen.sum(), NP, 100.0 * seen.mean()))
print('  U unconditional: %d/%d = 100.00%% (one global frame-frequency row)' % (NP, NP))
print('  R shuffled     : %d/%d = %.2f%% (same count as C, wrong sentence)' % (seen[perm].sum(), NP, 100.0 * seen[perm].mean()))

for nm, res in ARMS.items():
    print('\n=== %s ===' % nm)
    print('  gamma fired  fix break   net     acc   d_acc_vs_incumbent')
    for g, f, fx, br, a in res:
        print('  %5.2f %5d %5d %5d %+6d  %7.4f  %+7.4f' % (g, f, fx, br, fx - br, a, a - INC))
    best = max(res, key=lambda r: r[2] - r[3])
    print('  BEST net %+d at gamma %.2f (acc %.4f, %+0.4f vs incumbent %.4f)'
          % (best[2] - best[3], best[0], best[4], best[4] - INC, INC))

print('\n[summary] net rows gained per arm at its own best gamma')
for nm, res in ARMS.items():
    best = max(res, key=lambda r: r[2] - r[3])
    print('  %s  gamma %.2f  net %+d  acc %.4f' % (nm, best[0], best[2] - best[3], best[4]))
print('CONTROL_PROBE_DONE', flush=True)
