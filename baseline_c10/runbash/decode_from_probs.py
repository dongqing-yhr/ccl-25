#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""CPU decode of dumped member probabilities -> a task3 submission file.

The member grid (runbash/grid_member_search.py) only ever printed scores; this script turns one of
its rows into the delivered prediction file. It re-implements the decode of
runbash/ens_task3.py (the incumbent task-3 recipe) so that the same
rule which was validated on dev is what produces the delivered bytes:

    P = sum(w_m * P_m)                      # each member dumped as (n_candidates, n_roles+1)
    if P[NULL] > tau:            drop       # abstain line
    name = argmax(P[:n_roles]);  conf = max(P[:n_roles])
    if sentence has a legal-role list and name not in it:
        if gate == plain:                        emit
        elif gate == veto:                       drop
        elif gate == rescue and conf >= release: emit
        else:                                    drop
    emit [sentence_id, start, end, name]

`plain` means the frame-legal role list is not consulted at all, `veto` means it blocks hard,
`rescue` means it blocks unless that role's own probability is already >= --release.

Usage (dev, scores the file it writes against dev gold):
  PYTHONPATH=. python3 runbash/decode_from_probs.py \
      --members fn=dataset/dev/cmp1022/probs_union_m0p25_m0p75/fn.npy,la=.../la.npy \
      --weights 0.5/0.5 --tau 0.10 --gate veto \
      --task1 ./dataset/dev/dev_task1_test.json --gold ./dataset/cfn-dev.json \
      --out /tmp/check.json --expect_emitted 5221 --expect_correct 3161 --expect_ri 56.8781
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_official as so  # noqa: E402
from score_official import build_gold, load_json, score_ri  # noqa: E402

# span_to_chars reads a module-level ARGS that only exists under __main__; the grid script sets the
# same default (gold spans are already character indices, no word->char expansion).
so.ARGS = argparse.Namespace(expand_words=False)


def die(msg):
    print('[FATAL] %s' % msg)
    sys.exit(1)


ap = argparse.ArgumentParser()
ap.add_argument('--members', required=True, help='comma separated name=path.npy, order matches --weights')
ap.add_argument('--weights', required=True, help='slash separated, must sum to 1, e.g. 0.45/0.45/0.05/0.05')
ap.add_argument('--tau', type=float, required=True, help='abstain line on the NULL probability')
ap.add_argument('--gate', default='veto', choices=['plain', 'veto', 'rescue'])
ap.add_argument('--release', type=float, default=None, help='role probability that overrides the role list (gate=rescue only)')
ap.add_argument('--meta', required=True, help='meta.json written next by dump_member_probs.py')
ap.add_argument('--labels', required=True, help='labels.json written next by dump_member_probs.py')
ap.add_argument('--task1', required=True, help='task1 frame predictions: gives the frame-legal role list')
ap.add_argument('--frame_roles', default='./dataset/frame_roles.json')
ap.add_argument('--gold', default=None, help='set on dev only; scores the written file with the official RI metric')
ap.add_argument('--out', required=True)
ap.add_argument('--expect_emitted', type=int, default=None, help='assert the number of kept rows')
ap.add_argument('--expect_correct', type=int, default=None, help='dev only: assert the number of rows that match gold')
ap.add_argument('--expect_ri', type=float, default=None, help='dev only: assert RI to 4 decimals')
A = ap.parse_args()
if A.gate == 'rescue' and A.release is None:
    die('gate=rescue needs --release')

names, paths = [], []
for s in A.members.split(','):
    parts = s.split('=')
    if len(parts) != 2:
        die('member spec must be name=path.npy, got "%s"' % s)
    if not os.path.exists(parts[1]):
        die('member %s: no dump at %s' % (parts[0], parts[1]))
    names.append(parts[0])
    paths.append(parts[1])
W = [float(x) for x in A.weights.split('/')]
if len(W) != len(names):
    die('%d weights but %d members (%s)' % (len(W), len(names), ','.join(names)))
if abs(sum(W) - 1.0) > 1e-9:
    die('weights must sum to 1 (got %.6f) - these are probability weights' % sum(W))

meta = load_json(A.meta)
idx2label = load_json(A.labels)
NL = len(idx2label)
P = None
for n, p, w in zip(names, paths, W):
    m = np.load(p)
    if m.shape[0] != len(meta):
        die('%s has %d rows but meta.json has %d - dumps and metadata are from different cells'
            % (n, m.shape[0], len(meta)))
    if m.shape[1] != NL + 1:
        die('%s has %d columns but labels.json lists %d roles (+NULL = %d)'
            % (n, m.shape[1], NL, NL + 1))
    P = w * m if P is None else P + w * m
N = len(meta)

fr = load_json(A.frame_roles)
allowed = {}
for sid, frame in load_json(A.task1):
    allowed.setdefault(int(sid), set()).update(fr.get(frame, []))

top = P[:, :NL].argmax(axis=1)
conf = P[np.arange(N), top]
pnull = P[:, NL]
keep_null = pnull <= A.tau

out = []
n_null = n_veto = n_resc = 0
for i in range(N):
    sid, st, en = meta[i]
    if not keep_null[i]:
        n_null += 1
        continue
    name = idx2label[int(top[i])]
    if sid in allowed and name not in allowed[sid]:
        if A.gate == 'rescue' and conf[i] >= A.release:
            n_resc += 1
        elif A.gate == 'plain':
            pass
        else:
            n_veto += 1
            continue
    out.append([sid, st, en, name])

d = os.path.dirname(os.path.abspath(A.out))
if d and not os.path.isdir(d):
    die('output directory does not exist: %s' % d)
json.dump(out, open(A.out, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
summ = ('members=%s weights=%s tau=%s gate=%s release=%s | 候选%d 弃权%d 挡掉%d 越线放行%d 保留%d -> %s'
        % (','.join(names), A.weights, A.tau, A.gate, A.release, N, n_null, n_veto, n_resc, len(out), A.out))
print('[decode] ' + summ)
if A.expect_emitted is not None and len(out) != A.expect_emitted:
    die('kept %d rows, expected %d - this is not the decode you think it is' % (len(out), A.expect_emitted))
if A.gold:
    gold = build_gold(load_json(A.gold))
    ri, rp, rr = score_ri(gold, out)
    gt = {sid: set(g['triples']) for sid, g in gold.items()}
    ok = sum(1 for p in out if (p[1], p[2], p[3]) in gt.get(p[0], set()))
    summ += ' | RI=%.4f (p %.4f r %.4f) 判对%d/%d' % (ri, rp, rr, ok, len(out))
    print('[score ] ' + summ)
    if A.expect_correct is not None and ok != A.expect_correct:
        die('gold-matching rows = %d, expected %d' % (ok, A.expect_correct))
    if A.expect_ri is not None and abs(ri - A.expect_ri) > 5e-5:
        die('RI = %.4f, expected %.4f' % (ri, A.expect_ri))
elif A.expect_correct is not None or A.expect_ri is not None:
    die('--expect_correct/--expect_ri need --gold')
print('DECODE_DONE', flush=True)
