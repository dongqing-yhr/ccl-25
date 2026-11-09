#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R48 (CPU, zero training): two-member probability average vs each member alone.

Same decode rule as the incumbent recipe (abstain line on the NULL column, frame-legal role list
blocks hard), only the weight vector changes. Writes the row-level source account next to the three
F1 readings so the paper table can say how many rows the average actually moved.

Anchor: fn+la at 0.5/0.5 must reproduce 5221 kept rows / 3161 gold-matching rows / RI 56.8781.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_official as so  # noqa: E402
from score_official import build_gold, load_json, score_ri  # noqa: E402

so.ARGS = argparse.Namespace(expand_words=False)


def log(msg):
    print('[%s] %s' % (time.strftime('%FT%TZ', time.gmtime()), msg), flush=True)


def die(msg):
    print('[FATAL] %s' % msg, flush=True)
    sys.exit(1)


ap = argparse.ArgumentParser()
ap.add_argument('--dump', default='dataset/dev/cmp1022/probs_union_m0p25_m0p75')
ap.add_argument('--pair', default='fn,la')
ap.add_argument('--task1', default='dataset/dev/dev_task1_test.json')
ap.add_argument('--gold', default='dataset/cfn-dev.json')
ap.add_argument('--frame_roles', default='dataset/frame_roles.json')
ap.add_argument('--tau', type=float, default=0.10)
ap.add_argument('--gate', default='veto', choices=['plain', 'veto'])
ap.add_argument('--expect_emitted', type=int, default=5221)
ap.add_argument('--expect_correct', type=int, default=3161)
ap.add_argument('--expect_ri', type=float, default=56.8781)
A = ap.parse_args()

a, b = A.pair.split(',')
meta_path = os.path.join(A.dump, 'meta.json')
labels_path = os.path.join(A.dump, 'labels.json')
for p in (meta_path, labels_path, os.path.join(A.dump, a + '.npy'), os.path.join(A.dump, b + '.npy')):
    if not os.path.exists(p):
        die('missing input: %s' % p)

meta = load_json(meta_path)
idx2label = load_json(labels_path)
NL = len(idx2label)
Pa = np.load(os.path.join(A.dump, a + '.npy'))
Pb = np.load(os.path.join(A.dump, b + '.npy'))
if Pa.shape != Pb.shape:
    die('member tables differ: %s %s vs %s %s' % (a, Pa.shape, b, Pb.shape))
if Pa.shape[0] != len(meta):
    die('%s has %d rows but meta.json has %d' % (a, Pa.shape[0], len(meta)))
if Pa.shape[1] != NL + 1:
    die('%s has %d columns, labels.json lists %d roles (+NULL)' % (a, Pa.shape[1], NL))
log('tables: %s/%s shape=%s, %d role columns + NULL, %d candidates' % (a, b, Pa.shape, NL, len(meta)))

fr = load_json(A.frame_roles)
allowed = {}
for sid, frame in load_json(A.task1):
    allowed.setdefault(int(sid), set()).update(fr.get(frame, []))

gold = build_gold(load_json(A.gold))
gt = {sid: set(g['triples']) for sid, g in gold.items()}
n_gold = sum(len(v) for v in gt.values())
log('gold: %d sentences, %d (start,end,role) triples' % (len(gt), n_gold))


def decide(P):
    """Return per-candidate emitted role name, or None when the row is not submitted."""
    top = P[:, :NL].argmax(axis=1)
    conf = P[np.arange(P.shape[0]), top]
    keep_null = P[:, NL] <= A.tau
    name = np.empty(P.shape[0], dtype=object)
    name[:] = None
    for i in range(P.shape[0]):
        if not keep_null[i]:
            continue
        cand = idx2label[int(top[i])]
        sid = int(meta[i][0])
        if cand not in allowed.get(sid, set()) and sid in allowed:
            if A.gate == 'veto':
                continue
        name[i] = cand
    return name


def rows_of(nm):
    out = []
    for i in range(len(meta)):
        if nm[i] is not None:
            out.append([meta[i][0], meta[i][1], meta[i][2], nm[i]])
    return out


def score(nm, tag):
    rows = rows_of(nm)
    ri, rp, rr = score_ri(gold, rows)
    ok = sum(1 for p in rows if (p[1], p[2], p[3]) in gt.get(p[0], set()))
    log('%-14s emitted %5d  correct %5d  RI %.4f  (p %.4f r %.4f)' % (tag, len(rows), ok, ri, rp, rr))
    return rows, ok, ri


log('rule: tau=%s gate=%s  weights: single-%s / single-%s / 0.5-0.5 average' % (A.tau, A.gate, a, b))
na = decide(Pa)
nb = decide(Pb)
navg = decide(0.5 * Pa + 0.5 * Pb)

ra, oka, ria = score(na, 'single ' + a)
rb, okb, rib = score(nb, 'single ' + b)
rv, okv, riv = score(navg, 'average')

if abs(riv - A.expect_ri) > 5e-5 or okv != A.expect_correct or len(rv) != A.expect_emitted:
    die('anchor drifted: average gives %d rows / %d correct / RI %.4f, expected %d / %d / %.4f. '
        'Do not read differences off this table.'
        % (len(rv), okv, riv, A.expect_emitted, A.expect_correct, A.expect_ri))
log('anchor OK: the average arm reproduces %d rows / %d correct / RI %.4f'
    % (A.expect_emitted, A.expect_correct, A.expect_ri))

# row-level source account; the denominator of every percentage below is the candidate pool
correct = {}
for tag, nm in ((a, na), (b, nb), ('avg', navg)):
    correct[tag] = set(i for i in range(len(meta)) if nm[i] is not None
                       and (meta[i][1], meta[i][2], nm[i]) in gt.get(int(meta[i][0]), set()))
    log('correct-set %s = %d rows' % (tag, len(correct[tag])))

inter_ab = correct[a] & correct[b]
log('overlap of the two single-member correct sets = %d rows '
    '(a only %d, b only %d, union %d)'
    % (len(inter_ab), len(correct[a] - correct[b]), len(correct[b] - correct[a]),
       len(correct[a] | correct[b])))


for solo in (a, b):
    g = correct['avg'] - correct[solo]
    l = correct[solo] - correct['avg']
    log('average vs single %s : moved to right %d rows, moved to wrong %d rows, net %+d'
        % (solo, len(g), len(l), len(g) - len(l)))

gains_both = correct['avg'] - correct[a] - correct[b]
log('rows the average got right that BOTH singles missed = %d' % len(gains_both))
losses_both = (correct[a] & correct[b]) - correct['avg']
log('rows BOTH singles got right that the average lost = %d' % len(losses_both))

changed = [i for i in range(len(meta)) if navg[i] != na[i] or navg[i] != nb[i]]
log('rows whose submitted role name changed under the average = %d of %d candidates (%.2f%%)'
    % (len(changed), len(meta), 100.0 * len(changed) / len(meta)))


def wdemo(i):
    return (meta[i][0], meta[i][1], meta[i][2], na[i], nb[i], navg[i])


sample = [wdemo(i) for i in (correct['avg'] - correct[a] - correct[b])]
log('gains-vs-both sample (sid,start,end,%s,%s,avg): %s' % (a, b, sample[:10]))
sample2 = [wdemo(i) for i in sorted(correct[a] & correct[b] - correct['avg'])]
log('losses-vs-both sample (sid,start,end,%s,%s,avg): %s' % (a, b, sample2[:10]))
print('R48_DONE', flush=True)
