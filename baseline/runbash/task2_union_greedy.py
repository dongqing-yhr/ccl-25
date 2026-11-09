#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Greedy union of MANY task2 decodes, judged by dev AI.

task2 AI is a character-cover F1, so the only ways to move it are to add chars that hit gold or to
stop emitting chars that miss. fuse_task2_models.py's rule search (task2_rule_search.py) showed no
two-model fusion rule beats union_all on dev, and that union_all's gain comes from added characters
rather than from the extra spans as objects (add_disjoint, which never touches a primary character,
is 0.22 AI WORSE than union_all). So the productive direction is more decodes, unioned: different
boundary refinement radii, different thresholds, different checkpoints -- each is a cheap
predict_task2 run and each contributes characters the others missed.

Forward selection starts from the production decode and repeatedly adds whichever file raises dev AI
the most, stopping when no addition helps. Every file it selects maps to one pasteable A-board
command (see the tail of the output), so the dev result is directly actionable.

    python3 runbash/task2_union_greedy.py --gold dataset/cfn-dev.json \
        --base dataset/dev/dev_task2_test.json \
        --files dataset/dev/cmp1021/new_m05_r0.json,dataset/dev/cmp1021/old_m05_r1.json,...
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_official as so  # noqa: E402
from score_official import build_gold, load_json  # noqa: E402

so.ARGS = argparse.Namespace(expand_words=False)

ap = argparse.ArgumentParser()
ap.add_argument('--gold', default='./dataset/cfn-dev.json')
ap.add_argument('--base', required=True, help='the production task2 file, always included')
ap.add_argument('--files', required=True, help='comma separated candidate decode files')
ap.add_argument('--names', default='', help='optional labels, same order as --files')
ap.add_argument('--boot', type=int, default=1000)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--max_add', type=int, default=6)
ap.add_argument('--out', default='./dataset/dev/cmp1022', help='where to write the chosen union file')
A = ap.parse_args()
random.seed(A.seed)

gold = build_gold(load_json(A.gold))
sids = sorted(gold.keys())


def spans_by_sid(rows):
    d = {}
    for it in rows:
        d.setdefault(int(it[0]), set()).add((int(it[1]), int(it[2])))
    return d


def chars(spans):
    c = set()
    for (s, e) in spans:
        c |= set(range(s, e + 1))
    return c


GC = {sid: gold[sid]['cover'] for sid in sids}
G_TOTAL = sum(len(GC[s]) for s in sids)


def cover_sets(path):
    d = spans_by_sid(load_json(path))
    return {sid: chars(d.get(sid, ())) for sid in sids}


def ai_of(cov):
    inter = pred = 0.0
    for sid in sids:
        pc = cov.get(sid, set())
        inter += len(pc & GC[sid])
        pred += len(pc)
    return 200.0 * inter / (pred + G_TOTAL) if (pred + G_TOTAL) else 0.0, inter, pred


cands = [f.strip() for f in A.files.split(',') if f.strip()]
names = [x.strip() for x in A.names.split(',')] if A.names else [os.path.basename(f) for f in cands]
assert len(names) == len(cands), 'names and files must be the same length'
COV = {}
print('[load] %-34s %9s %8s %8s %7s' % ('file', 'dev AI', 'prec', 'rec', 'chars'))
base_cov = cover_sets(A.base)
ai_b, hit_b, pc_b = ai_of(base_cov)
print('%-34s %9.4f %8.2f %8.2f %7d   <-- BASE (production decode)'
      % (os.path.basename(A.base), ai_b, 100 * hit_b / pc_b, 100 * hit_b / G_TOTAL, pc_b))
for nm, f in zip(names, cands):
    if not os.path.exists(f):
        print('%-34s  MISSING' % nm)
        continue
    COV[nm] = cover_sets(f)
    ai, hit, pc = ai_of(COV[nm])
    print('%-34s %9.4f %8.2f %8.2f %7d' % (nm, ai, 100 * hit / pc, 100 * hit / G_TOTAL, pc))

chosen = [A.base]
cur = dict(base_cov)
cur_ai = ai_b
print('\n[forward selection] each step adds the candidate whose union raises dev AI the most')
trail = [(0, 'base', cur_ai, 0.0)]
for step in range(1, A.max_add + 1):
    best = None
    for nm in COV:
        trial = {sid: cur.get(sid, set()) | COV[nm].get(sid, set()) for sid in sids}
        ai, _, _ = ai_of(trial)
        if best is None or ai > best[1]:
            best = (nm, ai, trial)
    if best is None or best[1] <= cur_ai + 1e-9:
        print('  step %d: nothing helps (best candidate %s would give %.4f <= %.4f) -> stop'
              % (step, best[0] if best else '-', best[1] if best else 0, cur_ai))
        break
    print('  step %d: add %-28s AI=%.4f  d=%+.4f' % (step, best[0], best[1], best[1] - cur_ai))
    chosen.append([f for n, f in zip(names, cands) if n == best[0]][0])
    cur, cur_ai = best[2], best[1]
    trail.append((step, best[0], cur_ai, cur_ai - ai_b))

if len(trail) < 2:
    print('\nNo candidate union improves on the production decode at all: the task2 axis needs new'
          ' decodes (retrain / different seed / different model), not a different combination of'
          ' what is already on disk.')
else:
    # bootstrap the final gain against the production decode, sentence level
    diffs = []
    sids_l = list(sids)
    for _ in range(A.boot):
        draw = [sids_l[random.randrange(len(sids_l))] for _ in range(len(sids_l))]
        h1 = p1 = h0 = p0 = g = 0.0
        for sid in draw:
            a = cur.get(sid, set()); b = base_cov.get(sid, set()); gc = GC[sid]
            h1 += len(a & gc); p1 += len(a)
            h0 += len(b & gc); p0 += len(b); g += len(gc)
        diffs.append(200 * h1 / (p1 + g) - 200 * h0 / (p0 + g))
    diffs.sort()
    lo, hi = diffs[int(0.025 * A.boot)], diffs[int(0.975 * A.boot)]
    p_le0 = sum(1 for d in diffs if d <= 0) / float(A.boot)
    print('\n[final] union of %d files: dev AI=%.4f  dAI=%+.4f  bootstrap 95%% CI [%+.4f, %+.4f]  P(<=0)=%.3f'
          % (len(chosen), cur_ai, cur_ai - ai_b, lo, hi, p_le0))
    print('        direct composite effect = 0.3 * dAI = %+.4f (task2 AI transfers to the leaderboard'
          ' at ~110%%: dev +0.4105 -> A +0.4537)' % (0.3 * (cur_ai - ai_b)))
    os.makedirs(A.out, exist_ok=True)
    sel = spans_by_sid(load_json(A.base))
    for f in chosen[1:]:
        d2 = spans_by_sid(load_json(f))
        for sid, s in d2.items():
            sel.setdefault(sid, set()).update(s)
    flat = [[sid, s, e] for sid in sorted(sel) for (s, e) in sorted(sel[sid])]
    p = os.path.join(A.out, 'task2_union_greedy.json')
    json.dump(flat, open(p, 'w', encoding='utf8'), indent=1, ensure_ascii=False)
    print('        wrote %d spans -> %s' % (len(flat), p))
    print('\n[files to reproduce on A] every one of these is one predict_task2 run on the A test set:')
    for f in chosen:
        print('        %s' % f)
print('GREEDY_UNION_DONE')
