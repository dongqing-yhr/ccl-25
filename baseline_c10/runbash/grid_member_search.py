#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""CPU-only grid over an ARBITRARY member list of the task3 role decoder, scored on DEV.

Why this exists: runbash/grid_fuse3_search.py hard-codes exactly three members (oof/la/fn) and only
knows the hard frame-legal-role gate. The T1 arms add new members, and the recorded dev gain for the
"role is illegal but its probability is high enough, so keep it anyway" gate is not in that grid
either. This script takes any number of probability dumps and any weight vector over them.

A member file is one float array of shape (n_candidates, n_roles + 1): the role probabilities of
that model for every candidate span task2 proposed, plus the NULL (abstain) probability in the last
column. Averaging members means averaging those rows, so a one-member grid with weight 1.0 is just
that model alone.

Decode rules are the frozen ones, in this order: abstain when p_null > tau; otherwise take the
argmax role; then apply the gate. gate=plain keeps it, gate=veto drops it when the predicted frame
of that sentence does not list the role, gate=rescue drops it too unless that role's own
probability is at least --rescue_min (so a very confident illegal role survives).

Scoring uses score_official.score_ri, and composite is rebuilt analytically as
    0.3*FI + 0.3*AI + 0.4*RI
because FI is fixed by the task1 file and AI by the task2 file - only RI moves with the grid point.

  PYTHONPATH=. python3 runbash/grid_member_search.py --selfcheck
  PYTHONPATH=. python3 runbash/grid_member_search.py \
      --members fn=dataset/dev/cmp1022/probs_union_m0p25_m0p75/fn.npy,la=...,neg=... \
      --weights auto,1/0/0,0.5/0.5/0 --tau_csv 0.06,0.08,0.10,0.12 --out dataset/dev/cmp1022/grid_members.csv

--selfcheck decodes the already-stored two-member average as a single member and must land on the
recorded frozen cell (5221 emitted rows, dev-RI 56.8781). It is the gate that says the rules above
are the same rules the submission was built with; run it before reading any other row of output.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_official as so  # noqa: E402
from score_official import build_gold, load_json, score_ai, score_fi, score_ri  # noqa: E402

# score_official.span_to_chars reads the module-level ARGS that only exists under __main__; the
# char-index convention is what every other script in this repo uses, so pin it here.
so.ARGS = argparse.Namespace(expand_words=False)

DUMP = 'dataset/dev/cmp1022/audit_union_m0p25_m0p75'

ap = argparse.ArgumentParser()
ap.add_argument('--members', default=None,
                help='comma separated name=path.npy list; paths must be per-member probability dumps')
ap.add_argument('--weights', default='auto',
                help='comma separated weight vectors, "0.5/0.5/0" style, one per grid point; '
                     '"auto" adds the equal-weight point')
ap.add_argument('--tau_csv', default='-1,0.0,0.02,0.04,0.06,0.08,0.10,0.12,0.15,0.20',
                help='abstain lines; -1 means the baseline rule (abstain when NULL is top-1), '
                     'which is what the trainer selects on')
ap.add_argument('--gates', default='veto,plain,rescue')
ap.add_argument('--rescue_csv', default='0.90,0.92,0.95', help='rescue_min values for gate=rescue')
ap.add_argument('--meta', default=DUMP + '.meta.json')
ap.add_argument('--labels', default=DUMP + '.labels.json')
ap.add_argument('--task2', default='dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json')
ap.add_argument('--task1', default='./dataset/dev/dev_task1_test.json')
ap.add_argument('--gold', default='./dataset/cfn-dev.json')
ap.add_argument('--frame_info', default='./dataset/frame_info.json')
ap.add_argument('--frame_roles', default='./dataset/frame_roles.json')
ap.add_argument('--out', default='dataset/dev/cmp1022/grid_members.csv')
ap.add_argument('--cell', default='union_m0p25_m0p75')
ap.add_argument('--selfcheck', action='store_true',
                help='decode %s.prob.npy alone and assert 5221 rows / RI 56.8781' % DUMP)
A = ap.parse_args()

t0 = time.time()


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def die(m):
    print('[%s] FATAL %s' % (time.strftime('%H:%M:%S'), m), flush=True)
    sys.exit(1)


if A.selfcheck:
    A.members = 'avg=' + DUMP + '.prob.npy'
    A.weights = '1.0'
    A.tau_csv = '0.10'
    A.gates = 'veto,rescue'
    A.rescue_csv = '0.92'
    A.out = '/tmp/grid_selfcheck.csv'
    log('selfcheck: single member %s, tau=0.10, gates veto + rescue@0.92' % (DUMP + '.prob.npy'))

if not A.members:
    die('nothing to do: pass --members name=path.npy,... (or --selfcheck)')

# ---------------------------------------------------------------- gold / frames
gold_data = load_json(A.gold)
gold = build_gold(gold_data)
pred1 = load_json(A.task1)
pred2 = load_json(A.task2)
fi, fc, ft = score_fi(gold_data, pred1)
ai, aip, air = score_ai(gold, pred2)
log('cell=%s  FI=%.4f (%d/%d)  AI=%.4f (p %.4f / r %.4f)  task2_spans=%d'
    % (A.cell, fi, fc, ft, ai, aip, air, len(pred2)))
if fc == 0:
    die('FI hits = 0 -> the task1 file does not align with this gold; you are scoring A-board '
        'output against dev gold.')

idx2label = []
for line in load_json(A.frame_info):
    for fes in line['fes']:
        if fes['fe_name'] not in idx2label:
            idx2label.append(fes['fe_name'])
fr = load_json(A.frame_roles)
allowed = {}
for sid, frame in pred1:
    allowed.setdefault(int(sid), set()).update(fr.get(frame, []))
log('roles in the table: %d, sentences with a legal-role list: %d' % (len(idx2label), len(allowed)))

# ---------------------------------------------------------------- members
names, mats = [], []
for spec in A.members.split(','):
    if '=' not in spec:
        die('member spec must be name=path.npy, got "%s"' % spec)
    nm, path = spec.split('=', 1)
    if not os.path.exists(path):
        die('member %s: no probability dump at %s' % (nm, path))
    m = np.load(path).astype(np.float64)
    log('[load] %-6s %s  shape=%s' % (nm, path, m.shape))
    names.append(nm)
    mats.append(m)
N = len(mats[0])
for nm, m in zip(names, mats):
    if m.shape != mats[0].shape:
        die('member %s has shape %s, first member has %s - they were not decoded from the same '
            'task2 span file, refusing to average' % (nm, m.shape, mats[0].shape))
NL = mats[0].shape[1] - 1
if NL != len(idx2label):
    die('probability has %d role columns but the role table has %d - wrong label space'
        % (NL, len(idx2label)))

meta = load_json(A.meta)
if len(meta) != N:
    die('meta has %d rows but the dumps have %d - the dump belongs to another task2 cell'
        % (len(meta), N))
keys = [(int(r[0]), int(r[1]), int(r[2])) for r in meta]
sids = [k[0] for k in keys]
spans = [(k[1], k[2]) for k in keys]

gold_triples = {sid: set(g['triples']) for sid, g in gold.items()}
n_gold_total = sum(len(v) for v in gold_triples.values())
log('candidates=%d  gold triples=%d over %d annotated sentences'
    % (N, n_gold_total, len(gold_triples)))

# ---------------------------------------------------------------- grid
WEIGHTS = []
for w in A.weights.split(','):
    w = w.strip()
    if not w:
        continue
    if w == 'auto':
        WEIGHTS.append([1.0 / len(names)] * len(names))
        continue
    parts = [float(x) for x in w.split('/')]
    if len(parts) != len(names):
        die('weight vector "%s" has %d values but there are %d members (%s)'
            % (w, len(parts), len(names), ','.join(names)))
    if abs(sum(parts) - 1.0) > 1e-9:
        die('weight vector "%s" must sum to 1 (got %.6f) - these are probability weights' % (w, sum(parts)))
    WEIGHTS.append(parts)
if not WEIGHTS:
    die('no usable weight vectors in --weights %r' % A.weights)
TAUS = [float(x) for x in A.tau_csv.split(',') if x.strip()]
GATES = [g.strip() for g in A.gates.split(',') if g.strip()]
for g in GATES:
    if g not in ('plain', 'veto', 'rescue'):
        die('unknown gate "%s" - allowed: plain, veto, rescue' % g)
RESCUE = [float(x) for x in A.rescue_csv.split(',') if x.strip()] if 'rescue' in GATES else []
log('grid: %d weight point(s) x %d tau x gates %s%s'
    % (len(WEIGHTS), len(TAUS), GATES, (' x rescue_min %s' % RESCUE) if RESCUE else ''))

COLS = ['cell', 'members', 'weights', 'tau', 'gate', 'rescue_min', 'RI', 'ri_p', 'ri_r',
        'composite', 'emitted', 'abstain', 'vetoed', 'correct', 'fi', 'ai']
new = not os.path.exists(A.out)
fh = open(A.out, 'a', encoding='utf8', newline='')
wtr = csv.DictWriter(fh, fieldnames=COLS)
if new:
    wtr.writeheader()
    fh.flush()

rows_out = []
best = None
best_comp = -1.0
for wv in WEIGHTS:
    Pm = sum(w * m for w, m in zip(wv, mats))
    top = Pm[:, :NL].argmax(axis=1)
    pnull = Pm[:, NL]
    conf = Pm[np.arange(N), top]
    names_row = [idx2label[j] for j in top]
    legal = [(sids[i] not in allowed) or (names_row[i] in allowed[sids[i]]) for i in range(N)]
    for gate in GATES:
        rescues = RESCUE if gate == 'rescue' else [0.0]
        for rmin in rescues:
            for tau in TAUS:
                pred3 = []
                n_null = n_veto = 0
                # tau < 0 is the baseline rule instead of a scalar line: abstain when NULL itself is
                # the top-1 class. That is exactly the metric train_task3.py selects its checkpoint
                # on, so a member can be compared with the recorded epoch numbers of the T1 arms.
                if tau < 0:
                    keep = pnull <= Pm[:, :NL].max(axis=1)
                else:
                    keep = pnull <= tau
                for i in range(N):
                    if not keep[i]:
                        n_null += 1
                        continue
                    if gate != 'plain' and not legal[i]:
                        if gate == 'rescue' and conf[i] >= rmin:
                            pass
                        else:
                            n_veto += 1
                            continue
                    pred3.append([sids[i], spans[i][0], spans[i][1], names_row[i]])
                ri, rip, rir = score_ri(gold, pred3)
                n_correct = sum(1 for p in pred3
                                if (p[1], p[2], p[3]) in gold_triples.get(p[0], set()))
                comp = 0.3 * fi + 0.3 * ai + 0.4 * ri
                rec = {'cell': A.cell, 'members': '+'.join(names),
                       'weights': '/'.join('%.3f' % x for x in wv),
                       'tau': '%.3f' % tau, 'gate': gate,
                       'rescue_min': ('%.2f' % rmin) if gate == 'rescue' else '-',
                       'RI': '%.4f' % ri, 'ri_p': '%.4f' % rip, 'ri_r': '%.4f' % rir,
                       'composite': '%.4f' % comp, 'emitted': len(pred3),
                       'abstain': n_null, 'vetoed': n_veto, 'correct': n_correct,
                       'fi': '%.4f' % fi, 'ai': '%.4f' % ai}
                wtr.writerow(rec)
                rows_out.append(rec)
                if comp > best_comp:
                    best, best_comp = rec, comp
                log('  w=%s tau=%.3f %-6s rmin=%s RI=%.4f (p %.4f/r %.4f) comp=%.4f '
                    'emitted=%d abstain=%d vetoed=%d correct=%d'
                    % (rec['weights'], tau, gate, rec['rescue_min'], ri, rip, rir, comp,
                       len(pred3), n_null, n_veto, n_correct))
fh.close()

log('')
log('=== BEST of this run: w=%s tau=%s gate=%s rmin=%s RI=%s composite=%s emitted=%s correct=%s ==='
    % (best['weights'], best['tau'], best['gate'], best['rescue_min'], best['RI'],
       best['composite'], best['emitted'], best['correct']))
srt = sorted(rows_out, key=lambda r: -float(r['composite']))
log('=== top 8 by composite ===')
for r in srt[:8]:
    log('  w=%-18s tau=%-6s %-6s rmin=%-5s RI=%-9s comp=%-9s emitted=%s correct=%s'
        % (r['weights'], r['tau'], r['gate'], r['rescue_min'], r['RI'], r['composite'],
           r['emitted'], r['correct']))

if A.selfcheck:
    r = [x for x in rows_out if x['gate'] == 'veto']
    if len(r) != 1:
        die('selfcheck produced %d veto-gate rows, expected 1' % len(r))
    r = r[0]
    log('selfcheck frozen cell: emitted=%s abstain=%s vetoed=%s correct=%s of gold=%d, RI=%s'
        % (r['emitted'], r['abstain'], r['vetoed'], r['correct'], n_gold_total, r['RI']))
    if int(r['emitted']) != 5221:
        die('SELFCHECK FAIL: emitted %s rows, the frozen cell has 5221' % r['emitted'])
    if abs(float(r['RI']) - 56.8781) > 0.01:
        die('SELFCHECK FAIL: RI %s, the frozen cell is 56.8781' % r['RI'])
    log('SELFCHECK OK - these decode rules reproduce the frozen submitted cell (5221 rows, 56.8781)')
    r2 = [x for x in rows_out if x['gate'] == 'rescue']
    if r2:
        r2 = r2[0]
        # 57.4662 is the recorded dev-RI of the same dump with the illegal-role rescue line at 0.92
        # (the +0.5881 gain measured in the drop audit). A mismatch here is reported, not fatal:
        # the assertion above is what certifies the decode rules.
        log('selfcheck rescue@0.92: RI=%s emitted=%s correct=%s  (recorded 57.4662 / %s delta)'
            % (r2['RI'], r2['emitted'], r2['correct'],
               '%+.4f' % (float(r2['RI']) - float(r['RI']))))
        if abs(float(r2['RI']) - 57.4662) > 0.01:
            log('WARNING rescue point does not match the recorded 57.4662 - re-check the gate '
                'definition before trusting any rescue row below')

log('GRID_MEMBERS_DONE %s  (%d points, %.0fs, -> %s)' % (A.cell, len(rows_out), time.time() - t0, A.out))
