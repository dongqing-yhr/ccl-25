#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Turn the production decoder's --save_logits sidecar into one float32 table per member.

Why this exists: R48 (average two members), R37 (vote count / abstention) and R50 (same prior table
train-time vs decode-time) all need, per candidate span, the FULL softmax over the role labels plus
the NULL column. The sidecar already carries exactly that, written by predict_task3.py itself, so
nothing here re-implements the model or the input construction.

Two checks per member, both fatal on failure:
  1. argmax over the dumped row, with "NULL on top means do not submit that row", must reproduce the
     deployment decode file for that member row for row. That proves the table is the same decision
     the recorded reading was scored from.
  2. score_official's own role-side F1, recomputed off that reconstruction, must equal the recorded
     reading passed in --expect. Catches a mis-ordered label table, which check 1 alone can hide if
     every member shares the mistake.

  PYTHONPATH=. python3 runbash/probs_t3_to_npy_1108.py \
      --probs_dir dataset/devladder_c10/probs \
      --arms t3_c10_L0,t3_c10_L1,t3_c10_FN,t3_c10_L3 \
      --expect t3_c10_L0=62.3051,t3_c10_L1=62.9820,t3_c10_FN=62.8143,t3_c10_L3=62.8614

Row alignment across members is checked too: the candidate rows come from the task2 file alone, so
all members of one resource must carry an identical [sentence_id, start, end] list.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

ap = argparse.ArgumentParser()
ap.add_argument('--probs_dir', required=True)
ap.add_argument('--arms', required=True, help='comma separated member names')
ap.add_argument('--gold', default='./dataset/task3_dev.json')
ap.add_argument('--frame_info', default='./dataset/frame_info.json')
ap.add_argument('--expect', default='', help='arm=RI reading, comma separated; empty skips check 2')
A = ap.parse_args()


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def die(m):
    print('[%s] FATAL %s' % (time.strftime('%H:%M:%S'), m), flush=True)
    sys.exit(1)


arms = [s.strip() for s in A.arms.split(',') if s.strip()]
if not arms:
    die('no arms parsed from --arms')
expect = {}
for s in A.expect.split(','):
    s = s.strip()
    if s:
        k, v = s.split('=')
        expect[k] = v

for f in (A.gold, A.frame_info):
    if not os.path.exists(f):
        die('input file missing: %s' % f)

# the role table is built by the dataset class the same way for every member: first-seen order of
# fe_name over frame_info, and the NULL column sits right behind it
idx2label = []
for line in json.load(open(A.frame_info, encoding='utf8')):
    for fes in line['fes']:
        if fes['fe_name'] not in idx2label:
            idx2label.append(fes['fe_name'])
NL = len(idx2label)
log('role labels from %s: %d, NULL column index = %d' % (A.frame_info, NL, NL))

import argparse as _ap
import score_official as so
so.ARGS = _ap.Namespace(expand_words=False)
gold_data = json.load(open(A.gold, encoding='utf8'))
gt = so.build_gold(gold_data)
n_gold = sum(len(v['triples']) for v in gt.values())
log('gold role triples (start, end, role) over %d merged sentences: %d'
    % (len(gt), n_gold))

meta_ref = None
summary = []
for arm in arms:
    side = os.path.join(A.probs_dir, 'probs_%s.json' % arm)
    dec = os.path.join(A.probs_dir, 'dec_%s.json' % arm)
    for f in (side, dec):
        if not os.path.exists(f):
            die('%s missing - the dump stage did not finish this member' % f)
    rows = json.load(open(side, encoding='utf8'))
    bad = [r for r in rows if len(r) != NL + 4]
    if bad:
        die('%s: %d sidecar rows do not carry %d columns (3 index columns + %d role columns '
            'including NULL); first short row has %d columns'
            % (arm, len(bad), NL + 4, NL + 1, len(bad[0])))
    meta = [[int(r[0]), int(r[1]), int(r[2])] for r in rows]
    if meta_ref is None:
        meta_ref = meta
        log('candidate pool from %s: %d rows' % (arm, len(meta)))
    elif meta != meta_ref:
        die('%s candidate list differs from the first member - the tables would not be '
            'averageable' % arm)
    P = np.asarray([r[3:] for r in rows], dtype=np.float32)
    np.save(os.path.join(A.probs_dir, 'P_%s.npy' % arm), P)

    # check 1: re-decode off the table, must equal the deployment decode file
    top = P[:, :NL + 1].argmax(axis=1)
    pred = [[meta[i][0], meta[i][1], meta[i][2], idx2label[int(top[i])]]
            for i in range(len(meta)) if int(top[i]) < NL]
    prod = json.load(open(dec, encoding='utf8'))
    prod_cmp = [[int(r[0]), int(r[1]), int(r[2]), r[3]] for r in prod]
    if pred != prod_cmp:
        diff = [(i, pred[i], prod_cmp[i]) for i in range(min(len(pred), len(prod_cmp)))
                if pred[i] != prod_cmp[i]][:5]
        die('%s: re-decode off the dumped table is NOT the recorded decode '
            '(%d rows rebuilt vs %d recorded, first diffs %s)' % (arm, len(pred), len(prod_cmp), diff))
    n_abst = len(meta) - len(pred)

    # check 2: the role-side F1 recomputed from this reconstruction
    ri, prec, rec = so.score_ri(gt, [list(r) for r in prod_cmp])
    got = '%.4f' % ri
    if arm in expect and got != expect[arm]:
        die('%s: recorded reading is %s but this table gives %s (precision %.4f recall %.4f) '
            '- the table is not the artefact that reading came from'
            % (arm, expect[arm], got, prec, rec))
    summary.append('%s: %d candidates, %d emitted, %d abstained(NULL on top), RI %s '
                   '(p %.4f / r %.4f), P%s bytes'
                   % (arm, len(meta), len(prod_cmp), n_abst, got, prec, rec, P.shape))
    for s in summary[-1:]:
        log(s)

json.dump(meta_ref, open(os.path.join(A.probs_dir, 'meta.json'), 'w'))
json.dump(idx2label, open(os.path.join(A.probs_dir, 'labels.json'), 'w'))
log('wrote meta.json (%d rows) and labels.json (%d roles) for the whole panel'
    % (len(meta_ref), NL))
print('PROBS_TO_NPY_DONE ' + ' | '.join(s.split(':')[0] for s in summary), flush=True)
