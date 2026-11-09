#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R38 companion: task2 member score-scale quantiles and changed-span accounting.

Read-only. Two independent readouts, both needed by the R38 deliverable:
  1) per-member logit distribution quantiles (raw, no offset applied) and the shift that aligns
     each member's median to a chosen reference level;
  2) for a produced cell vs the incumbent cell, how many span-level decisions actually differ.

Run the operator cells themselves with runbash/task2_shift_mean.py; this script never rebuilds an
answer file, it only compares files that were already written.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_official import load_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--lists', required=True, help='comma separated name=path [sid,start,end,logit]')
ap.add_argument('--q', default='0.05,0.10,0.25,0.50,0.75,0.90,0.95')
ap.add_argument('--ref_member', default='', help='member whose median is the alignment target')
ap.add_argument('--incumbent', default='dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json')
ap.add_argument('--cell', action='append', default=[],
                help='label=path of a produced cell to diff against the incumbent; repeatable')
ap.add_argument('--emit_offset', default='',
                help='write the median-aligning --offset argument string to this file (no cells needed)')
A = ap.parse_args()

QS = [float(x) for x in A.q.split(',') if x]


def quantile(sorted_vals, q):
    if not sorted_vals:
        raise SystemExit('[FATAL] empty score list')
    if not 0.0 <= q <= 1.0:
        raise SystemExit('[FATAL] quantile %r outside [0,1]' % q)
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def span_set(path, name):
    rows = load_json(path)
    bad = [r for r in rows if len(r) != 3]
    if bad:
        raise SystemExit('[FATAL] %s: %d rows are not [sid,start,end]; first=%s' % (name, len(bad), bad[0]))
    out = {}
    for sid, st, en in rows:
        out.setdefault(int(sid), set()).add((int(st), int(en)))
    dup = sum(len(v) for v in out.values()) - len({(k, t) for k, v in out.items() for t in v})
    return out, sum(len(v) for v in out.values()), dup


print('[part] 1/2  member score-scale quantiles (raw logits, no offset applied)', flush=True)
MED = {}
print('%-6s %7s %7s %9s %s' % ('member', 'rows', 'spans', 'mean_logit', ' '.join('p%02d' % int(q * 100) for q in QS)), flush=True)
for kv in A.lists.split(','):
    name, path = kv.split('=', 1)
    if not os.path.isfile(path):
        raise SystemExit('[FATAL] missing %s (%s)' % (path, name))
    rows = load_json(path)
    bad = [r for r in rows if len(r) != 4]
    if bad:
        raise SystemExit('[FATAL] %s: %d rows are not [sid,start,end,logit]; first=%s' % (path, len(bad), bad[0]))
    d = {}
    for sid, st, en, sc in rows:
        k = (int(sid), int(st), int(en))
        d[k] = max(d.get(k, -1e9), float(sc))
    vs = sorted(d.values())
    med = quantile(vs, 0.50)
    MED[name] = med
    print('%-6s %7d %7d %9.3f %s' % (name, len(rows), len(d), sum(vs) / len(vs),
                                      ' '.join('%+8.3f' % quantile(vs, q) for q in QS)), flush=True)

REF = A.ref_member or (sorted(MED)[0] if MED else '')
if REF not in MED:
    raise SystemExit('[FATAL] --ref_member %r not among loaded members %s' % (REF, sorted(MED)))
print('[align] reference member %s median %+.3f -> shift applied to each member = its own median minus %s median'
      % (REF, MED[REF], REF), flush=True)
for name in sorted(MED):
    print('  %-6s median %+.3f  offset %+.3f  (i.e. --offset %s=%.3f)' % (name, MED[name], MED[name] - MED[REF], name, MED[name] - MED[REF]), flush=True)

if A.emit_offset:
    with open(A.emit_offset, 'w') as fh:
        fh.write(','.join('%s=%.3f' % (n, MED[n] - MED[REF]) for n in sorted(MED)))
    print('[emit ] %s' % A.emit_offset, flush=True)
    raise SystemExit(0)

print('', flush=True)
if not A.cell:
    raise SystemExit('[FATAL] nothing to diff: pass --cell label=path (or --emit_offset for part 1 only)')
print('[part] 2/2  span-level decisions changed vs incumbent', flush=True)
if not os.path.isfile(A.incumbent):
    raise SystemExit('[FATAL] incumbent cell missing: %s' % A.incumbent)
INC, inc_n, inc_dup = span_set(A.incumbent, 'incumbent')
if inc_dup:
    raise SystemExit('[FATAL] incumbent cell has %d duplicate span rows' % inc_dup)
print('[base ] %s  sentences %d  spans %d' % (os.path.basename(A.incumbent), len(INC), inc_n), flush=True)
for item in A.cell:
    label, path = item.split('=', 1)
    if not os.path.isfile(path):
        raise SystemExit('[FATAL] missing cell %s (%s)' % (path, label))
    C, c_n, c_dup = span_set(path, label)
    if c_dup:
        raise SystemExit('[FATAL] %s has %d duplicate span rows' % (label, c_dup))
    keys = set(INC) | set(C)
    added = removed = diff_sent = 0
    for sid in keys:
        a = INC.get(sid, set())
        b = C.get(sid, set())
        if a != b:
            diff_sent += 1
        added += len(b - a)
        removed += len(a - b)
    universe = inc_n + added
    print('[diff ] %-28s spans %6d | extra %5d | withdrawn %5d | net %+6d | changed %d / %d sentences (%.2f %% of the union of both files)'
          % (label, c_n, added, removed, c_n - inc_n, diff_sent, len(keys), 100.0 * diff_sent / len(keys)), flush=True)
    print('        spans whose answer differs %d (%.2f %% of %d spans in the union of both files: incumbent %d plus %d put in by this cell)'
          % (added + removed, 100.0 * (added + removed) / universe, universe, inc_n, added), flush=True)
print('R38_SCALE_DONE', flush=True)
