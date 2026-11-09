#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R38 follow-up: is `mean after per-member shift, one threshold` exactly `mean of raw logits with
the threshold moved by offset/N`?

Arithmetic says yes for a span BOTH members scored: (p + (a-o))/2 >= t-o/2  <=>  (p+a)/2 >= t.
It says NO for a span only one member scored, because the patch value used for the missing member
(--fill=-3.0) is a fixed constant that does NOT move with the offset. So a prim-only span keeps
(p-3)/2 in both runs while its threshold drops by o/2 => more prim-only spans pass after alignment.

This script measures which span group the disagreements fall in, so the paper sentence can state
the exact condition under which "alignment is only a threshold shift" holds.

Usage: r38_idcheck_diag_1108.py --lists prim=...,aux=... --offset 1.1588... --pairs t,rawFile,alnFile
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument('--lists', required=True)
ap.add_argument('--offset', type=float, required=True, help='aux offset used for the aligned run')
ap.add_argument('--pair', action='append', required=True,
                help='threshold,raw_cell_file,aligned_cell_file  (repeatable)')
A = ap.parse_args()

M = {}
for kv in A.lists.split(','):
    name, path = kv.split('=', 1)
    if not os.path.isfile(path):
        raise SystemExit('[FATAL] missing %s (%s)' % (path, name))
    d = {}
    for sid, st, en, sc in json.load(open(path)):
        k = (int(sid), int(st), int(en))
        d[k] = max(d.get(k, -1e9), float(sc))
    M[name] = d
    print('[load ] %-5s spans %d' % (name, len(d)), flush=True)
if 'prim' not in M or 'aux' not in M:
    raise SystemExit('[FATAL] this check needs members named prim and aux, got %s' % sorted(M))

BOTH = set(M['prim']) & set(M['aux'])
PONY = set(M['prim']) - set(M['aux'])
AONY = set(M['aux']) - set(M['prim'])
print('[group] both %d  prim-only %d  aux-only %d  union %d'
      % (len(BOTH), len(PONY), len(AONY), len(BOTH | PONY | AONY)), flush=True)

for item in A.pair:
    t, rf, af = item.split(',')
    for f in (rf, af):
        if not os.path.isfile(f):
            raise SystemExit('[FATAL] missing cell %s' % f)
    R = {(int(s), int(a), int(b)) for s, a, b in json.load(open(rf))}
    B = {(int(s), int(a), int(b)) for s, a, b in json.load(open(af))}
    dis = (B - R) | (R - B)
    print('[pair ] raw thr %s -> %d spans ; aligned thr %.6f (= %s - offset/2) -> %d spans ; differ %d (+%d/-%d)'
          % (t, len(R), float(t) - A.offset / 2.0, t, len(B), len(dis), len(B - R), len(R - B)), flush=True)
    for nm, ks in (('both', BOTH), ('prim-only', PONY), ('aux-only', AONY)):
        print('   %-9s pool %5d  differ %4d  (%.2f %% of that pool, %.2f %% of all disagreements)'
              % (nm, len(ks), len(dis & ks), 100.0 * len(dis & ks) / len(ks) if ks else 0.0,
                 100.0 * len(dis & ks) / len(dis) if dis else 0.0), flush=True)
    if dis and not (dis <= PONY):
        print('   [NOTE] disagreements outside the prim-only pool: %d' % len(dis - PONY), flush=True)
    print('   [VERDICT] both-scored subset identical: %s ; aux-only subset identical: %s'
          % (len(dis & BOTH) == 0, len(dis & AONY) == 0), flush=True)
print('IDCHECK_DONE', flush=True)
