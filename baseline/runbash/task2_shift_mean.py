#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N6: an aligned-average task-2 fusion operator -- per-member offset, arithmetic mean, one threshold.

The reference recipe this imitates (a competing team's unpublished write-up): for each of the four span-score matrices do
`mat -= offset` (offsets -0.30/+0.30/0.00/0.00 in the order rb8/rb6/rb15/mc1), average the four
shifted matrices, then emit every (start,end) whose averaged value clears ONE threshold.

Our sidecars are row lists [sid,start,end,logit] dumped at one low floor, not dense matrices, so a
span a member did not emit has no value at all. Their dense matrix would have used the real (low)
logit for it. `--fill` picks how to patch that gap:
  present (default) -- average over the members that did emit this span (their dense value was high)
  <float>           -- treat a missing span as if that member had scored it this logit
Both are approximations of a dense mean; which one is closer is measured, not assumed.

Two rules live here so the SAME loader and the SAME character-union scorer produce both numbers:
  union / inter  -- our own production operators (per-member thresholds, no averaging)
  mean           -- theirs

Self-check (aborts, no silent garbage):
  --rule union --offset 0 --per_thr prim=-0.25,aux=-0.75 must write a span file byte-identical to
  dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json (6801 spans, dev AI 88.9700).

    python3 runbash/task2_shift_mean.py \
        --lists prim=dataset/dev/cmp1022/task2_prim_floor.json,aux=dataset/dev/cmp1022/task2_aux_floor.json \
        --rule mean --mean_thr_csv -1.5,-1.0,-0.75,-0.5,-0.25,0.0 \
        --gold dataset/cfn-dev.json \
        --ref_file dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_official as so  # noqa: E402
from score_official import build_gold, load_json  # noqa: E402

so.ARGS = argparse.Namespace(expand_words=False)

ap = argparse.ArgumentParser()
ap.add_argument('--lists', required=True, help='comma separated name=path [sid,start,end,logit]')
ap.add_argument('--offset', default='', help='name=value, shifted = raw - offset; missing name = 0.0')
ap.add_argument('--rule', default='mean', choices=['mean', 'union', 'inter'])
ap.add_argument('--mean_thr_csv', default='-1.5,-1.0,-0.75,-0.5,-0.25,0.0', help='rule=mean sweep')
ap.add_argument('--per_thr', default='', help='rule=union/inter: name=value per member, e.g. prim=-0.25,aux=-0.75')
ap.add_argument('--fill', default='present', help="'present' or a float -- see the header for what this patches")
ap.add_argument('--gold', default='./dataset/cfn-dev.json')
ap.add_argument('--no_gold', action='store_true', help='test boards: score nothing, only build/write cells')
ap.add_argument('--ref_file', default='', help='existing task2 file to diff added/removed characters against')
ap.add_argument('--check_file', default='', help='self-check: the built cell must be byte-identical to this file')
ap.add_argument('--expect_ai', type=float, default=None, help='assert dev AI of the single mean/union cell')
ap.add_argument('--out_dir', default='', help='write each cell as task2_<tag>.json here')
A = ap.parse_args()

names = [kv.split('=')[0] for kv in A.lists.split(',')]
OFF = {}
for kv in [x for x in A.offset.split(',') if x]:
    k, v = kv.split('=')
    if k not in names:
        raise SystemExit('[FATAL] --offset names %s, --lists names %s' % (k, names))
    OFF[k] = float(v)
FILL = None if A.fill == 'present' else float(A.fill)
if A.rule in ('union', 'inter') and not A.per_thr:
    raise SystemExit('[FATAL] rule=%s needs --per_thr name=value per member' % A.rule)
PER = {}
for kv in [x for x in A.per_thr.split(',') if x]:
    k, v = kv.split('=')
    if k not in names:
        raise SystemExit('[FATAL] --per_thr names %s, --lists names %s' % (k, names))
    PER[k] = float(v)
if A.rule in ('union', 'inter') and sorted(PER) != sorted(names):
    raise SystemExit('[FATAL] --per_thr must cover every member: have %s, need %s' % (sorted(PER), sorted(names)))


def load_scored(path, name):
    rows = load_json(path)
    bad = [r for r in rows if len(r) != 4]
    if bad:
        raise SystemExit('[FATAL] %s: %d rows are not [sid,start,end,logit]; first=%s'
                         % (path, len(bad), bad[0]))
    d = {}
    dup = 0
    for sid, st, en, sc in rows:
        k = (int(sid), int(st), int(en))
        if k in d:
            dup += 1
            d[k] = max(d[k], float(sc))
        else:
            d[k] = float(sc)
    off = OFF.get(name, 0.0)
    if off:
        d = {k: v - off for k, v in d.items()}
    vs = sorted(d.values())
    print('[load] %-6s %-52s %6d rows -> %6d spans | dup %d | raw p50 %+.3f | shifted min %+.3f '
          'p50 %+.3f max %+.3f | offset %+.2f'
          % (name, os.path.basename(path), len(rows), len(d), dup, vs[len(vs) // 2], vs[0],
             vs[len(vs) // 2], vs[-1], -off), flush=True)
    if not d:
        raise SystemExit('[FATAL] %s is empty' % path)
    return d


S = {n: load_scored(p, n) for n, p in [kv.split('=', 1) for kv in A.lists.split(',')]}
ALLK = set().union(*[set(d) for d in S.values()])
NM = len(S)
cnt = {}
for d in S.values():
    for k in d:
        cnt[k] = cnt.get(k, 0) + 1
dist = {j: sum(1 for k in ALLK if cnt[k] == j) for j in range(1, NM + 1)}
print('[cover] %d distinct spans over %d members | present in exactly k members: %s'
      % (len(ALLK), NM, ', '.join('k=%d: %d' % (j, dist[j]) for j in sorted(dist))), flush=True)
print('[cfg  ] rule=%s offset=%s fill=%s mean_thr=%s per_thr=%s'
      % (A.rule, {n: OFF.get(n, 0.0) for n in names}, A.fill, A.mean_thr_csv, A.per_thr or '-'), flush=True)

gold = {} if A.no_gold else build_gold(load_json(A.gold))
SIDS = sorted({k[0] for k in ALLK})
NGOLD = sum(1 for s in SIDS if s in gold)
if not A.no_gold and NGOLD != len(SIDS):
    print('[WARN] %d of %d sids have no gold row -- ids do not line up with %s'
          % (len(SIDS) - NGOLD, len(SIDS), A.gold), flush=True)


def chars(spans):
    c = set()
    for (s, e) in spans:
        c |= set(range(s, e + 1))
    return c


def sel_from(flat):
    out = {}
    for sid, st, en in [r[:3] for r in flat]:
        out.setdefault(int(sid), set()).add((int(st), int(en)))
    return out


def cells():
    """yield (tag, {sid: {(s,e),...}}) for every threshold setting this run asks for."""
    if A.rule == 'mean':
        for thr in [float(x) for x in A.mean_thr_csv.split(',') if x != '']:
            sel = {}
            for k in ALLK:
                vals = [S[n][k] for n in names if k in S[n]]
                if FILL is not None:
                    vals = vals + [FILL] * (NM - len(vals))
                if sum(vals) / len(vals) >= thr:
                    sel.setdefault(k[0], set()).add((k[1], k[2]))
            yield ('mean%.4g' % thr, sel)
    else:
        tag = '%s_%s' % (A.rule, '_'.join('%s_m%s' % (n, str(PER[n]).replace('-', 'm').replace('.', 'p'))
                                          for n in names))
        sel = {}
        for k in ALLK:
            hits = [n for n in names if k in S[n] and S[n][k] >= PER[n]]
            if hits and (A.rule == 'union' or len(hits) == NM):
                sel.setdefault(k[0], set()).add((k[1], k[2]))
        yield (tag, sel)


REF = sel_from(load_json(A.ref_file)) if A.ref_file else None
ref_chars = {sid: chars(REF.get(sid, ())) for sid in SIDS} if REF is not None else None
rows = []
for tag, sel in cells():
    flat = [[sid, s, e] for sid in sorted(sel) for (s, e) in sorted(sel[sid])]
    ai = p = g = inter = 0.0
    per = {sid: chars(sel.get(sid, ())) for sid in SIDS}
    addn = addhit = drobn = drohit = 0
    if ref_chars is not None:
        for sid in SIDS:
            a = per.get(sid, set()) - ref_chars[sid]
            d0 = ref_chars[sid] - per.get(sid, set())
            gc = gold[sid]['cover'] if sid in gold else set()
            addn += len(a); addhit += len(a & gc)
            drobn += len(d0); drohit += len(d0 & gc)
    if not A.no_gold:
        for sid in SIDS:
            gc = gold[sid]['cover'] if sid in gold else set()
            pc = per[sid]
            inter += len(pc & gc)
            p += len(pc)
            g += len(gc)
        ai = 200.0 * inter / (p + g) if p + g else 0.0
        prec = 100.0 * inter / p if p else 0.0
        rec = 100.0 * inter / g if g else 0.0
    else:
        prec = rec = 0.0
    line = '%s,%d,%.4f,%.4f,%.4f,%d,%d,%.2f,%d,%d,%.2f' % (
        tag, len(flat), ai, prec, rec, addn, addhit, 100.0 * addhit / addn if addn else 0.0,
        drobn, drohit, 100.0 * drohit / drobn if drobn else 0.0)
    rows.append(line)
    print('[cell ] %s  (spans, AI, P, R | added chars / hits / hit%% | removed / hits / hit%%)'
          % line, flush=True)
    if A.out_dir:
        os.makedirs(A.out_dir, exist_ok=True)
        t = tag.replace('-', 'm').replace('.', 'p')
        out = os.path.join(A.out_dir, 'task2_%s.json' % t)
        json.dump(flat, open(out, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
        print('[write] %s spans=%d' % (out, len(flat)), flush=True)
        if A.check_file:
            mine = open(out, 'rb').read()
            theirs = open(A.check_file, 'rb').read()
            if mine != theirs:
                raise SystemExit('[FATAL] self-check: %s is not byte-identical to %s (%d vs %d bytes)'
                                 % (out, A.check_file, len(mine), len(theirs)))
            print('[check] byte-identical to %s -- loader/threshold/char-union scoring all agree '
                  'with the production cell' % A.check_file, flush=True)

if A.expect_ai is not None and len(rows) == 1:
    got = float(rows[0].split(',')[2])
    if abs(got - A.expect_ai) > 5e-4:
        raise SystemExit('[FATAL] AI %.4f != expected %.4f' % (got, A.expect_ai))
    print('[expect] AI %.4f matches' % got, flush=True)

print('[hdr  ] cell,spans,dev_ai,prec,recall,added_chars,added_hit_chars,added_hit_pct,'
      'removed_chars,removed_hit_chars,removed_hit_pct')
print('SHIFT_MEAN_DONE', flush=True)
