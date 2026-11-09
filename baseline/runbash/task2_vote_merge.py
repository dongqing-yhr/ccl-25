#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge several task2 decodes by "how many models flagged this span" (vote count), offline on CPU.

Why this exists: predict_task2 emits a span when its span-head logit >= its own threshold and nothing
else (no NMS, no boundary refinement), so one decode dumped at a low floor threshold already contains
every stricter decode of that same model as a subset. That makes the merge pure arithmetic over the
scored sidecars [sid, start, end, logit]: give each model its own threshold, count for every candidate
span how many models emitted it, and keep the spans with at least k agreeing models.

Counting agreements rather than averaging logits is deliberate: the models score on different scales
(measured: the anchor-augmented checkpoint runs about 0.8 higher than the primary one), so a shared
"average probability" threshold means something different per model. A vote does not.

Same yardstick as runbash/task2_fuse_grid.py: dev AI is the character-overlap F1 from score_official,
computed on the task2 file itself; task3 is not touched here.

  # self-check: two members, k=1 must reproduce the current production cell exactly
  python3 runbash/task2_vote_merge.py \
      --member prim:dataset/dev/cmp1022/scores_prim.json:-0.25 \
      --member aux:dataset/dev/cmp1022/scores_aux.json:-0.75 \
      --votes 1 --expect_k 1 --expect 88.9700 \
      --ref_file dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json

  # four members, all vote levels
  python3 runbash/task2_vote_merge.py \
      --member rP:.../scores_roberta_prim.json:-0.25 --member rA:.../scores_roberta_aux.json:-0.75 \
      --member mP:.../scores_macbert_plain.json:-0.25 --member mA:.../scores_macbert_aug.json:-0.75 \
      --votes 1,2,3,4 --ref_file dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json --boot 1000

  # main panel = the production pair, second panel = the two new models used only as negative evidence
  python3 runbash/task2_vote_merge.py \
      --member rP:...:N6029 --member rA:...:N6351 --member mP:...:N6029 --member mA:...:N6351 \
      --group_a rP,rA --group_b mP,mA --require_b 0,1,2 \
      --ref_file dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json

A threshold may also be given as a span count, "NAME:SIDECAR:N6029": keep that model's 6029
highest-scored candidate spans whatever its logits are worth. Use it when a model comes from a
different backbone -- MacBERT's span-head logits do not sit on the RoBERTa scale, so "the same
threshold number" is not "the same operating point", while "the same number of emitted spans" is.

Every cell is read out two ways, because dev (1995 sentences) cannot resolve small aggregate
differences (~+-0.5 AI) but can resolve per-batch evidence:
  1) dev AI, delta against the reference file, and the spread of that delta over sentence-level
     resampling (how many of the draws still say "not better than the reference");
  2) the marginal gate: the characters this cell newly adds over the reference must hit gold more
     often than AI/2 of the reference -- that is the exact break-even for an F1 numerator.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_official as so  # noqa: E402
from score_official import build_gold, load_json  # noqa: E402

so.ARGS = argparse.Namespace(expand_words=False)

ap = argparse.ArgumentParser()
ap.add_argument('--member', action='append', required=True,
                help='NAME:SIDECAR_PATH:THRESHOLD, repeatable. sidecar rows are [sid,start,end,logit]')
ap.add_argument('--gold', default='./dataset/cfn-dev.json')
ap.add_argument('--votes', default='', help='k values to score (keep spans agreed by >= k models); '
                'empty means every level from 1 to the number of --member entries')
ap.add_argument('--ref_file', default='', help='current production task2 file to compare against')
ap.add_argument('--boot', type=int, default=1000, help='resampling draws per cell (per-sid stats are '
                'precomputed, so this is cheap -- unlike task2_fuse_grid.py)')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--expect', type=float, default=None, help='hard assert: this AI must be reproduced')
ap.add_argument('--expect_k', type=int, default=1, help='which vote level --expect applies to')
ap.add_argument('--check_file', action='append', default=[],
                help='NAME=PRODUCTION_FILE: that member, at its own threshold, must emit exactly this file')
ap.add_argument('--no_gold', action='store_true', help='test boards: build and write cells, score nothing')
ap.add_argument('--solo_counts', default='4000,5000,6000,6801,7500,8500',
                help='emission volumes to test each model at alone; gives "is this model weaker by itself, '
                     'or just not complementary" without a threshold grid')
ap.add_argument('--group_a', default='', help='names, comma separated. With --group_b this replaces the '
                'single vote count: a span is kept only if at least one group_a model emitted it AND at '
                'least j group_b models emitted it. group_a alone is the starting file.')
ap.add_argument('--group_b', default='', help='names, comma separated. The panel used as second opinion.')
ap.add_argument('--require_b', default='', help='j values to scan, e.g. "0,1,2". j=0 keeps all of group_a '
                'and must reproduce the group_a union -- that is the self-check for this scan.')
ap.add_argument('--write', default='', help='k values to persist as task2 files, e.g. "2,3"')
ap.add_argument('--outdir', default='./dataset/dev/cmp1026/vote')
ap.add_argument('--out_prefix', default='task2')
ap.add_argument('--csv', default='./logfile/task2_vote_merge.csv')
A = ap.parse_args()
random.seed(A.seed)

# ---- members ----
MEM = []
for spec in A.member:
    parts = spec.split(':')
    if len(parts) != 3:
        raise SystemExit('[FATAL] --member must be NAME:SIDECAR:THRESH(or N<count>), got %r' % spec)
    name, path, tspec = parts[0], parts[1], parts[2].strip()
    if not os.path.isfile(path):
        raise SystemExit('[FATAL] member %s: sidecar missing: %s' % (name, path))
    if any(m[0] == name for m in MEM):
        raise SystemExit('[FATAL] duplicate member name %r -- names key the output files' % name)
    if tspec[:1] in ('N', 'n'):                      # N<count> = keep that many top spans
        MEM.append((name, path, None, int(tspec[1:])))
    else:
        MEM.append((name, path, float(tspec), 0))
if len(MEM) < 2:
    raise SystemExit('[FATAL] need at least 2 --member specs (this tool is about agreement between models)')
N = len(MEM)
if not A.votes.strip():
    A.votes = ','.join(str(i) for i in range(1, N + 1))
THR = {}
print('[members] %d' % N)
for name, path, thr, cnt in MEM:
    THR[name] = thr
    print('   %-6s %-9s %s' % (name, ('N%d spans' % cnt) if thr is None else ('thr=%+.3f' % thr), path))

EMIT = {}
SORTED = {}
UNION_KEYS = set()
for name, path, thr, cnt in MEM:
    rows = load_json(path)
    bad = [r for r in rows if len(r) != 4]
    if bad:
        raise SystemExit('[FATAL] %s: %d rows are not [sid,start,end,logit]; first=%s' % (path, len(bad), bad[0]))
    sc = {}
    dup = 0
    for sid, st, en, v in rows:
        k = (int(sid), int(st), int(en))
        if k in sc:
            dup += 1
            sc[k] = max(sc[k], float(v))
        else:
            sc[k] = float(v)
    if not sc:
        raise SystemExit('[FATAL] %s is empty' % path)
    UNION_KEYS |= set(sc)
    vals = sorted(sc.values())
    if thr is None:                                  # resolve the span-count operating point
        if cnt < 1 or cnt > len(sc):
            raise SystemExit('[FATAL] member %s asked for %d spans but only %d unique candidates exist'
                             % (name, cnt, len(sc)))
        thr = sorted(sc.values(), reverse=True)[cnt - 1]
        THR[name] = thr
        print('[thresh] %-6s N%d -> logit %+.4f (ties at that value are all kept, so it may emit a few more)'
              % (name, cnt, thr))
    emitted = set(k for k, v in sc.items() if v >= thr)
    EMIT[name] = emitted
    SORTED[name] = sorted(sc.items(), key=lambda kv: -kv[1])
    print('[load] %-6s rows=%d uniq=%d dup_collapsed=%d | logit min %+.3f p50 %+.3f p90 %+.3f max %+.3f '
          '| emitted@%+.2f = %d'
          % (name, len(rows), len(sc), dup, vals[0], vals[len(vals) // 2], vals[int(0.9 * len(vals))], vals[-1],
             thr, len(emitted)))
    if len(emitted) == 0:
        raise SystemExit('[FATAL] member %s emits nothing at %+.2f -- threshold above its whole logit range'
                         % (name, thr))

# ---- scale / agreement diagnostics: is a vote informative at all? ----
names = [m[0] for m in MEM]
print('\n[agreement] pairwise overlap of emitted span sets')
for i in range(N):
    for j in range(i + 1, N):
        a, b = EMIT[names[i]], EMIT[names[j]]
        inter = len(a & b)
        print('   %s vs %s: both=%d only-%s=%d only-%s=%d  jaccard=%.3f'
              % (names[i], names[j], inter, names[i], len(a - b), names[j], len(b - a),
                 inter / float(len(a | b))))

ALLK = set()
for name in names:
    ALLK |= EMIT[name]
VOTES = Counter()
for name in names:
    for k in EMIT[name]:
        VOTES[k] += 1
print('[votes] candidate spans agreed by exactly k models: %s'
      % ', '.join('%d:%d' % (k, sum(1 for v in VOTES.values() if v == k)) for k in range(1, N + 1)))

# ---- gold + reference ----
gold = {} if A.no_gold else build_gold(load_json(A.gold))
# sentence list taken from the union of all scored candidates (unthresholded), exactly as
# runbash/task2_fuse_grid.py does it, so the AI numbers are on the same denominator
SIDS = sorted({k[0] for k in UNION_KEYS})
if not A.no_gold:
    miss = sum(1 for s in SIDS if s not in gold)
    if miss:
        print('[WARN] %d of %d sids have no gold row in %s -- they count as empty' % (miss, len(SIDS), A.gold))
REF = None
if A.ref_file:
    if not os.path.isfile(A.ref_file):
        raise SystemExit('[FATAL] --ref_file missing: %s' % A.ref_file)
    REF = {}
    for it in load_json(A.ref_file):
        REF.setdefault(int(it[0]), set()).add((int(it[1]), int(it[2])))
    print('[ref] %s -> %d spans, %d sentences (this is the current production emission file)'
          % (A.ref_file, sum(len(v) for v in REF.values()), len(REF)))


def chars(spans):
    c = set()
    for (s, e) in spans:
        c |= set(range(s, e + 1))
    return c


def sel_at(k):
    out = {}
    for key, v in VOTES.items():
        if v >= k:
            out.setdefault(key[0], set()).add((key[1], key[2]))
    return out


# ---- per-sentence quantities, computed once so the resampling is cheap ----
class Stats(object):
    pass


def stats_of(sel):
    st = Stats()
    st.per = {}
    for sid in SIDS:
        pc = chars(sel.get(sid, ())) if sel else set()
        gc = gold[sid]['cover'] if (sid in gold and gold) else set()
        st.per[sid] = (len(pc & gc), len(pc), len(gc))
    tot = [sum(x[i] for x in st.per.values()) for i in range(3)]
    st.ai = 200.0 * tot[0] / (tot[1] + tot[2]) if (tot[1] + tot[2]) else 0.0
    st.pr = 100.0 * tot[0] / tot[1] if tot[1] else 0.0
    st.rc = 100.0 * tot[0] / tot[2] if tot[2] else 0.0
    st.spans = sum(len(v) for v in sel.values()) if sel else 0
    return st


REFST = None
if REF is not None and not A.no_gold:
    REFST = stats_of(REF)
    print('[ref] dev AI of the reference file = %.4f (prec %.2f / recall %.2f), marginal bar = AI/2 = %.2f%%'
          % (REFST.ai, REFST.pr, REFST.rc, REFST.ai / 2.0))
elif REF is not None:
    print('[ref] --no_gold: reference scored on the boards is impossible, only span counts are compared')


def dref_boot(st_cell, st_ref, draws):
    """Resample sentences with replacement, recompute both F1s on each draw, look at the spread.

    Plain steps: draw len(SIDS) sentences with replacement from the dev sentences -> sum their
    precomputed numerators -> two F1s -> their difference. Repeat `draws` times. If nearly all draws
    come out positive the gap is real; if both signs show up, the gap is which sentences got drawn.
    """
    sid_l = list(SIDS)
    n = len(sid_l)
    c = [st_cell.per.get(s, (0, 0, 0)) for s in sid_l]
    r = [st_ref.per.get(s, (0, 0, 0)) for s in sid_l]
    diffs = []
    for _ in range(draws):
        a1 = ap1 = a2 = ap2 = g = 0.0
        for _i in range(n):
            j = random.randrange(n)
            x = c[j]
            a1 += x[0]
            ap1 += x[1]
            y = r[j]
            a2 += y[0]
            ap2 += y[1]
            g += x[2]
        f1 = 200.0 * a1 / (ap1 + g) if (ap1 + g) else 0.0
        f2 = 200.0 * a2 / (ap2 + g) if (ap2 + g) else 0.0
        diffs.append(f1 - f2)
    diffs.sort()
    lo = diffs[int(0.025 * draws)]
    hi = diffs[min(draws - 1, int(0.975 * draws))]
    ple = sum(1 for d in diffs if d <= 0) / float(draws)
    return lo, hi, ple


def gate_vs_ref(sel):
    """Characters this cell adds over the reference, and characters it drops -- each against AI/2."""
    add_c = add_h = drop_c = drop_h = 0
    for sid in set(sel) | set(REF):
        pc = chars(sel.get(sid, ()))
        rc = chars(REF.get(sid, ())) if REF else set()
        gc = gold[sid]['cover'] if (sid in gold and gold) else set()
        fresh = pc - rc
        gone = rc - pc
        add_c += len(fresh)
        add_h += len(fresh & gc)
        drop_c += len(gone)
        drop_h += len(gone & gc)
    return add_c, add_h, drop_c, drop_h


def report_cell(label, sel, rows):
    """One cell, read out both ways: the aggregate gap (with resampling spread) and the two marginal batches."""
    st = stats_of(sel)
    d = st.ai - REFST.ai if REFST else 0.0
    lo = hi = ple = 0.0
    if REFST and A.boot:
        lo, hi, ple = dref_boot(st, REFST, A.boot)
    add_c = add_h = drop_c = drop_h = 0
    if REF is not None:
        add_c, add_h, drop_c, drop_h = gate_vs_ref(sel)
    ar = 100.0 * add_h / add_c if add_c else 0.0
    dr = 100.0 * drop_h / drop_c if drop_c else 0.0
    print('%-8s %7d %9.4f %+9.4f %8.4f %8.4f %7.3f %9d %9.2f %9d %9.2f'
          % (label, st.spans, st.ai, d, lo, hi, ple, add_c, ar, drop_c, dr))
    rows.append([label, st.spans, st.ai, d, lo, hi, ple, st.pr, st.rc, add_c, add_h, ar, drop_c, drop_h, dr])
    return st


# ---- member-at-its-own-threshold must equal a production file (fail loudly) ----
for spec in A.check_file:
    name, _, path = spec.partition('=')
    if name not in EMIT:
        raise SystemExit('[FATAL] --check_file names unknown member %r' % name)
    want = {}
    for it in load_json(path):
        want.setdefault(int(it[0]), set()).add((int(it[1]), int(it[2])))
    got = {}
    thr = THR[name]
    for (sid, s, e) in EMIT[name]:
        got.setdefault(sid, set()).add((s, e))
    if {k: v for k, v in got.items() if v} != {k: v for k, v in want.items() if v}:
        raise SystemExit('[FATAL] member %s at %+.2f emits %d spans but %s has %d -- sidecar does not '
                         'reproduce the production decode, nothing below is comparable'
                         % (name, thr, len(got), path, len(want)))
    print('[check] member %s @%+.2f == %s -- sidecar is faithful' % (name, thr, path))

if A.no_gold:
    os.makedirs(A.outdir, exist_ok=True)
    for k in [int(x) for x in A.votes.split(',') if x.strip()]:
        sel = sel_at(k)
        flat = [[sid, s, e] for sid in sorted(sel) for (s, e) in sorted(sel[sid])]
        out = os.path.join(A.outdir, '%s_vote%d.json' % (A.out_prefix, k))
        json.dump(flat, open(out, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
        n_span = len(flat)
        n_ref = sum(len(v) for v in REF.values()) if REF else 0
        print('[write] %s spans=%d (reference file had %d)' % (out, n_span, n_ref))
    print('TASK2_VOTE_MERGE_DONE (no gold)')
    raise SystemExit(0)

KS = sorted(set(int(x) for x in A.votes.split(',') if x.strip()))
for k in KS:
    if k < 1 or k > N:
        raise SystemExit('[FATAL] --votes has %d but only %d members were given' % (k, N))

print('\n%-8s %7s %9s %9s %8s %8s %7s %9s %9s %9s %9s'
      % ('vote>=k', 'spans', 'dev AI', 'd_ref', 'ci lo', 'ci hi', 'P<=0', 'added ch', 'add hit%',
         'drop ch', 'drop hit%'))
rows = []
for k in KS:
    st = report_cell(k, sel_at(k), rows)
    if A.expect is not None and k == A.expect_k:
        if abs(st.ai - A.expect) > 0.0001:
            raise SystemExit('[FATAL] vote>=%d gives dev AI %.4f, expected %.4f -- wrong sidecars, wrong '
                             'thresholds, or the reference cell is not what this was built from'
                             % (k, st.ai, A.expect))
        print('   [expect OK] vote>=%d reproduces dev AI %.4f exactly -- same yardstick as the live recipe'
              % (k, st.ai))

print('\nHow to read this: d_ref is the aggregate gap (0.3*d_ref is what task2 alone contributes to the '
      'composite, and the measured task2->task3 carry-over adds 0.4*0.24*d_ref). P<=0 is the fraction of '
      'resampled fake exam papers where this cell is still not better than the reference -- 0.5 means a '
      'coin toss. "add hit%%" is the share of the newly added characters that are inside a gold span; the '
      'break-even for an F1 numerator is the reference F1 divided by 2 (%.2f%% here). Adding a batch whose '
      'hit rate is below that bar costs composite points even when dev says it is close.'
      % ((REFST.ai / 2.0) if REFST else 0.0))

# ---- each model on its own, over a ladder of emission volumes ----
# Tells apart the two reasons a merge can fail: this model is weaker by itself, or it is strong but
# says the same thing as the others (then agreement adds nothing and only the junk differs).
if gold:
    LAD = [int(x) for x in A.solo_counts.split(',') if x.strip()]
    print('\n=== solo: each model alone, keeping its top-c scored candidate spans (dev AI) ===')
    print('%-8s%s   best' % ('model', ''.join('%9d' % c for c in LAD)))
    for name in names:
        vals = []
        for c in LAD:
            sel = {}
            for (sid, s, e), _v in SORTED[name][:c]:
                sel.setdefault(sid, set()).add((s, e))
            vals.append(stats_of(sel).ai)
        b = max(vals)
        print('%-8s%s  %7.4f@%d' % (name, ''.join('%9.4f' % v for v in vals), b, LAD[vals.index(b)]))

# ---- which agreement level is worth merging: batch by how many models agree ----
if REF is not None:
    print('\n=== per-agreement batches, measured over the reference file (hit rate must clear %.2f%%) ==='
          % ((REFST.ai / 2.0) if REFST else 0.0))
    print('%-14s %8s %9s %9s %9s %10s' % ('agreed models', 'spans', 'new chars', 'gold hits', 'hit rate', 'verdict'))
    ref_all = {sid: chars(v) for sid, v in REF.items()}
    for kk in range(N, 0, -1):
        keys = [key for key, v in VOTES.items() if v == kk]
        add_c = add_h = 0
        for (sid, s, e) in keys:
            gc = gold[sid]['cover'] if (sid in gold and gold) else set()
            fresh = set(range(s, e + 1)) - ref_all.get(sid, set())
            add_c += len(fresh)
            add_h += len(fresh & gc)
        rate = 100.0 * add_h / add_c if add_c else 0.0
        bar = REFST.ai / 2.0 if REFST else 0.0
        verdict = 'merge' if add_c and rate > bar else ('nothing new' if not add_c else 'stop')
        print('%-14s %8d %9d %9d %8.2f%% %10s' % ('exactly %d' % kk, len(keys), add_c, add_h, rate, verdict))

# ---- two panels: keep what the main panel emits, filtered by the second panel's agreement count ----
# A single vote count conflates two opposite moves: raising k both deletes main-panel spans and keeps
# second-panel junk. Splitting the members into a main panel and a second-opinion panel lets the removal
# be measured on its own -- j=0 keeps the whole main panel (that row must show 0 added / 0 dropped chars
# against the reference, otherwise the main panel is not the production file), and j>0 drops exactly the
# main-panel spans the second panel refuses to confirm, adding nothing.
if A.group_b:
    ga = [x.strip() for x in A.group_a.split(',') if x.strip()] or names
    gb = [x.strip() for x in A.group_b.split(',') if x.strip()]
    unknown = sorted(set(x for x in ga + gb if x not in EMIT))
    if unknown:
        raise SystemExit('[FATAL] --group_a/--group_b names unknown member(s) %s' % unknown)
    both = sorted(set(ga) & set(gb))
    if both:
        raise SystemExit('[FATAL] a member cannot sit in both panels: %s' % both)
    cA = Counter()
    for n in ga:
        cA.update(EMIT[n])
    cB = Counter()
    for n in gb:
        cB.update(EMIT[n])
    JS = sorted(set(int(x) for x in A.require_b.split(',') if x.strip())) or list(range(0, len(gb) + 1))
    for j in JS:
        if j < 0 or j > len(gb):
            raise SystemExit('[FATAL] require_b=%d but panel_b has only %d members' % (j, len(gb)))
    print('\n=== second panel: keep spans emitted by >=1 of [%s] AND >=j of [%s] ==='
          % (','.join(ga), ','.join(gb)))
    print('%-8s %7s %9s %9s %8s %8s %7s %9s %9s %9s %9s'
          % ('j', 'spans', 'dev AI', 'd_ref', 'ci lo', 'ci hi', 'P<=0', 'added ch', 'add hit%',
             'drop ch', 'drop hit%'))
    for j in JS:
        sel = {}
        for key, v in cA.items():
            if v >= 1 and cB.get(key, 0) >= j:
                sel.setdefault(key[0], set()).add((key[1], key[2]))
        st = report_cell('j=%d' % j, sel, rows)
        if j == 0 and REF is not None and rows[-1][9] == 0 and rows[-1][12] == 0:
            print('   [j=0 OK] the main panel alone reproduces the reference file (%.4f) -- the rows below '
                  'are pure removals' % st.ai)

with open(A.csv, 'w', encoding='utf8', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['cell', 'spans', 'dev_ai', 'd_ref', 'ci_lo', 'ci_hi', 'p_le_0', 'prec', 'recall',
                 'added_chars', 'added_hits', 'added_hit_pct', 'drop_chars', 'drop_hits', 'drop_hit_pct'])
    wr.writerows(rows)
print('[csv] %s (%d rows)' % (A.csv, len(rows)))

if A.write:
    os.makedirs(A.outdir, exist_ok=True)
    for k in [int(x) for x in A.write.split(',') if x.strip()]:
        sel = sel_at(k)
        flat = [[sid, s, e] for sid in sorted(sel) for (s, e) in sorted(sel[sid])]
        out = os.path.join(A.outdir, '%s_vote%d.json' % (A.out_prefix, k))
        json.dump(flat, open(out, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
        print('[write] %s spans=%d' % (out, len(flat)))
print('TASK2_VOTE_MERGE_DONE')
