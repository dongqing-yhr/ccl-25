#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""T1 prep: build the NULL negative pool the champion's task3 uses - four families instead of our
current one.

What we mine (family names are literal descriptions, no abbreviations):
  boundary_drift   a gold argument span with its start and/or end moved by 1-2 characters
  other_target     a span that is an argument of a *different* target word of the same sentence
  random_span      a random in-text span that overlaps neither a gold span nor the target word
  model_false_pos  what runbash/mine_deploy_negatives.py already produced: spans the task2 model
                   predicted on the training set that are not gold

Output format is the one dataset_task3.py already reads: a list of [sentence_id, start, end].
That loader attaches every negative of a sentence to the LAST target-word record of that sentence
and drops any negative that overlaps that record's gold spans or its target, so this script applies
exactly the same two bounds filters and the same overlap rule before counting against the budget.
Rows are then de-duplicated across families, so a family's "accepted" count is lower than the same
filter run on that file alone (the deploy pool is 4875 rows for the trainer but 2220 of them are
already in another family here - mostly the cross-target one). The number that must match the
training log's "加入部署匹配负例: N 条" is the selected total, and it should equal it exactly.

Budget: the champion uses about 2x the positive count, capped at 16 negatives per sentence. --ratio
controls the multiplier, default 1.0 so the first run costs roughly 4 GPU-hours instead of 6.5.
"""
import argparse, json, os, random, sys, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--train_file', default='./dataset/cfn-train.json')
ap.add_argument('--deploy_negatives_file', default='./dataset/train_deploy_negatives.json',
                help='model_false_pos pool from runbash/mine_deploy_negatives.py; empty string disables it')
ap.add_argument('--out', default='./dataset/train_null_negatives.json')
ap.add_argument('--detail_out', default='', help='optional [sid, start, end, family] file for analysis')
ap.add_argument('--ratio', type=float, default=1.0, help='negatives per positive')
ap.add_argument('--cap_per_sid', type=int, default=16)
ap.add_argument('--drift', default='1,2', help='character offsets for boundary drift')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--dry_run', action='store_true', help='print the plan, write nothing')
A = ap.parse_args()

DRIFT = [int(x) for x in A.drift.split(',') if x.strip()]


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def die(m):
    print('[%s] FATAL %s' % (time.strftime('%H:%M:%S'), m), flush=True)
    sys.exit(1)


t0 = time.time()
if not os.path.exists(A.train_file):
    die('missing %s' % A.train_file)
log('=' * 68)
log('NULL negative mining for task3 (T1 prep)')
log('  train_file            = %s' % A.train_file)
log('  model_false_pos file  = %s' % (A.deploy_negatives_file or 'DISABLED'))
log('  ratio negatives/pos   = %.2f' % A.ratio)
log('  cap per sentence      = %d' % A.cap_per_sid)
log('  drift offsets         = %s' % DRIFT)
log('  seed                  = %d' % A.seed)
log('=' * 68)

data = json.load(open(A.train_file, encoding='utf8'))
# Same record the loader builds: one entry per sentence, taken from the last line with that id,
# with the +1/+3 character shift already baked in the way dataset_task3 does it.
sid2rec = {}
sid_lines = defaultdict(list)
n_pos = 0
for line in data:
    sid = line['sentence_id']
    tgt = [line['target'][-1]['start'] + 1, line['target'][-1]['end'] + 1]
    gold = [(s['start'], s['end']) for s in line['cfn_spans']]
    sid_lines[sid].append((tgt, gold))
    n_pos += len(gold)
    sid2rec[sid] = {'text': line['text'], 'target': tgt, 'gold': gold}
log('positives (gold span-role pairs): %d   sentences: %d   lines: %d'
    % (n_pos, len(sid2rec), len(data)))
budget_total = int(round(A.ratio * n_pos))
log('negative budget: %.2f x %d = %d rows, cap %d per sentence'
    % (A.ratio, n_pos, budget_total, A.cap_per_sid))


def loadable(sid, st, en):
    """Mirror dataset_task3.py's acceptance test. Returns True if the trainer will keep this row."""
    rec = sid2rec.get(sid)
    if rec is None:
        return False
    n = len(rec['text'])
    if st < 0 or en < st or en >= n:                      # first bounds filter
        return False
    if any(g[0] <= st <= g[1] or g[0] <= en <= g[1] for g in rec['gold']):
        return False                                      # overlaps a gold span of that record
    tg = rec['target']
    if en + 1 < tg[0]:
        li = [st + 1, en + 1]
    elif st + 1 > tg[1]:
        li = [st + 3, en + 3]
    else:
        return False                                      # overlaps the target word
    if li[1] >= n + 4:                                    # second bounds filter
        return False
    return True


pools = defaultdict(list)      # family -> list of (sid, st, en)
seen = set()


def add(family, sid, st, en):
    key = (sid, st, en)
    if key in seen:
        return False
    if not loadable(sid, st, en):
        return False
    seen.add(key)
    pools[family].append(key)
    return True


rng = random.Random(A.seed)
n_multi_target_sid = 0
for sid, rec in sid2rec.items():
    text, gold, tg = rec['text'], rec['gold'], rec['target']
    n = len(text)
    # 1. boundary drift: every gold span, both ends nudged by every offset pair
    for (gs, ge) in gold:
        for ds in [0] + DRIFT + [-d for d in DRIFT]:
            for de in [0] + DRIFT + [-d for d in DRIFT]:
                if ds == 0 and de == 0:
                    continue
                add('boundary_drift', sid, gs + ds, ge + de)
    # 2. other target word: gold spans of the sentence's other lines
    if len(sid_lines[sid]) > 1:
        n_multi_target_sid += 1
        for (other_tg, other_gold) in sid_lines[sid]:
            if other_tg == tg:
                continue
            for (gs, ge) in other_gold:
                add('other_target', sid, gs, ge)
    # 3. random spans: short random windows, biased to lengths the gold spans actually have
    glens = [ge - gs + 1 for gs, ge in gold] or [2]
    n_rand = max(4, min(24, 3 * len(gold)))
    for _ in range(n_rand):
        L = rng.choice(glens)
        st = rng.randint(0, max(0, n - L))
        add('random_span', sid, st, st + L - 1)

if A.deploy_negatives_file:
    if not os.path.exists(A.deploy_negatives_file):
        die('missing %s (pass --deploy_negatives_file "" to skip this family)' % A.deploy_negatives_file)
    raw = json.load(open(A.deploy_negatives_file, encoding='utf8'))
    kept = 0
    for item in raw:
        if add('model_false_pos', item[0], item[1], item[2]):
            kept += 1
    log('model_false_pos: %d raw rows read, %d accepted' % (len(raw), kept))

log('')
log('pool sizes after the trainer\'s own filters (all sentences, before the budget bites):')
for fam in ('boundary_drift', 'other_target', 'random_span', 'model_false_pos'):
    log('  %-16s %7d rows   (%d sentences)' % (fam, len(pools[fam]),
                                               len(set(k[0] for k in pools[fam]))))
log('  %-16s %7d rows' % ('TOTAL', sum(len(v) for v in pools.values())))
log('  sentences with more than one target word: %d of %d' % (n_multi_target_sid, len(sid2rec)))

# ------------------------------------------------------------------ budgeted selection
# Round-robin over families so no single family eats the budget, then the per-sentence cap.
# Sentences are visited in a shuffled order and each contributes at most cap_per_sid rows, so the
# pool stays spread over the corpus instead of concentrating on the longest sentences.
fams = [f for f in ('model_false_pos', 'boundary_drift', 'other_target', 'random_span') if pools[f]]
if not fams:
    die('every family is empty - nothing to write')
avail = defaultdict(dict)          # sid -> family -> list of (start, end)
for f in fams:
    for (sid, st, en) in pools[f]:
        avail[sid].setdefault(f, []).append((st, en))
sid_order = sorted(avail.keys(), key=lambda s: rng.random())
fam_taken = defaultdict(int)
cur = defaultdict(lambda: defaultdict(int))   # sid -> family -> how many of that family already taken
per_sid = defaultdict(int)
chosen = []                          # (sid, start, end, family)
log('')
log('selecting up to %d rows, round-robin over %s, cap %d per sentence'
    % (budget_total, fams, A.cap_per_sid))
# Quota rounds instead of one long fill: pass 1 gives every sentence one negative, pass 2 a second,
# and so on up to the cap. Filling sentence by sentence would spend the whole budget on the few
# hundred sentences that happened to come first in the shuffle and leave most sentences with none.
quota = 0
while len(chosen) < budget_total and quota < A.cap_per_sid:
    quota += 1
    before = len(chosen)
    for sid in sid_order:
        if len(chosen) >= budget_total:
            break
        pool = avail[sid]
        while per_sid[sid] < quota and len(chosen) < budget_total:
            pick, fewest = None, None
            for f in fams:                            # of the families this sentence still has,
                if cur[sid].get(f, 0) >= len(pool.get(f, [])):
                    continue                          # take the one used least overall so far
                if fewest is None or fam_taken[f] < fewest:
                    pick, fewest = f, fam_taken[f]
            if pick is None:
                break
            st, en = pool[pick][cur[sid][pick]]
            cur[sid][pick] += 1
            fam_taken[pick] += 1
            per_sid[sid] += 1
            chosen.append((sid, st, en, pick))
    log('  quota round %2d: %d rows selected (%.0f%% of budget, %d sentences touched)'
        % (quota, len(chosen), 100.0 * len(chosen) / budget_total,
           len(set(c[0] for c in chosen))))
    if len(chosen) == before:
        log('  no further row available, stopping the rounds')
        break
if len(chosen) < budget_total:
    log('pools exhausted at the cap: %d of the %d budgeted rows are available'
        % (len(chosen), budget_total))
for f in fams:
    log('  final mix %-16s %7d of %7d available' % (f, fam_taken[f], len(pools[f])))
log('selected total: %d rows over %d sentences (mean %.2f per used sentence)'
    % (len(chosen), len(set(c[0] for c in chosen)),
       len(chosen) / max(1, len(set(c[0] for c in chosen)))))

if A.dry_run:
    log('--dry_run: nothing written')
    sys.exit(0)

out_rows = [[sid, st, en] for sid, st, en, _f in chosen]
if os.path.exists(A.out) and not os.path.exists(A.out + '.bak_pre_t1'):
    os.rename(A.out, A.out + '.bak_pre_t1')
    log('existing %s moved aside to %s' % (A.out, A.out + '.bak_pre_t1'))
with open(A.out, 'w', encoding='utf8', newline='\n') as fh:
    json.dump(out_rows, fh, ensure_ascii=False)
log('wrote %s  (%d rows, %.1f KB)' % (A.out, len(out_rows), os.path.getsize(A.out) / 1024.0))
if A.detail_out:
    with open(A.detail_out, 'w', encoding='utf8', newline='\n') as fh:
        json.dump([[sid, st, en, f] for sid, st, en, f in chosen], fh, ensure_ascii=False)
    log('wrote %s  (%d rows with family labels)' % (A.detail_out, len(chosen)))
if not os.path.exists(A.out) or os.path.getsize(A.out) < 1000:
    die('output file missing or suspiciously small')
log('done in %.1f s' % (time.time() - t0))
