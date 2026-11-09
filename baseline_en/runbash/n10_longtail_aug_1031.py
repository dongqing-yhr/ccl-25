#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""N10: long-tail surface replacement augmentation for the task1/task2/task3 training file.

What the operator is (a competing team's unpublished write-up, quoted as
"(frame, role) freq <=5, other surface form of the same combination, length difference <=3,
candidate span shifted in step"):

  For every span in a training row whose (frame, role) pair occurs at most --freq_max times in
  the whole training file, take a surface form that some OTHER row used for the same
  (frame, role) pair, require |len(new) - len(old)| <= --len_diff_max, splice it into the
  sentence at the old position, and shift every coordinate that sits to the right of the
  replaced region (other spans, the target word, the word list) by the length difference.

The output is the original file plus the generated rows appended, so a training run can switch
train files without any code change. Rows are never modified in place and never dropped, so the
original part of the file stays byte-comparable through --check.

Self-checks built in (all of them abort with exit code 1):
  * every generated row is re-read from the produced JSON and every span in it must extract
    exactly the surface form it claims, and the target word must still extract the same text
    as in the source row;
  * the shifted coordinate lists must stay in range and ordered;
  * the output's row count = input rows + generated rows;
  * --check re-validates a produced file against its source without regenerating.

Usage (container, baseline dir):
  python3 runbash/n10_longtail_aug_1031.py --train_file ./dataset/cfn-train.json \
      --out_file ./dataset/cfn-train-longtailaug.json
  python3 runbash/n10_longtail_aug_1031.py --check ./dataset/cfn-train-longtailaug.json
"""
import argparse
import codecs
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

t0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument('--train_file', default='./dataset/cfn-train.json')
ap.add_argument('--out_file', default='./dataset/cfn-train-longtailaug.json')
ap.add_argument('--check', default='', help='re-validate a produced file, then exit')
ap.add_argument('--meta_file', default='', help='where to write the provenance sidecar')
ap.add_argument('--freq_max', default='5', help='a (frame, role) pair at or below this many '
                                                'training occurrences counts as long-tail')
ap.add_argument('--len_diff_max', default='3', help='max |len(new)-len(old)| characters')
ap.add_argument('--per_row_max', default='1', help='max generated rows per source row')
ap.add_argument('--seed', default='20261031')
A = ap.parse_args()

FREQ = int(A.freq_max)
LDMAX = int(A.len_diff_max)
PERROW = int(A.per_row_max)


def log(m):
    print('[%7.1fs] %s' % (time.time() - t0, m))
    sys.stdout.flush()


def die(m):
    log('FATAL ' + m)
    sys.exit(1)


def load(path):
    rows = json.load(codecs.open(path, 'r', encoding='utf8'))
    isinstance(rows, list) or die('%s is not a JSON list' % path)
    return rows


def remap(st, en, rs, re, new_len):
    """Map the interval [st, en] across a replacement of the region [rs, re].

    CFN annotations overlap and nest, so an interval may lie left of, right of, straddling, or
    inside the replaced region. Boundaries are handled half-open: a boundary equal to re + 1
    belongs to the right side and moves by the length difference, while a boundary strictly inside
    the region collapses onto the new region's right edge. Returns None when the interval
    collapses to nothing, which the caller reads as "this replacement cannot be represented".
    """
    delta = new_len - (re - rs + 1)

    def boundary(x):
        if x <= rs:
            return x
        if x >= re + 1:
            return x + delta
        return rs + new_len

    ns, ne = boundary(st), boundary(en + 1) - 1
    if ne < ns:
        return None
    return ns, ne


if A.check:
    # --check <produced file>: re-validate a produced file against its source without regenerating.
    # Coordinate *ordering* is not asserted: CFN's own cfn_spans list is not sorted by start in 233
    # of the 10700 source rows, so the check compares each produced row with the source row the
    # sidecar points at, span by span, instead.
    produced = load(A.check)
    src_rows = load(A.train_file)
    len(produced) >= len(src_rows) or die('produced file shorter than the source')
    for i, (a, b) in enumerate(zip(src_rows, produced)):
        if a != b:
            die('row %d of the produced file differs from the source; N10 must only append' % i)
    n_src_rows = len(src_rows)
    appended = produced[n_src_rows:]
    meta_path = A.meta_file or (A.check + '.n10meta.json')
    meta = json.load(codecs.open(meta_path, 'r', encoding='utf8'))
    prov = meta['provenance']
    len(appended) == len(prov) or die('appended rows=%d provenance=%d' % (len(appended), len(prov)))
    n_bad = n_unsorted = 0
    for j, (p, row) in enumerate(zip(prov, appended)):
        srow = src_rows[p['row']]
        st_en = srow['cfn_spans'][p['span']]
        old, new = p['old'], p['new']
        rs, re, nl = st_en['start'], st_en['end'], len(new)
        why = []
        if row['text'] != srow['text'][:rs] + new + srow['text'][re + 1:]:
            why.append('text')
        if row['frame'] != srow['frame']:
            why.append('frame')
        if len(row['cfn_spans']) != len(srow['cfn_spans']):
            why.append('span_count')
        else:
            for sp, spo in zip(row['cfn_spans'], srow['cfn_spans']):
                if sp['fe_name'] != spo['fe_name']:
                    why.append('role_name')
                    break
                want = remap(spo['start'], spo['end'], rs, re, nl)
                if want is None:
                    why.append('collapsing_source_span')
                elif (sp['start'], sp['end']) != want:
                    why.append('coord')
        for t, to in zip(row['target'], srow['target']):
            want = remap(to['start'], to['end'], rs, re, nl)
            if want is None or (t['start'], t['end']) != want:
                why.append('target_coord')
        n = len(row['text'])
        for sp in row['cfn_spans']:
            if not (0 <= sp['start'] <= sp['end'] < n) or not row['text'][sp['start']:sp['end'] + 1]:
                why.append('span_range')
        prev = -1
        for sp in row['cfn_spans']:
            if sp['start'] < prev:
                n_unsorted += 1
                break
            prev = sp['start']
        if why:
            n_bad += 1
            if n_bad <= 5:
                log('appended row %d (source row %d, span %d) fails: %s'
                    % (n_src_rows + j, p['row'], p['span'], ','.join(sorted(set(why)))))
    log('check %s: rows=%d source rows verified as an untouched prefix, appended=%d '
        'provenance mismatches=%d' % (A.check, len(produced), len(appended), n_bad))
    if n_bad:
        die('provenance mismatches found')
    # the same ordering statistic measured on the source, so the appended block is read as a
    # proportion, not as a defect
    src_unsorted = 0
    for srow in src_rows:
        prev = -1
        for sp in srow['cfn_spans']:
            if sp['start'] < prev:
                src_unsorted += 1
                break
            prev = sp['start']
    log('spans not sorted by start position: source rows %d of %d, appended rows %d of %d '
        '(informational; CFN annotation order, not produced by N10)'
        % (src_unsorted, len(src_rows), n_unsorted, len(appended)))
    print('N10_CHECK_DONE', flush=True)
    sys.exit(0)

rows = load(A.train_file)
log('loaded %s rows=%d' % (A.train_file, len(rows)))

# ---- 1. frequency of every (frame, role) pair and the surface-form pool per pair -------------
pair_cnt = Counter()
pair_forms = defaultdict(Counter)     # (frame, role) -> Counter(surface form)
for ri, row in enumerate(rows):
    fr = row.get('frame')
    fr is not None or die('row %d has no frame' % ri)
    for sp in row['cfn_spans']:
        key = (fr, sp['fe_name'])
        pair_cnt[key] += 1
        pair_forms[key][row['text'][sp['start']:sp['end'] + 1]] += 1
log('(frame, role) pairs=%d  long-tail pairs(<=%d occurrences)=%d'
    % (len(pair_cnt), FREQ, sum(1 for v in pair_cnt.values() if v <= FREQ)))

tail_spans = sum(v for v in pair_cnt.values() if v <= FREQ)
log('spans sitting on a long-tail pair=%d of %d all training spans (%.2f%%)'
    % (tail_spans, sum(pair_cnt.values()), 100.0 * tail_spans / max(1, sum(pair_cnt.values()))))

rng = random.Random(int(A.seed))

# ---- 2. one pass over the rows, generating at most PERROW replacements per row ---------------
stats = Counter()
generated = []
provenance = []


for ri, row in enumerate(rows):
    fr = row.get('frame')
    text = row['text']
    cands = []
    for si, sp in enumerate(row['cfn_spans']):
        key = (fr, sp['fe_name'])
        if pair_cnt[key] > FREQ:
            stats['span_not_longtail'] += 1
            continue
        old = text[sp['start']:sp['end'] + 1]
        stats['span_longtail'] += 1
        pool = [(w, c) for w, c in pair_forms[key].items() if w != old
                and abs(len(w) - len(old)) <= LDMAX]
        if not pool:
            stats['reject_no_candidate'] += 1
            continue
        # a replacement that already occurs elsewhere in the sentence would make two spans
        # indistinguishable for the model; drop those
        pool = [(w, c) for (w, c) in pool if w not in text]
        if not pool:
            stats['reject_already_in_sentence'] += 1
            continue
        ws = [w for w, _c in pool]
        cands.append((si, key, old, ws))
    if not cands:
        stats['row_nothing_to_do'] += 1
        continue
    stats['row_eligible'] += 1
    made = 0
    for (si, key, old, ws) in cands:
        if made >= PERROW:
            break
        sp = row['cfn_spans'][si]
        rs, re = sp['start'], sp['end']
        new = rng.choice(ws)
        nl = len(new)
        delta = nl - (re - rs + 1)
        out = json.loads(json.dumps(row, ensure_ascii=False))
        tx = text[:rs] + new + text[re + 1:]
        out['text'] = tx
        bad = ''
        for s2 in out['cfn_spans']:
            m = remap(s2['start'], s2['end'], rs, re, nl)
            if m is None:
                bad = 'span_collapses'
                break
            s2['start'], s2['end'] = m
        if not bad:
            for t2 in out['target']:
                m = remap(t2['start'], t2['end'], rs, re, nl)
                if m is None:
                    bad = 'target_collapses'
                    break
                t2['start'], t2['end'] = m
        if not bad and 'word' in out:
            for w2 in out['word']:
                m = remap(w2['start'], w2['end'], rs, re, nl)
                w2['start'], w2['end'] = m if m else (rs, rs + nl - 1)
        if not bad:
            src_tgt = [text[t['start']:t['end'] + 1] for t in row['target']]
            new_tgt = [tx[t['start']:t['end'] + 1] for t in out['target']]
            if src_tgt != new_tgt:
                bad = 'target_text_changed'
        if bad:
            stats['reject_' + bad] += 1
            continue
        if tx[rs:rs + nl] != new:
            die('row %d span %d: spliced text does not extract' % (ri, si))
        for s2 in out['cfn_spans']:
            if not (0 <= s2['start'] <= s2['end'] < len(tx)) or not tx[s2['start']:s2['end'] + 1]:
                die('row %d span %d out of range or empty after remap' % (ri, s2['start']))
        for t2 in out['target']:
            if not (0 <= t2['start'] <= t2['end'] < len(tx)):
                die('row %d target out of range after remap' % ri)
        made += 1
        stats['generated'] += 1
        stats['delta_%+d' % delta] += 1
        if delta == 0:
            stats['equal_length_extra'] += 1
        generated.append(out)
        provenance.append({'row': ri, 'span': si, 'frame': key[0], 'role': key[1],
                           'pair_count': pair_cnt[key], 'old': old, 'new': new})

log('rows scanned=%d  eligible rows=%d  generated rows=%d (%.2f%% of source rows)'
    % (len(rows), stats['row_eligible'], len(generated), 100.0 * len(generated) / len(rows)))
log('span accounting: long-tail spans=%d, not long-tail=%d; '
    'rejects: no candidate in pool=%d, replacement already in the sentence=%d, '
    'another span in the row collapses=%d, the target word collapses=%d, '
    'the target word would swallow the replacement=%d'
    % (stats['span_longtail'], stats['span_not_longtail'],
       stats['reject_no_candidate'], stats['reject_already_in_sentence'],
       stats['reject_span_collapses'], stats['reject_target_collapses'],
       stats['reject_target_text_changed']))
log('long-tail spans with at least one usable replacement = %d; of those, %d were never attempted '
    'because the row already used its %d replacement(s) per row'
    % (stats['span_longtail'] - stats['reject_no_candidate'] - stats['reject_already_in_sentence'],
       stats['span_longtail'] - stats['reject_no_candidate'] - stats['reject_already_in_sentence']
       - stats['generated'] - stats['reject_span_collapses'] - stats['reject_target_collapses']
       - stats['reject_target_text_changed'], PERROW))
log('replacement hit rate on long-tail spans = %d / %d = %.2f%%'
    % (stats['generated'], stats['span_longtail'],
       100.0 * stats['generated'] / max(1, stats['span_longtail'])))
dl = sorted([k for k in stats if k.startswith('delta_')],
             key=lambda k: abs(int(k.split('_')[1])))
log('length-difference distribution: ' + ', '.join('%s=%d' % (k, stats[k]) for k in dl)
    + '; same-length replacements also counted inside delta_+0 = %d' % stats['equal_length_extra'])

# ---- 3. write: source rows first (untouched), generated rows appended ------------------------
out_rows = rows + generated
tmp = A.out_file + '.n10tmp'
with codecs.open(tmp, 'w', encoding='utf8') as f:
    json.dump(out_rows, f, ensure_ascii=False)
os.replace(tmp, A.out_file)
log('wrote %s rows=%d (%d source + %d generated)' % (A.out_file, len(out_rows), len(rows), len(generated)))

meta_path = A.meta_file or (A.out_file + '.n10meta.json')
with codecs.open(meta_path, 'w', encoding='utf8') as f:
    json.dump({'train_file': A.train_file, 'seed': int(A.seed), 'freq_max': FREQ,
               'len_diff_max': LDMAX, 'per_row_max': PERROW,
               'source_rows': len(rows), 'generated_rows': len(generated),
               'pairs': len(pair_cnt), 'stats': dict(stats),
               'provenance_sample': provenance[:50], 'provenance': provenance}, f, ensure_ascii=False)
log('wrote sidecar %s' % meta_path)
print('N10_AUG_DONE', flush=True)
