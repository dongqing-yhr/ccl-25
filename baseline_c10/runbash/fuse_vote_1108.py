#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""R48 (two members averaged vs each alone) and R37 (vote count / abstention switches), one CPU pass.

Both read the same per-member softmax tables, so they share all loading and row-alignment checks.
Nothing here touches the GPU or re-implements the model: the tables come from the production decoder
(`predict_task3.py --save_logits`) via runbash/probs_t3_to_npy_1108.py.

Two decode rules exist in this project and they are NOT interchangeable, so the rule is a switch:
  prod        argmax over the role columns + NULL, NULL on top means the row is not submitted.
              This is what the ladder readings of the two comparison resources were scored from, so
              it is the only rule under which those recorded numbers can be reproduced.
  incumbent   argmax over the role columns only, then drop the row when P(NULL) > tau, then drop it
              when the role name is not legal for that sentence's 框架名称 (the frame-legal list comes
              from the 任务一 predicted 框架名称). This is the 赛题 CFN 2.2 recipe that produced the
              recorded 5221 / 3161 / 56.8781.

Vote count (子项 3.3) is defined per candidate row: each member's own top answer (role name, or NULL
when that member abstains), and the count is how many members put the SAME non-NULL role name first.
The row is submitted with that role name once the count reaches the threshold being scanned.
Ties between two role names are broken by the summed probability, and the split is reported so the
number of affected rows is visible rather than hidden by a tie-break rule.

No board-side quantity is computed anywhere in this file: no weighted total, no realization rate, no
half-row threshold. Every reading is one resource's own validation set.

  PYTHONPATH=. python3 runbash/fuse_vote_1108.py --dump dataset/devladder_c10/probs \
      --arms t3_c10_L0,t3_c10_L1,t3_c10_FN,t3_c10_L3 --pair t3_c10_FN,t3_c10_L3 --rule prod
  PYTHONPATH=. python3 runbash/fuse_vote_1108.py --rule incumbent --tau 0.10 --gate veto \
      --dump dataset/dev/cmp1022/probs_union_m0p25_m0p75 --arms fn,la,ctl,neg,arg,mtl --pair fn,la \
      --task1 dataset/devladder_c22/dev_task1_test.json \
      --expect_emitted 5221 --expect_correct 3161 --expect_ri 56.8781
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import score_official as so  # noqa: E402
from score_official import build_gold, load_json, score_ri  # noqa: E402

so.ARGS = argparse.Namespace(expand_words=False)

ap = argparse.ArgumentParser()
ap.add_argument('--dump', required=True, help='dir with P_<arm>.npy (new resources) or <arm>.npy')
ap.add_argument('--prefix', default='P_', help='table filename prefix; 赛题 dump 用空串')
ap.add_argument('--arms', required=True, help='comma separated member names, vote pool')
ap.add_argument('--pair', default='', help='the two members R48 averages; empty = skip R48')
ap.add_argument('--rule', default='prod', choices=['prod', 'incumbent'])
ap.add_argument('--tau', type=float, default=0.10, help='incumbent NULL line (丙档 table note)')
ap.add_argument('--gate', default='veto', choices=['plain', 'veto'], help='incumbent frame-legal list')
ap.add_argument('--task1', default='dataset/devladder_c22/dev_task1_test.json')
ap.add_argument('--frame_roles', default='dataset/frame_roles.json')
ap.add_argument('--gold', default=None, help='default depends on the rule: 赛题 cfn-dev, else task3_dev')
ap.add_argument('--weights', default='0.5', help='weight of the first member of --pair')
ap.add_argument('--null_scan', default='0,0.05,0.10,0.20,0.30',
                 help='NULL-column thresholds to scan for the abstention switch (prod rule)')
ap.add_argument('--expect_emitted', type=int, default=0)
ap.add_argument('--expect_correct', type=int, default=0)
ap.add_argument('--expect_ri', type=float, default=0.0)
A = ap.parse_args()


def log(m):
    print('[%s] %s' % (time.strftime('%FT%TZ', time.gmtime()), m), flush=True)


def die(m):
    print('[FATAL] %s' % m, flush=True)
    sys.exit(1)


def table_path(arm):
    for cand in (os.path.join(A.dump, A.prefix + arm + '.npy'),
                 os.path.join(A.dump, arm + '.npy')):
        if os.path.exists(cand):
            return cand
    die('no probability table for member %s in %s (tried prefix %r and no prefix)'
        % (arm, A.dump, A.prefix))


arms = [s.strip() for s in A.arms.split(',') if s.strip()]
if not arms:
    die('no members parsed from --arms')
gold = A.gold or ('dataset/cfn-dev.json' if A.rule == 'incumbent' else './dataset/task3_dev.json')
for f in (os.path.join(A.dump, 'meta.json'), os.path.join(A.dump, 'labels.json'), gold):
    if not os.path.exists(f):
        die('input file missing: %s' % f)

meta = load_json(os.path.join(A.dump, 'meta.json'))
idx2label = load_json(os.path.join(A.dump, 'labels.json'))
NL = len(idx2label)
P = {}
for arm in arms:
    p = table_path(arm)
    a = np.load(p)
    if a.shape != (len(meta), NL + 1):
        die('%s table is %s, this dump is %d candidates x %d roles + NULL - mixed label spaces or '
            'a stale table from another candidate pool' % (arm, a.shape, len(meta), NL))
    P[arm] = a
    log('table %-12s %s  %.1f MB' % (arm, a.shape, os.path.getsize(p) / 1e6))
log('candidate pool %d rows, %d role columns + NULL (index %d), rule=%s, gold=%s'
    % (len(meta), NL, NL, A.rule, gold))

allowed = {}
if A.rule == 'incumbent':
    for f in (A.task1, A.frame_roles):
        if not os.path.exists(f):
            die('the incumbent rule needs the frame-legal role list, %s is missing' % f)
    fr = load_json(A.frame_roles)
    for sid, frame in load_json(A.task1):
        allowed.setdefault(int(sid), set()).update(fr.get(frame, []))
    log('frame-legal list from %s: %d sentences covered' % (A.task1, len(allowed)))

gt = build_gold(load_json(gold))
gset = {sid: set(g['triples']) for sid, g in gt.items()}
n_gold = sum(len(v) for v in gset.values())
log('gold: %d sentences, %d (start,end,role) triples (denominator of every precision/recall below)'
    % (len(gset), n_gold))


L2I = {v: i for i, v in enumerate(idx2label)}


def decide(mat, tau=None, gate=None, min_role=0.0):
    """Per candidate row the submitted role name, or None when the row is not submitted.

    min_role only bites under the prod rule and filters ON TOP of "NULL on top means drop", so
    min_role=0 must reproduce the recorded decode exactly - that is the zero point of the
    abstention-strength scan, not a second decoder.
    """
    tau = A.tau if tau is None else tau
    gate = A.gate if gate is None else gate
    n = mat.shape[0]
    if A.rule == 'prod':
        top = mat.argmax(axis=1)
        keep = top < NL
        if min_role > 0.0:
            keep &= mat[np.arange(n), np.minimum(top, NL - 1)] >= min_role
        name = np.empty(n, dtype=object)
        name[:] = None
        for i in np.nonzero(keep)[0]:
            name[i] = idx2label[int(top[i])]
        return name
    top = mat[:, :NL].argmax(axis=1)
    nullp = mat[:, NL]
    thr = nullp <= tau
    name = np.empty(n, dtype=object)
    name[:] = None
    for i in range(n):
        if not thr[i]:
            continue
        cand = idx2label[int(top[i])]
        sid = int(meta[i][0])
        # a sentence the 任务一 file says nothing about is NOT vetoed (that is the incumbent recipe;
        # dropping it instead would move rows and break the recorded anchor)
        if gate == 'veto' and sid in allowed and cand not in allowed[sid]:
            continue
        name[i] = cand
    return name


def rows_of(nm):
    return [[meta[i][0], meta[i][1], meta[i][2], nm[i]] for i in range(len(meta))
            if nm[i] is not None]


def score(nm, tag, want=None):
    rows = rows_of(nm)
    ri, rp, rr = score_ri(gt, rows)
    ok = sum(1 for p in rows if (p[1], p[2], p[3]) in gset.get(p[0], set()))
    log('%-26s 提交 %5d 行  判对 %5d 行  严格匹配 F1 %8.4f  (查准 %.4f 查全 %.4f)'
        % (tag, len(rows), ok, ri, rp, rr))
    if want is not None:
        if (len(rows), ok, round(ri, 4)) != want:
            die('anchor drifted at %r: got %d rows / %d correct / %.4f, expected %d / %d / %.4f. '
                'Stop, do not read differences off this table.'
                % (tag, len(rows), ok, ri, *want))
        log('锚点一致：%s 复现 %d 行 / %d 对 / %.4f' % (tag, len(rows), ok, ri))
    return nm, rows, ok, ri


def correct_idx(nm):
    return set(i for i in range(len(meta)) if nm[i] is not None
               and (meta[i][1], meta[i][2], nm[i]) in gset.get(int(meta[i][0]), set()))


def account(dst, src, label):
    g = dst - src
    l = src - dst
    log('%s：改对 %d 行，改错 %d 行，净 %+d 行（分母是同一批 %d 个候选 span）'
        % (label, len(g), len(l), len(g) - len(l), len(meta)))
    return len(g), len(l)


# ---------------------------------------------------------------- R48
if A.pair:
    a, b = [s.strip() for s in A.pair.split(',')]
    if a not in P or b not in P:
        die('--pair %s,%s must both be inside --arms' % (a, b))
    w = float(A.weights)
    log('')
    log('==== R48：两名成员概率表加权平均 对 各自单用（权重 %s 对 %s，取法照搬赛题现用配方，不重扫）'
        % (w, 1.0 - w))
    na, ra, oka, ria = score(decide(P[a]), 'single ' + a)
    nb, rb, okb, rib = score(decide(P[b]), 'single ' + b)
    navg = decide(w * P[a] + (1.0 - w) * P[b])
    want = None
    if A.expect_ri:
        want = (A.expect_emitted, A.expect_correct, round(A.expect_ri, 4))
    rv, rws, okv, riv = score(navg, 'average %s+%s' % (a, b), want)
    ca, cb, cv = correct_idx(na), correct_idx(nb), correct_idx(navg)
    log('两名成员判对集合的重叠 %d 行（仅 %s 判对 %d、仅 %s 判对 %d、并集 %d、平均 %d）'
        % (len(ca & cb), a, len(ca - cb), b, len(cb - ca), len(ca | cb), len(cv)))
    account(cv, ca, '平均 对 单独 %s' % a)
    account(cv, cb, '平均 对 单独 %s' % b)
    log('两名单独都判错、平均判对的行 = %d；两名单独都判对、平均丢掉的行 = %d'
        % (len(cv - ca - cb), len((ca & cb) - cv)))
    chg = [i for i in range(len(meta)) if navg[i] != na[i] or navg[i] != nb[i]]
    log('平均之后角色名称真的换了的行 = %d 个候选里占 %d（%.2f%%）'
        % (len(meta), len(chg), 100.0 * len(chg) / len(meta)))

# ---------------------------------------------------------------- R37 票数
log('')
log('==== R37 按同意成员数判定：票数池 = %s（%d 名成员）' % (','.join(arms), len(arms)))
topv = np.stack([P[arm].argmax(axis=1) for arm in arms])          # (M, N)
vote = np.empty((len(arms), len(meta)), dtype=object)
vote[:] = None
for m in range(len(arms)):
    for i in np.nonzero(topv[m] < NL)[0]:
        vote[m, i] = idx2label[int(topv[m][i])]
POOL = sum(P[arm] for arm in arms) / len(arms)
base = decide(POOL)
cb0 = correct_idx(base)
ri0 = score_ri(gt, rows_of(base))[0]
log('对照基线（%d 名成员打分等权平均后按 %s 规则提交）：提交 %d 行 判对 %d 行 严格匹配 F1 %.4f'
    % (len(arms), A.rule, len(rows_of(base)), len(cb0), ri0))
no_vote = int(np.sum(~np.any(vote != None, axis=0)))
log('没有任何成员给出非 NULL 答案的行 = %d 个候选里占 %d（一票都数不出来，只能不提交）'
    % (len(meta), no_vote))


def vote_decide(k):
    nm = np.empty(len(meta), dtype=object)
    nm[:] = None
    for i in range(len(meta)):
        c = Counter(v for v in vote[:, i] if v is not None)
        if not c:
            continue
        best = max(c.values())
        if best < k:
            continue
        cands = sorted([v for v, n in c.items() if n == best])
        if len(cands) > 1:
            tot = {v: sum(float(P[arm][i, L2I[v]]) for arm in arms
                          if vote[arms.index(arm), i] == v) for v in cands}
            cands.sort(key=lambda v: -tot[v])
        nm[i] = cands[0]
    return nm


prev = None
for k in range(len(arms), 0, -1):
    nm = vote_decide(k)
    rows = rows_of(nm)
    ri, rp, rr = score_ri(gt, rows)
    ok = sum(1 for p in rows if (p[1], p[2], p[3]) in gset.get(p[0], set()))
    log('票数>=%d（共 %d 名）：提交 %5d 行  判对 %5d 行  严格匹配 F1 %8.4f  (查准 %.4f 查全 %.4f)'
        % (k, len(arms), len(rows), ok, ri, rp, rr))
    ci = correct_idx(nm)
    if prev is not None:
        account(ci, prev, '票数>=%d 对 票数>=%d' % (k, k + 1))
    account(ci, cb0, '票数>=%d 对 打分基线' % k)
    prev = ci
k1 = vote_decide(1)
log('票数>=1 与票数>=2 之间那一档提交得多出来的行数 = %d（分母同一批 %d 个候选）'
    % (len(rows_of(k1)) - len(rows_of(vote_decide(2))), len(meta)))

# ---------------------------------------------------------------- R37 弃权开关
log('')
log('==== R37 弃权开／关：开＝NULL 排第一就不交这一行（即部署解码规则），关＝只在角色列里取第一名、每行都交')
off = POOL[:, :NL].argmax(axis=1)
nm_off = np.array([idx2label[int(v)] for v in off], dtype=object)
rows = rows_of(nm_off)
ri, rp, rr = score_ri(gt, rows)
ok = sum(1 for p in rows if (p[1], p[2], p[3]) in gset.get(p[0], set()))
log('%-26s 提交 %5d 行  判对 %5d 行  严格匹配 F1 %8.4f  (查准 %.4f 查全 %.4f)'
    % ('弃权关（每个候选 span 都交）', len(rows), ok, ri, rp, rr))
account(correct_idx(nm_off), cb0, '弃权关 对 弃权开（打分基线）')
log('弃权开而不交、弃权关却交出来的行 = %d（这些行上模型把 NULL 排在第一名角色前面）'
    % (len(rows) - len(rows_of(base))))
if A.rule == 'prod':
    log('')
    log('---- 弃权强度扫描（在"NULL 排第一就不交"之上再加一条第一名概率下限，0.00 那一档必须与记录读数逐字相同）')
    for s in [x for x in A.null_scan.split(',') if x.strip() != '']:
        t = float(s)
        nm = decide(POOL, min_role=t)
        rows = rows_of(nm)
        ri, rp, rr = score_ri(gt, rows)
        ok = sum(1 for p in rows if (p[1], p[2], p[3]) in gset.get(p[0], set()))
        log('第一名概率下限 %.2f：提交 %5d 行  判对 %5d 行  严格匹配 F1 %8.4f' % (t, len(rows), ok, ri))
print('FUSE_VOTE_DONE', flush=True)
