#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Split-decision decode: fuse the emit/not-emit vote, do NOT fuse the role vote.

Why this exists. The 10-23 fused arm averaged the whole 1011-column probability vector
(1010 roles + NULL) over four members and let that one average answer both questions - "is this span
an argument" and "which role". Both came out negative on our own dev ruler. The reference recipe allocates the vote the other way
round: a NULL panel gets 0.70 against the frame-conditioned model's 0.30, and the argument-hood head
then gets 0.50 of the final say. Most of the voting power sits on the one-dimensional
emit/not-emit decision, which is exactly where our own attribution found the signal
("gains can only come from withdrawing more"), so those two runs never tested their shape.

This script keeps the role decision byte-identical to the incumbent recipe and moves only the
abstain score:

    P_role   = sum(w_m * P_m for m in --role)          # role source, default fn+la 0.5/0.5
    role     = argmax(P_role[:n_roles])                # <- unchanged from the incumbent recipe
    panel    = sum(w_m * P_m[NULL] for m in --panel)   # the abstain panel
    inner    = --panel_weight * panel + (1-panel_weight) * P[--frame][NULL]
    null     = --head_weight * P[--head][NULL] + (1-head_weight) * inner     (skipped if --head empty)
    votes    = how many of the --vote members cast their one vote for the emitted role name
    emit     = (null <= --tau) and votes >= --vote_min and the frame-legal role list allows it

The scale-free variant. --vote m1,m2,... makes every listed member cast ONE vote for the role name
its own argmax prefers, ignoring probability size. --role_rule plurality then picks the role with the
most votes (ties broken by P_role) and --vote_min k withdraws a row when the role it emits got fewer
than k votes. That reads "merge complementary members and filter low-quality results" without ever
comparing two members' numbers against each other, which is the thing that failed on 10-23: the five
weights sit on different abstain rulers (la reads p90 0.17 where arg reads 0.85), so averaging their
probabilities only moved the line. Here the votes are counts.

Reproducing the frozen unit is the anchor: --role fn:0.5,la:0.5 --panel fn:0.5,la:0.5
--panel_weight 1.0 and no --head makes `null` the NULL column of the plain average, so the first
run below must print 5221 rows / 3161 correct / RI 56.8781. Same anchor applies with --vote as long as
--role_rule stays avg and --vote_min stays 1.

Usage (dev):
  PYTHONPATH=. python3 runbash/decode_split.py \
      --members fn=dataset/dev/cmp1022/probs_union_m0p25_m0p75/fn.npy,la=.../la.npy,... \
      --meta dataset/dev/cmp1022/probs_union_m0p25_m0p75/meta.json \
      --labels dataset/dev/cmp1022/probs_union_m0p25_m0p75/labels.json \
      --role fn:0.5,la:0.5 --panel fn:0.5,la:0.5,ctl:0.25,neg:0.25 --panel_weight 0.70 --frame fn \
      --head arg --head_weight 0.50 --tau_csv 0.04,0.06,0.08,0.10,0.12,0.15 \
      --gate veto --task1 ./dataset/dev/dev_task1_test.json --gold ./dataset/cfn-dev.json \
      --baseline dataset/dev/cmp1022/verify_dev_frozen.json \
      --out_prefix dataset/dev/cmp1022/split_probe
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_official as so  # noqa: E402
from score_official import build_gold, load_json, score_ri  # noqa: E402

# score_official.span_to_chars reads a module-level ARGS that only exists under __main__.
so.ARGS = argparse.Namespace(expand_words=False)


def die(msg):
    print('[FATAL] %s' % msg, flush=True)
    sys.exit(1)


ap = argparse.ArgumentParser()
ap.add_argument('--members', required=True, help='comma separated name=path.npy for every dump available')
ap.add_argument('--role', required=True, help='weighted members that pick the role, e.g. fn:0.5,la:0.5')
ap.add_argument('--panel', required=True, help='weighted members whose NULL column forms the abstain panel')
ap.add_argument('--panel_weight', type=float, default=1.0, help='share of the panel in the inner abstain score')
ap.add_argument('--frame', default=None, help='the 1-panel_weight share: a single member name, empty = panel only')
ap.add_argument('--head', default=None, help='argument-hood member that gets --head_weight of the final say')
ap.add_argument('--head_weight', type=float, default=0.5)
ap.add_argument('--tau_csv', default='0.10', help='comma separated abstain lines to sweep')
ap.add_argument('--op', default='full', choices=['full', 'support'],
                help='full = weighted average of the whole 1010+NULL vector (the 10-23 fused arm, negative on dev). '
                     'support = the reference recipe's operator: per candidate keep only the top-2 roles of that '
                     'member plus NULL, renormalise those three, merge stepwise by role identity and renormalise '
                     'again. Under --op support the merged vector answers BOTH questions, so --role is ignored '
                     'and the frozen 56.8781 anchor no longer applies.')
ap.add_argument('--gate', default='veto', choices=['plain', 'veto', 'rescue'])
ap.add_argument('--vote', default=None,
                help='comma separated member names that each cast ONE vote for the role name they '
                     'prefer, ignoring probability size (e.g. fn,la,ctl,neg,arg). Empty = no vote.')
ap.add_argument('--role_rule', default='avg', choices=['avg', 'plurality'],
                help='avg = role from the weighted-average vector (the incumbent recipe); '
                     'plurality = role from the vote of --vote members, ties broken by the average')
ap.add_argument('--vote_min', type=int, default=1,
                help='withdraw rows whose chosen role got fewer than this many votes (the filter)')
ap.add_argument('--min_margin', type=float, default=0.0,
                help='0 = off, and off keeps the decode byte-identical. Above 0 this is the opponent\'s '
                     'vote2 acceptance rule: a row stays only when its averaged role probability beats '
                     '--tau AND the gap between the best and the second-best role name in that averaged '
                     'vector reaches this value; a row that --vote members ALL agree on is accepted '
                     'whatever the gap (their rule: K/K votes means the probability checks are skipped). '
                     'Needs --vote to make the all-agree bypass mean anything.')
ap.add_argument('--vote_min_conf', type=float, default=0.0,
                help='0 = off, and off keeps the decode byte-identical. The other half of the opponent\'s '
                     'vote2 rule 2: a row that is NOT unanimously agreed on also needs the best averaged '
                     'role probability (max over the averaged vector, before the plurality vote may move '
                     'the role) to reach this value. Pairs with --min_margin: unanimity bypasses both.')

ap.add_argument('--null_vote', default=None,
                help='comma separated members that each cast one abstain vote: the member votes abstain '
                     'when its OWN argmax over the 1010 roles plus NULL lands on NULL. Counts only, never '
                     'probability size, so the different abstain rulers cannot distort it.')
ap.add_argument('--null_withdraw', type=int, default=0,
                help='withdraw a row when this many or more of --null_vote members vote abstain (0 = off)')
ap.add_argument('--null_reinstate', type=int, default=-1,
                help='a row the abstain line dropped is put back when at most this many members vote '
                     'abstain (-1 = off)')
ap.add_argument('--release', type=float, default=None, help='role probability that overrides the role list (gate=rescue)')
ap.add_argument('--prior_beta', type=float, default=0.0,
                help='decode-time frame-conditional prior strength. 0 = off and the decode stays '
                     'byte-identical. Above 0, a row the role list vetoes gets a second role picked '
                     'by argmax(log P_role + beta*log p(role|frame)) and is emitted when that second '
                     'role IS in the list. Rows already emitted keep their role name untouched.')
ap.add_argument('--prior_clamp', type=float, default=0.0,
                help='floor the log-prior at -this value before multiplying by beta '
                     '(0 = raw log-frequency, which reaches -27 for an unseen pair)')
ap.add_argument('--prior_train', default='./dataset/cfn-train.json',
                help='training file the (frame, role) counts come from')
ap.add_argument('--prior_side', action='store_true',
                help='split the frame prior by where the span sits against the target word '
                     '(before / after / over). A row reads the (frame, side) cell when that cell '
                     'has at least --prior_side_min_n gold roles in the training file, otherwise '
                     'it falls back to the plain frame cell. Off keeps the decode byte-identical.')
ap.add_argument('--prior_side_min_n', type=int, default=3,
                help='smallest training gold-role count that makes a (frame, side) cell trusted')
ap.add_argument('--prior_side_src', default='',
                help='jsonl holding the target-word offsets of the sentences being decoded '
                     '(dev: ./dataset/cfn-dev.json, boards: ./dataset/cfn-test-A.json / -B.json)')
ap.add_argument('--role_uniq', type=int, default=0,
                help='0 = off. 1 = inside one sentence keep only the highest-confidence row for a '
                     'given role name; rows with the same span and same role name are already one '
                     'triple for the scorer, this one targets same role name on different spans')
ap.add_argument('--sent_cap_q', type=float, default=0.0,
                help='0 = off. Otherwise: for every frame take the q-quantile of "how many role rows '
                     'that frame\'s sentences have in the training file", cap each sentence at that '
                     'number, and cut its lowest-confidence rows down to the cap')
ap.add_argument('--sent_cap_train', default=None,
                help='training file for --sent_cap_q (default: --prior_train)')
ap.add_argument('--sent_cap_min', type=int, default=1,
                help='floor on the per-frame cap so no sentence is emptied')
ap.add_argument('--sent_floor_q', type=float, default=0.0,
                help='0 = off. Mirror of --sent_cap_q for the other direction: dev row counts show we '
                     'emit FEWER rows per sentence than gold (p50 2 vs 3), so when a sentence comes in '
                     'short of its frame\'s q-quantile count, put back the highest-confidence rows that '
                     'only the abstain line refused, provided the role list and the votes allow them')
ap.add_argument('--t2_file', default=None,
                help='task2 submission file for the same split ([sid, start, end] spans). Off unless '
                     '--t2_fill names at least one reason; giving the file alone must not change a byte')
ap.add_argument('--t2_fill', default='',
                help='comma list of drop reasons that the task2 span list is allowed to override: '
                     'abstain (the abstain line refused it), vote (too few members agreed), '
                     'legal (the frame\'s role list refused it). A row is put back only when its span '
                     'is one task2圈出的 and the reason it was dropped is listed here')
ap.add_argument('--t2_fill_min_conf', type=float, default=0.0,
                help='put a dropped row back only when the panel role probability reaches this value '
                     '(0 = no filter). The abstain line dropped exactly the low-confidence rows, so '
                     'this is the knob that says how sure the role head has to be before task2 is '
                     'allowed to overrule it')
ap.add_argument('--fill_ignores_t2', action='store_true',
                help='with --t2_fill + a positive --t2_fill_min_conf, release a dropped row on the role '
                     'probability alone; the span no longer has to be one of task2\'s. This says whether '
                     'the task2 list contributes anything on top of the confidence filter')
ap.add_argument('--second_frame', default=None,
                help='per-sentence task1 top-2 file ([sid, frame1, conf1, frame2, conf2]); only read '
                     'when --second_frame_min_conf is above 0')
ap.add_argument('--second_frame_min_conf', type=float, default=0.0,
                help='0 = off. Otherwise: a row the first frame\'s role list refuses is still emitted '
                     'when its role name IS in the second-candidate frame\'s role list and that second '
                     'candidate\'s own probability reaches this value')
ap.add_argument('--second_role_max_gap', type=float, default=0.0,
                help='0 = off, and off keeps the file byte-identical. Otherwise: on every row we already '
                     'emit, ALSO emit a second role name on the same span, as an extra row. The score is '
                     'the F1 of the set of (start, end, role name) triples, so an extra name can only add '
                     'a wrong triple, it never rewrites a row that was already right; the added batch pays '
                     'for itself when its own hit rate beats role-identification F1 / 2. This is the value '
                     'the best alternative name may trail the emitted one by, in panel probability')
ap.add_argument('--second_role_min', type=float, default=0.0,
                help='floor on that alternative role name\'s own panel probability')
ap.add_argument('--second_role_scope', default='panel', choices=['panel', 'legal'],
                help='panel = the runner-up over every role name in the label space; legal = the best name '
                     'this sentence\'s frame role list allows')
ap.add_argument('--meta', required=True)
ap.add_argument('--labels', required=True)
ap.add_argument('--task1', required=True)
ap.add_argument('--frame_roles', default='./dataset/frame_roles.json')
ap.add_argument('--gold', default=None)
ap.add_argument('--baseline', default=None, help='task3 json to diff every row against (the frozen unit)')
ap.add_argument('--out_prefix', default=None, help='write <prefix>_t<tau>.json for every row when set')
ap.add_argument('--out_csv', default=None)
ap.add_argument('--expect_emitted', type=int, default=None)
ap.add_argument('--expect_correct', type=int, default=None)
ap.add_argument('--expect_ri', type=float, default=None)
A = ap.parse_args()
if A.gate == 'rescue' and A.release is None:
    die('gate=rescue needs --release')


def parse_spec(text, what):
    out = []
    for chunk in text.split(','):
        parts = chunk.split(':')
        if len(parts) != 2:
            die('%s spec must be name:weight, got "%s"' % (what, chunk))
        out.append((parts[0], float(parts[1])))
    total = sum(w for _, w in out)
    if abs(total - 1.0) > 1e-9:
        die('%s weights sum to %.6f, not 1 (%s)' % (what, total, text))
    for n, _ in out:
        if n not in M:
            die('%s names member "%s" which is not in --members (%s)' % (what, n, ','.join(M)))
    return out


M = {}
for chunk in A.members.split(','):
    parts = chunk.split('=')
    if len(parts) != 2:
        die('member spec must be name=path.npy, got "%s"' % chunk)
    if not os.path.exists(parts[1]):
        die('member %s: no dump at %s' % (parts[0], parts[1]))
    M[parts[0]] = np.load(parts[1])

meta = load_json(A.meta)
idx2label = load_json(A.labels)
NL = len(idx2label)
N = len(meta)
for n, m in M.items():
    if m.shape != (N, NL + 1):
        die('member %s has shape %s, expected (%d, %d) from meta/labels' % (n, m.shape, N, NL + 1))

role_spec = parse_spec(A.role, '--role')
panel_spec = parse_spec(A.panel, '--panel')
if A.frame and A.frame not in M:
    die('--frame member "%s" not in --members' % A.frame)
if A.head and A.head not in M:
    die('--head member "%s" not in --members' % A.head)
if A.frame is None and abs(A.panel_weight - 1.0) > 1e-9:
    die('--panel_weight %.3f without --frame drops %.3f of the probability mass' % (A.panel_weight, 1 - A.panel_weight))

VOTE = []
if A.vote:
    for n in A.vote.split(','):
        if n not in M:
            die('--vote names member "%s" which is not in --members (%s)' % (n, ','.join(M)))
        VOTE.append(n)
if A.op == 'support' and VOTE:
    die('--vote is wired for --op full only: the support operator already replaces the role decision, '
        'so there is nothing for the vote to change')
if A.role_rule == 'plurality' and not VOTE:
    die('--role_rule plurality needs --vote, otherwise there are no votes to count')
if A.vote_min > 1 and not VOTE:
    die('--vote_min %d needs --vote, otherwise no row can be counted' % A.vote_min)

NULL_VOTE = []
if A.null_vote:
    for n in A.null_vote.split(','):
        if n not in M:
            die('--null_vote names member "%s" which is not in --members (%s)' % (n, ','.join(M)))
        NULL_VOTE.append(n)
if (A.null_withdraw or A.null_reinstate >= 0) and not NULL_VOTE:
    die('--null_withdraw/--null_reinstate need --null_vote, otherwise there are no abstain votes to count')
if A.null_withdraw and (A.null_withdraw > len(NULL_VOTE)):
    die('--null_withdraw %d is above the %d voters in --null_vote: no row could ever survive'
        % (A.null_withdraw, len(NULL_VOTE)))
if A.null_reinstate >= len(NULL_VOTE):
    die('--null_reinstate %d is at or above the %d voters in --null_vote: every abstained row would be '
        'put back, which disables --tau rather than tuning it' % (A.null_reinstate, len(NULL_VOTE)))

P_role = np.zeros((N, NL + 1))
for n, w in role_spec:
    P_role += w * M[n]


def support3(mat, i):
    """The three numbers the winning operator keeps for one candidate: its top-2 roles and NULL.
    Role id -1 stands for NULL. The three are renormalised to sum 1, so a member that hedges over 40
    roles no longer brings its whole tail of tiny probabilities into the merge."""
    row = mat[i, :NL]
    k = np.argpartition(-row, 1)[:2]
    d = {-1: float(mat[i, NL])}
    for j in k:
        d[int(j)] = float(row[j])
    s = sum(d.values())
    return {r: v / s for r, v in d.items()} if s > 0 else d


def merge3(a, b, wa):
    """Merge two three-point supports by role identity, then renormalise. A role that only one member
    ranked top-2 keeps only that member's share of the vote - which is the point of the operator."""
    out = {r: wa * a.get(r, 0.0) + (1.0 - wa) * b.get(r, 0.0) for r in set(a) | set(b)}
    s = sum(out.values())
    return {r: v / s for r, v in out.items()} if s > 0 else out


top = P_role[:, :NL].argmax(axis=1)
conf = P_role[np.arange(N), top]

null_panel = np.zeros(N)
for n, w in panel_spec:
    null_panel += w * M[n][:, NL]
if A.frame:
    inner = A.panel_weight * null_panel + (1.0 - A.panel_weight) * M[A.frame][:, NL]
else:
    inner = null_panel
if A.head:
    null_final = A.head_weight * M[A.head][:, NL] + (1.0 - A.head_weight) * inner
else:
    null_final = inner

if A.op == 'support':
    # Both decisions come out of one merged three-point vector per candidate.
    top = np.zeros(N, dtype=np.int64)
    conf = np.zeros(N)
    null_final = np.zeros(N)
    for i in range(N):
        acc = {}
        for n, w in panel_spec:
            for r, p in support3(M[n], i).items():
                acc[r] = acc.get(r, 0.0) + w * p
        st = {r: p / sum(acc.values()) for r, p in acc.items()} if sum(acc.values()) > 0 else acc
        if A.frame:
            st = merge3(st, support3(M[A.frame], i), A.panel_weight)
        if A.head:
            # merge3(a, b, wa) gives a the weight wa, so the head model is passed as 1 - head_weight
            # to match the op=full formula head_weight*head + (1-head_weight)*inner.
            st = merge3(st, support3(M[A.head], i), 1.0 - A.head_weight)
        best_r, best_p = -1, -1.0
        for r, p in st.items():
            if r >= 0 and p > best_p:
                best_r, best_p = r, p
        top[i] = best_r
        conf[i] = best_p
        null_final[i] = st.get(-1, 0.0)
    print('[note ] op=support: --role %s is not consulted, the merged support picks the role' % A.role,
          flush=True)
    role_full = P_role[:, :NL].argmax(axis=1)
    print('[check] role agrees with the --role argmax on %d/%d rows (%.2f%%) | abstain score '
          'mean %.4f p50 %.4f p90 %.4f -> sweep --tau_csv across that range'
          % (int((top == role_full).sum()), N, 100.0 * float((top == role_full).mean()),
             float(null_final.mean()), float(np.percentile(null_final, 50)),
             float(np.percentile(null_final, 90))), flush=True)

MARGIN = None
if A.min_margin > 0:
    # 均值向量第一名与第二名角色名称的概率差: 只看 --role 加权出来的那个向量, 与 --vote 最终选了哪个
    # 角色无关, 这就是对手 vote2 里那条 margin。--min_margin 0 时整段跳过, 解码逐字节不变。
    _p = P_role[:, :NL]
    _idx = np.argpartition(-_p, 1, axis=1)[:, :2]
    _v = np.take_along_axis(_p, _idx, axis=1)
    MARGIN = _v.max(axis=1) - _v.min(axis=1)
    _nfail = int((MARGIN < A.min_margin).sum())
    print('[margn] --role 均值向量第一名减第二名 < %.3f 的跨度 %d/%d = %.2f%% | 这一批的分布 '
          'p10=%.3f p50=%.3f p90=%.3f | 对手那条规矩: 全员同意的跨度不看概率差照样交'
          % (A.min_margin, _nfail, N, 100.0 * _nfail / N,
             float(np.percentile(MARGIN, 10)), float(np.percentile(MARGIN, 50)),
             float(np.percentile(MARGIN, 90))), flush=True)

MEAN_TOP1 = None
if A.vote_min_conf > 0:
    # 对手规则二看的是"平均概率第一名"，取加权向量里的最大值。plurality 之后 conf[] 会变成多数票角色那一格的
    # 概率，两者在意见不一致的行上不同，所以在投票改判之前先把第一名概率存下来。
    # --vote_min_conf 0 时整段跳过，解码逐字节不变。
    MEAN_TOP1 = P_role[:, :NL].max(axis=1)
    _nlow = int((MEAN_TOP1 < A.vote_min_conf).sum())
    print('[cminf] --role 均值向量第一名 < %.3f 的跨度 %d/%d = %.2f%% | 这一批的分布 '
          'p10=%.3f p50=%.3f p90=%.3f | 对手那条规矩: 全员同意的跨度不看概率大小照样交'
          ' | low_top1=%d/%d'
          % (A.vote_min_conf, _nlow, N, 100.0 * _nlow / N,
             float(np.percentile(MEAN_TOP1, 10)), float(np.percentile(MEAN_TOP1, 50)),
             float(np.percentile(MEAN_TOP1, 90)), _nlow, N), flush=True)

votes = np.zeros(N, dtype=np.int64)
disagree_all = 0
if VOTE:
    vm = np.stack([M[n][:, :NL].argmax(axis=1) for n in VOTE])   # (K, N) role index per member
    for i in range(N):
        cnt = {}
        for k in range(len(VOTE)):
            r = int(vm[k, i])
            cnt[r] = cnt.get(r, 0) + 1
        if A.role_rule == 'plurality':
            # counts are integers and P_role is < 1, so sorting by (count, probability) lets the votes
            # decide and the averaged probability only break ties between equal vote counts.
            best = max(cnt, key=lambda r: (cnt[r], float(P_role[i, r])))
            if best != int(top[i]):
                disagree_all += 1
            top[i] = best
            conf[i] = float(P_role[i, best])
        votes[i] = cnt.get(int(top[i]), 0)
    print('[vote ] %d votes from %s | role_rule=%s | plurality moves the role on %d/%d rows (%.2f%%) | '
          'a row needs %d vote(s) to stay in'
          % (len(VOTE), A.vote, A.role_rule, disagree_all, N, 100.0 * disagree_all / N, A.vote_min), flush=True)

nullvotes = np.zeros(N, dtype=np.int64)
if NULL_VOTE:
    nv = np.stack([M[n].argmax(axis=1) for n in NULL_VOTE])       # (K, N) the member's own top choice
    nullvotes = (nv == NL).sum(axis=0)
    print('[nsvote] %d voters from %s | rows by abstain-vote count: %s | withdraw at >= %d, '
          'reinstate at <= %d'
          % (len(NULL_VOTE), A.null_vote,
             ' '.join('%d=%d' % (j, int((nullvotes == j).sum())) for j in range(len(NULL_VOTE) + 1)),
             A.null_withdraw, A.null_reinstate), flush=True)

fr = load_json(A.frame_roles)
allowed = {}
frame_of = {}
frames_of = {}
for sid, frame in load_json(A.task1):
    allowed.setdefault(int(sid), set()).update(fr.get(frame, []))
    frame_of[int(sid)] = frame
    frames_of.setdefault(int(sid), set()).add(frame)

ALLOWED2 = {}
if A.second_frame_min_conf > 0:
    if not A.second_frame:
        die('--second_frame_min_conf=%.2f needs --second_frame' % A.second_frame_min_conf)
    n_diff = 0
    for r in load_json(A.second_frame):
        sid, f2, c2 = int(r[0]), r[3], float(r[4])
        if c2 >= A.second_frame_min_conf:
            s2 = set(fr.get(f2, []))
            ALLOWED2[sid] = s2
            if s2 - allowed.get(sid, set()):
                n_diff += 1
    print('[2nd  ] 第二候选框架名自身概率 >= %.2f 的句子 %d/%d 个；其中允许的角色名称比第一候选多'
          '（有第一候选覆盖不到的角色名）的 %d 个'
          % (A.second_frame_min_conf, len(ALLOWED2), len(frames_of), n_diff), flush=True)

TOP_PRIOR = None
if A.prior_beta > 0:
    # same estimator train_task3.py uses for the la member (la_mode=frame, la_alpha=1.0 smoothing
    # toward the global role frequency), applied at decode time instead of inside the loss, so the
    # strength becomes a knob. Counting only the gold role labels of the training file.
    cnt = {}
    per_frame = {}
    frame_seen = set()
    for line in load_json(A.prior_train):
        f = line['frame']
        frame_seen.add(f)
        d = cnt.setdefault(f, {})
        for sp in line['cfn_spans']:
            d[sp['fe_name']] = d.get(sp['fe_name'], 0) + 1
            per_frame[f] = per_frame.get(f, 0) + 1
    glob = {}
    tot = 0
    for d in cnt.values():
        for r, c in d.items():
            glob[r] = glob.get(r, 0) + c
            tot += c
    PFRAME = {}
    for f in cnt:
        nf = per_frame[f]
        PFRAME[f] = {r: (cnt[f].get(r, 0) + 1.0 * glob.get(r, 0) / max(tot, 1)) / (nf + 1.0)
                     for r in glob}

    def side_of(st, en, t0, t1):
        if en <= t0:
            return 'before'
        if st >= t1:
            return 'after'
        return 'over'

    SIDE_T = {}
    SCNT = {}
    STOT = {}
    n_amb = 0
    if A.prior_side:
        if not A.prior_side_src:
            die('--prior_side needs --prior_side_src: the file holding the target-word offsets '
                'of the sentences being decoded')
        for line in load_json(A.prior_side_src):
            tg = line.get('target') or []
            if not tg:
                continue
            sid = int(line['sentence_id'])
            t = (int(tg[0]['start']), int(tg[0]['end']))
            if sid in SIDE_T and SIDE_T[sid] != t:
                n_amb += 1                      # same id used by two different target words
                continue
            SIDE_T[sid] = t
        for line in load_json(A.prior_train):
            tg = line.get('target') or []
            if not tg or not line.get('cfn_spans'):
                continue
            t0, t1 = int(tg[0]['start']), int(tg[0]['end'])
            for sp in line['cfn_spans']:
                k = (line['frame'], side_of(int(sp['start']), int(sp['end']), t0, t1))
                d = SCNT.setdefault(k, {})
                d[sp['fe_name']] = d.get(sp['fe_name'], 0) + 1
                STOT[k] = STOT.get(k, 0) + 1
    LP = np.zeros((N, NL), dtype=np.float64)
    n_known = n_side = 0
    for i in range(N):
        f = frame_of.get(int(meta[i][0]))
        if f is None or f not in frame_seen:
            continue                     # frame unknown to train -> no shift, as in the trainer
        row = per_frame[f]
        d = cnt[f]
        use_side = False
        if A.prior_side:
            ts = SIDE_T.get(int(meta[i][0]))
            if ts:
                k = (f, side_of(int(meta[i][1]), int(meta[i][2]), ts[0], ts[1]))
                if STOT.get(k, 0) >= A.prior_side_min_n:
                    row, d, use_side = STOT[k], SCNT[k], True
                    n_side += 1
        back = PFRAME[f] if use_side else glob
        for j, rname in enumerate(idx2label):
            b = back.get(rname, 0.0) / (max(tot, 1) if not use_side else 1.0)
            LP[i, j] = np.log((d.get(rname, 0) + 1.0 * b) / (row + 1.0))
        n_known += 1
    if A.prior_clamp > 0:
        LP = np.clip(LP, -A.prior_clamp, 0.0)
    TOP_PRIOR = (np.log(np.clip(P_role[:, :NL], 1e-12, None)) + A.prior_beta * LP).argmax(axis=1)
    print('[prior ] beta=%.3f clamp=%s | %d/%d rows have a frame the train file knows | train gives '
          '%d frames, %d (frame,role) pairs, %d role labels | prior role differs from the panel '
          'role on %d/%d rows (%.2f%%)'
          % (A.prior_beta, A.prior_clamp or 'raw', n_known, N, len(cnt), sum(len(d) for d in cnt.values()),
             tot, int((TOP_PRIOR != top).sum()), N, 100.0 * float((TOP_PRIOR != top).mean())), flush=True)
    if A.prior_side:
        print('[prior ] 位置这一维：训练集 (框架名称,位置) 格 %d 个（计数 >= %d 的 %d 个）| '
              '%d/%d 行用上位置格，其余退回只按框架名称查 | 待解码句子里目标词有冲突编号的 %d 句'
              % (len(SCNT), A.prior_side_min_n, sum(1 for v in STOT.values() if v >= A.prior_side_min_n),
                 n_side, N, n_amb), flush=True)

base = None
if A.baseline:
    base = set((int(p[0]), p[1], p[2], p[3]) for p in load_json(A.baseline))
gold = build_gold(load_json(A.gold)) if A.gold else None
gt = {sid: set(g['triples']) for sid, g in gold.items()} if gold else None

FILL = set()
SP2 = {}
if A.t2_fill:
    for x in A.t2_fill.split(','):
        if not x:
            continue
        if x not in ('abstain', 'vote', 'legal'):
            die('--t2_fill only accepts abstain/vote/legal, got "%s"' % x)
        FILL.add(x)
    if not A.t2_file and not A.fill_ignores_t2:
        die('--t2_fill=%s needs --t2_file (or --fill_ignores_t2)' % A.t2_fill)
    if A.fill_ignores_t2 and A.t2_fill_min_conf <= 0:
        die('--fill_ignores_t2 releases rows on role probability alone, so --t2_fill_min_conf must be > 0')
if A.t2_file:
    for it in load_json(A.t2_file):
        SP2.setdefault(int(it[0]), set()).add((int(it[1]), int(it[2])))
    print('[t2   ] 任务二 %d 个句子、%d 个跨度 | 开关放行的砍掉原因: %s'
          % (len(SP2), sum(len(v) for v in SP2.values()), ','.join(sorted(FILL)) or '无（只登记不放行）'),
          flush=True)


def in_t2(sid, st, en):
    return (int(st), int(en)) in SP2.get(int(sid), ())

CAPF = None
FLOORF = None
if A.sent_cap_q > 0 or A.sent_floor_q > 0:
    # One number per frame: how many role rows a sentence of that frame actually carries in the
    # training file (counted on the scorer's own triple set, so a span reported twice counts once).
    # np.percentile's default linear interpolation is the quantile definition used by both knobs.
    ctr = load_json(A.sent_cap_train or A.prior_train)
    gtr = build_gold(ctr)
    per = {}
    for sid, g in gtr.items():
        legal = {f: set(fr.get(f, [])) for f in g['frame']}
        for f in g['frame']:
            k = sum(1 for t in g['triples'] if t[2] in legal[f])
            per.setdefault(f, []).append(k)
    allv = [x for v in per.values() for x in v]

    def _q(vals, p):
        return max(1, int(round(float(np.percentile(vals, 100.0 * p)))))

    def _hist(d):
        h = {}
        for v in d.values():
            h[v] = h.get(v, 0) + 1
        return ' '.join('%d:%d' % kv for kv in sorted(h.items()))

    gcap = gfloor = -1
    if A.sent_cap_q > 0:
        CAPF = {f: max(A.sent_cap_min, _q(v, A.sent_cap_q)) for f, v in per.items()}
        gcap = max(A.sent_cap_min, _q(allv, A.sent_cap_q))
    if A.sent_floor_q > 0:
        FLOORF = {f: _q(v, A.sent_floor_q) for f, v in per.items()}
        gfloor = _q(allv, A.sent_floor_q)
    print('[count] train sentences %d, frames %d | q_cap=%.2f -> global %d, per-frame %s | '
          'q_floor=%.2f -> global %d, per-frame %s | dev sentences with a trained frame %d/%d'
          % (len(gtr), len(per), A.sent_cap_q, gcap, _hist(CAPF) if CAPF else '-',
             A.sent_floor_q, gfloor, _hist(FLOORF) if FLOORF else '-',
             sum(1 for s in frames_of if frames_of[s] & set(per)), len(frames_of)), flush=True)

taus = [float(x) for x in A.tau_csv.split(',')]
print('[cfg  ] role=%s | panel=%s weight=%.2f frame=%s | head=%s weight=%.2f | gate=%s release=%s | '
      'vote=%s rule=%s min=%d margin=%s | nsvote=%s wd=%d ri=%d | prior beta=%s clamp=%s | taus=%s | '
      'role_uniq=%d sent_cap_q=%s sent_floor_q=%s t2_fill=%s min_conf=%.2f%s second_frame_min_conf=%.2f'
      % (A.role, A.panel, A.panel_weight, A.frame or '-', A.head or '-', A.head_weight, A.gate, A.release,
         A.vote or '-', A.role_rule, A.vote_min, A.min_margin,
         A.null_vote or '-', A.null_withdraw, A.null_reinstate,
         A.prior_beta, A.prior_clamp or 'raw', A.tau_csv, A.role_uniq, A.sent_cap_q or 'off',
         A.sent_floor_q or 'off', A.t2_fill or 'off', A.t2_fill_min_conf,
         ' 不看任务二集合' if A.fill_ignores_t2 else '', A.second_frame_min_conf),
      flush=True)
header = ('tau,emitted,correct,RI,ri_p,ri_r,abstain,vetoed,rescued,new_rows,new_correct,new_hit,'
          'gone_rows,gone_correct,disagree,lowvote,nsvote_withdrawn,nsvote_reinstated,prior_rescued,'
          'uniq_dropped,cap_dropped,floor_added,t2_abstain,t2_vote,t2_legal,t2_correct,second_frame_rows,'
          'second_role_rows,second_role_correct,margin_dropped,margin_saved')
print('[hdr  ] ' + header)
LIDX = {r: j for j, r in enumerate(idx2label)}
ORD4 = np.argsort(-P_role[:, :NL], axis=1)[:, :4] if A.second_role_max_gap > 0 else None
if A.second_role_max_gap > 0:
    print('[2role] 同跨度加交第二个角色名称：概率差 <= %.2f，第二个名字自身概率 >= %.2f，候选取 %s | '
          '已交出的跨度 %d 条（加行不改行，得分是 (起,止,角色名称) 三元组集合的 F1）'
          % (A.second_role_max_gap, A.second_role_min,
             '本句框架名称允许列表内' if A.second_role_scope == 'legal' else '全体角色名称里的次高',
             N), flush=True)

rows = []
for tau in taus:
    out = []
    oc = []
    refused = []
    n_null = n_veto = n_resc = n_low = n_dis = n_nsw = n_nsi = n_pri = n_2f = 0
    n_margin = n_msave = 0
    n_conf = n_csave = 0
    sr_rows = []
    t2put = {'abstain': [], 'vote': [], 'legal': []}
    for i in range(N):
        sid, st, en = meta[i]
        ok2 = (conf[i] >= A.t2_fill_min_conf) if A.fill_ignores_t2 else \
              (bool(SP2) and conf[i] >= A.t2_fill_min_conf and in_t2(sid, st, en))
        why = None                      # which gate this row only passed because task2 圈出了它
        if null_final[i] > tau:
            if ok2 and 'abstain' in FILL:
                why = 'abstain'
            elif A.null_reinstate >= 0 and nullvotes[i] <= A.null_reinstate:
                n_nsi += 1
            else:
                n_null += 1
                if FLOORF is not None:
                    rname = idx2label[int(top[i])]
                    if (not VOTE or votes[i] >= A.vote_min) and rname in allowed.get(sid, ()):
                        refused.append([sid, st, en, rname, float(conf[i])])
                continue
        if A.null_withdraw and nullvotes[i] >= A.null_withdraw:
            n_nsw += 1
            continue
        if VOTE and votes[i] < A.vote_min:
            if ok2 and 'vote' in FILL and why is None:
                why = 'vote'
            else:
                n_low += 1
                continue
        if MARGIN is not None and MARGIN[i] < A.min_margin:
            # 对手的规矩: 投票成员全体同意同一个角色名称时跳过概率检查, 照样交
            if VOTE and votes[i] == len(VOTE):
                n_msave += 1
            else:
                n_margin += 1
                continue
        if MEAN_TOP1 is not None and MEAN_TOP1[i] < A.vote_min_conf:
            if VOTE and votes[i] == len(VOTE):
                n_csave += 1
            else:
                n_conf += 1
                continue
        name = idx2label[int(top[i])]
        if sid in allowed and name not in allowed[sid]:
            if A.gate == 'rescue' and conf[i] >= A.release:
                n_resc += 1
            elif A.gate == 'plain':
                pass
            elif TOP_PRIOR is not None:
                alt = idx2label[int(TOP_PRIOR[i])]
                if alt in allowed[sid]:
                    n_pri += 1
                    name = alt
                elif ALLOWED2 and name in ALLOWED2.get(sid, ()):
                    n_2f += 1
                elif ALLOWED2 and alt in ALLOWED2.get(sid, ()):
                    n_2f += 1
                    name = alt
                elif ok2 and 'legal' in FILL:
                    t2put['legal'].append([sid, st, en, name, float(conf[i])])
                    continue
                else:
                    n_veto += 1
                    continue
            elif ALLOWED2 and name in ALLOWED2.get(sid, ()):
                n_2f += 1
            elif ok2 and 'legal' in FILL:
                t2put['legal'].append([sid, st, en, name, float(conf[i])])
                continue
            else:
                n_veto += 1
                continue
        if why is not None:
            t2put[why].append([sid, st, en, name, float(conf[i])])
        out.append([sid, st, en, name])
        oc.append(float(conf[i]))
        if ORD4 is not None:
            j_name = LIDX.get(name)
            alt = None
            if j_name is not None:
                if A.second_role_scope == 'legal':
                    pool = [j for j in range(NL) if idx2label[j] in allowed.get(int(sid), ())] \
                        if int(sid) in allowed else []
                else:
                    pool = [int(j) for j in ORD4[i]]
                for j in pool:
                    if j == j_name:
                        continue
                    p = float(P_role[i, j])
                    if alt is None or p > alt[1]:
                        alt = (j, p)
            p_name = float(P_role[i, j_name]) if j_name is not None else 0.0
            if alt and p_name - alt[1] <= A.second_role_max_gap and alt[1] >= A.second_role_min:
                out.append([sid, st, en, idx2label[alt[0]]])
                oc.append(alt[1])
                sr_rows.append([sid, st, en, idx2label[alt[0]], alt[1]])
        if VOTE and votes[i] < len(VOTE):
            n_dis += 1

    n_t2 = {}
    for k in ('abstain', 'vote', 'legal'):
        n_t2[k] = len(t2put[k])
        if k == 'legal':
            for r in t2put[k]:
                out.append(r[:4])
                oc.append(r[4])
    if any(n_t2.values()):
        pos = {}
        for j, (ms, mst, men) in enumerate(meta):
            pos.setdefault((int(ms), int(mst), int(men)), j)
        z = sorted(range(len(out)), key=lambda j: pos.get((out[j][0], out[j][1], out[j][2]), 1 << 30))
        out = [out[j] for j in z]
        oc = [oc[j] for j in z]
        msg = ' '.join('%s=%d' % (k, n_t2[k]) for k in ('abstain', 'vote', 'legal'))
        hit = ''
        cf = [r[4] for k in ('abstain', 'vote', 'legal') for r in t2put[k]]
        if gold:
            hit = ' | 判对 %d/%d = %.2f%%（门槛 28.44%%）' % (
                sum(1 for k in n_t2 for r in t2put[k]
                    if (r[1], r[2], r[3]) in gt.get(int(r[0]), set())),
                sum(n_t2.values()),
                100.0 * sum(1 for k in n_t2 for r in t2put[k]
                            if (r[1], r[2], r[3]) in gt.get(int(r[0]), set()))
                / max(sum(n_t2.values()), 1))
        print('[t2put] tau=%.2f %s被放回的行: %s%s' % (tau,
              '只看角色概率（不看任务二集合）' if A.fill_ignores_t2 else '因任务二圈出而', msg, hit),
              flush=True)
        if cf:
            print('        放回的这些行自己的角色概率分位: p50=%.2f p75=%.2f p90=%.2f'
                  % (float(np.percentile(cf, 50)), float(np.percentile(cf, 75)),
                     float(np.percentile(cf, 90))), flush=True)

    n_uniq = n_cap = n_floor = 0
    if A.role_uniq:
        best = {}
        for j, row in enumerate(out):
            key = (row[0], row[3])
            if key not in best or oc[j] > oc[best[key]]:
                best[key] = j
        keep = set(best.values())
        n_uniq = len(out) - len(keep)
        out = [r for j, r in enumerate(out) if j in keep]
        oc = [c for j, c in enumerate(oc) if j in keep]
    if CAPF is not None:
        bysid = {}
        for j, row in enumerate(out):
            bysid.setdefault(row[0], []).append(j)
        drop = set()
        for sid, js in bysid.items():
            c = max([CAPF.get(f, gcap) for f in frames_of[sid]] or [gcap])
            if len(js) > c:
                drop.update(sorted(js, key=lambda j: oc[j])[:len(js) - c])
        n_cap = len(drop)
        out = [r for j, r in enumerate(out) if j not in drop]
        oc = [c for j, c in enumerate(oc) if j not in drop]
    if FLOORF is not None:
        have = {}
        for row in out:
            have[row[0]] = have.get(row[0], 0) + 1
        need = {}
        for sid, frames in frames_of.items():
            t = max([FLOORF.get(f, gfloor) for f in frames])
            d = t - have.get(sid, 0)
            if d > 0:
                need[sid] = d
        for r in sorted(refused, key=lambda x: -x[4]):
            if need.get(r[0], 0) > 0:
                out.append(r[:4])
                oc.append(r[4])
                need[r[0]] -= 1
                n_floor += 1
    if n_floor:
        pos = {}
        for j, (ms, mst, men) in enumerate(meta):
            pos.setdefault((int(ms), int(mst), int(men)), j)
        z = sorted(range(len(out)), key=lambda j: pos.get((out[j][0], out[j][1], out[j][2]), 1 << 30))
        out = [out[j] for j in z]
        oc = [oc[j] for j in z]
    if A.role_uniq or CAPF is not None or FLOORF is not None:
        print('[post ] tau=%.2f 同角色名重复砍 %d 行 | 每句上限砍 %d 行 | 每句下限补 %d 行（被砍掉的弃权行里挑的，'
              '补之前交 %d 行）| 余 %d 行'
              % (tau, n_uniq, n_cap, n_floor, len(out) - n_floor, len(out)), flush=True)

    n_sr = len(sr_rows)
    sr_ok = -1
    if n_sr:
        sr_ok = sum(1 for r in sr_rows if gold and (r[1], r[2], r[3]) in gt.get(int(r[0]), set()))
        print('[2role] tau=%.2f 在同一跨度上加交第二个角色名称 %d 行，这批判对 %d = %.2f%% | 第二个名字自身'
              '概率 p50=%.2f p90=%.2f | 判赚门槛 = 本链角色识别 F1 的一半（dev 28.44%% / A 30.61%% / B 29.75%%）'
              % (tau, n_sr, sr_ok, 100.0 * sr_ok / n_sr,
                 float(np.percentile([r[4] for r in sr_rows], 50)),
                 float(np.percentile([r[4] for r in sr_rows], 90))), flush=True)
    if A.out_prefix:
        path = '%s_t%s.json' % (A.out_prefix, str(tau).replace('-', 'm').replace('.', 'p'))
        json.dump(out, open(path, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    ri = rp = rr = float('nan')
    ok = -1
    if gold:
        ri, rp, rr = score_ri(gold, out)
        ok = sum(1 for p in out if (p[1], p[2], p[3]) in gt.get(int(p[0]), set()))
    new_c = new_hit = gone_c = -1
    new_n = gone_n = 0
    if base is not None:
        cur = set((int(p[0]), p[1], p[2], p[3]) for p in out)
        added = cur - base
        gone = base - cur
        new_n, gone_n = len(added), len(gone)
        if gold:
            new_c = sum(1 for t in added if (t[1], t[2], t[3]) in gt.get(t[0], set()))
            gone_c = sum(1 for t in gone if (t[1], t[2], t[3]) in gt.get(t[0], set()))
            new_hit = 100.0 * new_c / new_n if new_n else 0.0
    t2_ok = sum(1 for k in ('abstain', 'vote', 'legal') for r in t2put[k]
                if gold and (r[1], r[2], r[3]) in gt.get(int(r[0]), set()))
    line = ('%.2f,%d,%d,%.4f,%.4f,%.4f,%d,%d,%d,%d,%d,%.2f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d'
            % (tau, len(out), ok, ri, rp, rr, n_null, n_veto, n_resc, new_n, new_c, new_hit, gone_n, gone_c,
               (n_dis if VOTE else -1), n_low, n_nsw, n_nsi, n_pri, n_uniq, n_cap, n_floor,
               n_t2['abstain'], n_t2['vote'], n_t2['legal'], t2_ok, n_2f, n_sr, sr_ok,
               n_margin, n_msave))
    print('[row  ] ' + line + ('   <- new-row hit rate must beat 28.44%% to pay for itself'
                               if new_n and gold else ''), flush=True)
    if MEAN_TOP1 is not None:
        # 只在 --vote_min_conf 开着时打这一行，默认路径的 stdout 一个字节都不多。
        print('[cmin ] tau=%.2f 均值向量第一名 < %.3f 砍掉 %d 行 | 其中投票全员同意、按规矩不看概率照样留 %d 行'
              ' | conf_dropped=%d conf_saved=%d'
              % (tau, A.vote_min_conf, n_conf, n_csave, n_conf, n_csave), flush=True)
    rows.append(line)
    if A.expect_emitted is not None and len(taus) == 1:
        if len(out) != A.expect_emitted:
            die('kept %d rows, expected %d' % (len(out), A.expect_emitted))
        if gold and A.expect_correct is not None and ok != A.expect_correct:
            die('gold-matching rows = %d, expected %d' % (ok, A.expect_correct))
        if gold and A.expect_ri is not None and abs(ri - A.expect_ri) > 5e-5:
            die('RI = %.4f, expected %.4f' % (ri, A.expect_ri))

if A.out_csv:
    with open(A.out_csv, 'w', encoding='utf8') as fh:
        fh.write('cell,' + header + '\n')
        for line in rows:
            fh.write('%s,%s\n' % (os.path.basename(A.out_prefix or 'split'), line))
    print('[csv  ] wrote %d rows -> %s' % (len(rows), A.out_csv))
print('SPLIT_DONE', flush=True)
