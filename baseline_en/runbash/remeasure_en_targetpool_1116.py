#!/usr/bin/env python3
"""Recompute English RI ablations under the target-admitted candidate pool.

The fixed pool (3321 rows) omits target-word candidates; the deployed English cascade adds
them via the self-fe rule (share>=tau, modal role). This script checks each ablation arm
against its fixed-pool anchor, then unions the self-fe target rows and re-scores, so the
role-prior and fusion readings use the deployed protocol. Pure CPU: target rows come from
the frequency rule, not from member scores.

Anchors (old pool, strict RI F1): fn 36.7977 / la(=L3) 32.3513 / avg2 36.9239 /
pool4 37.4491 / agree k=3 37.8231 / decode-prior(L1,clamped tau.5) 37.2449.
"""
import argparse, collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import score_official as so  # noqa: E402
from score_official import build_gold, score_ri  # noqa: E402
so.ARGS = argparse.Namespace(expand_words=False)

AP = argparse.ArgumentParser()
AP.add_argument('--dump', required=True, help='dir with meta.json/labels.json/P_<arm>.npy (3321 pool)')
AP.add_argument('--prefix', default='P_t3_en_', help='npy filename prefix before the arm name')
AP.add_argument('--gold', default='dataset/task3_dev.json')
AP.add_argument('--train', default='dataset/task3_train.json')
AP.add_argument('--task1', required=True, help='dev task1 prediction json [[sid,frame],...]')
AP.add_argument('--tau_selffe', type=float, default=0.25)
AP.add_argument('--alpha', type=float, default=1.0)
AP.add_argument('--clamp', type=float, default=2.0)
AP.add_argument('--tau_prior', type=float, default=0.5)
AP.add_argument('--curve_out', default='', help='optional JSON path for the aligned count-shrunk tau/F1 curve')
A = AP.parse_args()


def log(m):
    print(m, flush=True)


def die(m):
    print('FATAL: ' + m, flush=True)
    sys.exit(1)


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


for f in (os.path.join(A.dump, 'meta.json'), os.path.join(A.dump, 'labels.json'), A.gold, A.train, A.task1):
    if not os.path.exists(f):
        die('missing input %s' % f)
meta = load(os.path.join(A.dump, 'meta.json'))
idx2label = load(os.path.join(A.dump, 'labels.json'))
NL = len(idx2label)
keys = {int(r[0]): r[1] for r in load(A.task1)}
gt = build_gold(load(A.gold))
gset = {sid: set(g['triples']) for sid, g in gt.items()}
dev_recs = load(A.gold)
train_recs = load(A.train)


def table(arm):
    p = os.path.join(A.dump, A.prefix + arm + '.npy')
    a = np.load(p)
    if a.shape != (len(meta), NL + 1):
        die('%s shape %s != pool %d x %d' % (arm, a.shape, len(meta), NL + 1))
    return a


ARMS = ['FN', 'L3', 'L0', 'L1']
P = {a: table(a) for a in ARMS}
log('pool %d rows, %d roles + NULL; arms %s' % (len(meta), NL, ','.join(ARMS)))


def decide_prod(mat):
    top = mat.argmax(axis=1)
    keep = top < NL
    nm = np.empty(mat.shape[0], dtype=object)
    nm[:] = None
    for i in np.nonzero(keep)[0]:
        nm[i] = idx2label[int(top[i])]
    return nm


def rows_of(nm):
    return [[meta[i][0], meta[i][1], meta[i][2], nm[i]] for i in range(len(meta)) if nm[i] is not None]


def score_rows(rows, tag):
    ri, rp, rr = score_ri(gt, rows)
    ok = sum(1 for p in rows if (p[1], p[2], p[3]) in gset.get(p[0], set()))
    log('%-22s emit %5d correct %5d strictF1 %8.4f (P %.4f R %.4f)' % (tag, len(rows), ok, ri, rp, rr))
    return ri, ok, len(rows)


# ---- self-fe target admission (replicates selffe_emit_1111 learn_train/emit) ----
def target_span(rec):
    tg = rec.get('target') or []
    if not tg:
        return None
    t = tg[-1]
    return (t['start'], t['end'])


def self_role(rec):
    ts = target_span(rec)
    if ts is None:
        return []
    return [s['fe_name'] for s in rec.get('cfn_spans', [])
            if (s['start'], s['end']) == ts and s['fe_name'] not in ('Not_a_FrameElement', 'EMPTY_FE', '_', '')]


n_rec, n_self = collections.Counter(), collections.Counter()
roles = collections.defaultdict(collections.Counter)
for rec in train_recs:
    f = rec.get('frame') or ''
    if not f:
        continue
    n_rec[f] += 1
    rs = self_role(rec)
    if rs:
        n_self[f] += 1
        for r in rs:
            roles[f][r] += 1
selftab = {f: dict(share=n_self[f] / float(n_rec[f]),
                   role=(roles[f].most_common(1)[0][0] if roles[f] else None)) for f in n_rec}


def with_target(base_rows):
    add = []
    for rec in dev_recs:
        f = keys.get(rec['sentence_id'], rec.get('frame') or '')
        e = selftab.get(f)
        if not e or e['share'] < A.tau_selffe or e['role'] is None:
            continue
        ts = target_span(rec)
        if ts is None:
            continue
        add.append((rec['sentence_id'], ts[0], ts[1], e['role']))
    seen = set((r[0], r[3]) for r in base_rows)
    extra = [[s, st, en, ro] for (s, st, en, ro) in add if (s, ro) not in seen]
    return list(base_rows) + extra, len(extra)


# ---- frame-conditional role prior (clamped), keyed by task1 predicted frame ----
cnt = collections.defaultdict(lambda: np.zeros(NL))
rowtot = collections.Counter()
for rec in train_recs:
    f = rec.get('frame') or ''
    if not f:
        continue
    for s in rec.get('cfn_spans', []):
        r = s['fe_name']
        if r in idx2label:
            cnt[f][idx2label.index(r)] += 1
            rowtot[f] += 1
# deployed count table includes deployment-matched negatives (NULL rows): they enlarge
# rowtot/tot exactly as the training loader does (r50 fingerprint total=roles+negatives).
neg_path = os.path.join(os.path.dirname(A.train), 'train_deploy_negatives.json')
if os.path.exists(neg_path):
    sid2frame = {r['sentence_id']: (r.get('frame') or '') for r in train_recs}
    sid2target = {r['sentence_id']: target_span(r) for r in train_recs}
    nneg = 0
    for row in load(neg_path):
        if sid2target.get(row[0]) == (row[1], row[2]):
            continue  # loader drops target-overlapping negatives (6 on English)
        f = sid2frame.get(row[0], '')
        if f:
            rowtot[f] += 1
            nneg += 1
    log('count table += %d deployment negatives (NULL rows)' % nneg)
glob = np.zeros(NL)
tot = sum(rowtot.values())
for f in cnt:
    glob += cnt[f]
glob /= max(tot, 1)
logpi = {}
for f in cnt:
    p = (cnt[f] + A.alpha * glob) / (rowtot[f] + A.alpha)
    logpi[f] = np.clip(np.log(p + 1e-12), -A.clamp, 0.0)
log('role prior: %d frames, clamp %.1f tau %.2f' % (len(logpi), A.clamp, A.tau_prior))
# frames unseen in training: cnt=0,rowtot=0 => p=glob, so clamped/unclamped offset is
# clip(log glob)/log glob (not zero); count-shrunk weight=0 there.
FB_CLAMP = np.clip(np.log(glob + 1e-12), -A.clamp, 0.0)
FB_RAW = np.log(glob + 1e-12)
FB_EB = np.zeros(NL)


def decode_prior(arm):
    lg = np.log(P[arm] + 1e-12)
    for i in range(len(meta)):
        f = keys.get(int(meta[i][0]))
        v = logpi.get(f, FB_CLAMP)
        if v is not None:
            lg[i, :NL] += A.tau_prior * v
    lg -= lg.max(axis=1, keepdims=True)
    pr = np.exp(lg)
    pr /= pr.sum(axis=1, keepdims=True)
    return pr


def agree(mat_list, k):
    votes = np.zeros((len(meta), NL), dtype=int)
    for m in mat_list:
        top = m.argmax(axis=1)
        for i in np.nonzero(top < NL)[0]:
            votes[i, top[i]] += 1
    best = votes.argmax(axis=1)
    nm = np.empty(len(meta), dtype=object)
    nm[:] = None
    for i in range(len(meta)):
        if votes[i, best[i]] >= k:
            nm[i] = idx2label[int(best[i])]
    return nm


ANCHOR = {'fn': 36.7977, 'la': 32.3513, 'avg2': 36.9239, 'pool4': 37.4491, 'agree3': 37.8231, 'decode_prior': 37.2449, 'noprior': 36.0599}
results = {}


def run(tag, nm, anchor_key, fatal=True):
    base = rows_of(nm)
    ri, ok, n = score_rows(base, tag + ' [old pool]')
    if abs(ri - ANCHOR[anchor_key]) > 1e-3:
        msg = 'anchor drift %s: %.4f vs %.4f' % (tag, ri, ANCHOR[anchor_key])
        if fatal:
            die(msg)
        log('WARN: ' + msg + ' (reconstructed prior, tolerated)')
    aligned, nadd = with_target(base)
    ri2, ok2, n2 = score_rows(aligned, tag + ' [+target]')
    log('   +%d target rows; delta %+.4f' % (nadd, ri2 - ri))
    results[tag] = (ri, ri2)


run('fn', decide_prod(P['FN']), 'fn')
run('la(L3)', decide_prod(P['L3']), 'la')
run('noprior(L1)', decide_prod(P['L1']), 'noprior')
run('avg2', decide_prod(0.5 * P['FN'] + 0.5 * P['L3']), 'avg2')
run('pool4', decide_prod(sum(P[a] for a in ARMS) / 4.0), 'pool4')
run('agree3', agree([P[a] for a in ARMS], 3), 'agree3')
run('decode_prior(L1)', decide_prod(decode_prior('L1')), 'decode_prior', fatal=False)

log('==== aligned (target-admitted) English RI, for Table IV role rows / Table V ====')
for t, (old, new) in results.items():
    log('%-18s old %8.4f -> aligned %8.4f  (%+.4f)' % (t, old, new, new - old))
np_old, np_new = results['noprior(L1)']
dp_old, dp_new = results['decode_prior(L1)']
la_old, la_new = results['la(L3)']
log('aligned role-prior decode delta %+.4f ; train delta %+.4f' % (dp_new - np_new, la_new - np_new))
log('aligned avg2 vs better-member(fn) %+.4f ; agree3 vs pool4 %+.4f'
    % (results['avg2'][1] - results['fn'][1], results['agree3'][1] - results['pool4'][1]))

# ---- aligned agreement ladder + withholding on/off (verify "all three" claims) ----
poolmat = sum(P[a] for a in ARMS) / 4.0
for k in (1, 2, 3, 4):
    nm = agree([P[a] for a in ARMS], k)
    rows, _ = with_target(rows_of(nm))
    ri, _, _ = score_ri(gt, rows)
    log('aligned agree k=%d F1 %8.4f' % (k, ri))
top = poolmat[:, :NL].argmax(axis=1)
nm_nw = np.empty(len(meta), dtype=object)
for i in range(len(meta)):
    nm_nw[i] = idx2label[int(top[i])]
rows_nw, _ = with_target(rows_of(nm_nw))
ri_nw, _, _ = score_ri(gt, rows_nw)
log('aligned pool4 withholding-OFF F1 %8.4f (withholding-ON %8.4f, gain %+.4f)'
    % (ri_nw, results['pool4'][1], results['pool4'][1] - ri_nw))

# ---- aligned best-of-grid shape sweep (for Table IV English role max rows) ----
rawlp, eblp = {}, {}
lglob = np.log(glob + 1e-12)
cntmat = np.stack([cnt[f] for f in cnt]) if cnt else np.zeros((0, NL))
rt = np.array([rowtot[f] for f in cnt], dtype=float)
with np.errstate(divide='ignore', invalid='ignore'):
    var = np.where(cntmat > 0, 1.0 / np.maximum(cntmat, 1e-300) - 1.0 / np.maximum(rt[:, None], 1e-300), np.inf)
var = np.maximum(var, 0.0)
obs = cntmat > 0
delta = np.log((cntmat + A.alpha * glob) / (rt[:, None] + A.alpha) + 1e-12) - lglob[None, :]
sig = max(float(np.var(delta[obs]) - np.mean(var[obs])), 0.0) if obs.sum() else 0.0
with np.errstate(divide='ignore', invalid='ignore'):
    w = sig / (sig + var)
w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
for j, f in enumerate(cnt):
    p = (cnt[f] + A.alpha * glob) / (rowtot[f] + A.alpha)
    rawlp[f] = np.log(p + 1e-12)
    eblp[f] = w[j] * (rawlp[f] - lglob)
log('shape sweep: sig %.6f' % sig)


def aligned_f1(mat):
    nm = decide_prod(mat)
    rows, _ = with_target(rows_of(nm))
    ri, _, _ = score_ri(gt, rows)
    return ri


base_lg = np.log(P['L1'] + 1e-12)


def correct_set(mat):
    nm = decide_prod(mat)
    rows, _ = with_target(rows_of(nm))
    return set((r[0], r[1], r[2], r[3]) for r in rows
               if (r[1], r[2], r[3]) in gset.get(r[0], set()))


def mcnemar(a, b):
    from math import comb
    bb = len(a - b)
    cc = len(b - a)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return bb, cc, min(p, 1.0)


np_set = correct_set(P['L1'])
for tag, mat in (('decode_fixed', decode_prior('L1')), ('train_L3', P['L3'])):
    bb, cc, p = mcnemar(np_set, correct_set(mat))
    log('aligned McNemar %-12s b=%d c=%d p=%.6g' % (tag, bb, cc, p))
for shape, vec, fb in (('clamped', logpi, FB_CLAMP), ('unclamped', rawlp, FB_RAW), ('count-shrunk', eblp, FB_EB)):
    best = -1.0
    best_tau = 0.0
    best_mat = None
    curve = []
    for tau in np.arange(0.0, 2.0001, 0.05):
        lg = base_lg.copy()
        for i in range(len(meta)):
            v = vec.get(keys.get(int(meta[i][0])), fb)
            if v is not None:
                lg[i, :NL] += tau * v
        lg -= lg.max(axis=1, keepdims=True)
        pr = np.exp(lg)
        pr /= pr.sum(axis=1, keepdims=True)
        f1 = aligned_f1(pr)
        curve.append([round(float(tau), 2), float(f1)])
        if f1 > best:
            best, best_tau, best_mat = f1, tau, pr
    bb, cc, p = mcnemar(np_set, correct_set(best_mat))
    log('aligned max %-13s F1 %8.4f at tau %.2f ; delta vs noprior %+.4f ; McNemar b=%d c=%d p=%.6g'
        % (shape, best, best_tau, best - np_new, bb, cc, p))
    if shape == 'count-shrunk' and A.curve_out:
        assert len(curve) == 41 and abs(curve[0][1] - np_new) < 1e-6
        with open(A.curve_out, 'w', encoding='utf-8') as f:
            json.dump({'noprior': np_new, 'best_tau': best_tau, 'best_f1': best,
                       'points': curve}, f, ensure_ascii=False, indent=1)
        log('aligned count-shrunk curve written: %s (41 points)' % A.curve_out)
# aligned input-injection decode side: fn (frame-name in input) vs L1 (no injection)
fn_set = correct_set(P['FN'])
bb, cc, pp = mcnemar(np_set, fn_set)
log('aligned input-injection decode: fn %8.4f - L1 %8.4f = %+.4f ; McNemar b=%d c=%d p=%.4g'
    % (results['fn'][1], np_new, results['fn'][1] - np_new, bb, cc, pp))
log('REMEASURE_EN_TARGETPOOL_DONE')