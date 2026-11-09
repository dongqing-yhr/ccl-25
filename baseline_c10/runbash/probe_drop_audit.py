#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Audit what the frozen task3 recipe throws away.

Stage 1 (GPU): run both members on one task2 cell and dump the averaged probability matrix, so
every later question is CPU-only.  Stage 2 (CPU): reproduce the recorded cell (row count and RI
must match the chain log), then break the dropped rows down by abstain score and by veto reason,
and recompute RI exactly for each threshold, veto-handling and oracle variant.

  PYTHONPATH=. python3 runbash/probe_drop_audit.py              # stage 1 + 2, needs the GPU
  PYTHONPATH=. python3 runbash/probe_drop_audit.py --analyze    # stage 2 only, CPU
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from functools import partial
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer
from model_task3 import Model

ap = argparse.ArgumentParser()
ap.add_argument('--task2_file', default='dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json')
ap.add_argument('--tau', type=float, default=0.10)
ap.add_argument('--gold', default='./dataset/cfn-dev.json')
ap.add_argument('--task1', default='./dataset/dev/dev_task1_test.json')
ap.add_argument('--frame_info', default='./dataset/frame_info.json')
ap.add_argument('--frame_roles', default='./dataset/frame_roles.json')
ap.add_argument('--dump', default='dataset/dev/cmp1022/audit_union_m0p25_m0p75')
ap.add_argument('--analyze', action='store_true')
A = ap.parse_args()


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def die(m):
    print('[%s] FATAL %s' % (time.strftime('%H:%M:%S'), m), flush=True)
    sys.exit(1)


MC = './chinese-macbert-large/config.json'
MV = './chinese-macbert-large/vocab.txt'
TEST = './dataset/cfn-dev.json'
FN_W = './saves/model_task3_fn_best.bin'
LA_W = './saves/model_task3_la_best.bin'


class Dataset(torch.utils.data.Dataset):
    """Verbatim copy of runbash/ens_task3.py Dataset - keep in sync or the audit is meaningless."""

    def __init__(self, jf, lf, t2f, t1f, tok, with_frame_name=False):
        self.wfn = with_frame_name
        self.tok = tok
        self.all_data = json.load(open(jf, encoding='utf8'))
        self.ori_labels = json.load(open(lf, encoding='utf8'))
        self.sid2frame = {}
        if with_frame_name:
            for it in json.load(open(t1f, encoding='utf8')):
                self.sid2frame[it[0]] = it[1]
        self.idx2label = []
        for line in self.ori_labels:
            for fes in line['fes']:
                if fes['fe_name'] not in self.idx2label:
                    self.idx2label.append(fes['fe_name'])
        self.dd = {}
        for line in self.all_data:
            self.dd[line['sentence_id']] = {'text': line['text'],
                                            'target': [line['target'][-1]['start'] + 1,
                                                       line['target'][-1]['end'] + 1]}
        self.data = []
        for line in json.load(open(t2f, encoding='utf8')):
            rec = self.dd.get(line[0])
            if rec is None:
                continue
            text = rec['text']
            tg = rec['target']
            st, en = line[1], line[2]
            n = len(text)
            if st < 0 or en >= n or st > en:
                continue
            if en + 1 < tg[0]:
                li = [st + 1, en + 1]
            elif st + 1 > tg[1]:
                li = [st + 3, en + 3]
            else:
                continue
            if li[1] >= n + 4:
                continue
            self.data.append({'text': text, 'label_idx': li, 'sentence_id': line[0],
                              'frame': self.sid2frame.get(line[0]), 'target': tg,
                              'ori_target': [st, en]})
        self.num_labels = len(self.idx2label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        e = self.tok.encode_plus(list(d['text']))
        ids = e.data['input_ids']
        am = e.data['attention_mask']
        t = d['target']
        ids = ids[0:t[0]] + [1] + ids[t[0]:t[1] + 1] + [2] + ids[t[1] + 1:]
        am = am + [1, 1]
        if self.wfn and d.get('frame'):
            f = self.tok.convert_tokens_to_ids(list(d['frame']))
            ids = ids + [self.tok.sep_token_id] + f
            am = am + [1] * (len(f) + 1)
        return ids, am, d['label_idx'], d['sentence_id'], d['ori_target']


def collate(data, device=None):
    ml = max(len(x[0]) for x in data)
    pad = lambda d, v=0: d + [v] * (ml - len(d))
    ids = torch.from_numpy(np.array([pad(x[0]) for x in data], dtype=np.compat.long)).to(device)
    am = torch.from_numpy(np.array([pad(x[1]) for x in data], dtype=np.compat.long)).to(device)
    return ids, am, [x[2] for x in data], [x[3] for x in data], [x[4] for x in data]


PROB = A.dump + '.prob.npy'
META = A.dump + '.meta.json'
LABS = A.dump + '.labels.json'

if not os.path.exists(PROB):
    if A.analyze:
        die('no probability dump at %s - run stage 1 without --analyze first' % PROB)
    log('stage 1: forward both members on %s' % A.task2_file)
    if not torch.cuda.is_available():
        die('cuda not available, stage 1 needs the GPU (or reuse an existing dump with --analyze)')
    dev = torch.device('cuda')
    tok = BertTokenizer(vocab_file=MV, do_lower_case=True)
    ds_fn = Dataset(TEST, A.frame_info, A.task2_file, A.task1, tok, True)
    ds_la = Dataset(TEST, A.frame_info, A.task2_file, A.task1, tok, False)
    if len(ds_fn) != len(ds_la):
        die('member datasets differ: %d vs %d' % (len(ds_fn), len(ds_la)))
    NL = ds_fn.num_labels
    recs = ds_fn.data
    log('candidates scored: %d, roles: %d' % (len(ds_fn), NL))

    def probs(path, ds):
        cfg = BertConfig.from_json_file(MC)
        cfg.num_labels = NL
        cfg.with_null = True
        m = Model(cfg)
        m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        m = m.to(dev).eval()
        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                        collate_fn=partial(collate, device=dev))
        P = np.zeros((len(ds), NL + 1), dtype=np.float32)
        pos = 0
        with torch.no_grad():
            for ids, am, tg, sid, ori in dl:
                p = F.softmax(m(input_ids=ids, attention_mask=am, target=tg, labels=None,
                                device=dev, for_test=True)['logits'], dim=-1).cpu().numpy()
                P[pos:pos + len(ori)] = p
                pos += len(ori)
                if (pos // 800) != ((pos - len(ori)) // 800):
                    log('  %-28s %d/%d' % (os.path.basename(path), pos, len(ds)))
        del m
        torch.cuda.empty_cache()
        if pos != len(ds):
            die('%s covered %d of %d' % (path, pos, len(ds)))
        return P

    Pfn = probs(FN_W, ds_fn)
    log('  FN member done')
    Pla = probs(LA_W, ds_la)
    log('  LA member done')
    np.save(PROB, (Pfn + Pla) / 2.0)
    json.dump([[r['sentence_id'], r['ori_target'][0], r['ori_target'][1]] for r in recs], open(META, 'w'))
    json.dump(ds_fn.idx2label, open(LABS, 'w'))
    log('  dump written: %s (%.1f MB)' % (PROB, os.path.getsize(PROB) / 1e6))
    del Pfn, Pla
else:
    log('stage 1 skipped, reusing %s' % PROB)

log('stage 2: analyse (CPU only)')
avg = np.load(PROB).astype(np.float32)
meta = json.load(open(META))
idx2label = json.load(open(LABS))
NL = len(idx2label)
if avg.shape != (len(meta), NL + 1):
    die('dump shape %s does not match meta %d / labels %d' % (avg.shape, len(meta), NL + 1))

gold_data = json.load(open(A.gold, encoding='utf8'))
gold = {}
for line in gold_data:
    slot = gold.setdefault(line['sentence_id'], set())
    for s in line.get('cfn_spans', []):
        slot.add((s['start'], s['end'], s['fe_name']))
gspans = {sid: {(a, b) for a, b, _ in t} for sid, t in gold.items()}
n_gold = sum(len(t) for t in gold.values())
log('gold triples: %d over %d annotated sentences' % (n_gold, len(gold)))

fr = json.load(open(A.frame_roles, encoding='utf8'))
allowed = {}
for sid, frame in json.load(open(A.task1, encoding='utf8')):
    allowed.setdefault(sid, set()).update(fr.get(frame, []))
log('sentences with a legal-role list: %d' % len(allowed))

role2idx = {r: i for i, r in enumerate(idx2label)}
allowed_idx = {sid: [role2idx[r] for r in sorted(s) if r in role2idx] for sid, s in allowed.items()}

pnull = avg[:, NL]
top = np.argmax(avg[:, :NL], axis=1)
conf = avg[np.arange(len(meta)), top]

rows = []
for i in range(len(meta)):
    sid, st, en = meta[i]
    name = idx2label[top[i]]
    if pnull[i] > A.tau:
        status = 'abstain'
    elif sid in allowed_idx and name not in allowed[sid]:
        status = 'veto'
    else:
        status = 'emit'
    rows.append({'i': i, 'sid': sid, 'st': st, 'en': en, 'role': name,
                 'pnull': float(pnull[i]), 'conf': float(conf[i]), 'status': status,
                 'ok': (st, en, name) in gold.get(sid, set()),
                 'span_ok': (st, en) in gspans.get(sid, set())})


def ri(triples):
    """Mirrors score_official.score_ri exactly: per-sentence set of (start,end,role), and only the
    annotated sentences contribute to n_pred / n_gold (predictions elsewhere are ignored there too)."""
    per = defaultdict(set)
    for sid, st, en, role in triples:
        per[sid].add((st, en, role))
    inter = n_pred = 0
    for sid, gg in gold.items():
        pp = per.get(sid, set())
        inter += len(pp & gg)
        n_pred += len(pp)
    p = inter / n_pred if n_pred else 0.0
    r = inter / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return f * 100.0, p * 100.0, r * 100.0, n_pred


base = [(r['sid'], r['st'], r['en'], r['role']) for r in rows if r['status'] == 'emit']
F1b, pb, rb, nb = ri(base)
log('reproduced cell: file rows=%d, scored rows=%d, RI=%.4f (p %.4f / r %.4f)' % (len(base), nb, F1b, pb, rb))
log('expected from the chain log: file rows=5221 RI=56.8781')
if abs(F1b - 56.8781) > 0.005 or len(base) != 5221:
    die('reproduction mismatch - the audit decode has drifted from the frozen pipeline, stop')
log('reproduction OK')

gate = F1b / 2.0
print('')
print('===== A. what the frozen recipe drops (tau=%.2f, whitelist on) =====' % A.tau)
print('marginal gate: a batch handed in must be at least %.2f%% fully correct to raise RI' % gate)
print('%-18s %7s %8s %8s %9s %11s %10s' % ('bucket', 'rows', 'correct', 'corr%', 'span_in_gold',
                                           'role_wrong', 'fake_span'))
for lo, hi in [(0.10, 0.12), (0.12, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 9.99)]:
    b = [r for r in rows if r['status'] == 'abstain' and lo < r['pnull'] <= hi]
    if not b:
        print('%-18s %7d' % ('abstain %.2f-%.2f' % (lo, hi), 0))
        continue
    c = sum(1 for r in b if r['ok'])
    s = sum(1 for r in b if r['span_ok'])
    print('%-18s %7d %8d %7.2f%% %9d %11d %10d' % ('abstain %.2f-%.2f' % (lo, hi), len(b), c,
                                                    100.0 * c / len(b), s, s - c, len(b) - s))
vb = [r for r in rows if r['status'] == 'veto']
c = sum(1 for r in vb if r['ok'])
s = sum(1 for r in vb if r['span_ok'])
print('%-18s %7d %8d %7.2f%% %9d %11d %10d' % ('whitelist veto', len(vb), c,
                                               100.0 * c / len(vb) if vb else 0.0, s, s - c, len(vb) - s))
eb = [r for r in rows if r['status'] == 'emit']
c = sum(1 for r in eb if r['ok'])
print('%-18s %7d %8d %7.2f%%  <- the part we hand in' % ('emitted', len(eb), c, 100.0 * c / len(eb)))

print('')
print('===== B. abstain line moved, whitelist kept (exact recompute) =====')
print('%-8s %8s %10s %10s %10s %10s' % ('tau', 'rows', 'RI', 'dRI', 'p', 'r'))
for t in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    tr = [(r['sid'], r['st'], r['en'], r['role']) for r in rows
          if r['pnull'] <= t and not (r['sid'] in allowed_idx and r['role'] not in allowed[r['sid']])]
    F, p, rr, n = ri(tr)
    print('%-8.2f %8d %10.4f %+10.4f %10.4f %10.4f' % (t, n, F, F - F1b, p, rr))

print('')
print('===== C. veto handling variants (tau=%.2f fixed) =====' % A.tau)
tr = [(r['sid'], r['st'], r['en'], r['role']) for r in rows if r['pnull'] <= A.tau]
F, p, rr, n = ri(tr)
print('%-46s rows=%5d RI=%8.4f  dRI=%+7.4f' % ('C1 whitelist removed entirely', n, F, F - F1b))

tr = list(base)
for r in rows:
    if r['status'] != 'veto':
        continue
    ai = allowed_idx.get(r['sid'])
    if not ai:
        tr.append((r['sid'], r['st'], r['en'], r['role']))
        continue
    j = ai[int(np.argmax(avg[r['i'], ai]))]
    tr.append((r['sid'], r['st'], r['en'], idx2label[j]))
F, p, rr, n = ri(tr)
print('%-46s rows=%5d RI=%8.4f  dRI=%+7.4f' % ('C2 vetoed span keeps best legal-role instead', n, F, F - F1b))

for lo in [0.50, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99, 1.01]:
    kept = [r for r in rows if r['status'] == 'veto' and r['conf'] >= lo]
    tr = list(base) + [(r['sid'], r['st'], r['en'], r['role']) for r in kept]
    F, p, rr, n = ri(tr)
    c = sum(1 for r in kept if r['ok'])
    print('%-46s rows=%5d RI=%8.4f  dRI=%+7.4f  rescued=%4d correct=%4d (%.2f%%)'
          % ('C3 keep a vetoed row if role prob >= %.2f' % lo, n, F, F - F1b, len(kept), c,
             100.0 * c / len(kept) if kept else 0.0))

print('')
print('===== C5. symmetric idea on the abstain side (role prob high, abstain score over tau) =====')
for lo in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
    kept = [r for r in rows if r['status'] == 'abstain' and r['conf'] >= lo]
    tr = list(base) + [(r['sid'], r['st'], r['en'], r['role']) for r in kept]
    F, p, rr, n = ri(tr)
    c = sum(1 for r in kept if r['ok'])
    print('%-46s rows=%5d RI=%8.4f  dRI=%+7.4f  rescued=%4d correct=%4d (%.2f%%)'
          % ('C5 keep an abstained row if role prob >= %.2f' % lo, n, F, F - F1b, len(kept), c,
             100.0 * c / len(kept) if kept else 0.0))

print('')
print('===== D. oracle ceilings (upper bound, not reachable without a discriminator) =====')
for tag, keep in [('D1 perfect abstain (only correct drops come back)', 'abstain'),
                  ('D2 perfect whitelist (only correct vetoes come back)', 'veto')]:
    tr = base + [(r['sid'], r['st'], r['en'], r['role']) for r in rows if r['status'] == keep and r['ok']]
    F, p, rr, n = ri(tr)
    print('%-46s rows=%5d RI=%8.4f  dRI=%+7.4f' % (tag, n, F, F - F1b))
tr = base + [(r['sid'], r['st'], r['en'], r['role']) for r in rows
             if r['status'] in ('abstain', 'veto') and r['ok']]
F, p, rr, n = ri(tr)
print('%-46s rows=%5d RI=%8.4f  dRI=%+7.4f' % ('D3 both oracles', n, F, F - F1b))
allsp = [(r['sid'], r['st'], r['en'], r['role']) for r in rows]
F, p, rr, n = ri(allsp)
print('%-46s rows=%5d RI=%8.4f  dRI=%+7.4f' % ('D4 no abstain, no whitelist at all', n, F, F - F1b))
log('done')
