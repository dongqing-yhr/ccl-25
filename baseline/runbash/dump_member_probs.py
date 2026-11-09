#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""GPU stage 1 for the member grid: dump per-model role probabilities for one task2 dev cell.

runbash/probe_drop_audit.py stores only the FN+LA average, so its dump cannot answer "what if the
new T1 member gets weight 0.3" or "is this member useful on its own". This script writes ONE
probability matrix per member - shape (n_candidates, n_roles + 1), last column = NULL - so
runbash/grid_member_search.py can average any subset on CPU afterwards without touching the GPU
again.

  PYTHONPATH=. python3 runbash/dump_member_probs.py \
      --members fn:saves/model_task3_fn_best.bin:wfn,la:saves/model_task3_la_best.bin:nofn, \
                neg:saves/model_task3_t1neg_best.bin:wfn,arg:saves/model_task3_t1arg_best.bin:wfn \
      --out_dir dataset/dev/cmp1022/probs_union_m0p25_m0p75 --expect_rows 6726

One member spec is name:checkpoint:mode, the three fields split by COLONS (commas separate the
members). mode=wfn means the frame name is appended to the input (that is how the FN member and all
three T1 arms were trained); nofn means plain input, like the LA member; wfn2 appends the runner-up
frame after a second [SEP] as well (the FN2 member) and needs --task1_file2 to point at a 5-column
[sid, f1, p1, f2, p2] confidence file.

Row alignment is the whole point, so it is checked three ways: every member must produce exactly the
same candidate count, that count can be pinned with --expect_rows against the number the frozen cell
is known to have, and meta.json is written once from the shared dataset so the CPU stage can prove
the dumps and the metadata agree. A member whose dump already exists with the right shape is skipped
unless --force, so an interrupted run can be resumed.
"""
import argparse
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument('--members', required=True, help='comma separated name:checkpoint:wfn|nofn list')
ap.add_argument('--task2_file', default='dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json')
ap.add_argument('--task1_file', default='./dataset/dev/dev_task1_test.json')
ap.add_argument('--task1_file2', default='dataset/dev/dev_task1_conf.json',
                help='5-column [sid, f1, p1, f2, p2] runner-up frame source for mode=wfn2; the top-1 '
                     'in this file is ignored, only column 4 is read, so the f1 the model sees stays '
                     'the one in --task1_file')
ap.add_argument('--gold', default='./dataset/cfn-dev.json')
ap.add_argument('--frame_info', default='./dataset/frame_info.json')
ap.add_argument('--test_file', default='./dataset/cfn-dev.json',
                help='the sentence file the spans index into; dev for a dev cell')
ap.add_argument('--out_dir', required=True)
ap.add_argument('--batch_size', type=int, default=16)
ap.add_argument('--expect_rows', type=int, default=0,
                help='fail if the candidate count differs from this (0 = any)')
ap.add_argument('--model_module', default='model_task3',
                help='Model 类从哪个模块取; 现役五成员留默认, 换了角色头的新臂(如 model_task3_span)要显式写')
ap.add_argument('--force', action='store_true', help='recompute a member even if its dump exists')
A = ap.parse_args()

import importlib                       # 同上, 解析完参数才知道用哪个模型类
Model = getattr(importlib.import_module(A.model_module), "Model")

MC = './chinese-macbert-large/config.json'
MV = './chinese-macbert-large/vocab.txt'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def die(m):
    print('[%s] FATAL %s' % (time.strftime('%H:%M:%S'), m), flush=True)
    sys.exit(1)


class Dataset(torch.utils.data.Dataset):
    """Verbatim copy of runbash/probe_drop_audit.py Dataset - keep in sync or the dumps are
    not comparable with the recorded frozen cell."""

    def __init__(self, jf, lf, t2f, t1f, tok, with_frame_name=False, t1f2=None):
        self.wfn = with_frame_name
        self.tok = tok
        self.all_data = json.load(open(jf, encoding='utf8'))
        self.ori_labels = json.load(open(lf, encoding='utf8'))
        self.sid2frame = {}
        self.sid2frame2 = {}
        if with_frame_name:
            for it in json.load(open(t1f, encoding='utf8')):
                self.sid2frame[it[0]] = it[1]
        if t1f2 is not None:
            rows = json.load(open(t1f2, encoding='utf8'))
            bad = [r for r in rows if len(r) < 4]
            if bad:
                die('%s is not a 5-column [sid, f1, p1, f2, p2] file (%d short rows) - '
                    'wfn2 would have no runner-up frame to append' % (t1f2, len(bad)))
            for it in rows:
                if it[3]:
                    self.sid2frame2[it[0]] = it[3]
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
                              'frame2': self.sid2frame2.get(line[0]),
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
            if d.get('frame2') and d['frame2'] != d['frame']:
                f2 = self.tok.convert_tokens_to_ids(list(d['frame2']))
                ids = ids + [self.tok.sep_token_id] + f2
                am = am + [1] * (len(f2) + 1)
        return ids, am, d['label_idx'], d['sentence_id'], d['ori_target']


def collate(data, device=None):
    ml = max(len(x[0]) for x in data)
    pad = lambda d, v=0: d + [v] * (ml - len(d))
    ids = torch.from_numpy(np.array([pad(x[0]) for x in data], dtype=np.compat.long)).to(device)
    am = torch.from_numpy(np.array([pad(x[1]) for x in data], dtype=np.compat.long)).to(device)
    return ids, am, [x[2] for x in data], [x[3] for x in data], [x[4] for x in data]


specs = []
for s in A.members.split(','):
    s = s.strip()
    if not s:
        continue
    parts = s.split(':')
    if len(parts) != 3 or parts[2] not in ('wfn', 'nofn', 'wfn2'):
        die('member spec must be name:checkpoint:wfn|nofn|wfn2, got "%s"' % s)
    if not os.path.exists(parts[1]):
        die('member %s: checkpoint %s does not exist' % (parts[0], parts[1]))
    specs.append(parts)
if not specs:
    die('no members parsed from --members %r' % A.members)
names = [s[0] for s in specs]
if len(set(names)) != len(names):
    die('duplicate member names in --members')
log('members: %s' % ', '.join('%s(%s)' % (n, s[2]) for n, s in zip(names, specs)))
log('task2 spans %s  task1 frames %s  sentences %s' % (A.task2_file, A.task1_file, A.test_file))
for f in (A.task2_file, A.task1_file, A.test_file, A.frame_info, MC, MV):
    if not os.path.exists(f):
        die('input file missing: %s' % f)
if any(s[2] == 'wfn2' for s in specs) and not os.path.exists(A.task1_file2):
    die('a wfn2 member needs the runner-up frame file, but %s is missing' % A.task1_file2)

os.makedirs(A.out_dir, exist_ok=True)
tok = BertTokenizer(vocab_file=MV, do_lower_case=True)

# One dataset build per mode, reused across members with the same mode - the candidate rows are
# decided by the task2 file alone, so all members must see the same list in the same order.
ds_cache = {}
NL = None
recs = None
idx2label = None
for name, ckpt, mode in specs:
    path = os.path.join(A.out_dir, name + '.npy')
    if os.path.exists(path) and not A.force:
        m = np.load(path)
        log('%-6s dump already there, shape=%s - skipping (use --force to recompute)' % (name, m.shape))
        if NL is None:
            NL = m.shape[1] - 1
        elif m.shape[1] - 1 != NL:
            die('%s has %d role columns, another member has %d - mixed label spaces'
                % (name, m.shape[1] - 1, NL))
        if A.expect_rows and m.shape[0] != A.expect_rows:
            die('%s dump has %d rows, this cell is expected to have %d - stale dump from another '
                'task2 file, delete it or pass --force' % (name, m.shape[0], A.expect_rows))
        continue
    if mode not in ds_cache:
        ds_cache[mode] = Dataset(A.test_file, A.frame_info, A.task2_file, A.task1_file,
                                 tok, mode in ('wfn', 'wfn2'),
                                 t1f2=(A.task1_file2 if mode == 'wfn2' else None))
        if recs is None:
            recs = ds_cache[mode].data
            NL = ds_cache[mode].num_labels
            idx2label = ds_cache[mode].idx2label
            log('candidates scored: %d, roles: %d (from the %s dataset)'
                % (len(ds_cache[mode]), NL, mode))
        ds = ds_cache[mode]
        if len(ds) != len(recs):
            die('%s dataset has %d rows but the first one has %d - the two modes must produce the '
                'same candidate list, they do not' % (mode, len(ds), len(recs)))
        if A.expect_rows and len(ds) != A.expect_rows:
            die('candidate count %d != --expect_rows %d; this is not the frozen cell'
                % (len(ds), A.expect_rows))
    ds = ds_cache[mode]
    if not torch.cuda.is_available():
        die('cuda not available - this is the GPU stage')
    free, total = torch.cuda.mem_get_info()
    log('%-6s forward %s  (gpu free %.1f GB of %.1f GB)'
        % (name, ckpt, free / 1e9, total / 1e9))
    if free < 20e9:
        die('only %.1f GB free on the card - wait for the training arms to release memory '
            'instead of risking a mid-run CUDA OOM' % (free / 1e9))
    cfg = BertConfig.from_json_file(MC)
    cfg.num_labels = NL
    cfg.with_null = True          # the head has NL roles + the NULL column
    m = Model(cfg)
    loaded = m.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
    # A member trained with the L2 frame head (mtl) carries two extra tensors the role-only model
    # here does not have, and a logit-adjustment member carries the prior buffer. Ignoring those is
    # intended; ignoring anything else would silently dump a half-loaded model, so refuse.
    def _tolerated(k):
        return k.startswith('frame_head.') or k == 'logit_prior'
    stray = [k for k in loaded.unexpected_keys if not _tolerated(k)]
    if loaded.missing_keys or stray:
        die('%s checkpoint does not match the role-only model: missing=%s unexpected(not tolerated)=%s'
            % (name, loaded.missing_keys, stray))
    if loaded.unexpected_keys:
        log('  %-6s 只读角色那条路, 忽略键 %s' % (name, loaded.unexpected_keys))
    dl = DataLoader(ds, batch_size=A.batch_size, shuffle=False, num_workers=0,
                    collate_fn=partial(collate, device=torch.device('cuda')))
    P = np.zeros((len(ds), NL + 1), dtype=np.float32)
    pos = 0
    m = m.to('cuda').eval()
    with torch.no_grad():
        for ids, am, tg, sid, ori in dl:
            p = F.softmax(m(input_ids=ids, attention_mask=am, target=tg, labels=None,
                            device=torch.device('cuda'), for_test=True)['logits'],
                          dim=-1).cpu().numpy()
            P[pos:pos + len(ori)] = p
            pos += len(ori)
            if (pos // 1000) != ((pos - len(ori)) // 1000):
                log('  %-6s %d/%d' % (name, pos, len(ds)))
    if pos != len(ds):
        die('%s covered %d of %d candidates' % (name, pos, len(ds)))
    np.save(path, P)
    log('%-6s wrote %s (%.1f MB)' % (name, path, os.path.getsize(path) / 1e6))
    del m, P
    torch.cuda.empty_cache()

if recs is None:
    # every member was skipped, so rebuild the row list and the role table from the cell alone
    ds = Dataset(A.test_file, A.frame_info, A.task2_file, A.task1_file, tok,
                 specs[0][2] in ('wfn', 'wfn2'),
                 t1f2=(A.task1_file2 if specs[0][2] == 'wfn2' else None))
    recs = ds.data
    NL = ds.num_labels
    idx2label = ds.idx2label
json.dump([[r['sentence_id'], r['ori_target'][0], r['ori_target'][1]] for r in recs],
          open(os.path.join(A.out_dir, 'meta.json'), 'w'))
json.dump(idx2label, open(os.path.join(A.out_dir, 'labels.json'), 'w'))
have = []
for name in names:
    path = os.path.join(A.out_dir, name + '.npy')
    if os.path.exists(path):
        sh = np.load(path).shape
        if sh[0] != len(recs) or sh[1] != NL + 1:
            die('%s dump shape %s does not match this cell (%d candidates, %d roles + NULL)'
                % (name, sh, len(recs), NL))
        have.append('%s%s' % (name, sh))
    else:
        die('member %s has no dump at the end of the run (%s)' % (name, path))
log('done: %s' % ' '.join(have))
log('next (CPU): PYTHONPATH=. python3 runbash/grid_member_search.py --members %s --out %s/grid.csv'
    % (','.join('%s=%s/%s.npy' % (n, A.out_dir, n) for n in names), A.out_dir))
print('DUMP_MEMBER_PROBS_DONE', flush=True)
