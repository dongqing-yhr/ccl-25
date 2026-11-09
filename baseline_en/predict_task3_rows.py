#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按行展开推理。不改训练、不改 predict_task3.py, 本脚本自带一套解码。

背景: 金标文件里一句有几行就代表几个目标词(每行自带 target 与自己的 cfn_spans)。
训练按行枚举(dataset_task3.py:38-60), 推理却用 data_dict[sid] 建表(predict_task3.py:74-78),
同一句后一行覆盖前一行, 于是非保留行的跨度都在错误目标词的上下文里打分,
与保留目标词重叠的跨度更是直接被丢掉(predict_task3.py:94-101)。

ACTION=forward  一次 GPU 前向, 把每个 (句子, 行, 跨度) 的弃权概率与最佳角色写进 sidecar
ACTION=decode   纯 CPU 从 sidecar 解码三个版本并打分, 不碰 GPU
ACTION=all      先 forward 再 decode

三个解码版本(输出格式都与现管线一致, 每条形如 [sentence_id, start, end, role]):
  V0 = 现管线的复现: 只用每个句子最后一行, 一个 span 出一条
  V1 = 精度轴: 同一个 span 在各行里各打一次分, 只留角色概率最高的那条, 一句一段文字仍只挂一个角色
  V2 = 召回轴: 各行结果取并集, 只去重完全相同的三元组, 允许同一段文字挂两个角色
  V3 = 仅 --sid_mode near 时多出一个文件: 每个 span 只喂字符距离最近的那个目标词那一行,
       距离并列时取行号小的那条。默认 --sid_mode last 不生成 V3, 上面三份输出逐字节不变。
       对方没有给出"最近"的度量、平票取法、帧名跟哪一行, 所以 V3 是我方按字面的实现,
       不能当成对方那一档的复现; dev 上 V3 与 V1 的差在分辨力以内(见 runbash/n7_sid_near_decode_1031.py)。

env: T3_MODEL 权重路径  TAG 输出标签  BS 批大小  MIN_FREE_GB 共享卡显存门槛
用法(在容器 baseline 目录下, 与其它 python 代码同级):
  T3_MODEL=saves/model_task3_oof_best.bin ACTION=all \
  python3 predict_task3_rows.py --test_file ./dataset/cfn-dev.json \
      --config_file ./chinese-macbert-large/config.json \
      --vocab_file ./chinese-macbert-large/vocab.txt --null_threshold 0.08
自检: V0 的 RI 应当等于现管线在同一弃权阈值下的 dev 分数(OOF 模型 @0.08 = 56.028)。
"""

import os
import sys
import time
import json
import codecs
from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from params import args
from model_task3 import Model

# 与 train_task3.py / predict_task3.py 保持一致的 NULL 弃权配置
args.with_null = True
args.null_weight = 0.5

# 权重来源优先级: 环境变量 T3_MODEL > 命令行 --model_path > 默认 task3 权重
MODEL_PATH = os.environ.get('T3_MODEL') or args.model_path or 'saves/model_task3_best.bin'
TAG = os.environ.get('TAG', 'rows')
BS = int(os.environ.get('BS', '16'))
MIN_FREE_GB = float(os.environ.get('MIN_FREE_GB', '20'))
ACTION = os.environ.get('ACTION', 'all')
NULL_THR = args.null_threshold

SIDECAR = '{}/{}task3_rows_{}.json'.format(args.out_dir, args.out_prefix, TAG)


def log(msg):
    print('[{}] [task3_rows] {}'.format(time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def build_row_items(all_data, task2_data):
    """按行展开: 每个 (句子, 行) × 该句全部 task2 候选跨度生成一条打分样本。
    分支换算与 dataset_task3.py:45-50 逐字一致, 保证与训练看到的编码方式相同。"""
    sid2spans = defaultdict(list)
    for line in task2_data:
        sid2spans[line[0]].append((line[1], line[2]))

    sid2rows = {}
    for ri, line in enumerate(all_data):
        rec = sid2rows.get(line["sentence_id"])
        text = line["text"]
        if rec is None:
            sid2rows[line["sentence_id"]] = {"text": text, "rows": [(ri, text, line)]}
        else:
            if rec["text"] != text:
                raise RuntimeError('句子 {} 的两行 text 不一致, 展开无法进行'.format(line["sentence_id"]))
            rec["rows"].append((ri, text, line))

    items = []
    drop_overlap = drop_bad = 0
    n_multi = 0
    orphan_sid = {s for s in sid2spans.keys() if s not in sid2rows}
    if orphan_sid:
        log('task2 结果里有 {} 个句子 id 在本文件中不存在, 这些跨度不打分'.format(len(orphan_sid)))
    for sid, rec in sid2rows.items():
        if len(rec["rows"]) > 1:
            n_multi += 1
        spans = sid2spans.get(sid)
        if not spans:
            continue
        last_row_idx = rec["rows"][-1][0]
        for row_idx, text, line in rec["rows"]:
            target = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
            n = len(text)
            for st, en in spans:
                if st < 0 or en >= n or st > en:
                    drop_bad += 1
                    continue
                if en + 1 < target[0]:
                    label_idx = [st + 1, en + 1]
                elif st + 1 > target[1]:
                    label_idx = [st + 3, en + 3]
                else:
                    drop_overlap += 1
                    continue
                if label_idx[1] >= n + 4:
                    drop_bad += 1
                    continue
                items.append({
                    'text': text,
                    'label_idx': label_idx,
                    'sentence_id': sid,
                    'row_idx': row_idx,
                    'frame': None,
                    'target': target,
                    'ori_target': [st, en],
                    'is_last_row': 1 if row_idx == last_row_idx else 0,
                })
    log('句子 {} 个(共 {} 行), 其中 {} 句有多行(多个目标词), task2 候选跨度 {} 条, 展开后打分样本 {} 条'.format(
        len(sid2rows), len(all_data), n_multi, len(task2_data), len(items)))
    log('其中与所在行目标词重叠而跳过的组合 {} 个, 越界/起止倒置跳过 {} 个'.format(drop_overlap, drop_bad))
    return items, len(sid2rows)


class RowDataset(torch.utils.data.Dataset):
    """编码路径与 predict_task3.py:122-140 完全相同, 只是样本粒度换成 (句子, 行, 跨度)。"""

    def __init__(self, items, tokenizer):
        self.items = items
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        d1 = self.items[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = d1['target']
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        return input_ids, attention_mask, d1['label_idx'], [d1['sentence_id'], d1['row_idx'],
                                                            d1['ori_target'][0], d1['ori_target'][1],
                                                            d1['is_last_row']]


def get_model_input(data, device=None):
    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))

    max_len = max([len(x[0]) for x in data])
    input_ids_list = [pad(d[0], max_len, 0) for d in data]
    attention_mask_list = [pad(d[1], max_len, 0) for d in data]
    meta = [d[3] for d in data]

    input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(device)
    return input_ids, attention_mask, [d[2] for d in data], meta


def do_forward():
    if not torch.cuda.is_available():
        raise RuntimeError('容器内看不到 GPU, 拒跑(前向需要卡)。先按排障流程恢复, 不要退回 CPU。')
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb = free_b / 1024.0 ** 3
    log('卡 {} 总显存 {:.1f} GB, 空闲 {:.1f} GB'.format(torch.cuda.get_device_name(0), total_b / 1024.0 ** 3, free_gb))
    if free_gb < MIN_FREE_GB:
        raise RuntimeError('空闲显存 {:.1f} GB 低于门槛 {:.1f} GB, 卡上是共享的, 等一等再跑'.format(free_gb, MIN_FREE_GB))

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    with codecs.open(args.test_file, 'r', encoding='utf8') as f:
        all_data = json.load(f)
    with codecs.open(args.task2_result, 'r', encoding='utf8') as f:
        task2_data = json.load(f)
    with codecs.open('./dataset/frame_info.json', 'r', encoding='utf8') as f:
        ori_labels = json.load(f)
    idx2label = []
    for line in ori_labels:
        for fes in line["fes"]:
            if fes["fe_name"] not in idx2label:
                idx2label.append(fes["fe_name"])
    num_labels = len(idx2label)
    log('角色表 {} 类, NULL 索引 {}'.format(num_labels, num_labels))

    items, n_sid = build_row_items(all_data, task2_data)
    ds = RowDataset(items, tokenizer)
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = num_labels
    config.with_null = True
    model = Model(config)
    state = torch.load(MODEL_PATH, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    log('权重 {} 加载完成, 缺失 {} 项 / 多余 {} 项'.format(MODEL_PATH, len(missing), len(unexpected)))
    if len(missing) > 8 or len(unexpected) > 8:
        raise RuntimeError('与权重结构差异过大(缺失 {} / 多余 {}), 先核对底座 config_file 是否配对'.format(
            len(missing), len(unexpected)))
    model = model.to(device='cuda')
    model.eval()

    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=0,
                        collate_fn=partial(get_model_input, device='cuda'), drop_last=False)
    rows = []
    n_null_top1 = 0
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc='forward'):
            input_ids, attention_mask, target, meta = batch
            out = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                        labels=None, device='cuda', for_test=True)
            probs = F.softmax(out['logits'], dim=-1)
            real = probs[:, :num_labels]
            rp, ridx = real.max(dim=-1)
            pn = probs[:, num_labels]
            for i in range(len(meta)):
                sid, row_idx, st, en, is_last = meta[i]
                rows.append([int(sid), int(row_idx), int(st), int(en), int(is_last),
                             round(float(pn[i]), 6), int(ridx[i]), round(float(rp[i]), 6)])
    if len(rows) != len(ds):
        raise RuntimeError('打分条数 {} 与样本数 {} 不符, sidecar 不可用'.format(len(rows), len(ds)))
    for r in rows:
        if r[5] > r[7]:
            n_null_top1 += 1
    payload = {'meta': {'model_path': MODEL_PATH, 'test_file': args.test_file,
                        'task2_result': args.task2_result, 'null_threshold': NULL_THR,
                        'n_rows_file': n_sid, 'n_items': len(rows), 'tag': TAG},
               'idx2label': idx2label,
               'items': rows}
    os.makedirs(args.out_dir, exist_ok=True)
    with codecs.open(SIDECAR, 'w', encoding='utf8') as f:
        json.dump(payload, f, ensure_ascii=False)
    log('sidecar 已写 {} ({} 条, NULL 进 top1 的 {} 条)'.format(SIDECAR, len(rows), n_null_top1))


def abstained(p_null, role_prob, thr):
    """弃权规则与管线一致: thr>=0 时 p_null>thr 才弃权; thr<0 时退回旧规则(NULL 进 top1 就弃权)。"""
    if thr is None or thr < 0:
        return p_null > role_prob
    return p_null > thr


def score(pred, gold, n_gold):
    inter = 0
    n_pred = 0
    for sid, tset in pred.items():
        n_pred += len(tset)
        inter += len(tset & gold.get(sid, set()))
    p = 100.0 * inter / n_pred if n_pred else 0.0
    r = 100.0 * inter / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {'N': n_pred, 'TP': inter, 'P': p, 'R': r, 'RI': f}


def dump_pred(pred, path):
    out = []
    for sid in sorted(pred.keys()):
        for (st, en, role) in sorted(pred[sid]):
            out.append([sid, st, en, role])
    with codecs.open(path, 'w', encoding='utf8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return len(out)


def do_decode():
    with codecs.open(SIDECAR, 'r', encoding='utf8') as f:
        payload = json.load(f)
    idx2label = payload['idx2label']
    items = payload['items']
    thr = payload['meta'].get('null_threshold', NULL_THR)
    if thr != NULL_THR:
        log('sidecar 记的弃权阈值 {} 与本次命令行 {} 不同, 按命令行重算'.format(thr, NULL_THR))
        thr = NULL_THR
    log('sidecar {} 条打分样本, 弃权阈值 {}'.format(len(items), thr))

    with codecs.open(args.test_file, 'r', encoding='utf8') as f:
        all_data = json.load(f)
    gold = defaultdict(set)
    for line in all_data:
        for s in line.get('cfn_spans', []):
            gold[line['sentence_id']].add((s['start'], s['end'], s['fe_name']))
    n_gold = sum(len(v) for v in gold.values())
    log('金标三元组(按句子取并集去重) {} 条, 覆盖 {} 句'.format(n_gold, len(gold)))
    if thr is None or thr < 0:
        log('弃权规则: 旧规则(NULL 进 top1 就丢), 命令行 --null_threshold 为 {}'.format(thr))
    if n_gold == 0:
        log('提示: 该文件没有金标标注(A/B 测试集), 下面的分数无意义, 只用它生成预测文件')

    v0 = defaultdict(set)
    v1 = defaultdict(set)
    v1_best = {}
    v2 = defaultdict(set)
    n_keep = n_abstain = 0
    for sid, row_idx, st, en, is_last, p_null, ridx, rprob in items:
        role = idx2label[ridx]
        if abstained(p_null, rprob, thr):
            n_abstain += 1
            continue
        n_keep += 1
        if is_last:
            v0[sid].add((st, en, role))
        v2[sid].add((st, en, role))
        key = (sid, st, en)
        cur = v1_best.get(key)
        if cur is None or rprob > cur[0]:
            v1_best[key] = (rprob, role)
    for (sid, st, en), (rprob, role) in v1_best.items():
        v1[sid].add((st, en, role))
    log('打分样本 {} 条: 通过弃权 {} 条, 弃权掉 {} 条'.format(len(items), n_keep, n_abstain))

    v3 = None
    if args.sid_mode == 'near':
        # 距离换算与 build_row_items 的重叠判定同一套: 行目标词起止取 target 起止各 +1。
        row_tgt = {}
        for ri, line in enumerate(all_data):
            t = line['target'][-1]
            row_tgt[ri] = (t['start'] + 1, t['end'] + 1)
        near = {}
        for sid, row_idx, st, en, is_last, p_null, ridx, rprob in items:
            if abstained(p_null, rprob, thr):
                continue
            ts, te = row_tgt[row_idx]
            if en + 1 < ts:
                gap = ts - (en + 1)
            elif st + 1 > te:
                gap = (st + 1) - te
            else:
                raise RuntimeError('句子 {} 跨度 [{},{}] 与行 {} 的目标词 [{},{}] 重叠, '
                                   '与 build_row_items 的跳过判定矛盾'.format(sid, st, en, row_idx, ts, te))
            key = (sid, st, en)
            cur = near.get(key)
            if cur is None or (gap, row_idx) < cur[0]:
                near[key] = ((gap, row_idx), idx2label[ridx])
        v3 = defaultdict(set)
        for (sid, st, en), (_k, role) in near.items():
            v3[sid].add((st, en, role))
        log('V3(--sid_mode near): {} 个跨度各取字符距离最近的目标词行, 距离并列时取行号小的那条'.format(len(near)))

    variants = [('V0', '现管线复现(只用每句最后一行)', v0),
                ('V1', '跨行取概率最高的一段一角色', v1),
                ('V2', '跨行取并集(允许一段挂两个角色)', v2)]
    if v3 is not None:
        variants.append(('V3', '每段只喂字符距离最近的目标词行(near 的字面读法)', v3))
    scores = {}
    print('')
    print('| 版本(名字后是它在改什么) | 发射条数 | TP | 精度 P | 召回 R | RI |')
    print('|---|---|---|---|---|---|')
    for tag, desc, pred in variants:
        s = score(pred, gold, n_gold)
        scores[tag] = s
        print('| {} {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |'.format(
            tag, desc, s['N'], s['TP'], s['P'], s['R'], s['RI']))
    base = scores['V0']
    bar = base['RI'] / 2.0
    print('')
    log('F1/2 判据: 基准 V0 的 RI {:.4f}, 门槛 {:.4f}%。新增那批里对的比例高于门槛才该加, 被删那批低于门槛才该删。'.format(
        base['RI'], bar))
    for tag, _desc, pred in variants[1:]:
        delta = scores[tag]['RI'] - base['RI']
        added = set()
        removed = set()
        for sid in set(list(pred.keys()) + list(v0.keys())):
            pset = pred.get(sid, set())
            oset = v0.get(sid, set())
            added |= {(sid, x) for x in (pset - oset)}
            removed |= {(sid, x) for x in (oset - pset)}
        a_hit = sum(1 for (sid, x) in added if x in gold.get(sid, set()))
        r_hit = sum(1 for (sid, x) in removed if x in gold.get(sid, set()))
        log('{}: RI 相对 V0 {:+.4f}; 新增 {} 条(命中 {} 条, {}%), 删除 {} 条(其中 V0 原本命中 {} 条, {}%)'.format(
            tag, delta, len(added), a_hit,
            '{:.1f}'.format(100.0 * a_hit / len(added)) if added else '0.0',
            len(removed), r_hit,
            '{:.1f}'.format(100.0 * r_hit / len(removed)) if removed else '0.0'))

    out_paths = []
    for tag, desc, pred in variants:
        path = '{}/{}task3_{}_{}.json'.format(args.out_dir, args.out_prefix, TAG, tag.lower())
        n = dump_pred(pred, path)
        out_paths.append((tag, path))
        log('写出 {} ({} 条)'.format(path, n))
    print('')
    log('下一步(纯 CPU, 用官方打分器复核这三个文件): ')
    t1 = '{}/{}task1_test.json'.format(args.out_dir, args.out_prefix)
    t2 = '{}/{}task2_test.json'.format(args.out_dir, args.out_prefix)
    for tag, path in out_paths:
        log('  {} : python3 score_official.py --gold {} --task1 {} --task2 {} --task3 {}'.format(
            tag, args.test_file, t1, t2, path))
    if 'dev' in os.path.basename(args.test_file):
        log('对照点: OOF 模型 @0.08 的历史 dev RI = 56.028; V0 若明显偏离说明本脚本复现没对上, 先查 V0')


if __name__ == '__main__':
    log('参数: test_file={} task2={} 权重={} config={} vocab={} null_threshold={} ACTION={}'.format(
        args.test_file, args.task2_result, MODEL_PATH, args.config_file, args.vocab_file, NULL_THR, ACTION))
    if ACTION in ('forward', 'all'):
        do_forward()
    if ACTION in ('decode', 'all'):
        if not os.path.exists(SIDECAR):
            raise RuntimeError('找不到 sidecar {}, 先跑 ACTION=forward'.format(SIDECAR))
        do_decode()
    log('完成')
