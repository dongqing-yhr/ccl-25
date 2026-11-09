#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
task1 多成员概率融合，支持跨底座，并把"冠军那个算子"并排跑出来做对照。

一、成员写法（--models 空格分隔，每个成员一项）
    保存路径                                   用全局 --config_file/--vocab_file（同底座多种子的老写法，保持字节兼容）
    保存路径:底座目录                           底座目录里必须有 config.json 和 vocab.txt，这个成员自己配一套
  跨底座融合时，非默认底座那一个必须写成第二种，否则会用错词表和错层数。

二、两种合并算子（同一次前向里都算出来，不重复跑 GPU）
    full  = 每个成员的完整框架 softmax 等权平均后 argmax。现役任务一融合用的就是这个算子。
    topk  = 冠军的说法："各模型输出前 5 个框架概率，分别归一化后等权融合，选择最高分框架"。
            即每个成员先只留自己的前 k 个框架概率、在这 k 个上重新归一化（其余列置 0），再等权平均、argmax。
            k=1 时每个成员只剩一票（one-hot），等权平均就是多数投票，正好接上 10-23 中午 task3 那条投票探针。
  --op/--topk 决定写进 --out 的那一档；--arms 列出的每一档另外各写一个文件到 --arm_dir，并在末尾打对照表。

三、成员加权（--weights）
    与 --models 同序的一组小数，会被归一化成和为 1。不给这个参数就走原来的等权公式（一个比特都不变，
    等权那份的逐字节回归门槛才守得住）。加权的动机：dev 上 roberta 单模比 macbert 高 1.74，
    两个底座的强弱随验证划分改变，所以"以 roberta 为主、按权重少量采纳 macbert"这一档
    在 dev 上接近持平。
    注意 --op topk 那一档里每个成员先在自己前 k 列上归一化，权重乘的是归一化之后的那份。

四、防呆（都是硬中断）
  1) 从检查点自己的张量形状反推 (hidden_size, 层数, 词表大小, 框架类数)，与底座 config.json 和数据集标签数比对，
     不符就停。这一条拦得住"base 权重配 large 底座目录"和反向；拦不住 macbert 与 roberta 互换——已核实两者
     config.json 只差 bos/eos/output_past/gradient_checkpointing 四个无关键，hidden 1024、24 层、词表 21128 全同，
     三个底座目录的 vocab.txt md5 也都是 3b5b76c4aef48ecf8cb3abaafe960f09，所以换底座目录不改变结构。
  2) 成员数、推理行数、框架类数都要对上；成员重名要停（档名会撞）。
  3) 有金标时，除了自己的逐行正确率，还要用官方 score_official.score_fi 再算一遍 FI 打出来，两个数不一致就说明输出格式跟槽位文件不一致。

用法（容器 baseline 目录）：
  # 对照档一次跑齐（dev，有金标）
  PYTHONPATH=. python3 runbash/fuse_task1_seeds.py \
      --test_file ./dataset/cfn-dev.json \
      --models saves/model_task1_macbert.bin:./chinese-macbert-large \
               saves/model_task1_roberta.bin:./chinese-roberta-wwm-ext-large \
      --op full --out dataset/dev/cmp1023/task1_xbase_full.json \
      --arms 1,3,5,10,full --arm_dir dataset/dev/cmp1023/t1arms
  # s42+s43 等权融合的逐字节回归对照（需当时那两份 seed 权重在手，产物与记录在案的答卷逐字节一致）
  python3 runbash/fuse_task1_seeds.py --test_file ./dataset/cfn-test-A.json \
      --models saves/model_task1_macbert_s42_best.bin saves/model_task1_macbert_s43_best.bin \
      --op full --out dataset/A/cmp1023/task1_s42s43_check.json
"""
import os
import json
import argparse
import codecs

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer


ap = argparse.ArgumentParser()
ap.add_argument('--test_file', default='./dataset/cfn-dev.json')
ap.add_argument('--models', nargs='+', required=True,
                help='每个成员一项：保存路径 或 保存路径:底座目录（底座目录里要有 config.json 与 vocab.txt）')
ap.add_argument('--config_file', default='./chinese-macbert-large/config.json',
                help='只给没写底座目录的成员当默认值')
ap.add_argument('--vocab_file', default='./chinese-macbert-large/vocab.txt')
ap.add_argument('--label_file', default='./dataset/frame_info.json')
ap.add_argument('--op', default='full', choices=['full', 'topk'], help='写进 --out 的那一档算子')
ap.add_argument('--topk', type=int, default=5, help='--op topk 时保留几个框架')
ap.add_argument('--weights', default='', help='与 --models 同序的权重，如 0.7,0.3；留空＝等权（走原公式，字节不变）')
ap.add_argument('--dump_dir', default=None,
                help='把每个成员自己的 (句数,框架数) 概率表存成 <成员名>.npy，之后换权重只在 CPU 上算，不必再占卡')
ap.add_argument('--arms', default='', help='逗号分隔的 k 值或 full，每一档都算并写文件（需 --arm_dir）')
ap.add_argument('--arm_dir', default=None, help='对照档输出目录，每个档名一个 <档名>.json')
ap.add_argument('--out', required=True)
ap.add_argument('--out_conf', default=None)
ap.add_argument('--model_module', default='model_task1',
                help='Model 类从哪个模块取; 现役成员留默认, 换了模型类的新臂(如 model_task1_def)要显式写')
ap.add_argument('--def_weight', type=float, default=0.0,
                help='--model_module model_task1_def 时给同一个值; 给小了那只头不创建, 加载会静默丢权重')
ap.add_argument('--def_dim', type=int, default=256)
ap.add_argument('--batch_size', type=int, default=16)
ARGS = ap.parse_args()

import importlib                       # 模型类要在 parse_args 之后才能按开关取
Model = getattr(importlib.import_module(ARGS.model_module), "Model")
print("[cfg  ] Model 类来自 %s, def_weight=%s def_dim=%s"
      % (ARGS.model_module, ARGS.def_weight, ARGS.def_dim), flush=True)

if ARGS.arms and not ARGS.arm_dir:
    raise SystemExit('[FATAL] --arms 需要一起给 --arm_dir，否则对照档只打印不落盘，下次没法复算')
if ARGS.arm_dir and not ARGS.arms:
    ARGS.arms = '1,3,5,10,full'


def die(msg):
    print('[FATAL] %s' % msg, flush=True)
    raise SystemExit(1)


def parse_member(spec):
    """`ckpt` or `ckpt:base_dir`. A base_dir is only recognised when it really is a directory holding both
    config.json and vocab.txt, so a path that happens to contain a colon cannot be silently mis-split."""
    if ':' not in spec:
        ck, cfg, voc = spec, ARGS.config_file, ARGS.vocab_file
    else:
        ck, base = spec.rsplit(':', 1)
        if not os.path.isdir(base):
            die('成员 %s 的底座目录 %s 不存在（要么它不是目录，要么整段路径写错了）' % (spec, base))
        cfg, voc = os.path.join(base, 'config.json'), os.path.join(base, 'vocab.txt')
    if not os.path.isfile(ck):
        die('成员 %s：找不到权重 %s' % (spec, ck))
    if not os.path.isfile(cfg) or not os.path.isfile(voc):
        die('成员 %s：底座里缺 config.json 或 vocab.txt（%s / %s）' % (spec, cfg, voc))
    name = os.path.basename(ck)[:-4] if ck.endswith('.bin') else os.path.basename(ck)
    return name, ck, cfg, voc


MEMBERS = [parse_member(s) for s in ARGS.models]
names = [m[0] for m in MEMBERS]
if len(set(names)) != len(names):
    die('成员名重复 %s：档名会撞，请把权重改个名再跑' % ','.join(names))

WEIGHTS = None
if ARGS.weights.strip():
    w = [float(x) for x in ARGS.weights.split(',')]
    if len(w) != len(MEMBERS):
        die('--weights 给了 %d 个数，成员有 %d 个：%s' % (len(w), len(MEMBERS), ARGS.weights))
    if min(w) < 0 or sum(w) <= 0:
        die('--weights 里出现负数或全 0：%s' % ARGS.weights)
    WEIGHTS = [x / sum(w) for x in w]
ARGS.arms = ARGS.arms.strip()
if ARGS.dump_dir:
    os.makedirs(ARGS.dump_dir, exist_ok=True)
print('[cfg  ] 成员 %d 个 | 算子 --op %s --topk %d | 对照档 %s | 权重 %s | 概率表落盘 %s | test=%s'
      % (len(MEMBERS), ARGS.op, ARGS.topk, ARGS.arms or '-',
         ('-' if WEIGHTS is None else ' '.join('%s=%.4f' % (n, w) for n, w in zip(names, WEIGHTS))),
         ARGS.dump_dir or '-', ARGS.test_file), flush=True)
for name, ck, cfg, voc in MEMBERS:
    print('[member] %-32s ckpt=%s\n          base=%s' % (name, ck, cfg), flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file, label_file, tokenizer):
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.idx2label = [line['frame_name'] for line in self.ori_labels]
        self.label2idx = {name: i for i, name in enumerate(self.idx2label)}
        self.num_labels = len(self.idx2label)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        target = [d1['target'][-1]['start'] + 1, d1['target'][-1]['end'] + 1]
        return data.data['input_ids'], data.data['attention_mask'], target, d1['sentence_id']


def get_model_input(data, device=None):
    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))
    max_len = max(len(x[0]) for x in data)
    ids = torch.from_numpy(np.array([pad(d[0], max_len, 0) for d in data], dtype=np.compat.long)).to(device)
    am = torch.from_numpy(np.array([pad(d[1], max_len, 0) for d in data], dtype=np.compat.long)).to(device)
    target = [d[2] for d in data]
    sid = [d[3] for d in data]
    return ids, am, target, sid


def shape_from_state(state):
    """(hidden, layers, vocab, out_dim) read off the checkpoint itself, without trusting any config file."""
    hidden = state['dense.weight'].shape[1]
    out_dim = state['dense.weight'].shape[0]
    vocab = state['bert.embeddings.word_embeddings.weight'].shape[0]
    layers = len(set(int(k.split('.')[3]) for k in state if k.startswith('bert.encoder.layer.')))
    return hidden, layers, vocab, out_dim


TOKENS = {}
N, NL, IDX2LABEL, SIDS = None, None, None, None
GOLD_IDX, HAS_GOLD = None, None
mats = {}
for name, ck, cfg_path, voc_path in MEMBERS:
    if voc_path not in TOKENS:
        tok = BertTokenizer(vocab_file=voc_path, do_lower_case=True)
        ds = Dataset(ARGS.test_file, ARGS.label_file, tok)
        loader = DataLoader(ds, batch_size=ARGS.batch_size, shuffle=False, num_workers=0,
                            collate_fn=partial(get_model_input, device=device), drop_last=False)
        if N is None:
            N, NL = len(ds), ds.num_labels
            IDX2LABEL = ds.idx2label
            SIDS = [d['sentence_id'] for d in ds.all_data]
            gold = [d.get('frame') for d in ds.all_data]
            # cfn-test-A.json 每行写的是 "frame": ""（10-23 实测 dev 2300/2300 非空、A 4400/4400 空串）。
            # 只判 is not None 会把无金标的 A 当成有金标，打印出 0.0000% 这种假读数。
            nonempty = sum(1 for g in gold if g)
            HAS_GOLD = nonempty == len(gold)
            if 0 < nonempty < len(gold):
                die('%s 里 %d/%d 行有金标框架、其余为空串：这份文件口径不一致，不能拿来算正确率'
                    % (ARGS.test_file, nonempty, len(gold)))
            if not HAS_GOLD:
                print('[gold ] 测试文件 %d/%d 行金标框架为空，按无金标处理：本次不打印单模/融合正确率'
                      % (nonempty, len(gold)), flush=True)
            GOLD_IDX = np.array([ds.label2idx.get(g, -1) for g in gold]) if HAS_GOLD else None
        elif len(ds) != N or ds.num_labels != NL:
            die('底座 %s 的句子数或框架数与第一个成员不一致（%d/%d vs %d/%d）：两份词表对应的预处理不是一回事'
                % (voc_path, len(ds), ds.num_labels, N, NL))
        TOKENS[voc_path] = (ds, loader)
    ds, loader = TOKENS[voc_path]

    config = BertConfig.from_json_file(cfg_path)
    config.num_labels = NL
    config.def_weight = ARGS.def_weight
    config.def_dim = ARGS.def_dim
    model = Model(config)
    state = torch.load(ck, map_location='cpu')
    raw_n = len(state)
    state = {k: v for k, v in state.items() if not k.startswith('lstm.')}
    h, l, v, od = shape_from_state(state)
    if (h, l, v) != (config.hidden_size, config.num_hidden_layers, config.vocab_size):
        die('%s：检查点自己是 hidden=%d 层=%d 词表=%d，但底座目录 %s 说 hidden=%d 层=%d 词表=%d，配错了'
            % (ck, h, l, v, os.path.dirname(cfg_path), config.hidden_size, config.num_hidden_layers,
               config.vocab_size))
    if od != NL * 64 * 2:
        die('%s：分类头输出维 %d != 框架数 %d × 128，权重和数据集标签表不是同一套' % (ck, od, NL))
    res = model.load_state_dict(state, strict=False)
    # model_task1 builds an LSTM and a BERT pooler that forward() never uses (the LSTM call is commented
    # out in model_task1.py), so a checkpoint without them is normal. Anything else missing is fatal.
    missing = [k for k in res.missing_keys if not (k.startswith('lstm.') or k.startswith('bert.pooler.'))]
    if missing:
        die('%s：检查点缺 %d 个真正参与前向的参数（例 %s），这份权重不能用来融合' % (ck, len(missing), missing[:2]))
    print('[member] %-32s 检查点 %d 个键，剥掉 %d 个 lstm.*（forward 不用），剩下 %d 个键里模型没有的 %d 个'
          % (name, raw_n, raw_n - len(state), len(state), len(res.unexpected_keys)), flush=True)
    # 非现役模型类(带新头的那种)必须整份装上: 少装一只头照样能跑完, 但算出来的是另一份打分,
    # 那比报错更糟。所以这里把"只打印"升成硬中断, 并核定义表到底冻进去没有。
    if ARGS.model_module != 'model_task1':
        if res.unexpected_keys:
            die('%s：检查点里有 %s 这些键在 %s.Model 里不存在 —— 模型类或 def_weight 传错了，'
                '这份权重没被完整装配' % (ck, res.unexpected_keys[:3], ARGS.model_module))
        if ARGS.model_module == 'model_task1_def':
            if ARGS.def_weight <= 0:
                die('%s：加载 model_task1_def 却没给 --def_weight(现在是 %s)，那只头连创建都不会创建'
                    % (ck, ARGS.def_weight))
            if float(model.def_ready.item()) != 1.0:
                die('%s：def_ready=%s，定义表不在权重里或没冻好，这份权重的定义分支是空的'
                    % (ck, float(model.def_ready.item())))
            print('[member] %-32s 定义分支已装配: def_weight=%s, def_emb 非零占比 %.3f'
                  % (name, ARGS.def_weight, float((model.def_emb != 0).float().mean())), flush=True)
    model = model.to(device)
    model.eval()
    probs = np.zeros((N, NL), dtype=np.float64)
    pos = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='model %s' % name):
            ids, am, target, sid = batch
            out = model(input_ids=ids, attention_mask=am, target=target, labels=None, device=device, for_test=True)
            p = F.softmax(out['logits'], dim=-1).cpu().numpy()
            probs[pos:pos + p.shape[0]] += p
            pos += p.shape[0]
    assert pos == N, 'model %s 推理行数 %d != 句子数 %d' % (ck, pos, N)
    mats[name] = probs
    if HAS_GOLD:
        print('[member] %-32s 单模逐行正确率 %.4f%%（%d/%d）'
              % (name, 100.0 * float((probs.argmax(1) == GOLD_IDX).mean()),
                 int((probs.argmax(1) == GOLD_IDX).sum()), N), flush=True)
    if ARGS.dump_dir:
        p = os.path.join(ARGS.dump_dir, '%s.npy' % name)
        np.save(p, probs.astype(np.float32))
        print('[dump ] %-32s -> %s %s' % (name, p, probs.astype(np.float32).shape), flush=True)
    del model
    torch.cuda.empty_cache() if device.type == 'cuda' else None

PROBS = [mats[n] for n in names]
if ARGS.dump_dir:
    man = {'members': names, 'n_rows': N, 'n_labels': NL, 'idx2label': IDX2LABEL,
           'sids': [int(s) for s in SIDS], 'test_file': ARGS.test_file, 'has_gold': bool(HAS_GOLD),
           'weights': (None if WEIGHTS is None else dict(zip(names, WEIGHTS)))}
    if HAS_GOLD:
        man['gold_frames'] = [d.get('frame') for d in ds.all_data]
    mp = os.path.join(ARGS.dump_dir, 'members.json')
    with open(mp, 'w', encoding='utf8', newline='\n') as f:
        json.dump(man, f, ensure_ascii=False)
    print('[dump ] manifest -> %s | 成员 %s | 行 %d | 类 %d | 金标 %s'
          % (mp, ','.join(names), N, NL, '有' if HAS_GOLD else '无'), flush=True)


def renorm_topk(p, k):
    """Keep only this member's own top-k frame probabilities and renormalise those k to sum 1.
    Everything else becomes 0, which is what makes a member that hedges over 1049 frames contribute the
    same total mass as a member that is already sure."""
    idx = np.argsort(-p, axis=1)[:, :k]
    v = np.take_along_axis(p, idx, axis=1)
    s = v.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    out = np.zeros_like(p)
    np.put_along_axis(out, idx, v / s, axis=1)
    return out


def merged(k):
    if WEIGHTS is None:
        if k is None:
            return sum(PROBS) / len(PROBS)
        return sum(renorm_topk(p, k) for p in PROBS) / len(PROBS)
    if k is None:
        return sum(w * p for w, p in zip(WEIGHTS, PROBS))
    return sum(w * renorm_topk(p, k) for w, p in zip(WEIGHTS, PROBS))


ARMS = []
for token in ARGS.arms.split(','):
    if not token.strip():
        continue
    ARMS.append(None if token.strip() == 'full' else int(token))
MAIN_K = None if ARGS.op == 'full' else ARGS.topk
if MAIN_K not in ARMS:
    ARMS.append(MAIN_K)
if HAS_GOLD and len(names) > 1:
    n_ok = (np.stack([p.argmax(1) == GOLD_IDX for p in PROBS])).sum(0)
    print('[cmp  ] 判对成员数分布（每个句子有几个成员押中金标框架）：'
          + ' '.join('%d个成员=%d句' % (j, int((n_ok == j).sum())) for j in range(len(names) + 1))
          + ' | 只有 1 个成员押中的句子 %d/%d = %.2f%% 是融合能吃到的互补量'
          % (int((n_ok == 1).sum()), N, 100.0 * float((n_ok == 1).mean())), flush=True)

os.makedirs(os.path.dirname(ARGS.out) or '.', exist_ok=True)


def dump_rows(avg, path):
    idx = avg.argmax(1)
    rows = [[SIDS[i], IDX2LABEL[idx[i]]] for i in range(N)]
    with open(path, 'w', encoding='utf8', newline='\n') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    return rows


def arm_label(k):
    return 'full' if k is None else 'top%d' % k


if ARGS.arm_dir:
    os.makedirs(ARGS.arm_dir, exist_ok=True)
main_rows = None
TABLE = []
for k in ARMS:
    avg = merged(k)
    label = arm_label(k)
    targets = []
    if k == MAIN_K:
        targets.append(ARGS.out)
    if ARGS.arm_dir:
        targets.append(os.path.join(ARGS.arm_dir, '%s.json' % label))
    rows = None
    for path in targets:
        rows = dump_rows(avg, path)
    if k == MAIN_K:
        main_rows = rows
    acc = float((avg.argmax(1) == GOLD_IDX).mean()) if HAS_GOLD else None
    TABLE.append((label, ' '.join(targets), acc))
    print('[arm  ] %-6s 融合后逐行正确率 %s  -> %s'
          % (label, ('%s%%' % ('%.4f' % (100 * acc)) if acc is not None else '无金标'),
             targets and ' '.join(targets) or '(不落盘)'), flush=True)

if ARGS.out_conf:
    avg = merged(MAIN_K)
    order = np.argsort(-avg, axis=1)
    conf = []
    for i in range(N):
        a, b = order[i][0], order[i][1]
        conf.append([SIDS[i], IDX2LABEL[a], round(float(avg[i][a]), 4), IDX2LABEL[b], round(float(avg[i][b]), 4)])
    with open(ARGS.out_conf, 'w', encoding='utf8', newline='\n') as f:
        json.dump(conf, f, indent=1, ensure_ascii=False)

print('=' * 72, flush=True)
print('成员数 %d  句子数 %d  框架类数 %d  金标 %s' % (len(names), N, NL, '有' if HAS_GOLD else '无'), flush=True)
for label, path, acc in TABLE:
    print('  档 %-6s %s' % (label, '%.4f%%' % (100 * acc) if acc is not None else '-'), flush=True)
if HAS_GOLD:
    from score_official import score_fi
    gold_json = json.load(codecs.open(ARGS.test_file, 'r', encoding='utf8'))
    print('  官方 score_fi(--out 那份) = %s' % (score_fi(gold_json, main_rows),), flush=True)
print('输出 -> %s' % ARGS.out, flush=True)
if ARGS.arm_dir:
    print('对照档 -> %s' % ARGS.arm_dir, flush=True)
print('=' * 72, flush=True)
print('FUSE_TASK1_DONE', flush=True)
