#!/usr/bin/python3

import torch
import codecs
import json
import os
from functools import partial
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from params import args
import model_task2
from model_task2 import Model


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.label2idx = {}
        self.label2cls = {}
        self.num_labels = 0
        for k, line in enumerate(self.ori_labels):
            frame_name = line["frame_name"]
            self.label2cls[frame_name] = k
            if frame_name not in self.label2idx:
                self.label2idx[frame_name] = {}
            for i, fes in enumerate(line["fes"]):
                self.label2idx[frame_name][fes["fe_name"]] = i
            if self.num_labels < len(line["fes"]):
                self.num_labels = len(line["fes"])
        self.max_cls = len(self.label2idx)

        pass

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[
                                                                                                target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]

        return input_ids, attention_mask, target, sentence_id


def get_model_input(data, device=None):
    """

    :param data: input_ids1, input_ids2, label_starts, label_ends, true_label
    :return:
    """

    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))

    bs = len(data)
    max_len = max([len(x[0]) for x in data])

    input_ids_list = []
    attention_mask_list = []
    target = []
    sentence_id = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        sentence_id.append(d[3])

    input_ids = np.array(input_ids_list, dtype=np.compat.long)
    attention_mask = np.array(attention_mask_list, dtype=np.compat.long)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)


    return input_ids, attention_mask, target, sentence_id


def span_iou(s1, s2):
    # s = (start, end)，闭区间
    inter = max(0, min(s1[1], s2[1]) - max(s1[0], s2[0]) + 1)
    union = max(s1[1], s2[1]) - min(s1[0], s2[0]) + 1
    return inter / union if union > 0 else 0.0


def nms_spans(cands, iou_thr):
    # cands: [(start, end, score)]，按分数从高到低保留，抑制 IoU 超过阈值的低分 span
    cands = sorted(cands, key=lambda x: x[2], reverse=True)
    kept = []
    for c in cands:
        if all(span_iou(c[:2], k[:2]) <= iou_thr for k in kept):
            kept.append(c)
    return kept


def refine_span(logits, mask, i, j, r, min_gain, lo, hi):
    """在 (i,j) 的 ±r 邻域内用跨度头自己的 logit 重挑边界，索引限制在 [lo,hi]（不跨目标词标记）。
    只有当邻域最高 logit 比当前边界高出 min_gain 才移动，否则原样返回。"""
    cur = logits[i, j]
    best = cur
    bi, bj = i, j
    for ii in range(max(lo, i - r), min(hi, i + r) + 1):
        if mask[ii] < 0.5:
            continue
        for jj in range(max(lo, ii, j - r), min(hi, j + r) + 1):
            if mask[jj] < 0.5:
                continue
            s = logits[ii, jj]
            if s > best:
                best, bi, bj = s, ii, jj
    return (bi, bj) if best > cur + min_gain else (i, j)


def test(model, val_loader):
    model.eval()
    predicts = []
    dumped = []
    R = args.refine_boundary
    n_refined = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, sentence_id = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None,
                           device=device, for_test=True)
            H_attention_mask = torch.triu(
                torch.matmul(attention_mask.unsqueeze(2).float(), attention_mask.unsqueeze(1).float()), diagonal=0)
            H_pred = torch.where(
                output["logits"] >= args.logit_threshold,
                torch.ones(output["logits"].shape).to(device),
                torch.zeros(output["logits"].shape).to(device)
            ) * H_attention_mask

            logits_np = output["logits"].detach().cpu().numpy()
            am_np = attention_mask.detach().cpu().numpy()
            predict_idx = torch.nonzero(H_pred)
            pred = [[] for i in range(len(H_attention_mask))]
            for idx in predict_idx:
                b, i, j = idx[0].item(), idx[1].item(), idx[2].item()
                if j < target[b][0]:
                    lo, hi, off = 1, target[b][0] - 1, 1
                elif i > target[b][1]:
                    lo, hi, off = target[b][1] + 3, logits_np.shape[-1] - 1, 3
                else:
                    continue
                if R > 0 and lo <= i <= j <= hi:
                    ni, nj = refine_span(logits_np[b], am_np[b], i, j, R, args.refine_min_gain, lo, hi)
                    if (ni, nj) != (i, j):
                        n_refined += 1
                    i, j = ni, nj
                pred[b].append((i - off, j - off, logits_np[b][i, j]))
            nms_drop = 0
            for b in range(len(pred)):
                cands = pred[b]
                if args.with_nms:
                    kept = nms_spans(cands, args.nms_iou)
                    nms_drop += len(cands) - len(kept)
                    cands = kept
                for start, end, score in cands:
                    predicts.append([sentence_id[b], start, end])
                    if args.span_dump:
                        dumped.append([int(sentence_id[b]), int(start), int(end), float(score)])
            pass
    if args.with_nms:
        print(f'NMS: dropped {nms_drop} spans, kept {len(predicts)}')
    if R > 0:
        print(f'REFINE: moved {n_refined} of {len(predicts)} spans, r={R}, min_gain={args.refine_min_gain}')
    data_json = json.dumps(predicts, indent=1, ensure_ascii=False)
    out_name = getattr(args, 'out_name', None)
    out_path = out_name if out_name else '{}/{}task2_test.json'.format(args.out_dir, args.out_prefix)
    with open(out_path, 'w', encoding='utf8', newline='\n') as f:
        f.write(data_json)
    if args.span_dump:
        d_dir = os.path.dirname(args.span_dump)
        if d_dir:
            os.makedirs(d_dir, exist_ok=True)
        with open(args.span_dump, 'w', encoding='utf8', newline='\n') as f:
            f.write(json.dumps(dumped, indent=1, ensure_ascii=False))
        print('SPANDUMP: %d scored spans -> %s' % (len(dumped), args.span_dump))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 解码必须和训练用同一套结构开关, 否则权重形状对不上且是被静默 strict=False 吞掉的
    model_task2.LAYER_FUSION = args.layer_fusion or 'last'
    model_task2.INNER_DIM = args.inner_dim or 64
    print('[predict_task2] 结构开关: layer_fusion={} inner_dim={}'.format(
        model_task2.LAYER_FUSION, model_task2.INNER_DIM))

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    test_dataset = Dataset(args.test_file,
                            "./dataset/frame_info.json",
                            tokenizer)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = 1
    config.max_cls = test_dataset.max_cls
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    _t2_ckpt = args.model_path
    if not _t2_ckpt or 'task3' in _t2_ckpt:
        _t2_ckpt = 'saves/model_task2_best.bin'
    print('[predict_task2] 加载权重: {}'.format(_t2_ckpt))
    state = torch.load(_t2_ckpt, map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    # strict=False 会把形状不符的跨度头权重静默丢掉 -> 随机初始化照样出预测。inner_dim 一改就会踩这条, 所以单卡住 dense。
    _dense_miss = [k for k in msg.missing_keys if k.startswith('dense.')]
    if _dense_miss:
        raise RuntimeError('跨度头权重没装进去(missing %d): %s  (检查 --inner_dim / --layer_fusion 与训练时是否一致)'
                           % (len(_dense_miss), _dense_miss[:5]))
    # model.load_state_dict(torch.load('', map_location='cpu'))
    model = model.to(device)


    test_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=test_dataset,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device),
        drop_last=False
    )
    test(model, test_loader)
