#!/usr/bin/python3

import torch
import codecs
import json
from functools import partial
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from params import args
from model_task3 import Model
import model_task3

# NULL 弃权机制: 与 train_task3.py 保持一致
args.with_null = True
args.null_weight = 0.5


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, task2_file, task1_file, tokenizer, for_test=False,
                 with_frame_name=False, soft_cascade=False, sc_high=0.8, sc_low=0.5,
                 task1_conf='./dataset/A_task1_conf.json', frame_roles='./dataset/frame_roles.json',
                 frame_topk=1):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        # 是否把框架名作为额外输入 token 拼进句子; 推理时框架来自 task1 预测结果
        self.with_frame_name = with_frame_name
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        with codecs.open(task2_file, 'r', encoding='utf8') as f:
            self.task2_data = json.load(f)
        # 仅在需要注入框架名时才读 task1 预测结果, 避免关闭时依赖该文件
        self.sid2frame = {}
        if with_frame_name:
            with codecs.open(task1_file, 'r', encoding='utf8') as f:
                task1_data = json.load(f)
            for item in task1_data:
                self.sid2frame[item[0]] = item[1]
        # frame_topk=2: 第二候选帧取自 task1_conf 文件 (行格式 [sid, f1, p1, f2, p2])。
        # 只读不建候选集(那是 soft_cascade 的事), 所以 with_frame_name 开着但 soft_cascade 关着时
        # 也能拿到 f2; 文件不覆盖的句子只注 top-1, 与训练端 sid2oof2 的缺省行为一致。
        self.frame_topk = frame_topk
        self.sid2frame2 = {}
        if with_frame_name and frame_topk >= 2:
            with codecs.open(task1_conf, 'r', encoding='utf8') as f:
                conf_rows = json.load(f)
            bad = [r for r in conf_rows if len(r) < 4]
            if bad:
                raise RuntimeError("[predict_task3] frame_topk=2 需要 conf 文件带第二候选列, "
                                   "但前 %d 行里最短只有 %d 列: %s -> 请确认传的是 task1_conf 而不是单帧文件"
                                   % (min(5, len(bad)), min(len(r) for r in bad), bad[0]))
            for row in conf_rows:
                if row[3]:
                    self.sid2frame2[row[0]] = row[3]
            print("[predict_task3] frame_topk=2: conf 覆盖 %d 句, 其中带第二候选帧 %d 句"
                  % (len(conf_rows), len(self.sid2frame2)))

        self.idx2label = []
        for line in self.ori_labels:
            for fes in line["fes"]:
                if fes["fe_name"] not in self.idx2label:
                    self.idx2label.append(fes["fe_name"])
        self.label2idx = {}
        for i in range(len(self.idx2label)):
            self.label2idx[self.idx2label[i]] = i

        # 软级联: 按 task1 置信度给每句定候选角色集, 低置信度句不 mask (回退全局)
        self.sid2cand = {}
        if soft_cascade:
            with codecs.open(frame_roles, 'r', encoding='utf8') as f:
                frame2roles = json.load(f)
            with codecs.open(task1_conf, 'r', encoding='utf8') as f:
                conf_data = json.load(f)
            n_high = n_mid = n_low = 0
            for sid, f1, p1, f2, p2 in conf_data:
                if p1 >= sc_high:
                    cand = frame2roles.get(f1, [])
                    n_high += 1
                elif p1 >= sc_low:
                    cand = list(dict.fromkeys(frame2roles.get(f1, []) + frame2roles.get(f2, [])))
                    n_mid += 1
                else:
                    cand = None
                    n_low += 1
                self.sid2cand[sid] = cand
            print("[predict_task3] 软级联分档: 高置信度 {} 句, 中置信度 {} 句, 低置信度(全局) {} 句".format(
                n_high, n_mid, n_low))

        self.data_dict = {}
        for line in self.all_data:
            text = line["text"]
            target = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
            self.data_dict[line["sentence_id"]] = {"text": text, "target": target}
        self.data = []
        dropped = 0
        for line in self.task2_data:
            rec = self.data_dict.get(line[0])
            if rec is None:
                dropped += 1
                continue
            text = rec["text"]
            target = rec["target"]
            st, en = line[1], line[2]
            n = len(text)
            # task2 预测出的畸形论元(越界/起止倒置)先丢掉
            if st < 0 or en >= n or st > en:
                dropped += 1
                continue
            if en + 1 < target[0]:
                label_idx = [st + 1, en + 1]
            elif st + 1 > target[1]:
                label_idx = [st + 3, en + 3]
            else:
                # 与目标词重叠的预测 span: 两个分支都不命中, label_idx 会未定义, 同样丢掉
                dropped += 1
                continue
            # 兜底: 换算后仍越过本句编码长度(CLS+句尾SEP+插入的两个标记=len+4)则丢掉
            if label_idx[1] >= n + 4:
                dropped += 1
                continue
            self.data.append({
                'text': text,
                "label_idx": label_idx,
                "sentence_id": line[0],
                "frame": self.sid2frame.get(line[0]),
                "frame2": self.sid2frame2.get(line[0]),
                "target": target,
                "ori_target": [st, en]
            })
        if dropped:
            print("[predict_task3] 丢弃 {} 个越界/与目标词重叠的畸形预测论元".format(dropped))
        self.num_labels = len(self.idx2label)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d1 = self.data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        label_idx = d1["label_idx"]
        target = d1["target"]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[
                                                                                                target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        # 框架名注入: 句尾追加 [SEP] 再拼预测框架名; 论元 span 位置在此之前, 索引不受影响
        if self.with_frame_name and d1.get("frame"):
            frame_ids = self.tokenizer.convert_tokens_to_ids(list(d1["frame"]))
            input_ids = input_ids + [self.tokenizer.sep_token_id] + frame_ids
            attention_mask = attention_mask + [1] * (len(frame_ids) + 1)
            # frame_topk=2: 再拼 [SEP] + 第二候选帧; topk=1 时这段整体跳过, token 序列与老权重训练时逐字节相同
            if self.frame_topk >= 2 and d1.get("frame2") and d1["frame2"] != d1["frame"]:
                f2_ids = self.tokenizer.convert_tokens_to_ids(list(d1["frame2"]))
                input_ids = input_ids + [self.tokenizer.sep_token_id] + f2_ids
                attention_mask = attention_mask + [1] * (len(f2_ids) + 1)
        sentence_id = d1["sentence_id"]
        ori_target = d1["ori_target"]

        return input_ids, attention_mask, label_idx, sentence_id, ori_target


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
    ori_target = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        sentence_id.append(d[3])
        ori_target.append(d[4])

    input_ids = np.array(input_ids_list, dtype=np.compat.long)
    attention_mask = np.array(attention_mask_list, dtype=np.compat.long)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)

    return input_ids, attention_mask, target, sentence_id, ori_target


def test(model, val_loader):
    model.eval()
    idx2label = val_loader.dataset.idx2label
    predicts = []
    if getattr(args, 'save_logits', False):
        globals()['logits_records'] = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, sentence_id, ori_target = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None,
                           device=device, for_test=True)
            logits = output["logits"]
            # 软级联 mask: 候选集外的角色 logit 置 -inf
            sid2cand = val_loader.dataset.sid2cand
            if sid2cand:
                label2idx = val_loader.dataset.label2idx
                for i in range(len(sentence_id)):
                    cand = sid2cand.get(sentence_id[i])
                    if cand:
                        idx = [label2idx[r] for r in cand if r in label2idx]
                        if idx:
                            keep = torch.zeros_like(logits[i], dtype=torch.bool)
                            keep[idx] = True
                            logits[i] = logits[i].masked_fill(~keep, float('-inf'))
            probs = F.softmax(logits, dim=-1)
            # NULL 类索引 = 原始类别数 (1009)
            null_idx = val_loader.dataset.num_labels
            null_thr = getattr(args, 'null_threshold', -1.0)
            second_thr = getattr(args, 'second_role_threshold', -1.0)
            save_logits = getattr(args, 'save_logits', False)
            n_null = 0
            for i in range(len(probs)):
                sid = int(sentence_id[i])
                st, en = int(ori_target[i][0]), int(ori_target[i][1])
                pvec = probs[i]
                if save_logits:
                    # sidecar: [sid, st, en, p0, p1, ..., p_null]
                    if 'logits_records' not in globals():
                        globals()['logits_records'] = []
                    globals()['logits_records'].append([sid, st, en] + [round(float(x), 6) for x in pvec.tolist()])
                if null_thr < 0:
                    # 旧规则: argmax 命中 NULL 就弃权
                    pred = int(torch.argmax(pvec))
                    if pred >= null_idx:
                        n_null += 1
                        continue
                else:
                    # 新规则: 仅当 p_null > 阈值才弃权, 否则在真实 FE 里取 argmax
                    if float(pvec[null_idx]) > null_thr:
                        n_null += 1
                        continue
                    pred = int(torch.argmax(pvec[:null_idx]))
                predicts.append([sid, st, en, idx2label[pred]])
                if second_thr >= 0.0:
                    # 同跨度多角色: 保留概率次高的 FE 角色
                    masked = pvec[:null_idx].clone()
                    masked[pred] = -1.0
                    pred2 = int(torch.argmax(masked))
                    if pred2 != pred and float(masked[pred2]) >= second_thr:
                        predicts.append([sid, st, en, idx2label[pred2]])
            if n_null:
                print("[predict_task3] 本 batch NULL 弃权 {} 个".format(n_null))

            pass
    if globals().get('logits_records'):
        logits_out = args.logits_out or '{}/{}task3_logits.json'.format(args.out_dir, args.out_prefix)
        with open(logits_out, 'w', encoding='utf8', newline='\n') as f:
            json.dump(globals()['logits_records'], f, ensure_ascii=False, indent=1)
        print("[predict_task3] 已保存 logits -> {} ({} spans)".format(logits_out, len(globals()['logits_records'])))
    data_json = json.dumps(predicts, indent=1, ensure_ascii=False)
    with open('{}/{}task3_test.json'.format(args.out_dir, args.out_prefix), 'w', encoding='utf8', newline='\n') as f:
        f.write(data_json)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    test_dataset = Dataset(args.test_file,
                            "./dataset/frame_info.json",
                            args.task2_result,
                            args.task1_result,
                            tokenizer,
                            with_frame_name=args.with_frame_name,
                            soft_cascade=args.soft_cascade,
                            sc_high=args.sc_high,
                            sc_low=args.sc_low,
                            task1_conf=args.task1_conf,
                            frame_roles=args.frame_roles,
                            frame_topk=args.frame_topk)

    config = BertConfig.from_json_file(args.config_file)
    model_task3.LAYER_FUSION = args.layer_fusion or 'avg4'
    model_task3.INNER_DIM = args.inner_dim or 64
    print('[predict_task3] 结构开关: layer_fusion={} inner_dim={}'.format(
        model_task3.LAYER_FUSION, model_task3.INNER_DIM))
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = test_dataset.num_labels
    config.with_null = args.with_null
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(args.model_path or 'saves/model_task3_best.bin', map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    # strict=False 会静默丢掉形状不符的打分头 -> 随机初始化照样出预测, 这里单独卡住 dense
    _dense_miss = [k for k in msg.missing_keys if k.startswith('dense.')]
    if _dense_miss:
        raise RuntimeError('打分头权重没装进去(missing %d): %s  (检查 --inner_dim / --layer_fusion 与训练时是否一致)'
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
