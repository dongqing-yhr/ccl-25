import torch
import codecs
import json
import numpy as np
from transformers import BertTokenizer


def load_frame_vocab(label_file):
    """Fixed frame-name -> index table, in frame_info.json order.

    The shared-encoder frame head (L2) needs a label space that does not depend on which frames
    happen to occur in the current file: dataset_task3's own self.frame2idx is built from the
    samples it sees, so a train-fold and a dev-fold would number the same frame differently.
    """
    with codecs.open(label_file, 'r', encoding='utf8') as f:
        info = json.load(f)
    vocab = {}
    for it in info:
        name = it.get("frame_name")
        if name and name not in vocab:
            vocab[name] = len(vocab)
    return vocab


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False, with_frame_name=False,
                 oof_frame_file=None, deploy_negatives_file=None, frame_topk=1):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        # 是否把框架名作为额外输入 token 拼进句子: 训练用金标框架, 推理用预测框架
        self.with_frame_name = with_frame_name
        # 1 = 只拼 top-1 帧 (与既有成员逐字节一致); 2 = 再拼一个 [SEP] + 第二候选帧
        self.frame_topk = frame_topk
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.idx2label = []
        for line in self.ori_labels:
            for fes in line["fes"]:
                if fes["fe_name"] not in self.idx2label:
                    self.idx2label.append(fes["fe_name"])
        self.label2idx = {}
        for i in range(len(self.idx2label)):
            self.label2idx[self.idx2label[i]] = i

        # OOF 帧: 用 sid -> 预测 frame 表覆盖金标 frame, 消除训练/推理分布差异
        # 行格式 [sid, f1, p1, f2, p2]; 第二候选帧只在 frame_topk>=2 且该行确实带第 4 列时才取
        sid2oof = {}
        sid2oof2 = {}
        if oof_frame_file:
            with codecs.open(oof_frame_file, 'r', encoding='utf8') as f:
                for row in json.load(f):
                    sid2oof[row[0]] = row[1]
                    if len(row) > 3 and row[3]:
                        sid2oof2[row[0]] = row[3]
            print('[dataset_task3] 加载 OOF 帧文件: {} 条'.format(len(sid2oof)))
            if self.frame_topk >= 2:
                print('[dataset_task3] frame_topk=2: 带第二候选帧的句 {} 条'.format(len(sid2oof2)))

        self.data = []
        for line in self.all_data:
            text = line["text"]
            target = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
            cfn_spans = line["cfn_spans"]
            # 帧来源: OOF 优先, 否则金标
            frame_used = sid2oof.get(line["sentence_id"], line.get("frame"))
            frame2_used = sid2oof2.get(line["sentence_id"])
            for spans in cfn_spans:
                if spans["end"] + 1 < target[0]:
                    label_idx = [spans["start"] + 1, spans["end"] + 1]
                elif spans["start"] + 1 > target[1]:
                    label_idx = [spans["start"] + 3, spans["end"] + 3]
                else:
                    continue
                fe_text = text[spans["start"]: spans["end"] + 1]
                self.data.append({
                    'text': text,
                    "label_class": self.label2idx[spans["fe_name"]],
                    "label_idx": label_idx,
                    "sentence_id": line["sentence_id"],
                    "frame": frame_used,
                    "frame2": frame2_used,
                    "gold_frame": line.get("frame"),
                    "target": target,
                    "fe_text": fe_text
                })

        # 部署匹配负例: 从 task2 在训练集上的假阳性 span 挖出, 归入 NULL 类
        null_idx = len(self.idx2label)  # NULL 类索引 = 原始类别数
        n_added_neg = 0
        if deploy_negatives_file:
            sid2record = {}
            for line in self.all_data:
                tgt = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
                sid2record[line["sentence_id"]] = {"text": line["text"], "target": tgt,
                                                   "gold_spans": [(s["start"], s["end"]) for s in line["cfn_spans"]],
                                                   "frame": sid2oof.get(line["sentence_id"], line.get("frame")),
                                                   "frame2": sid2oof2.get(line["sentence_id"]),
                                                   "gold_frame": line.get("frame")}
            with codecs.open(deploy_negatives_file, 'r', encoding='utf8') as f:
                neg_data = json.load(f)
            for item in neg_data:
                # item = [sentence_id, start, end]
                sid, st, en = item[0], item[1], item[2]
                rec = sid2record.get(sid)
                if rec is None:
                    continue
                # 越界防御: 挖掘器给的 span 可能超出原文本长度, 会让 forward 索引炸
                text_len = len(rec["text"])
                if st < 0 or en < st or en >= text_len:
                    continue
                target = rec["target"]
                # 与金标 span 严格重叠的不算负例
                if any(gs[0] <= st <= gs[1] or gs[0] <= en <= gs[1] for gs in rec["gold_spans"]):
                    continue
                if en + 1 < target[0]:
                    label_idx = [st + 1, en + 1]
                elif st + 1 > target[1]:
                    label_idx = [st + 3, en + 3]
                else:
                    continue
                # 二次防御: 移位后仍越界则丢弃
                if label_idx[1] >= text_len + 4:
                    continue
                self.data.append({
                    'text': rec["text"],
                    "label_class": null_idx,
                    "label_idx": label_idx,
                    "sentence_id": sid,
                    "frame": rec["frame"],
                    "frame2": rec.get("frame2"),
                    "gold_frame": rec.get("gold_frame"),
                    "target": target,
                    "fe_text": rec["text"][st:en+1]
                })
                n_added_neg += 1
            print('[dataset_task3] 加入部署匹配负例: {} 条 (NULL 类索引 {})'.format(n_added_neg, null_idx))

        self.num_labels = len(self.idx2label)

        # 帧索引表: 供 logit adjustment 的帧×角色先验按样本查帧号 (frame-conditional)
        self.frame2idx = {}
        for d in self.data:
            fr = d.get("frame")
            if fr and fr not in self.frame2idx:
                self.frame2idx[fr] = len(self.frame2idx)

        # 共享编码器帧头 (L2) 的标签: 固定 713 类表里的金标帧号, 与输入里拼的帧(OOF 预测)无关。
        # 金标帧缺失或不在表里给 -1, forward 里跳过这些句子的帧损失。
        self.frame_vocab = load_frame_vocab(label_file)
        n_ok = n_no_gold = n_off_table = 0
        for d in self.data:
            gf = d.get("gold_frame")
            gi = self.frame_vocab.get(gf) if gf else None
            d["gold_frame_idx"] = -1 if gi is None else gi
            if gi is not None:
                n_ok += 1
            elif not gf:
                n_no_gold += 1
            else:
                n_off_table += 1
        print('[dataset_task3] 帧头标签(固定 %d 类帧表): 有效 %d 行 / 无金标帧 %d 行 / 帧名不在表里 %d 行'
              % (len(self.frame_vocab), n_ok, n_no_gold, n_off_table))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d1 = self.data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        label_idx = d1["label_idx"]
        label = d1["label_class"]
        target = d1["target"]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[
                                                                                                target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        # 框架名注入: 在句尾追加一个 [SEP] 分隔再拼框架名字符。
        # 论元 span 的位置都在此之前, 索引不受影响; BERT 是双向注意力, 论元表示能读到这段框架名。
        if self.with_frame_name and d1.get("frame"):
            frame_ids = self.tokenizer.convert_tokens_to_ids(list(d1["frame"]))
            input_ids = input_ids + [self.tokenizer.sep_token_id] + frame_ids
            attention_mask = attention_mask + [1] * (len(frame_ids) + 1)
            # frame_topk=2: 再追一个 [SEP] + 第二候选帧。写在 top-1 之后, 所以 topk=1 的样本
            # 与本行存在时的 token 序列逐字节相同, 老权重的输入不会被改动 (可干净归因)。
            if self.frame_topk >= 2 and d1.get("frame2") and d1["frame2"] != d1["frame"]:
                f2_ids = self.tokenizer.convert_tokens_to_ids(list(d1["frame2"]))
                input_ids = input_ids + [self.tokenizer.sep_token_id] + f2_ids
                attention_mask = attention_mask + [1] * (len(f2_ids) + 1)
        sentence_id = d1["sentence_id"]
        frame_idx = self.frame2idx.get(d1.get("frame"), -1)
        gold_frame_idx = d1.get("gold_frame_idx", -1)

        return input_ids, attention_mask, label_idx, label, sentence_id, frame_idx, gold_frame_idx


class DevCandidateDataset(torch.utils.data.Dataset):
    """Deployment-matched validation set, used only to pick which epoch to keep.

    Rows are the spans the range model actually proposes for the dev sentences. A span that is a
    gold argument gets its gold role; every other span gets the NULL class. So this set contains the
    false positives the ordinary dev set cannot see, which is what makes the number it produces
    behave like the leaderboard instead of like accuracy-over-gold-spans.

    cand_file rows are [sentence_id, start, end], the same shape as dataset/dev/dev_task2_test.json.
    """

    def __init__(self, cand_file, json_file, label_file, tokenizer,
                 with_frame_name=False, frame_file=None, frame_topk=1):
        self.tokenizer = tokenizer
        self.with_frame_name = with_frame_name
        self.frame_topk = frame_topk
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.idx2label = []
        for line in self.ori_labels:
            for fes in line["fes"]:
                if fes["fe_name"] not in self.idx2label:
                    self.idx2label.append(fes["fe_name"])
        self.label2idx = {n: i for i, n in enumerate(self.idx2label)}
        self.num_labels = len(self.idx2label)
        self.null_idx = self.num_labels

        sid2rec = {}
        for line in self.all_data:
            sid = line["sentence_id"]
            tgt = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
            roles = {}
            for s in line["cfn_spans"]:
                roles.setdefault((s["start"], s["end"]), set()).add(s["fe_name"])
            # 与部署端 dataset 一致: 同一个 sid 有多条目标词记录时, 文本与目标词取最后一条,
            # 金标角色则把所有记录合起来(官方口径也是按句合并)。
            if sid in sid2rec:
                sid2rec[sid]["gold"].update(roles)
                sid2rec[sid]["text"] = line["text"]
                sid2rec[sid]["target"] = tgt
            else:
                sid2rec[sid] = {"text": line["text"], "target": tgt, "gold": roles}

        # 帧来源: 给了 frame_file ([[sid, frame], ...]) 就只用它, 查不到的句子不注入帧名, 与部署端一致;
        # 没给才退回金标帧(此时这个集合只能当训练期的近似选权重集用)。
        sid2frame = {}
        sid2frame2 = {}
        if frame_file:
            with codecs.open(frame_file, 'r', encoding='utf8') as f:
                for row in json.load(f):
                    try:
                        sid2frame[int(row[0])] = row[1]
                        if len(row) > 3 and row[3]:
                            sid2frame2[int(row[0])] = row[3]
                    except (TypeError, ValueError):
                        continue
        else:
            for line in self.all_data:
                if line.get("frame"):
                    sid2frame.setdefault(line["sentence_id"], line["frame"])

        # 官方口径的金标三元组: 每句一个集合, 同一 span 的多个角色都算, 全部标注句都进召回分母
        self.gold = {}
        for line in self.all_data:
            s = self.gold.setdefault(line["sentence_id"], set())
            for sp in line["cfn_spans"]:
                s.add((sp["start"], sp["end"], sp["fe_name"]))
        self.n_gold = sum(len(s) for s in self.gold.values())

        self.data = []
        n_ambiguous = 0
        with codecs.open(cand_file, 'r', encoding='utf8') as f:
            for row in json.load(f):
                sid, st, en = row[0], row[1], row[2]
                # 候选文件里的 sid 可能是字符串, 金标那边是整数; 不统一会静默丢光所有行
                try:
                    sid = int(sid)
                    st, en = int(st), int(en)
                except (TypeError, ValueError):
                    continue
                rec = sid2rec.get(sid)
                if rec is None:
                    continue
                text, tg = rec["text"], rec["target"]
                n = len(text)
                if st < 0 or en < st or en >= n:
                    continue
                if en + 1 < tg[0]:
                    label_idx = [st + 1, en + 1]
                elif st + 1 > tg[1]:
                    label_idx = [st + 3, en + 3]
                else:
                    continue
                if label_idx[1] >= n + 4:
                    continue
                names = rec["gold"].get((st, en))
                if names is None:
                    gold_role, label_class = None, self.null_idx
                else:
                    if len(names) > 1:
                        n_ambiguous += 1
                    gold_role = sorted(names)[0]
                    label_class = self.label2idx[gold_role]
                self.data.append({"text": text, "label_idx": label_idx, "label_class": label_class,
                                  "sentence_id": sid, "target": tg, "st": st, "en": en,
                                  "gold_role": gold_role,
                                  "frame": sid2frame.get(sid, None),
                                  "frame2": sid2frame2.get(sid, None) if frame_topk >= 2 else None})
        print("[DevCandidateDataset] %d candidate spans, %d of them gold arguments, "
              "%d spans carry more than one gold role (scored against the alphabetically first)"
              % (len(self.data), sum(1 for d in self.data if d["gold_role"] is not None),
                 n_ambiguous))
        print("[DevCandidateDataset] gold triples %d over %d annotated sentences; "
              "recall denominator uses all of them (spans task2 never proposed count as misses)"
              % (self.n_gold, len(self.gold)))
        print("[DevCandidateDataset] 帧来源: %s (覆盖 %d 句, with_frame_name=%s)"
              % (("预测帧 " + frame_file) if frame_file else "金标帧 (dev 文件自带)",
                 len(sid2frame), with_frame_name))
        if len(self.data) == 0 or self.n_gold == 0:
            raise RuntimeError("[DevCandidateDataset] nothing to score: %d candidate rows, %d gold "
                               "triples - check cand_file sid type / path" % (len(self.data), self.n_gold))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d1 = self.data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = d1["target"]
        input_ids = input_ids[0:target[0]] + [1] + input_ids[target[0]:target[1] + 1] + [2] + \
                    input_ids[target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        if self.with_frame_name and d1.get("frame"):
            frame_ids = self.tokenizer.convert_tokens_to_ids(list(d1["frame"]))
            input_ids = input_ids + [self.tokenizer.sep_token_id] + frame_ids
            attention_mask = attention_mask + [1] * (len(frame_ids) + 1)
            if self.frame_topk >= 2 and d1.get("frame2") and d1["frame2"] != d1["frame"]:
                f2_ids = self.tokenizer.convert_tokens_to_ids(list(d1["frame2"]))
                input_ids = input_ids + [self.tokenizer.sep_token_id] + f2_ids
                attention_mask = attention_mask + [1] * (len(f2_ids) + 1)
        return (input_ids, attention_mask, d1["label_idx"], d1["label_class"], d1["sentence_id"],
                d1["st"], d1["en"], d1["gold_role"])


def get_cand_input(data, device=None):
    """Collate for DevCandidateDataset: pads the token side, keeps the span bookkeeping as lists."""
    max_len = max(len(x[0]) for x in data)
    pad = lambda d, v=0: d + [v] * (max_len - len(d))
    ids = np.array([pad(x[0]) for x in data], dtype=np.compat.long)
    am = np.array([pad(x[1]) for x in data], dtype=np.compat.long)
    ids = torch.from_numpy(ids).to(device)
    am = torch.from_numpy(am).to(device)
    labels = torch.from_numpy(np.array([x[3] for x in data], dtype=np.compat.long)).to(device)
    return ids, am, [x[2] for x in data], labels, [x[4] for x in data], data


if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='/work/models/chinese-bert-wwm-ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset("./dataset/cfn-train.json",
                      "./dataset/frame_info.json",
                      tokenizer=tokenizer)

    dataset[0]
