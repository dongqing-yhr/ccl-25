import torch
import codecs
import json
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False, expand_targets=0):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.expand_targets = int(expand_targets)
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

    @staticmethod
    def expand_label(label, target, n_chars):
        """N1 (our own definition): also mark the +-1 character variants of every gold span as positive.

        Coordinates here are already the post-insertion ones used by the model, where index
        target[0] and target[1] + 2 are the two tokens wrapping the target word and the target
        word itself sits in between. A variant is dropped when it would touch one of those three
        blocks, run onto [CLS] or [SEP], or duplicate a span already in the list.
        """
        last_char = n_chars + 2
        seen = set(tuple(pair) for pair in label)
        out = list(label)
        for start, end in label:
            for s2, e2 in ((start - 1, end), (start, end + 1), (start - 1, end + 1)):
                if s2 < 1 or e2 > last_char:
                    continue
                if not (e2 < target[0] or s2 > target[1] + 2):
                    continue
                if (s2, e2) in seen:
                    continue
                seen.add((s2, e2))
                out.append([s2, e2])
        return out

    @staticmethod
    def merge_label(label):
        """N1 mode 2/3: grow each span by ABSORBING the next one when the two touch or nest.

        Chosen by the fingerprint scan (runbash/n1_shape_match_1101.py): the opponent's fused cell
        emits 4865 spans / 1994 sentences with mean width ~8.94 chars, i.e. fewer but wider spans
        than the gold (5947 / 7.75), while its character union exceeds gold. Merging touches no
        character, so under the official character-overlap ruler the merged label set scores the
        same as the gold one (lossless); it only removes narrow duplicates. Under a strict
        span-set ruler the inner spans of a nest are lost, which is why this is a switch and not
        a default.
        """
        out = []
        for start, end in sorted((tuple(pair) for pair in label)):
            if out and start <= out[-1][1] + 1:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([start, end])
        return out

    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        target_cls = self.label2cls[d1["frame"]]
        label = []
        for line in d1["cfn_spans"]:
            if line["end"] + 1 < target[0]:
                label.append([line["start"] + 1, line["end"] + 1])
            elif line["start"] + 1 > target[1]:
                label.append([line["start"] + 3, line["end"] + 3])
        if self.expand_targets == 1 and not self.for_test:
            label = self.expand_label(label, target, len(d1['text']))
        elif self.expand_targets in (2, 3) and not self.for_test:
            merged = self.merge_label(label)
            if self.expand_targets == 2:
                label = merged
            else:
                seen = set(tuple(pair) for pair in label)
                label = label + [pair for pair in merged if tuple(pair) not in seen]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]

        return input_ids, attention_mask, target, label, sentence_id, target_cls


if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='/work/models/chinese-bert-wwm-ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset("./dataset/cfn-train.json",
                      "./dataset/frame_info.json",
                      tokenizer=tokenizer)

    dataset[0]





