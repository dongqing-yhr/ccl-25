# !/usr/bin/python3

import codecs
import json
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from params import args
from model_task1 import Model


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.idx2label = []
        self.label2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.idx2label.append(line["frame_name"])
            self.label2idx[line["frame_name"]] = i

        self.num_labels = len(self.idx2label)
        pass

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
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


def test(model, test_loader):
    model.eval()
    all_dataset = []
    conf_dataset = []
    prob_dump = getattr(args, 'prob_dump', '') or ''
    prob_rows = []
    idx2label = test_loader.dataset.idx2label
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='eval'):
            input_ids, attention_mask, target, sentence_id = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None,
                           device=device, for_test=True)
            logits = output["logits"]
            probs = F.softmax(logits, dim=-1)
            top2p, top2i = torch.topk(probs, 2, dim=-1)
            pred = top2i[:, 0]
            for i in range(len(sentence_id)):
                all_dataset.append([sentence_id[i], idx2label[pred[i]]])
                # sidecar: top1/top2 框架及置信度, 供 task3 软级联使用
                conf_dataset.append([sentence_id[i], idx2label[int(top2i[i][0])], round(float(top2p[i][0]), 4),
                                     idx2label[int(top2i[i][1])], round(float(top2p[i][1]), 4)])
            if prob_dump:
                prob_rows.append(probs.detach().float().cpu().numpy())

    data_json = json.dumps(all_dataset, indent=1, ensure_ascii=False)
    out_name = getattr(args, 'out_name', None)
    out_path = out_name if out_name else '{}/{}task1_test.json'.format(args.out_dir, args.out_prefix)
    out_conf = (out_name.replace('task1_', 'task1conf_') if out_name
                else '{}/{}task1_conf.json'.format(args.out_dir, args.out_prefix))
    with open(out_path, 'w', encoding='utf8', newline='\n') as f:
        f.write(data_json)
    conf_json = json.dumps(conf_dataset, indent=1, ensure_ascii=False)
    with open(out_conf, 'w', encoding='utf8', newline='\n') as f:
        f.write(conf_json)

    if prob_dump:
        # 全 713 维概率: task1_conf.json 只有前两名的名字和置信度, 算不出 top-3/5/10/20 覆盖率。
        mat = np.concatenate(prob_rows, axis=0)
        np.savez(prob_dump, probs=mat.astype(np.float32),
                 sentence_id=np.array([r[0] for r in all_dataset], dtype=object))
        with open(prob_dump + '.labels.json', 'w', encoding='utf8') as f:
            json.dump(idx2label, f, ensure_ascii=False)
        print('[prob] 全量概率落盘 {} shape={} x {} (行序同 {}), 标签表 {}'.format(
            prob_dump, mat.shape[0], mat.shape[1], out_path, prob_dump + '.labels.json'), flush=True)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    test_dataset = Dataset(args.test_file,
                            "./dataset/frame_info.json",
                            tokenizer)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = test_dataset.num_labels
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    # state = torch.load(args.init_checkpoint, map_location='cpu')
    model_ckpt = args.model_path
    # 双保险: 空/旧 task3 默认都视为未指定, 走 task1 兜底
    if not model_ckpt or 'task3' in model_ckpt:
        model_ckpt = 'saves/model_task1_best.bin'
    state = torch.load(model_ckpt, map_location='cpu')
    # LSTM 头 forward 里没用到, 老 checkpoint 的 LSTM 尺寸可能与当前模型不匹配, 直接丢弃
    state = {k: v for k, v in state.items() if not k.startswith('lstm.')}

    msg = model.load_state_dict(state, strict=False)
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





