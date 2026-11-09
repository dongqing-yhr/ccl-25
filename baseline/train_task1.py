#!/usr/bin/python3

import os
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from dataset_task1 import Dataset
from params import args
from model_task1 import Model
from ema import seed_everything, EMA


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


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
    labels = []
    sentence_id = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])
        sentence_id.append(d[4])

    input_ids = np.array(input_ids_list, dtype=np.compat.long)
    attention_mask = np.array(attention_mask_list, dtype=np.compat.long)
    labels = np.array(labels, dtype=np.compat.long)

    input_ids = torch.from_numpy(input_ids)
    attention_mask = torch.from_numpy(attention_mask)
    labels = torch.from_numpy(labels)

    return input_ids, attention_mask, target, labels, sentence_id


def eval(model, val_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, labels, sentence_id = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device, for_test=True)
            logits = output["logits"]
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            correct += (pred == labels).sum().item() 
            total += len(pred)
            
    return correct / (total + 1e-6)


def train(model, train_loader, val_loader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # *******************  NoisyTune  ****************

    noise_lambda = args.noise_lambda
    for name, para in param_optimizer:
        model.state_dict()[name][:] += \
            (torch.rand(para.size()).to(device) - 0.5) * \
            noise_lambda * torch.std(para)

    # ***********************************************


    total_steps = int(len(train_loader) * args.num_train_epochs /
                      args.accumulate_gradients)

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr)

    # adv = FGM(bert_model) if args.with_adv_train else None
    global_step = 0
    best_acc = 0.0
    best_ema = 0.0
    fgm = None if args.no_fgm else FGM(model)
    ema = EMA(model, decay=args.ema_decay) if args.with_ema else None
    for i_epoch in range(1, 1 + args.num_train_epochs):
        total_loss = 0.0
        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch_{i_epoch} ')
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids, attention_mask, target, labels, sentence_id = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device)

            loss = output['loss']

            total_loss += loss.item()

            if (step + 1) % 1000 == 0:
                print(
                    f'loss: {total_loss / (step + 1)}')

            loss.backward()

            if fgm is not None:
                fgm.attack()
                loss_sum = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device)['loss']
                loss_sum.backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if (step + 1) % args.accumulate_gradients == 0:
                lr_this_step = args.lr * \
                               warmup_linear(global_step / total_steps,
                                             args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if ema is not None:
                    ema.update()

            # break

        acc = eval(model, val_loader)

        if acc > best_acc:
            print(f'saved! new best acc {acc}, ori_acc {best_acc}')
            best_acc = acc
            model_to_save = model.module if hasattr(model, 'module') else model
            os.makedirs('saves', exist_ok=True)
            torch.save(model_to_save.state_dict(), f'saves/{args.model_prefix}_best.bin')
        else:
            print(f'current acc: {acc}')

        if ema is not None:
            ema.apply()
            acc_ema = eval(model, val_loader)
            ema.restore()
            print(f'epoch {i_epoch}: raw_acc {acc}, ema_acc {acc_ema}')
            if acc_ema > best_ema:
                best_ema = acc_ema
                os.makedirs('saves', exist_ok=True)
                ema.save(f'saves/{args.model_prefix}_ema.bin')
                print(f'  saved EMA! new best ema_acc {acc_ema}')

        # train_loader.dataset.gen_data()


def load_pretrained_bert(bert_model, init_checkpoint):
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location='cpu')
        if 'model_bert_best' in init_checkpoint:
            bert_model.load_state_dict(state['model_bert'], strict=False)
        else:
            state = {k.replace('bert.', '').replace('roformer.', ''): v for k, v in state.items() if
                     not k.startswith('cls.')}
            # state['embeddings.token_type_embeddings.weight'] = state['embeddings.token_type_embeddings.weight'][:2, :]
            bert_model.load_state_dict(state, strict=False)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # 打印训练配置
    print("=" * 60)
    print("Task1 训练配置")
    print("=" * 60)
    print(f"底座模型: {args.config_file}")
    print(f"FGM 对抗训练: {'禁用' if args.no_fgm else '启用'} (--no_fgm 关闭; 开启时每步在 embedding 上做攻击并累加梯度)")
    print(f"NoisyTune 扰动幅度: {args.noise_lambda} (--noise_lambda; 不传=0.15, 就是原来写死的值; 0=不加噪)")
    print(f"EMA 权重平均: {'启用' if args.with_ema else '禁用'}")
    if args.with_ema:
        print(f"  EMA decay: {args.ema_decay}")
    print(f"训练轮数: {args.num_train_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"梯度累积: {args.accumulate_gradients}")
    print(f"学习率: {args.lr}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    train_dataset = Dataset(args.train_file,
                            "./dataset/frame_info.json",
                            tokenizer)
    dev_dataset = Dataset(args.val_file,
                          "./dataset/frame_info.json",
                          tokenizer)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = train_dataset.num_labels
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(args.init_checkpoint, map_location='cpu')
    # LSTM 头 forward 里没用到, 老 checkpoint 尺寸可能不匹配, 丢弃
    state = {k: v for k, v in state.items() if not k.startswith('lstm.')}
    msg = model.load_state_dict(state, strict=False)
    # model.load_state_dict(torch.load('', map_location='cpu'))
    model = model.to(device)

    train_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=train_dataset,
        shuffle=True,
        num_workers=4,
        collate_fn=partial(get_model_input, device=device),
        drop_last=True
    )

    val_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=dev_dataset,
        shuffle=False,
        num_workers=4,
        collate_fn=partial(get_model_input, device=device),
        drop_last=False
    )
    train(model, train_loader, val_loader)
