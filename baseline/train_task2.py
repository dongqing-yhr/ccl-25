#!/usr/bin/python3

import os
import sys
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from dataset_task2 import Dataset
from params import args
import model_task2
from model_task2 import Model
from ema import seed_everything, EMA


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
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
    target_cls = []
    labels = []
    sentence_id = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])
        sentence_id.append(d[4])
        target_cls.append(d[5])

    input_ids = np.array(input_ids_list, dtype=np.compat.long)
    attention_mask = np.array(attention_mask_list, dtype=np.compat.long)
    target_cls = np.array(target_cls, dtype=np.compat.long)

    input_ids = torch.from_numpy(input_ids)
    attention_mask = torch.from_numpy(attention_mask)
    target_cls = torch.from_numpy(target_cls)

    H_label = torch.zeros(bs, max_len, max_len)
    for i in range(bs):
        for idx in labels[i]:
            H_label[i][idx[0], idx[1]] = 1

    return input_ids, attention_mask, target, H_label, sentence_id, target_cls


def eval(model, val_loader):
    model.eval()
    H_correct = 0.0
    H_precision_total = 0.0
    H_recall_total = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, label, sentence_id, target_cls = batch
            input_ids, attention_mask, label, target_cls = input_ids.to(device), attention_mask.to(device), label.to(device), target_cls.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=label,
                           device=device, for_test=True)

            # output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)

            H_attention_mask = torch.triu(
                torch.matmul(attention_mask.unsqueeze(2).float(), attention_mask.unsqueeze(1).float()), diagonal=0)
            H_pred = torch.where(
                output["logits"] >= args.logit_threshold,
                torch.ones(output["logits"].shape).to(device),
                torch.zeros(output["logits"].shape).to(device)
            ) * H_attention_mask


            # H_pred = torch.argmax(F.softmax(output["logits"], dim=-1), dim=-1)

            H_correct += ((H_pred == label) * (label != 0)).sum().item()
            H_precision_total += (H_pred != 0).sum().item()
            H_recall_total += (label != 0).sum().item()

    H_precision = H_correct / (H_precision_total + 1e-6)
    H_recall = H_correct / (H_recall_total + 1e-6)


    return H_precision, H_recall


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
    best_f1 = 0.0
    best_ema = 0.0
    fgm = None if args.no_fgm else FGM(model)
    ema = EMA(model, decay=args.ema_decay) if args.with_ema else None
    for i_epoch in range(1, 1 + args.num_train_epochs):
        total_loss = 0.0
        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch_{i_epoch} ')
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids, attention_mask, target, label, sentence_id, target_cls = batch
            input_ids, attention_mask, label, target_cls = input_ids.to(device), attention_mask.to(device), label.to(device), target_cls.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=label,
                           device=device)

            # output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)

            loss = output['loss']

            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(
                    f'loss: {total_loss / (step + 1)}')

            loss.backward()

            if fgm is not None:
                fgm.attack()  # embedding被修改了
                # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
                loss_sum = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=label,
                                   device=device)[
                        'loss']
                # loss_sum = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)["loss"]
                loss_sum.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复Embedding的参数

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

        H_precision, H_recall = eval(model, val_loader)
        f1 = 2 * H_precision * H_recall / (H_precision + H_recall + 1e-6)
        if f1 > best_f1:
            print(f'saved! new best f1 {f1}, ori_f1 {best_f1}')
            print(f"H_precision: {H_precision}, H_recall: {H_recall}")
            best_f1 = f1
            model_to_save = model.module if hasattr(model, 'module') else model
            os.makedirs('saves', exist_ok=True)
            torch.save(model_to_save.state_dict(), f'saves/{args.model_prefix}_best.bin')
        else:
            print(f'current f1: {f1}')
            print(f" H_precision: {H_precision}, H_recall: {H_recall}")

        if ema is not None:
            ema.apply()
            H_precision_ema, H_recall_ema = eval(model, val_loader)
            f1_ema = 2 * H_precision_ema * H_recall_ema / (H_precision_ema + H_recall_ema + 1e-6)
            ema.restore()
            print(f'epoch {i_epoch}: raw_f1 {f1}, ema_f1 {f1_ema}')
            if f1_ema > best_ema:
                best_ema = f1_ema
                os.makedirs('saves', exist_ok=True)
                ema.save(f'saves/{args.model_prefix}_ema.bin')
                print(f'  saved EMA! new best ema_f1 {f1_ema}')

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
    # 轮数: 命令行传了 --num_train_epochs 就用命令行的; 没传时沿写 5
    # (旧代码在这里无条件 args.num_train_epochs = 5, 把命令行覆盖掉, 日志还打印 5, 看不出来)
    epochs_from_cli = any(a.split('=')[0] == '--num_train_epochs' for a in sys.argv[1:])
    if not epochs_from_cli:
        args.num_train_epochs = 5

    # 写进 model_task2 的模块级开关(未传参时保持现役取值)
    model_task2.LAYER_FUSION = args.layer_fusion or 'last'
    model_task2.INNER_DIM = args.inner_dim or 64
    model_task2.POS_WEIGHT = args.pos_weight

    # 打印训练配置
    print("=" * 60)
    print("Task2 训练配置")
    print("=" * 60)
    print(f"训练文件: {args.train_file}")
    print(f"验证文件: {args.val_file}")
    print(f"底座模型: {args.config_file}")
    print(f"层融合: {model_task2.LAYER_FUSION} (--layer_fusion, 不传=last)")
    print(f"inner_dim: {model_task2.INNER_DIM} (--inner_dim, 不传=64)")
    print(f"正样本损失权重 pos_weight: {model_task2.POS_WEIGHT} (1.0=关; GlobalPointer 正项整体乘该系数, 不等于 BCE 的 pos_weight)")
    print(f"目标词感知: 启用 (dataset_task2.py 中 target 位置编码)")
    print(f"目标 span 扩容增广: --expand_targets {args.expand_targets} "
          f"({{0: '关闭，训练标签逐字节等于现网', 1: '开启（±1 字加副本，11-01 实测判负）', "
          f"2: '开启（合并写法替换：相邻/嵌套的金标 span 并成一条更宽的）', "
          f"3: '开启（合并写法加副本：原 span 保留，另并出合并后的）'}}[args.expand_targets]); "
          f"只动训练集，验证集永不扩容")
    print(f"FGM 对抗训练: {'禁用' if args.no_fgm else '启用'} (--no_fgm 关闭; 开启时每步在 embedding 上做攻击并累加梯度)")
    print(f"NoisyTune 扰动幅度: {args.noise_lambda} (--noise_lambda; 不传=0.15, 就是原来写死的值; 0=不加噪)")
    print(f"EMA 权重平均: {'启用' if args.with_ema else '禁用'}")
    if args.with_ema:
        print(f"  EMA decay: {args.ema_decay}")
    print(f"训练轮数: {args.num_train_epochs} ({'命令行传入' if epochs_from_cli else '未传参, 用任务二默认的 5'})")
    print(f"解码阈值(选权重用的验证集判据): {args.logit_threshold}")
    print(f"Batch size: {args.batch_size}")
    print(f"梯度累积: {args.accumulate_gradients}")
    print(f"学习率: {args.lr}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    train_dataset = Dataset(args.train_file,
                            "./dataset/frame_info.json",
                            tokenizer,
                            expand_targets=args.expand_targets)
    dev_dataset = Dataset(args.val_file,
                          "./dataset/frame_info.json",
                          tokenizer)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = 1
    config.max_cls = train_dataset.max_cls
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(args.init_checkpoint, map_location='cpu')
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
