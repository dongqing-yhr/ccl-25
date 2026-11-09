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
from dataset_task3 import Dataset, DevCandidateDataset, get_cand_input
from params import args
from model_task3 import Model
import model_task3
from ema import seed_everything, EMA

# NULL 机制配置
args.with_null = True  # 启用 NULL 类
args.null_weight = 0.5  # NULL 损失权重


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
    frame_idx = []
    gold_frame_idx = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])
        sentence_id.append(d[4])
        frame_idx.append(d[5])
        gold_frame_idx.append(d[6] if len(d) > 6 else -1)

    input_ids = np.array(input_ids_list, dtype=np.compat.long)
    attention_mask = np.array(attention_mask_list, dtype=np.compat.long)
    labels = np.array(labels, dtype=np.compat.long)
    frame_idx = np.array(frame_idx, dtype=np.compat.long)
    gold_frame_idx = np.array(gold_frame_idx, dtype=np.compat.long)

    input_ids = torch.from_numpy(input_ids)
    attention_mask = torch.from_numpy(attention_mask)
    labels = torch.from_numpy(labels)
    frame_idx = torch.from_numpy(frame_idx)
    gold_frame_idx = torch.from_numpy(gold_frame_idx)

    return input_ids, attention_mask, target, labels, sentence_id, frame_idx, gold_frame_idx


def eval(model, val_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    # 帧头(dev)按句统计: 同句多个金标跨度只算第一条, 这样这个数字能和 task1 的句级 FI 对照
    frame_first = {}
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, labels, sentence_id, frame_idx, gold_frame_idx = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels,
                           device=device, for_test=True)
            logits = output["logits"]
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            correct += (pred == labels).sum().item()
            total += len(pred)
            fl = output.get("frame_logits")
            if fl is not None:
                pf = torch.argmax(fl, dim=-1).cpu().tolist()
                gf = gold_frame_idx.cpu().tolist()
                for sid, p, g in zip(sentence_id, pf, gf):
                    if g >= 0 and sid not in frame_first:
                        frame_first[sid] = (p == g)

    if frame_first:
        n_hit = sum(1 for v in frame_first.values() if v)
        print('  [frame_head] dev 帧 top-1 = %.4f  (%d/%d 句, 同句多跨度只计第一条; '
              '参照 task1 eq4 在 dev 的 top-1 命中 0.7529)'
              % (n_hit / len(frame_first), n_hit, len(frame_first)))

    return correct / (total + 1e-6)


def eval_ri(model, cand_loader, cand_dataset):
    """Score one checkpoint the way the leaderboard scores task3.

    Only the spans the range model would actually deliver are decoded here, so a false positive and
    an over-abstention both cost the same as they do on the board. Returns (RI, precision, recall,
    n_emitted, n_correct).
    """
    model.eval()
    null_idx = cand_dataset.null_idx
    idx2label = cand_dataset.idx2label
    per = {}
    n_rows = 0
    with torch.no_grad():
        for batch in tqdm(cand_loader, desc='eval_ri'):
            input_ids, attention_mask, target, labels, sentence_id, rows = batch
            output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                           target=target, labels=None, device=device, for_test=True)
            pred = torch.argmax(F.softmax(output['logits'], dim=-1), dim=-1).cpu().tolist()
            n_rows += len(pred)
            for p, sid, row in zip(pred, sentence_id, rows):
                if p == null_idx:
                    continue
                per.setdefault(sid, set()).add((row[5], row[6], idx2label[p]))
    inter = n_pred = 0
    for sid, gg in cand_dataset.gold.items():
        pp = per.get(sid, set())
        inter += len(pp & gg)
        n_pred += len(pp)
    if n_pred == 0:
        return 0.0, 0.0, 0.0, 0, 0
    p = inter / n_pred
    r = inter / max(cand_dataset.n_gold, 1)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1 * 100.0, p * 100.0, r * 100.0, n_pred, inter



def train(model, train_loader, val_loader, cand_loader=None):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # *******************  NoisyTune  ****************
    noise_lambda = 0.15
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
    best_ri = -1.0
    fgm = FGM(model)
    ema = EMA(model, decay=args.ema_decay) if args.with_ema else None
    for i_epoch in range(1, 1 + args.num_train_epochs):
        total_loss = 0.0
        total_frame_loss = 0.0
        n_frame_steps = 0
        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch_{i_epoch} ')
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids, attention_mask, target, labels, sentence_id, frame_idx, gold_frame_idx = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            frame_idx = frame_idx.to(device)
            gold_frame_idx = gold_frame_idx.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels,
                           device=device, frame_idx=frame_idx, gold_frame_idx=gold_frame_idx,
                           sentence_id=sentence_id)

            loss = output['loss']

            total_loss += loss.item()
            fl_item = output.get('frame_loss')
            if fl_item is not None:
                total_frame_loss += fl_item.item()
                n_frame_steps += 1

            if (step + 1) % 1000 == 0:
                print(
                    f'loss: {total_loss / (step + 1)}')
                if n_frame_steps:
                    fl_mean = total_frame_loss / n_frame_steps
                    print('  [mtl] 上面这个 loss = 角色项 + lambda*帧项; 帧交叉熵(未乘 lambda, 按句去重后) '
                          '%d 步均值 = %.4f, lambda=%.2f, 帧项占总 loss %.1f%%'
                          % (n_frame_steps, fl_mean, model.mtl_frame_weight,
                             100.0 * model.mtl_frame_weight * fl_mean / max(total_loss / (step + 1), 1e-6)))
                    total_frame_loss, n_frame_steps = 0.0, 0

            loss.backward()

            # fgm.attack()  # embedding被修改了
            # # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
            # loss_sum = \
            # model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device)[
            #     'loss']
            # loss_sum.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore()  # 恢复Embedding的参数

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

        if cand_loader is None:
            # 旧口径: 只按金标跨度的角色准确率选权重
            if acc > best_acc:
                print(f'saved! new best acc {acc}, ori_acc {best_acc}')
                best_acc = acc
                model_to_save = model.module if hasattr(model, 'module') else model
                os.makedirs('saves', exist_ok=True)
                torch.save(model_to_save.state_dict(), f'saves/{args.model_prefix}_best.bin')
            else:
                print(f'current acc: {acc}')
        else:
            # 部署口径: 按 task2 真正交上去的那批跨度的 micro-F1 选权重, 弃权过多和乱发同样都扣分
            ri, prec, rec, n_emit, n_tp = eval_ri(model, cand_loader, cand_loader.dataset)
            print(f'epoch {i_epoch}: acc {acc:.4f} | dev-RI {ri:.4f} '
                  f'(p {prec:.4f} / r {rec:.4f}, emitted {n_emit} of {len(cand_loader.dataset)}, '
                  f'correct {n_tp} of {cand_loader.dataset.n_gold} gold)')
            if acc > best_acc:
                best_acc = acc
                model_to_save = model.module if hasattr(model, 'module') else model
                os.makedirs('saves', exist_ok=True)
                torch.save(model_to_save.state_dict(), f'saves/{args.model_prefix}_bestacc.bin')
                print(f'  saved accuracy-best -> saves/{args.model_prefix}_bestacc.bin')
            if ri > best_ri:
                print(f'saved! new best dev-RI {ri:.4f}, ori_ri {best_ri:.4f}')
                best_ri = ri
                model_to_save = model.module if hasattr(model, 'module') else model
                os.makedirs('saves', exist_ok=True)
                torch.save(model_to_save.state_dict(), f'saves/{args.model_prefix}_best.bin')

        if ema is not None:
            ema.apply()
            acc_ema = eval(model, val_loader)
            ema.restore()
            if cand_loader is None:
                score_ema, tag_ema = acc_ema, 'acc'
                print(f'epoch {i_epoch}: raw_acc {acc}, ema_acc {acc_ema}')
            else:
                score_ema = eval_ri(model, cand_loader, cand_loader.dataset)[0]
                tag_ema = 'dev-RI'
                print(f'epoch {i_epoch}: ema acc {acc_ema:.4f} | ema dev-RI {score_ema:.4f}')
            if score_ema > best_ema:
                best_ema = score_ema
                os.makedirs('saves', exist_ok=True)
                ema.save(f'saves/{args.model_prefix}_ema.bin')
                print(f'  saved EMA! new best ema_{tag_ema} {score_ema}, ori_ema {best_ema}')

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
    # 自动检测 GPU/CPU（容器已配置 CUDA_VISIBLE_DEVICES）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存总量：{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"   可用 GPU 数量：{torch.cuda.device_count()}")
    else:
        print("❌ 错误：CUDA 不可用！")
        print("   请检查：")
        print("   1. 容器是否正确启动（--gpus all）")
        print("   2. 卡 4 是否可用")
        exit(1)
    seed_everything(args.seed)

    # 写进 model_task3 的模块级结构开关(不传参就保持现役: 最后四层平均 + inner_dim 64)
    model_task3.LAYER_FUSION = args.layer_fusion or 'avg4'
    model_task3.INNER_DIM = args.inner_dim or 64
    model_task3.NEG_BOOST = args.neg_boost

    # 打印训练配置
    print("=" * 60)
    print("Task3 训练配置")
    print("=" * 60)
    print(f"底座模型: {args.config_file}")
    print(f"层融合: {model_task3.LAYER_FUSION} (--layer_fusion, 不传=avg4)")
    print(f"inner_dim: {model_task3.INNER_DIM} (--inner_dim, 不传=64)")
    print(f"NULL 弃权机制: 启用 (代码顶部固定配置, null_weight={args.null_weight})")
    print(f"负样本倍率 neg_boost: {model_task3.NEG_BOOST} (--neg_boost, 不传=1.0 即关; "
          f"只把标签为 NULL 的跨度的交叉熵乘 {model_task3.NEG_BOOST}, 正样本项与分母不动; "
          f"类比对方 neg_boost 3/5 但不等价)")
    print(f"FGM 对抗训练: 禁用 (task3 训练循环里 attack/restore 是注释状态)")
    print(f"EMA 权重平均: {'启用' if args.with_ema else '禁用'}")
    if args.with_ema:
        print(f"  EMA decay: {args.ema_decay}")
    print(f"训练轮数: {args.num_train_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"梯度累积: {args.accumulate_gradients}")
    print(f"学习率: {args.lr}")
    print(f"Seed: {args.seed}")
    print(f"Logit adjustment: tau={args.la_tau} mode={args.la_mode} alpha={args.la_alpha} (tau=0 表示关闭)")
    print(f"框架名注入输入: {'启用' if args.with_frame_name else '禁用'}")
    print(f"注入帧数 frame_topk: {args.frame_topk} (1=只 top-1, 与老权重逐字节一致; 2=top-1+top-2 双帧)")
    print(f"OOF 帧文件: {args.oof_frame_file or '未使用 (训练吃金标帧)'}")
    print(f"部署匹配负例: {args.deploy_negatives_file or '未使用'}")
    print("是不是论元辅助损失: %s" % (('权重 %.2f' % args.arghood_weight) if args.arghood_weight else '关闭'))
    print("共享编码器帧头(L2): %s" % (('lambda=%.2f, 帧损失与角色损失同编码器反传' % args.mtl_frame_weight)
                                if args.mtl_frame_weight > 0 else '关闭 (lambda=0, 不创建帧头, 与现役配方逐字节一致)'))
    if args.dev_cand_file:
        print("选权重口径: 部署候选集 dev-RI  候选文件=%s  帧来源=%s"
              % (args.dev_cand_file, args.dev_frame_file or '金标帧'))
    else:
        print("选权重口径: 金标跨度准确率 (旧口径), 不加载候选集")
    print("=" * 60)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    train_dataset = Dataset(args.train_file,
                            "./dataset/frame_info.json",
                            tokenizer,
                            with_frame_name=args.with_frame_name,
                            oof_frame_file=args.oof_frame_file,
                            deploy_negatives_file=args.deploy_negatives_file,
                            frame_topk=args.frame_topk)
    dev_dataset = Dataset(args.val_file,
                          "./dataset/frame_info.json",
                          tokenizer,
                          with_frame_name=args.with_frame_name,
                          oof_frame_file=args.oof_frame_file,
                          frame_topk=args.frame_topk)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = train_dataset.num_labels
    config.arghood_weight = args.arghood_weight
    # NULL 机制：增加 NULL 类
    if args.with_null:
        config.num_labels += 1
        print(f"✅ 启用 NULL 机制，总类别数：{config.num_labels}（含 NULL 类）")
        print(f"   NULL 损失权重：{args.null_weight}")
        print(f"   训练样本数：{len(train_dataset.data)}")
        print(f"   验证样本数：{len(dev_dataset.data)}")
    # ---- L2 共享编码器帧头: 只有 lambda>0 才把 num_frames 写进 config ----
    # lambda=0 时帧头连创建都不创建, 参数名/形状与现役五成员逐字节一致, 老权重能直接 load。
    if args.mtl_frame_weight > 0:
        config.num_frames = len(train_dataset.frame_vocab)
        config.mtl_frame_weight = args.mtl_frame_weight
        print("✅ 帧头就绪: %d 类固定帧表 (frame_info.json), lambda=%.2f, 池化区=目标词跨距(含两端标记)"
              % (config.num_frames, args.mtl_frame_weight))
    else:
        config.num_frames = 0
        config.mtl_frame_weight = 0.0

    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        state = torch.load(args.init_checkpoint, map_location='cpu')
        msg = model.load_state_dict(state, strict=False)
        print(f"✅ 加载预训练权重：{args.init_checkpoint}")
        print(f"   加载消息：{msg}")
    else:
        print(f"⚠️  未加载预训练权重，从头开始训练")
    model = model.to(device)

    # ---- logit adjustment 先验矩阵 (训练期把 tau*log(prior) 加进 loss, 推理不加) ----
    if args.la_tau > 0:
        NF = len(train_dataset.frame2idx)
        NL = config.num_labels
        cnt = np.zeros((NF, NL))
        for d in train_dataset.data:
            f = train_dataset.frame2idx.get(d.get("frame"), -1)
            c = d["label_class"]
            if 0 <= f < NF and 0 <= c < NL:
                cnt[f, c] += 1
        tot = cnt.sum()
        glob = cnt.sum(0) / max(tot, 1.0)                       # 全局角色频率
        if args.la_mode == 'global':
            p = np.tile(glob, (NF, 1))
        else:                                                    # frame-conditional + 平滑
            row = cnt.sum(1, keepdims=True)
            p = (cnt + args.la_alpha * glob) / (row + args.la_alpha)
        logpi = np.log(p + 1e-12)
        if args.la_clamp > 0:
            logpi = np.clip(logpi, -args.la_clamp, 0.0)          # 封顶调整幅度, 防 -27 级深坑
        if args.with_null:
            logpi[:, NL - 1] = 0.0                              # NULL 列不调, 弃权行为不变
        model.register_buffer('logit_prior', torch.tensor(logpi, dtype=torch.float, device=device))
        model.la_tau = args.la_tau
        print("[logit_adj] mode={} tau={} alpha={} clamp={} 帧数={} 类别数={}".format(
            args.la_mode, args.la_tau, args.la_alpha, args.la_clamp, NF, NL))
        print("[logit_adj] 角色先验范围 log pi ∈ [{:.2f}, {:.2f}] (NULL 列已置 0)".format(
            logpi[:, :NL - 1].min(), logpi[:, :NL - 1].max()))
    else:
        model.la_tau = 0.0
    # ---------------------------------------------------------------------------

    train_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=train_dataset,
        shuffle=True,
        num_workers=0,  # 改为 0 避免多进程阻塞
        collate_fn=partial(get_model_input, device=device),
        drop_last=True
    )

    val_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=dev_dataset,
        shuffle=False,
        num_workers=0,  # 改为 0 避免多进程阻塞
        collate_fn=partial(get_model_input, device=device),
        drop_last=False
    )
    cand_loader = None
    if args.dev_cand_file:
        cand_dataset = DevCandidateDataset(args.dev_cand_file,
                                           args.val_file,
                                           "./dataset/frame_info.json",
                                           tokenizer,
                                           with_frame_name=args.with_frame_name,
                                           frame_file=(args.dev_frame_file or None),
                                           frame_topk=args.frame_topk)
        cand_loader = DataLoader(
            batch_size=args.batch_size,
            dataset=cand_dataset,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(get_cand_input, device=device),
            drop_last=False
        )
        print("[train_task3] 候选集 loader 就绪: %d 批 (batch_size=%d)" % (len(cand_loader), args.batch_size))

    train(model, train_loader, val_loader, cand_loader)
