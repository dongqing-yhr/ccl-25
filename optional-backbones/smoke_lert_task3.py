#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""任务三 LERT 底座冒烟（在容器 baseline 目录跑）：不过就退出码 1，不发十小时的训练。

  闸1 分词对齐：训练器按字符列表分词、目标词位置按"第几个字 + 1（[CLS]）"索引，
      所以必须一个字符恰好一个 token。抽 50 句训练集句子逐句核。
  闸2 装配：LERT 原始 pytorch_model.bin 装进 model_task3.Model（num_labels=1010，含 NULL 列），
      编码器可学习参数一个都不许缺，missing 只允许 dense.* / lstm.* 这两个头，形状不一致 0 个。
  闸3 前向：GPU 上按生产配方（层融合 avg4、inner_dim 64）跑两个批量，
      logits 形状 = (批量, 1010)、数值有限。
  闸4 反向 + LA：注册 logit_prior 缓冲、la_tau=0.5，走带先验的损失一次 backward，
      loss 有限、角色头 dense 有非零梯度。这一闸专门覆盖 la 臂相对 fn 臂唯一多出来的那段代码。

用法：python3 smoke_lert_task3.py [卡号]   卡号写进 CUDA_VISIBLE_DEVICES，默认 0
"""
import json
import os
import sys

BASE = "/work/workspace/ccl25/baseline"
BERT = os.path.join(BASE, "models", "hfl--LERT-large")
GPU = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
sys.path.insert(0, BASE)

import numpy as np
import torch
from transformers import BertConfig, BertTokenizer


def log(m):
    print(m, flush=True)


def die(m):
    log("[FATAL] " + m)
    sys.exit(1)


fi = json.load(open(os.path.join(BASE, "dataset", "frame_info.json"), encoding="utf8"))
roles = []
for line in fi:
    for fe in line["fes"]:
        if fe["fe_name"] not in roles:
            roles.append(fe["fe_name"])
frames = []
for line in fi:
    if line["frame_name"] not in frames:
        frames.append(line["frame_name"])
NL = len(roles) + 1
log("[cfg] 角色 %d + NULL = %d 类   帧 %d 类   卡号 CUDA_VISIBLE_DEVICES=%s" % (len(roles), NL, len(frames), GPU))

tok = BertTokenizer(vocab_file=os.path.join(BERT, "vocab.txt"), do_lower_case=True)
rows = json.load(open(os.path.join(BASE, "dataset", "cfn-train.json"), encoding="utf8"))
bad = 0
for d in rows[:50]:
    ids = tok.encode_plus(list(d["text"]))["input_ids"]
    if len(ids) != len(d["text"]) + 2:
        bad += 1
        if bad <= 3:
            log("  闸1 反例 sid=%s：%d 字 -> %d 个 token" % (d.get("sentence_id"), len(d["text"]), len(ids)))
if bad:
    die("闸1 没过：50 句里 %d 句不是一个字符一个 token，目标词位置索引会错位" % bad)
log("[闸1   ] 50/50 句一个字符一个 token（含 [CLS]/[SEP] 各 1 个）")

import model_task3
model_task3.LAYER_FUSION = "avg4"
model_task3.INNER_DIM = 64
config = BertConfig.from_json_file(os.path.join(BERT, "config.json"))
config.num_labels = NL
model = model_task3.Model(config)
state = torch.load(os.path.join(BERT, "pytorch_model.bin"), map_location="cpu")
msg = model.load_state_dict(state, strict=False)
bert_params = dict(model.bert.named_parameters())
have = set(k[len("bert."):] for k in state if k.startswith("bert."))
pm = sum(1 for k in bert_params if k in have)
head_missing = [k for k in msg.missing_keys if k.startswith(("dense.", "lstm.", "dropout"))]
other_missing = [k for k in msg.missing_keys if not k.startswith(("dense.", "lstm.", "dropout"))]
mismatch = [(k, tuple(v.shape), tuple(model.state_dict()[k].shape))
            for k, v in state.items()
            if k in model.state_dict() and tuple(model.state_dict()[k].shape) != tuple(v.shape)]
log("[闸2   ] 编码器覆盖率 %d/%d = %.2f%%  形状不一致 %d 个  missing 头 %d 个  其他 missing %d 个  unexpected %d 个"
    % (pm, len(bert_params), 100.0 * pm / len(bert_params), len(mismatch),
       len(head_missing), len(other_missing), len(msg.unexpected_keys)))
for k, a, b in mismatch[:10]:
    log("          %s ckpt%s vs model%s" % (k, a, b))
if other_missing:
    log("          其他缺失键（前 10）：%s" % other_missing[:10])
if pm != len(bert_params) or mismatch or other_missing:
    die("闸2 没过：编码器没被 LERT 原始权重填满，等于随机初始化训练")

if not torch.cuda.is_available():
    die("闸3 没过：容器里 CUDA 不可用（卡号 %s）" % GPU)
dev = torch.device("cuda")
model = model.to(dev)
log("[闸3   ] 上卡完成，显存已分配 %.2f GB" % (torch.cuda.memory_allocated(dev) / 1e9))

rnd = np.random.RandomState(42)
for bs, L in ((2, 24), (8, 64)):
    ids = torch.from_numpy(rnd.randint(3, 21000, size=(bs, L)).astype("int64")).to(dev)
    am = torch.ones_like(ids)
    t0 = torch.from_numpy(rnd.randint(2, L - 6, size=(bs,)).astype("int64"))
    tgt = [[int(a), int(a) + 2] for a in t0.numpy()]
    lab = torch.from_numpy(rnd.randint(0, NL, size=(bs,)).astype("int64")).to(dev)
    fidx = torch.from_numpy(rnd.randint(0, len(frames), size=(bs,)).astype("int64")).to(dev)
    out = model(input_ids=ids, attention_mask=am, target=tgt, labels=lab,
                device=dev, frame_idx=fidx, gold_frame_idx=fidx,
                sentence_id=list(range(bs)))
    lg = out["logits"]
    loss = out["loss"]
    if tuple(lg.shape) != (bs, NL):
        die("闸3 没过：logits 形状 %s，应为 (%d, %d)" % (tuple(lg.shape), bs, NL))
    if not torch.isfinite(lg).all() or not torch.isfinite(loss):
        die("闸3 没过：logits 或 loss 出现 NaN/Inf")
    log("          bs=%d L=%d  logits%s 有限  loss=%.4f" % (bs, L, tuple(lg.shape), loss.item()))

# ---- 闸4：LA 先验路径（la 臂相对 fn 臂唯一多出来的代码）----
cnt = np.zeros((len(frames), NL))
cnt[:, :NL - 1] = rnd.rand(len(frames), NL - 1) * 50 + 1
tot = cnt.sum()
glob = cnt.sum(0) / max(tot, 1.0)
row = cnt.sum(1, keepdims=True)
p = (cnt + 1.0 * glob) / (row + 1.0)
logpi = np.clip(np.log(p + 1e-12), -2.0, 0.0)
logpi[:, NL - 1] = 0.0
model.register_buffer("logit_prior", torch.tensor(logpi, dtype=torch.float, device=dev))
model.la_tau = 0.5

bs, L = 8, 64
ids = torch.from_numpy(rnd.randint(3, 21000, size=(bs, L)).astype("int64")).to(dev)
am = torch.ones_like(ids)
t0 = rnd.randint(2, L - 6, size=(bs,))
tgt = [[int(a), int(a) + 2] for a in t0]
lab = torch.from_numpy(rnd.randint(0, NL - 1, size=(bs,)).astype("int64")).to(dev)
fidx = torch.from_numpy(rnd.randint(0, len(frames), size=(bs,)).astype("int64")).to(dev)
model.zero_grad(set_to_none=True)
out = model(input_ids=ids, attention_mask=am, target=tgt, labels=lab,
            device=dev, frame_idx=fidx, gold_frame_idx=fidx, sentence_id=list(range(bs)))
loss_la = out["loss"]
loss_la.backward()
g = model.dense.weight.grad
if g is None or not torch.isfinite(g).all() or float(g.abs().sum()) == 0.0:
    die("闸4 没过：加了 LA 先验后角色头 dense 没拿到有限非零梯度")
log("[闸4   ] LA 先验生效：loss=%.4f 有限，dense 梯度绝对值和=%.4e（非零）"
    % (loss_la.item(), float(g.abs().sum())))

# 同一批数据在 la_tau=0 与 la_tau=0.5 下损失必须不同，否则先验没进损失
model.la_tau = 0.0
model.zero_grad(set_to_none=True)
out0 = model(input_ids=ids, attention_mask=am, target=tgt, labels=lab,
             device=dev, frame_idx=fidx, gold_frame_idx=fidx, sentence_id=list(range(bs)))
d0 = abs(float(out0["loss"]) - float(loss_la))
if not (d0 > 1e-6):
    die("闸4 没过：la_tau=0 与 0.5 的损失一样（差 %.2e），先验根本没作用到损失上" % d0)
log("[闸4b  ] la_tau=0 损失 %.4f 对 0.5 损失 %.4f，差 %.4f（先验确实改了损失）"
    % (float(out0["loss"]), float(loss_la), d0))

log("")
log("[结论] 冒烟通过：LERT 可以按我方任务三配方训练（fn 臂与 la 臂的代码路径都跑过了）")
