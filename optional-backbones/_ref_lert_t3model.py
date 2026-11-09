# -*- coding: utf-8 -*-
"""Task3 模型：target-anchored biaffine 角色分类。

与官方 model_task3.py 数学等价，两点改造（不改变数值）：
  1) 只在 (target[0], target[1]) 两个位置上取 hidden 过 dense，
     避免全序列 L*L*num_labels 的 logits（官方 batch=3 的根因）；
     由于 dense 是逐 token 的、RoPE 只依赖自身位置，结果完全一致；
  2) hidden 融合层数、inner_dim、dropout 参数化；LSTM 按 hidden_size 参数化。

forward(..., full=True) 走官方原路径，用于数值一致性校验。
"""
import math

import torch
import torch.nn as nn
from transformers import BertModel


def fuse_hidden(hs, mode):
    if mode == "last":
        return hs[-1]
    if mode == "avg2":
        return (hs[-1] + hs[-2]) / 2
    if mode == "avg3":
        return (hs[-1] + hs[-2] + hs[-3]) / 3
    if mode == "avg4":
        return (hs[-4] + hs[-3] + hs[-2] + hs[-1]) / 4
    if mode == "last2":  # 最后一层与倒数第二层的加权
        return 0.5 * hs[-1] + 0.5 * hs[-2]
    raise ValueError(mode)


def _rope(x, pos, dim):
    """x: (b, L, dim)；pos: (b,) 绝对位置。与官方 cos/sin 交错实现等价。"""
    half = dim // 2
    idx = torch.arange(0, half, dtype=torch.float32, device=x.device)
    inv = torch.pow(10000.0, -2.0 * idx / dim)
    ang = pos.to(torch.float32).unsqueeze(-1) * inv.unsqueeze(0)  # (b, half)
    cos = torch.cos(ang).repeat_interleave(2, dim=-1).unsqueeze(1)  # (b,1,dim)
    sin = torch.sin(ang).repeat_interleave(2, dim=-1).unsqueeze(1)
    x2 = torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)
    return x * cos + x2 * sin


class Model(nn.Module):
    def __init__(self, config, inner_dim=64, fusion="avg4", dropout=None,
                 use_lstm=False, null_cls=False):
        super(Model, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.inner_dim = inner_dim
        self.fusion = fusion
        self.use_lstm = use_lstm
        self.bert = BertModel(config)
        h = config.hidden_size
        if dropout is None:
            dropout = config.classifier_dropout if getattr(config, "classifier_dropout", None) is not None \
                else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(h, config.num_labels * inner_dim * 2)
        if use_lstm:
            self.lstm = nn.LSTM(h, h // 2, num_layers=1, batch_first=True, bidirectional=True)

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = fuse_hidden(out["hidden_states"], self.fusion)
        if self.use_lstm:
            h, _ = self.lstm(h)
        return h

    def _logits_fast(self, h, pair):
        b = h.size(0)
        dev = h.device
        t0 = torch.as_tensor([t[0] for t in pair], dtype=torch.long, device=dev)
        t1 = torch.as_tensor([t[1] for t in pair], dtype=torch.long, device=dev)
        ar = torch.arange(b, device=dev)
        n = self.num_labels
        d = self.inner_dim
        o0 = self.dense(h[ar, t0]).view(b, n, 2 * d)
        o1 = self.dense(h[ar, t1]).view(b, n, 2 * d)
        qw, _ = o0[..., :d], o0[..., d:]
        _, kw = o1[..., :d], o1[..., d:]
        qw = _rope(qw, t0, d)
        kw = _rope(kw, t1, d)
        return (qw * kw).sum(-1) / math.sqrt(d)

    def _logits_full(self, h, pair):
        """官方原路径（仅用于校验）。"""
        outputs = self.dense(h)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        b, L = h.size(0), h.size(1)
        pos = torch.arange(0, L, dtype=torch.float32, device=h.device).unsqueeze(0).repeat(b, 1)
        half = self.inner_dim // 2
        idx = torch.arange(0, half, dtype=torch.float32, device=h.device)
        inv = torch.pow(10000.0, -2.0 * idx / self.inner_dim)
        ang = pos.unsqueeze(-1) * inv  # (b,L,half)
        pos_emb = torch.stack([torch.sin(ang), torch.cos(ang)], -1).reshape(b, L, self.inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw) / self.inner_dim ** 0.5
        return torch.stack([logits[i][:, pair[i][0], pair[i][1]] for i in range(len(pair))], dim=0)

    def forward(self, input_ids=None, attention_mask=None, pair=None, labels=None,
                device=None, for_test=False, full=False):
        h = self.encode(input_ids, attention_mask)
        logits = self._logits_full(h, pair) if full else self._logits_fast(h, pair)
        loss = None
        if not for_test and labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"logits": logits, "loss": loss}
