#!/usr/bin/python3
"""第一梯队降方差工具: 固定随机种子 + PostHoc EMA(权重滑动平均)。
三条 train 脚本共用, 通过 params 里的 --seed / --with_ema / --ema_decay 开关。
"""
import random
import numpy as np
import torch


def seed_everything(seed):
    # seed < 0 表示不固定(保持 baseline 原本的随机行为)
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class EMA:
    """对 requires_grad 的参数维护指数滑动平均。
    训练结束时把 shadow 权重单独存成一个 *_ema.bin, 推理时用 --model_path 指过去对比。
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone()

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        p.data.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[name] = p.data.detach().clone()

    @torch.no_grad()
    def apply(self):
        # 把 shadow 换进模型用于评估, 原始权重先备份
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.detach().clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        # 以当前模型完整 state_dict 为底, 覆盖成 shadow 参数, 保证能被 strict=False 正常加载
        full = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        for name, v in self.shadow.items():
            full[name] = v.detach().clone()
        return full

    def save(self, path):
        torch.save(self.state_dict(), path)
