#!/usr/bin/env python3
# 三套资源上"解码期频次先验能不能外推"的适用性量（11-06，零训练、纯 CPU）。
# 两个读数：
#   一、框架名称分布的漂移量：以训练标注里各框架名称的比例为期望，对验证标注的计数做拟合优度卡方，
#       再按自由度摊（自由度取"训练里出现过的框架名称个数减一"）。摊出来的值接近 1 说明验证集与训练集同分布，
#       远大于 1 说明先验表要用的目标分布已经漂了。
#   二、只用表能拿多少分：用训练标注建"目标词到最常框架名称"的查表，在验证标注上数覆盖率与命中率。
#       这两个数就是解码期先验 γ 与训练期先验 LA 的天花板参照（表本身不带任何模型打分）。
import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESOURCES = [
    ("赛题 CFN 2.2", os.path.join(ROOT, "dataset", "cfn-train.json"),
     os.path.join(ROOT, "dataset", "cfn-dev.json")),
    ("CFN 1.0", os.path.join(ROOT, "datasets_paper", "cfn1.0_task3", "task3_train.json"),
     os.path.join(ROOT, "datasets_paper", "cfn1.0_task3", "task3_dev.json")),
    ("英文 FrameNet 1.7", os.path.join(ROOT, "datasets_paper", "framenet_task3", "task3_train.json"),
     os.path.join(ROOT, "datasets_paper", "framenet_task3", "task3_dev.json")),
]


def target_word(rec):
    w = rec.get("word")
    if isinstance(w, str) and w:
        return w
    tgt = rec.get("target")
    if isinstance(tgt, list) and tgt:
        last = tgt[-1]
        return rec["text"][last["start"]:last["end"]]
    return ""


def rows_with_frame(records):
    return [r for r in records if isinstance(r.get("frame"), str) and r["frame"].strip()]


def chi2_per_dof(train_frames, dev_frames):
    """dev 的框架名称计数 对 train 比例的拟合优度；返回 (卡方, 自由度, 摊每自由度, 训练未见框架名的行数)."""
    n_tr = len(train_frames)
    n_dev = len(dev_frames)
    p = Counter(train_frames)
    obs = Counter(dev_frames)
    chi = 0.0
    dof_cells = 0
    for frame, prob in p.items():
        expect = prob / n_tr * n_dev
        if expect <= 0:
            continue
        chi += (obs.get(frame, 0) - expect) ** 2 / expect
        dof_cells += 1
    unseen_rows = sum(c for f, c in obs.items() if f not in p)
    dof = max(dof_cells - 1, 1)
    return chi, dof, chi / dof, unseen_rows, n_dev


print("资源                    训练行  验证行  框架名称数(训/验/交)  卡方  自由度  卡方/自由度  训练里没见过的框架名占验证行")
print("-" * 118)
for name, tr_path, dev_path in RESOURCES:
    tr = json.load(open(tr_path, encoding="utf8"))
    dev = json.load(open(dev_path, encoding="utf8"))
    tr_f, dev_f = rows_with_frame(tr), rows_with_frame(dev)
    chi, dof, ratio, unseen, n_dev = chi2_per_dof([r["frame"] for r in tr_f], [r["frame"] for r in dev_f])
    s_tr, s_dev, s_int = set(r["frame"] for r in tr_f), set(r["frame"] for r in dev_f), set(r["frame"] for r in tr_f) & set(r["frame"] for r in dev_f)
    print("%-22s %6d %7d %10d/%-5d/%-5d %9.1f %7d %10.3f %12d / %d = %5.2f%%"
          % (name, len(tr_f), n_dev, len(s_tr), len(s_dev), len(s_int), chi, dof, ratio, unseen, n_dev, 100.0 * unseen / max(n_dev, 1)))

print()
print("资源                    查表覆盖的验证行  覆盖率    查表命中金标框架名称  命中率(覆盖内)  命中率(全部验证行)")
print("-" * 118)
for name, tr_path, dev_path in RESOURCES:
    tr = json.load(open(tr_path, encoding="utf8"))
    dev = json.load(open(dev_path, encoding="utf8"))
    tr_f, dev_f = rows_with_frame(tr), rows_with_frame(dev)
    best = {}
    pair = defaultdict(Counter)
    for r in tr_f:
        pair[target_word(r)][r["frame"]] += 1
    for w, c in pair.items():
        best[w] = c.most_common(1)[0][0]
    cov = hit = 0
    for r in dev_f:
        w = target_word(r)
        if w in best:
            cov += 1
            if best[w] == r["frame"]:
                hit += 1
    n_dev = len(dev_f)
    print("%-22s %12d / %-6d %7.2f%% %12d / %-6d %12.2f%% %14.2f%%"
          % (name, cov, n_dev, 100.0 * cov / max(n_dev, 1), hit, cov,
             100.0 * hit / max(cov, 1), 100.0 * hit / max(n_dev, 1)))
