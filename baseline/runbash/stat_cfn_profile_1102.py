# -*- coding: utf-8 -*-
# stat_cfn_profile_1102.py — P4 六格数字卡统计（官方发布的 CFN2.2，本地 dataset/）
# 用法: python stat_cfn_profile_1102.py   (在 CCL_v2025/ 下运行)
import json, io, os, sys, datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
D = os.path.join(ROOT, "dataset")

def load(fn):
    with io.open(os.path.join(D, fn), encoding="utf-8") as f:
        return json.load(f)

def log(m):
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), m), flush=True)

fi = load("frame_info.json")
fr = load("frame_roles.json")

labels = []
for line in fi:
    for fes in line["fes"]:
        if fes["fe_name"] not in labels:
            labels.append(fes["fe_name"])

splits = {}
for fn in ["cfn-train.json", "cfn-dev.json", "cfn-test-A.json", "cfn-test-B.json"]:
    splits[fn] = load(fn)

def nested_stats(rows):
    nsent = 0
    tot = inn = out = 0
    for r in rows:
        sp = [(s["start"], s["end"]) for s in r["cfn_spans"]]
        tot += len(sp)
        hit = False
        for i, (a, b) in enumerate(sp):
            for j, (c, d) in enumerate(sp):
                if i != j and (a, b) != (c, d):
                    if a >= c and b <= d:
                        inn += 1; hit = True
                    if c >= a and d <= b:
                        out += 1; hit = True
        nsent += 1 if hit else 0
    return nsent, tot, inn, out

log("卡1 框架名称定义数(frame_info.json): %d" % len(fi))
log("卡2 角色名去重(frame_info.json fes, 任务三标签空间): %d" % len(labels))

tr = splits["cfn-train.json"]
used = set()
for r in tr:
    for s in r["cfn_spans"]:
        used.add(s["fe_name"])
dead = [l for l in labels if l not in used]
log("卡5 死标签(1009 个标签中训练集 10700 句从未出现): %d = %.1f%%" % (dead.__len__(), 100.0 * len(dead) / len(labels)))

tot_rows = sum(len(v) for v in splits.values())
log("卡3 四切分行数: %s = %d" % ("+".join(str(len(v)) for v in splits.values()), tot_rows))

for fn in ["cfn-train.json", "cfn-dev.json"]:
    ns, tt, ii, oo = nested_stats(splits[fn])
    log("卡4 %s 嵌套句(严格包含): %d/%d = %.2f%% | span 总 %d, 内层 %d, 外层 %d, 嵌套相关 %d = %.2f%%"
        % (fn, ns, len(splits[fn]), 100.0 * ns / len(splits[fn]), tt, ii, oo, ii + oo, 100.0 * (ii + oo) / tt))
log("卡4 官方论文口径(含测试集金标): 690 句 / 22000 = 3.14%% (训练 423 / 开发 52 / A 81 / B 134)")

for fn in ["cfn-train.json", "cfn-dev.json"]:
    m = sum(1 for r in splits[fn] if len(r["target"]) > 1)
    log("卡6 %s 多目标词(构式)行: %d = %.2f%%" % (fn, m, 100.0 * m / len(splits[fn])))

fr_pairs = set()
for f, rs in fr.items():
    for x in rs:
        fr_pairs.add((f, x))
used_pairs = set()
for r in tr:
    for s in r["cfn_spans"]:
        used_pairs.add((r["frame"], s["fe_name"]))
log("参考 (框架名称,角色) 组合: 定义 %d, 训练集用到 %d, 未出现 %d" % (len(fr_pairs), len(used_pairs), len(fr_pairs - used_pairs)))
log("参考 角色名去重(frame_roles.json 口径): %d | train+dev 标注实际用到: %d" % (
    len({x for rs in fr.values() for x in rs}),
    len({s["fe_name"] for fn in ["cfn-train.json", "cfn-dev.json"] for r in splits[fn] for s in r["cfn_spans"]})))
print("STAT_CFN_PROFILE_DONE", flush=True)
