#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把三资源"部署口径"解码链的读数收成一张表（读日志，不动任何产物）。

一份 decode_ladder_<资源>.log 里交替出现：
    === 档位 <权重名>  <时间>  <这一档吃了哪些上游输出> ===
    [读数] 框架名称：命中 h / n 行 = F
    task1 框架识别 FI acc : ...
    task2 论元范围 AI f1  : ...  (p ... / r ...)
    task3 论元角色 RI f1  : ...  (p ... / r ...)
    综合分 0.3*FI+0.3*AI+0.4*RI : ...      （只有赛题这一格有）
本脚本按档位切片，输出六列：资源／槽位／口径／读数／分母／这一格在量什么。
"""
import os
import re
import sys

W = "/work/workspace/ccl25"
RES = [("赛题 2.2", "c22", "R32 裸基线（无部署匹配负例、框架名不进输入、训练期先验关）"),
       ("CFN 1.0", "c10", "四档 = 裸基线 / 加部署匹配负例 / 负例加框架名进输入 / 负例加训练期框架条件先验"),
       ("英文 1.7", "en", "四档 = 同上，只差底座是 bge-large-en-v1.5")]

RE_ARM = re.compile(r"^=== 档位 (\S+)\s+(.*?)\s*$")
RE_FI = re.compile(r"框架名称：命中 (\d+) / (\d+) 行 = ([0-9.]+)")
RE_ACC = re.compile(r"task1 框架识别 FI acc :\s*([0-9.]+)")
RE_AI = re.compile(r"task2 论元范围 AI f1  :\s*([0-9.]+)\s*\(p ([0-9.]+) / r ([0-9.]+)\)")
RE_RI = re.compile(r"task3 论元角色 RI f1  :\s*([0-9.]+)\s*\(p ([0-9.]+) / r ([0-9.]+)\)")
RE_COMB = re.compile(r"综合分 0\.3\*FI\+0\.3\*AI\+0\.4\*RI :\s*([0-9.]+)")
RE_DROP = re.compile(r"丢弃 (\d+) 个")

nfound = 0
for name, code, what in RES:
    path = os.path.join({"c22": W + "/baseline", "c10": W + "/baseline_c10", "en": W + "/baseline_en"}[code],
                        "logfile", "decode_ladder_%s.log" % code)
    print("=== %s  日志 %s" % (name, path))
    if not os.path.exists(path):
        print("    [未跑] 解码链还没执行过（任务一权重未落盘时脚本会在自检处中断）")
        print()
        continue
    with open(path, encoding="utf8", errors="replace") as f:
        lines = [x.rstrip("\n") for x in f]
    arm = None
    rows = []
    for ln in lines:
        m = RE_ARM.match(ln)
        if m:
            arm = {"arm": m.group(1), "fi": None, "ai": None, "ri": None, "comb": None,
                   "den": None, "hit": None, "abstain": 0, "dropped": 0}
            rows.append(arm)
            continue
        if arm is None:
            continue
        m = RE_FI.search(ln)
        if m:
            arm["hit"], arm["den"], arm["fi"] = int(m.group(1)), int(m.group(2)), float(m.group(3))
        for rx, key in ((RE_ACC, "fi"), (RE_AI, "ai"), (RE_RI, "ri"), (RE_COMB, "comb")):
            m = rx.search(ln)
            if m and arm.get(key) is None:
                arm[key] = float(m.group(1))
        m = RE_DROP.search(ln)
        if m:
            arm["dropped"] += int(m.group(1))
    if not rows:
        print("    [无档位] 日志里没有档位标记")
        print()
        continue
    print("    | 档位（任务三权重） | 框架名称准确率 FI | 跨度 F1 AI | 角色 F1 RI | 综合分 | 分母 |")
    print("    |---|---|---|---|---|---|")
    for r in rows:
        nfound += 1
        cell = lambda v: ("%.4f" % v) if v is not None else "缺"
        print("    | %s | %s | %s | %s | %s | %s |" % (r["arm"], cell(r["fi"]), cell(r["ai"]),
                                                       cell(r["ri"]), cell(r["comb"]),
                                                       ("%d 条记录" % r["den"]) if r["den"] else "未记"))
    print("    [说明] " + what)
    print("    [说明] 三档口径相同之处：跨度由该资源自己的任务二发（及格线 -0.5）、"
          "框架名称由该资源自己的任务一判、角色打分的 argmax 落在空类就不交该行。")
    print()

print("[合计] %d 个档位有读数；没跑齐的资源先补任务一训练（chain_t1_c10_1106.sh / chain_t1_en_1106.sh）。" % nfound)
