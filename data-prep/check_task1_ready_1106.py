#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发车任务一之前的只读自检：确认两套新资源的记录能直接被 dataset_task1 装载。

dataset_task1.py:31-36 每条记录只取四个字段：text（逐字符编码）、target[-1] 的起止、
frame（当类别标签，必须在该资源的 dataset/frame_info.json 的 frame_name 列里）、sentence_id。
所以"任务一格式转换"这件事的真正内容不是转格式，而是核这四条是否条条成立。
任何一条不成立就 exit 1，不允许带着缺口上卡。
"""
import json
import os
import sys

W = "/work/workspace/ccl25"
RES = [
    ("CFN 1.0", W + "/baseline_c10", "../baseline/chinese-macbert-large"),
    ("英文 1.7", W + "/baseline_en", "../baseline/bge-large-en-v1.5"),
]


def load(p):
    with open(p, encoding="utf8") as f:
        return json.load(f)


for name, d, enc in RES:
    print("=== %s  工作目录 %s ===" % (name, d))
    info = load(os.path.join(d, "dataset/frame_info.json"))
    frames = set(x["frame_name"] for x in info)
    for split in ("task3_train.json", "task3_dev.json"):
        data = load(os.path.join(d, "dataset", split))
        n = len(data)
        no_target = sum(1 for r in data if not r.get("target"))
        multi_target = sum(1 for r in data if len(r.get("target", [])) > 1)
        no_frame = sum(1 for r in data if not r.get("frame"))
        off_table = sum(1 for r in data if r.get("frame") and r["frame"] not in frames)
        no_sid = sum(1 for r in data if "sentence_id" not in r)
        maxlen = max(len(r["text"]) for r in data)
        # 任务一不拼框架名，只 CLS + 句尾 SEP + 目标词两侧标记 = 4 个额外位置
        over = sum(1 for r in data if len(r["text"]) + 4 > 512)
        dup_sid = n - len(set(r["sentence_id"] for r in data))
        print("[自检] %-16s 记录 %5d  唯一 sentence_id %5d（同句多目标词 %d 条）"
              % (split, n, len(set(r["sentence_id"] for r in data)), dup_sid))
        print("        缺目标词 %d / 一句多目标词 %d / 缺框架名称 %d / 框架名称不在类别表里 %d / 缺 sid %d"
              % (no_target, multi_target, no_frame, off_table, no_sid))
        print("        最长原文 %d 字符，加 4 个固定词元后越出 512 位置的记录 %d 条" % (maxlen, over))
        assert off_table == 0, "有框架名称不在该资源类别表里，任务一会 KeyError"
        assert no_frame == 0 and no_target == 0 and no_sid == 0, "有记录缺任务一必需字段"
        assert over == 0, "有记录编码后会越出位置表"
    for f in ("config.json", "vocab.txt"):
        p = os.path.join(W, "baseline", enc.split("/")[-1], f)
        assert os.path.exists(p), "缺底座文件 " + p
    ini = os.path.join(W, "baseline", enc.split("/")[-1], "pytorch_model.bin")
    ini_pref = os.path.join(W, "baseline", enc.split("/")[-1], "pytorch_model_bertprefixed.bin")
    print("[底座] %s  原始 %s  加前缀 %s"
          % (enc, "有" if os.path.exists(ini) else "缺",
             "有" if os.path.exists(ini_pref) else "缺（英文侧必须先跑 fix_bge_keys_1106.py）"))
    print()

print("[结论] 两套新资源的训练/验证文件都可被 dataset_task1 直接装载，任务一不需要单独做格式转换；")
print("       类别表由各工作目录下的 dataset/frame_info.json 提供（train_task1.py:238/242 硬编码该相对路径）。")
