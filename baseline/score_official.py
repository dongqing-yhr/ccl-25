#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
按 CCL25 CFSP Overview 论文 4.1/4.2/4.3 定义的官方打分器。
在 dev 上对整条 task1->task2->task3 流水线的预测结果打分，作为提交前的离线代理指标。

三个子任务口径不同：
  task1 框架识别 FI: 行级准确率 = correct / total (一行 = 一条 sentence_id 加它的目标词; dev 2300 行,
      去重后 1995 个 sentence_id, 分母取行不去重)
  task2 论元范围 AI: token(字符)重叠 F1, precision=InterSec/Len(pred), recall=InterSec/Len(gold)
  task3 论元角色 RI: 严格边界+角色的 (span, role) 集合 F1
综合分 = 0.3*FI + 0.3*AI + 0.4*RI  (各项按百分制)

坐标约定：实测 gold 的 cfn_spans 下标已超出 word 列表长度，说明它与 predict 输出一样的 list(text) 字符下标，直接同坐标比较即可，不做词->字符展平；如某份数据确为词级，加 --expand_words 走 word 列表展平。
自检 --selfcheck 用 gold 自身当预测，FI/AI/RI 都应 = 100。
"""

import argparse
import json
import codecs
from collections import defaultdict


def load_json(path):
    with codecs.open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def span_to_chars(line, start, end):
    """默认 gold cfn_spans 已是字符下标直接用；--expand_words 时按 word 列表把词级下标展成字符区间"""
    if not ARGS.expand_words:
        return start, end
    words = line["word"]
    cs = words[start]["start"]
    ce = words[end]["end"]
    return cs, ce


def char_cover(start, end):
    return set(range(start, end + 1))


def f1(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def build_gold(gold_data):
    """按 sentence_id 合并（一句可能有多行=多个目标词）：gold_frame 集合、字符cover 并集、(cs,ce,role) 三元组并集"""
    gold = {}
    for line in gold_data:
        sid = line["sentence_id"]
        slot = gold.setdefault(sid, {"frame": set(), "cover": set(), "triples": set()})
        if line.get("frame") is not None:
            slot["frame"].add(line["frame"])
        for s in line.get("cfn_spans", []):
            cs, ce = span_to_chars(line, s["start"], s["end"])
            slot["cover"] |= char_cover(cs, ce)
            slot["triples"].add((cs, ce, s["fe_name"]))
    return gold


def score_fi(gold_data, pred1):
    """FI 按行位置对齐：predict_task1 逐行读入与 gold 同序，pred1[i] 对应 gold_data[i]"""
    total = 0
    correct = 0
    mismatch_order = 0
    for i, line in enumerate(gold_data):
        gf = line.get("frame")
        if gf is None:
            continue
        total += 1
        if i >= len(pred1):
            continue
        if pred1[i][0] != line["sentence_id"]:
            mismatch_order += 1
        if pred1[i][0] == line["sentence_id"] and pred1[i][1] == gf:
            correct += 1
    if mismatch_order:
        print("[警告] FI 位置对齐有 {} 处 sentence_id 不匹配，预测文件顺序/行数与 gold 不一致".format(mismatch_order))
    return (100.0 * correct / total if total else 0.0), correct, total


def score_ai(gold, pred2):
    pred_cover = defaultdict(set)
    for item in pred2:
        sid, st, en = item[0], item[1], item[2]
        pred_cover[sid] |= char_cover(st, en)
    inter = 0.0
    len_pred = 0.0
    len_gold = 0.0
    for sid, g in gold.items():
        pc = pred_cover.get(sid, set())
        gc = g["cover"]
        inter += len(pc & gc)
        len_pred += len(pc)
        len_gold += len(gc)
    prec = inter / len_pred if len_pred else 0.0
    rec = inter / len_gold if len_gold else 0.0
    return 100.0 * f1(prec, rec), 100.0 * prec, 100.0 * rec


def score_ri(gold, pred3):
    pred_triples = defaultdict(set)
    for item in pred3:
        sid, st, en, role = item[0], item[1], item[2], item[3]
        pred_triples[sid].add((st, en, role))
    inter = 0.0
    n_pred = 0.0
    n_gold = 0.0
    for sid, g in gold.items():
        pp = pred_triples.get(sid, set())
        gg = g["triples"]
        inter += len(pp & gg)
        n_pred += len(pp)
        n_gold += len(gg)
    prec = inter / n_pred if n_pred else 0.0
    rec = inter / n_gold if n_gold else 0.0
    return 100.0 * f1(prec, rec), 100.0 * prec, 100.0 * rec


def gold_as_pred(gold_data):
    """自检用：把 gold 直接伪造成三个预测文件的内容"""
    p1, p2, p3 = [], [], []
    for line in gold_data:
        sid = line["sentence_id"]
        p1.append([sid, line.get("frame")])
        for s in line.get("cfn_spans", []):
            cs, ce = span_to_chars(line, s["start"], s["end"])
            p2.append([sid, cs, ce])
            p3.append([sid, cs, ce, s["fe_name"]])
    return p1, p2, p3


def report(fi, ai, ai_p, ai_r, ri, ri_p, ri_r):
    final = 0.3 * fi + 0.3 * ai + 0.4 * ri
    print("=" * 46)
    print("task1 框架识别 FI acc : {:8.4f}".format(fi))
    print("task2 论元范围 AI f1  : {:8.4f}  (p {:.4f} / r {:.4f})".format(ai, ai_p, ai_r))
    print("task3 论元角色 RI f1  : {:8.4f}  (p {:.4f} / r {:.4f})".format(ri, ri_p, ri_r))
    print("综合分 0.3*FI+0.3*AI+0.4*RI : {:8.4f}".format(final))
    print("=" * 46)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="dataset/cfn-dev.json")
    parser.add_argument("--task1", default="dataset/A_task1_test.json")
    parser.add_argument("--task2", default="dataset/A_task2_test.json")
    parser.add_argument("--task3", default="dataset/A_task3_test.json")
    parser.add_argument("--split", default=None, choices=["dev", "A", "B"],
                        help="指定后从 dataset/<split>/ 读 <prefix>task{1,2,3}_test.json, 覆盖上面三个默认值")
    parser.add_argument("--expand_words", action='store_true',
                        help="gold cfn_spans 是词级下标时加此参数，按 word 列表展成字符区间（默认按字符下标直接用）")
    parser.add_argument("--selfcheck", action='store_true',
                        help="用 gold 自身当预测，FI/AI/RI 应全 = 100")
    ARGS = parser.parse_args()

    if ARGS.split:
        prefix = 'dev_' if ARGS.split == 'dev' else ARGS.split + '_'
        ARGS.task1 = 'dataset/{}/{}task1_test.json'.format(ARGS.split, prefix)
        ARGS.task2 = 'dataset/{}/{}task2_test.json'.format(ARGS.split, prefix)
        ARGS.task3 = 'dataset/{}/{}task3_test.json'.format(ARGS.split, prefix)

    gold_data = load_json(ARGS.gold)
    gold = build_gold(gold_data)

    if ARGS.selfcheck:
        p1, p2, p3 = gold_as_pred(gold_data)
        fi, _, _ = score_fi(gold_data, p1)
        ai, aip, air = score_ai(gold, p2)
        ri, rip, rir = score_ri(gold, p3)
        print("[自检] gold vs gold，下面三项应全等于 100")
        report(fi, ai, aip, air, ri, rip, rir)
    else:
        p1 = load_json(ARGS.task1)
        p2 = load_json(ARGS.task2)
        p3 = load_json(ARGS.task3)
        fi, fc, ft = score_fi(gold_data, p1)
        ai, aip, air = score_ai(gold, p2)
        ri, rip, rir = score_ri(gold, p3)
        print("[官方口径 dev 打分] gold={}  FI命中 {}/{}".format(ARGS.gold, fc, ft))
        report(fi, ai, aip, air, ri, rip, rir)
