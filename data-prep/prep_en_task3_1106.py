# R60（新编号）：把英文 FrameNet 1.7 的三份切分转成任务三装载器认的记录格式（零训练，2026-11-06）
#
# 为什么要这一项：baseline/dataset_task3.py 要的记录有 text／target／cfn_spans／frame／sentence_id 五个字段，
# 英文侧手上只有 CoNLL。同时任务三装载器里 label_file 是硬编码的 ./dataset/frame_info.json，
# 所以英文侧还要一份同构的合法角色表（键名 frame_name 与 fes[].fe_name）。
#
# 跨度口径：装载器把句子按字符切开再逐字符送分词器，实测 1 个字符占 1 个 token（含空格，见评审意见 8.2），
# 所以这里把词元跨度换算成字符跨度：text = 词元用单个空格接起来，第 i 个词元的字符区间
# [前缀长度之和, 该区间加上词元长度减一]，两端都闭，与 CFN 的 cfn_spans 口径一致。
#
# 目标词口径：一个目标词＝一段连续带目标标记且帧名相同的词元（多词谓语算一个目标词）。
# census_splits_1106.py 里报的 20 105／2 382／6 872 数的是带目标标记的词元个数，比目标词个数多出
# 665／95／150 个续接词元；本脚本报的是目标词个数，两个整数都打印出来，表注里写清数的是哪一批。
#
# 两条装载级前置（11-06 自检 check_load_ready_1106.py 查出来的，不做则训练当场崩）：
#  1) 编码器位置上限 512。装载器把原文逐字符送分词器，英文实测 1 字符约 1 词元。
#     序列长＝原文长度 + 4（[CLS]、[SEP] 与目标词两端标记）；帧名注入那一档再 + 1 个 [SEP] + 框架名长度。
#     框架名注入用的名字在训练期取该行金标、推理期取任务一的预测，所以留的余量要按全表最长框架名算：
#     本资源最长框架名 'Cause_change_of_position_on_a_scale' 35 个字符 ⇒ 原文上限 512-4-1-35-2 = 470。
#     超过的记录直接剔除（占比千分位），剔除前后的记录数与跨度数都打印出来当分母。

#  2) 类别表要覆盖三份切分里出现过的每个框架名称与角色名称，否则 dataset_task2.py:87 与
#     dataset_task3.py:81 查表 KeyError。该数据源没有随切分分发的框架声明文件，
#     所以表的框架清单与角色列取三份切分实际出现过的配对；训练期先验与部署匹配负例仍只吃训练切分。

MAX_CHARS = 470
import hashlib
import json
import os
import sys

SRC = os.path.join("datasets_paper", "framenet_splits")
OUT = os.path.join("datasets_paper", "framenet_task3")
SPLITS = [("fn1.7.fulltext.train.syntaxnet.conll", "task3_train.json", 0),
          ("fn1.7.dev.syntaxnet.conll", "task3_dev.json", 10 ** 6),
          ("fn1.7.test.syntaxnet.conll", "task3_test.json", 2 * 10 ** 6)]


def surface(cell):
    """列 1 是 Python bytes 的字面量，形如 b'Some' 或 b"n't"，两种引号都要吃下。"""
    s = cell.strip()
    if s.startswith("b'") and s.endswith("'"):
        inner = s[2:-1]
        return inner.replace("\\'", "'").replace("\\\\", "\\")
    if s.startswith('b"') and s.endswith('"'):
        inner = s[2:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    return s


def iter_blocks(path):
    block = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if block:
                    yield block
                block = []
                continue
            p = line.split("\t")
            if len(p) >= 15:
                block.append(p)
    if block:
        yield block


def char_offsets(tokens):
    offs = []
    cur = 0
    for t in tokens:
        offs.append((cur, cur + len(t) - 1))
        cur += len(t) + 1
    return offs


def target_runs(rows):
    marks = [i for i, r in enumerate(rows) if r[12] not in ("_", "")]
    runs = []
    i = 0
    while i < len(marks):
        j = i
        while j + 1 < len(marks) and marks[j + 1] == marks[j] + 1 and rows[marks[j + 1]][13] == rows[marks[j]][13]:
            j += 1
        if rows[marks[i]][13] not in ("_", ""):
            runs.append((marks[i], marks[j], rows[marks[i]][13]))
        i = j + 1
    return runs, len(marks)


def role_spans(rows, offs):
    spans = []
    st = en = prev_name = None
    for idx, r in enumerate(rows):
        tag = r[14]
        if tag in ("O", "_", ""):
            if st is not None:
                spans.append((st, en, prev_name))
                st = None
            prev_name = None
            continue
        state = tag[0]
        name = tag.split("-", 1)[1] if "-" in tag else tag
        if st is None or state in ("S", "B") or name != prev_name:
            if st is not None:
                spans.append((st, en, prev_name))
            st = offs[idx][0]
        en = offs[idx][1]
        prev_name = name
    if st is not None:
        spans.append((st, en, prev_name))
    return spans


def main():
    os.makedirs(OUT, exist_ok=True)
    train_pairs = {}
    all_pairs = {}
    all_roles = set()
    split_roles = {}

    for src_name, out_name, sid_base in SPLITS:
        path = os.path.join(SRC, src_name)
        if not os.path.exists(path):
            raise SystemExit("[FAIL] 缺 %s" % path)
        records = []
        n_blocks = n_marks = n_spans = n_argless = n_targets = 0
        n_toolong = n_toolong_spans = 0
        frames = set()
        for bi, rows in enumerate(iter_blocks(path)):
            n_blocks += 1
            tokens = [surface(r[1]) for r in rows]
            text = " ".join(tokens)
            offs = char_offsets(tokens)
            runs, marks = target_runs(rows)
            n_marks += marks
            n_targets += len(runs)
            spans = role_spans(rows, offs)
            n_spans += len(spans)
            if not spans:
                n_argless += 1
            if not runs:
                continue
            too_long = len(text) > MAX_CHARS
            if too_long:
                n_toolong += 1
                n_toolong_spans += len(spans)
            for fe in spans:
                all_roles.add(fe[2])
                split_roles.setdefault(out_name, set()).add(fe[2])
            for k, (a, b, frame) in enumerate(runs):
                frames.add(frame)
                all_pairs.setdefault(frame, set())
                if out_name.endswith("train.json"):
                    train_pairs.setdefault(frame, set())
                for (s, e, nm) in spans:
                    all_pairs[frame].add(nm)
                    if out_name.endswith("train.json"):
                        train_pairs[frame].add(nm)
                if too_long:
                    continue
                sid = sid_base + bi
                records.append({
                    "sentence_id": sid,
                    "text": text,
                    "word": tokens,
                    "frame": frame,
                    "target": [{"start": offs[a][0], "end": offs[b][1], "pos": rows[a][5]}],
                    "cfn_spans": [{"start": s, "end": e, "fe_name": nm, "fe_abbr": nm}
                                  for (s, e, nm) in spans],
                })
        dst = os.path.join(OUT, out_name)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        sids = set(r["sentence_id"] for r in records)
        gold = sum(len(r["cfn_spans"]) for r in records)
        kept_targets = len(records)
        print("[产出] %-16s 块 %6d 带目标标记的词元 %6d 目标词 %6d 记录 %6d 独立句 %6d 金标跨度 %7d 空论元块 %5d 帧名 %4d"
              % (out_name, n_blocks, n_marks, n_targets, len(records), len(sids), gold, n_argless, len(frames)))
        print("        超长剔除(原文>%d 字符) 记录 %4d 跨度 %5d ；剔除后目标词 %d 跨度 %d"
              % (MAX_CHARS, n_toolong, n_toolong_spans, kept_targets, gold))
        print("        跨度对照：census 报的 arg_span 与本脚本换算出的跨度总数按块一致时，上一行的金标跨度应等于 census 读数（含无目标词块）")

    print("-" * 70)
    for k in sorted(split_roles):
        print("角色名称 %s 出现 %d 种" % (k, len(split_roles[k])))
    print("角色名称去重：三份切分并集 %d 种" % len(all_roles))
    # 类别表：框架清单与每帧角色清单取三份切分实际出现过的配对（该数据源没有随切分分发的声明文件），
    # 目的是装载器查表不 KeyError。训练期先验表与部署匹配负例只吃训练切分，不受这里影响。
    frame_info = []
    for fr in sorted(all_pairs):
        frame_info.append({"frame_name": fr,
                           "frame_ename": fr,
                           "frame_def": "",
                           "fes": [{"fe_name": nm, "fe_abbr": nm, "fe_ename": nm, "fe_def": ""}
                                   for nm in sorted(all_pairs[fr])]})
    dst = os.path.join(OUT, "frame_info_en.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(frame_info, f, ensure_ascii=False)
    per_frame = sorted(len(v) for v in all_pairs.values())
    train_frame_max_fes = max(len(v) for v in train_pairs.values())
    print("[类别表] 帧 %d 个（其中训练切分出现过的 %d 个），角色名称列 %d 个（加空类 %d 列），"
          "每帧角色数 中位 %d 均值 %.2f 最大 %d"
          % (len(frame_info), len(train_pairs), len(all_roles), len(all_roles) + 1,
             per_frame[len(per_frame) // 2], sum(per_frame) / len(per_frame), per_frame[-1]))
    print("        任务二第二个维度（角色下标上界）取训练切分：最大 %d" % train_frame_max_fes)
    print("[产出] %s" % dst)
    print("用法：给英文侧建 baseline 工作副本，把本目录的 frame_info_en.json 放到副本的 dataset/frame_info.json，"
          "三份记录文件放 dataset/ 下，装载器与训练脚本一行不改。")


if __name__ == "__main__":
    sys.exit(main())
