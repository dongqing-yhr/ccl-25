# R51: CFN 1.0 的任务三读入自检与 target 字段归一化（零训练，2026-11-06）
#
# 为什么要这一项：baseline/dataset_task3.py 第 66 行读 line["target"][-1]，
# CFN 2.2 的 target 是列表、CFN 1.0 的 target 是单个字典，直接挂 1.0 数据会 KeyError。
# 本脚本不改 baseline/ 里任何代码，只在数据侧把 1.0 的三份切分归一化成 2.2 的记录形状，
# 并逐字复现装载器第 63-89 行的样本构建逻辑，把四项整数一次打全。

import json
import os
import sys

SRC = os.path.join("datasets_paper", "cfn1.0")
OUT = os.path.join("datasets_paper", "cfn1.0_task3")
SPLITS = [("cfn-train.json", "task3_train.json"), ("cfn-dev.json", "task3_dev.json")]

os.makedirs(OUT, exist_ok=True)


def normalize(records, tag):
    n_dict = n_list = 0
    for line in records:
        t = line.get("target")
        if isinstance(t, dict):
            line["target"] = [t]
            n_dict += 1
        elif isinstance(t, list):
            n_list += 1
        else:
            raise SystemExit("[FAIL] %s 里出现第三种 target 形态: %r" % (tag, t))
    print("[归一化] %s: target 由字典转列表 %d 条, 原本已是列表 %d 条" % (tag, n_dict, n_list))
    return records


def build_label_table(frame_info_path):
    idx2label = []
    with open(frame_info_path, encoding="utf-8") as f:
        ori = json.load(f)
    for line in ori:
        for fe in line.get("fes", []):
            if fe["fe_name"] not in idx2label:
                idx2label.append(fe["fe_name"])
    return idx2label, ori


def emulate_loader(records, label2idx, tag):
    """逐字照 dataset_task3.py 第 63-89 行（金标帧来源，不挂 OOF 文件）。"""
    data = []
    skipped_overlap = 0
    missing_fe = {}
    spans_seen = 0
    for line in records:
        text = line["text"]
        target = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
        for spans in line["cfn_spans"]:
            spans_seen += 1
            if spans["fe_name"] not in label2idx:
                missing_fe[spans["fe_name"]] = missing_fe.get(spans["fe_name"], 0) + 1
                continue
            if spans["end"] + 1 < target[0]:
                label_idx = [spans["start"] + 1, spans["end"] + 1]
            elif spans["start"] + 1 > target[1]:
                label_idx = [spans["start"] + 3, spans["end"] + 3]
            else:
                skipped_overlap += 1
                continue
            data.append({"label_idx": label_idx,
                         "sentence_id": line["sentence_id"],
                         "max_idx": label_idx[1]})
    return data, spans_seen, skipped_overlap, missing_fe


def main():
    print("=" * 68)
    print("R51 CFN 1.0 任务三读入自检")
    print("=" * 68)

    frame_info = os.path.join(SRC, "frame_info.json")
    if not os.path.exists(frame_info):
        raise SystemExit("[FAIL] 缺 %s" % frame_info)
    idx2label, ori_frame_info = build_label_table(frame_info)
    label2idx = {n: i for i, n in enumerate(idx2label)}
    print("[检查二] frame_info.json 记录 %d 条; 去重后角色名 %d 个, 加 NULL 一类 = %d"
          % (len(ori_frame_info), len(idx2label), len(idx2label) + 1))
    if len(idx2label) + 1 != 991:
        print("          ⚠️ 与需求文档里写的 991 列不符, 以本行读数为准并停下报告")

    # 合法角色表松紧（表 I 用数：每帧允许角色数的中位数）
    per_frame = [len(l.get("fes", [])) for l in ori_frame_info if l.get("fes")]
    per_frame.sort()
    med = per_frame[len(per_frame) // 2] if per_frame else 0
    print("[材料] 每帧声明角色数: 中位 %d, 均值 %.2f, 最大 %d, 帧数 %d"
          % (med, sum(per_frame) / max(1, len(per_frame)), per_frame[-1] if per_frame else 0, len(per_frame)))

    for src_name, out_name in SPLITS:
        src = os.path.join(SRC, src_name)
        dst = os.path.join(OUT, out_name)
        if not os.path.exists(src):
            print("[SKIP] 缺 %s" % src)
            continue
        with open(src, encoding="utf-8") as f:
            records = json.load(f)
        print("-" * 68)
        print("[%s]" % src_name)
        records = normalize(records, src_name)

        sids = set(r["sentence_id"] for r in records)
        gold_spans = sum(len(r["cfn_spans"]) for r in records)
        frame_ok = sum(1 for r in records if r.get("frame"))
        multi_tgt = sum(1 for r in records if len(r["target"]) > 1)
        print("[检查一] 记录 %d 条 / 独立句 %d / 金标 span %d / 带框架名称 %d / 一句两个目标词 %d"
              % (len(records), len(sids), gold_spans, frame_ok, multi_tgt))

        data, spans_seen, skipped, missing = emulate_loader(records, label2idx, src_name)
        print("[检查三] 装载后计入样本 %d 行; 因跨度与目标词重叠而跳过 %d 行; 未计入合计 %d 行"
              % (len(data), skipped, spans_seen - len(data)))
        if missing:
            tot = sum(missing.values())
            print("[检查三·异常] 角色名不在 frame_info 声明表里的 span %d 个, 涉及 %d 种名称: %s"
                  % (tot, len(missing), sorted(missing.items(), key=lambda kv: -kv[1])[:10]))
        else:
            print("[检查三] 角色名全部能在 frame_info 声明表里查到, 无缺失")

        longest = max((d["max_idx"] for d in data), default=0)
        over500 = sum(1 for d in data if d["max_idx"] >= 500)
        print("[检查四] 最长跨度下标 %d; 跨度下标 >=500 的行 %d (512 位置上限的截断风险)" % (longest, over500))

        with open(dst, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        print("[产出] %s (%.1f MB)" % (dst, os.path.getsize(dst) / 1048576.0))

    print("=" * 68)
    print("归一化文件写在 %s; 训练时 --train_file/--dev_file 指到这里, baseline/ 代码不动" % OUT)


if __name__ == "__main__":
    sys.exit(main())
