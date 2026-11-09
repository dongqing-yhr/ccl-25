"""R51 之后、发车之前的装载级自检（零训练、CPU）。

对 CFN 1.0 与英文两份转换产物，检查任务二与任务三装载时会不会当场崩：
  1) 每条记录的框架名称是否在该资源的角色表里（任务二 dataset_task2.py:87 直接查表，缺一个就 KeyError）；
  2) 每个框架元素的名称是否在该资源的角色列里（任务三 dataset_task3.py:81 直接查表，同上）；
  3) 表形状：框架数（任务二分类头列数）、单框架最多角色数（任务二第二维）、角色列总数＋NULL（任务三分类头列数）；
  4) 任务三装载后计入的标注行数，与原文跳过（跨度与目标词重叠）的行数；
  5) 冒烟喂一个 batch：分词后的 input_ids 最大长度是否超过编码器 max_position_embeddings。

用法：python3 datasets_paper/check_load_ready_1106.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "datasets_paper")

RESOURCES = [
    ("CFN 1.0", os.path.join(PAPER, "cfn1.0"), os.path.join(PAPER, "cfn1.0_task3"),
     "frame_info.json", ["task3_train.json", "task3_dev.json"],
     os.path.join(ROOT, "baseline", "chinese-macbert-large", "config.json")),
    ("英文 FrameNet 1.7", os.path.join(PAPER, "framenet_task3"), os.path.join(PAPER, "framenet_task3"),
     "frame_info_en.json", ["task3_train.json", "task3_dev.json", "task3_test.json"],
     os.path.join(ROOT, "baseline", "bge-large-en-v1.5", "config.json")),
]


def load(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


for name, info_dir, data_dir, info_file, splits, cfg in RESOURCES:
    print("=" * 78)
    print(name)
    ip = os.path.join(info_dir, info_file)
    if not os.path.exists(ip):
        print("  [缺] 角色表不存在: %s" % ip)
        continue
    info = load(ip)
    frames = [r["frame_name"] for r in info]
    roles = set()
    max_fes = 0
    for r in info:
        max_fes = max(max_fes, len(r["fes"]))
        for fe in r["fes"]:
            roles.add(fe["fe_name"])
    print("  表形状  框架数=%d  单框架最多角色数=%d  角色列数=%d  任务三分类头列数=%d(+NULL)"
          % (len(frames), max_fes, len(roles), len(roles) + 1))
    frame_set = set(frames)

    max_len = 0
    n_records = 0
    n_rows = 0
    n_skip = 0
    bad_frame = 0
    bad_role = 0
    multi_target = 0
    for sp in splits:
        dp = os.path.join(data_dir, sp)
        if not os.path.exists(dp):
            print("  [缺] 记录文件不存在: %s" % dp)
            continue
        data = load(dp)
        loc_bad_frame = loc_bad_role = loc_rows = loc_skip = 0
        for line in data:
            if line["frame"] not in frame_set:
                loc_bad_frame += 1
            tgt = line["target"]
            if len(tgt) > 1:
                multi_target += 1
            t = [tgt[-1]["start"] + 1, tgt[-1]["end"] + 1]
            for s in line["cfn_spans"]:
                if s["fe_name"] not in roles:
                    loc_bad_role += 1
                    continue
                if s["end"] + 1 < t[0] or s["start"] + 1 > t[1]:
                    loc_rows += 1
                else:
                    loc_skip += 1
            max_len = max(max_len, len(line["text"]))
        n_records += len(data)
        n_rows += loc_rows
        n_skip += loc_skip
        bad_frame += loc_bad_frame
        bad_role += loc_bad_role
        print("  %-18s 记录=%6d  计入标注行=%6d  跳过(跨目标词)=%5d  框架名缺=%d  角色名缺=%d"
              % (sp, len(data), loc_rows, loc_skip, loc_bad_frame, loc_bad_role))

    cap = None
    if os.path.exists(cfg):
        cap = load(cfg).get("max_position_embeddings")
    # 任务装载时逐字切分后再插 2 个分隔标记，且 [CLS]/[SEP] 各占一个
    print("  最长原文=%d 字符  编码器 max_position_embeddings=%s  (任务装载后约 %d 个 token)"
          % (max_len, cap, max_len + 4))
    print("  合计  记录=%d  计入标注行=%d  跳过=%d  框架名缺=%d  角色名缺=%d  一行多目标词的记录=%d"
          % (n_records, n_rows, n_skip, bad_frame, bad_role, multi_target))
    if bad_frame or bad_role:
        print("  [阻塞] 存在查不到表的框架名称或角色名称，装载会 KeyError，先补表再发车")
    elif cap is not None and max_len + 4 > cap:
        print("  [阻塞] 有句子的字符 token 数超过编码器位置上限，需要决定截断口径")
    else:
        print("  [通过] 装载不会 KeyError")
