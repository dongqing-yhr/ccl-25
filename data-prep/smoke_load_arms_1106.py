"""上卡前的装载级冒烟（CPU，零训练产物，2026-11-06）。

为什么要这一步：链 B 与链 A 一共 12 次训练，任何一次在起跑几分钟后崩掉就白排一次队。
本脚本按 train_task3.py 的同一段代码走一遍，只走一个 batch，检查四件事：
  1) 记录文件与类别表能被 dataset_task3.Dataset 吃下（框架名称、角色名称查表不缺）；
  2) 逐字符分词后的序列长度不超过编码器位置上限（英文侧最容易撞）；
  3) 预训练权重的键名与模型对得上——只数 bert. 前缀那部分，缺一个就说明底座装不进去；
  4) 一个 batch 的前向＋反向能跑通（含训练期帧条件先验建表那一步）。

用法（在某个资源的工作副本目录里跑，副本的 dataset/ 下已放好该资源的表与记录）：
    cd /work/workspace/ccl25/baseline_c10
    python3 ../datasets_paper/smoke_load_arms_1106.py arms=L0,L1,L3,FN n=3
    cd /work/workspace/ccl25/baseline_en
    python3 ../datasets_paper/smoke_load_arms_1106.py enc=../baseline/bge-large-en-v1.5 arms=L0,L1 n=3
不带 --enc 时用 ../baseline/chinese-macbert-large。
"""
import os
import sys

OPTS = {}
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1)
        OPTS[k] = v
ENC = OPTS.get("enc", "../baseline/chinese-macbert-large")
ARMS = OPTS.get("arms", "L0,L1,L3,FN").split(",")
N = int(OPTS.get("n", "3"))
NEG = "./dataset/train_deploy_negatives.json"
# params.py 在 import 时就 parse_args()，先把本脚本的参数从 argv 里摘掉
sys.argv = [sys.argv[0]]
WORK = os.getcwd()
sys.path.insert(0, WORK)

import torch  # noqa: E402
from transformers import BertConfig, BertTokenizer  # noqa: E402
from params import args  # noqa: E402
import model_task3  # noqa: E402
from dataset_task3 import Dataset  # noqa: E402
from train_task3 import get_model_input  # noqa: E402

torch.manual_seed(0)
print("[冒烟] 工作副本=%s  预训练模型=%s  档位=%s" % (WORK, ENC, ARMS))
cfg_path, vocab_path = (os.path.join(ENC, n) for n in ("config.json", "vocab.txt"))
# 权重文件可以单独指：hfl 那份带 bert. 前缀直接用，bge 那份不带前缀，
# 要先跑 baseline/fix_bge_keys_1106.py 生成 pytorch_model_bertprefixed.bin 再传 init= 指它。
init_path = OPTS.get("init", os.path.join(ENC, "pytorch_model.bin"))
for p in (cfg_path, vocab_path, init_path):
    if not os.path.exists(p):
        raise SystemExit("[FAIL] 底座文件缺失: %s" % p)

tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
base = Dataset("./dataset/task3_train.json", "./dataset/frame_info.json", tokenizer)
cap = BertConfig.from_json_file(cfg_path).max_position_embeddings
n_rec_over = sum(1 for r in base.all_data if len(r["text"]) + 4 > cap)
print("[自检] 角色列=%d  空类索引=%d  框架数=%d  位置上限=%d  原文+4 就越出位置表的记录 %d 条（应为 0）"
      % (base.num_labels, base.num_labels, len(base.frame2idx), cap, n_rec_over))
assert not n_rec_over, "有记录超过位置上限，先在转换阶段剔除或改截断口径"


def make_arm(arm):
    ds = Dataset("./dataset/task3_train.json", "./dataset/frame_info.json", tokenizer,
                 with_frame_name=(arm == "FN"),
                 deploy_negatives_file=(NEG if arm in ("L1", "L3", "FN") and os.path.exists(NEG) else None))
    # 真正的闸门是送进分词器以后的序列长度：帧名注入那一档还要再加一个 [SEP] 与框架名字符，
    # 这里取原文最长的 3 行实测，不看估算式。
    probe = sorted(range(len(ds.data)), key=lambda i: len(ds.data[i]["text"]))[-3:]
    lens = [len(ds[i][0]) for i in probe]
    print("[长度] %s 原文最长 3 行送进模型后的序列长 %s，位置上限 %d" % (arm, lens, cap))
    assert max(lens) <= cap, "帧名注入后越出位置表，转换阶段的原文长度上限要再压低"
    order = sorted(range(len(ds.data)), key=lambda i: len(ds.data[i]["text"]))[:N]
    ds.data = [ds.data[i] for i in order]
    return ds


for arm in ARMS:
    ds = make_arm(arm)
    cfg = BertConfig.from_json_file(cfg_path)
    cfg.num_labels = ds.num_labels + 1          # 空类，与 train_task3.py:400 那句 args.with_null 一致
    cfg.arghood_weight = 0.0
    cfg.num_frames = 0
    cfg.mtl_frame_weight = 0.0
    model = model_task3.Model(cfg)
    state = torch.load(init_path, map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    msg = model.load_state_dict(state, strict=False)
    miss_bert = [k for k in msg.missing_keys if k.startswith("bert.")]
    unexp_bert = [k for k in msg.unexpected_keys if k.startswith("bert.")]
    shape_bad = []
    msd = model.state_dict()
    for k in msd:
        if k in state and tuple(state[k].shape) != tuple(msd[k].shape):
            shape_bad.append((k, tuple(state[k].shape), tuple(msd[k].shape)))
    print("[权重] %s 装配缺失 bert. 键 %d 个 / 多余 bert. 键 %d 个 / 形状不符 %d 个"
          % (arm, len(miss_bert), len(unexp_bert), len(shape_bad)))
    if miss_bert:
        print("        缺失示例: %s" % miss_bert[:3])
    if shape_bad:
        print("        形状不符示例: %s" % shape_bad[:3])
    del state

    if arm == "L3":
        import numpy as np
        NF, NL = len(ds.frame2idx), cfg.num_labels
        cnt = np.zeros((NF, NL))
        for d in ds.data:
            f = ds.frame2idx.get(d.get("frame"), -1)
            if 0 <= f < NF:
                cnt[f, d["label_class"]] += 1
        tot = cnt.sum()
        glob = cnt.sum(0) / max(tot, 1.0)
        p = (cnt + 1.0 * glob) / (cnt.sum(1, keepdims=True) + 1.0)
        logpi = np.clip(np.log(p + 1e-12), -2.0, 0.0)
        logpi[:, NL - 1] = 0.0
        model.register_buffer("logit_prior", torch.tensor(logpi, dtype=torch.float))
        model.la_tau = 0.5
        print("[先验] 帧数=%d 类别数=%d  log pi ∈ [%.2f, %.2f]"
              % (NF, NL, logpi[:, :NL - 1].min(), logpi[:, :NL - 1].max()))
    else:
        model.la_tau = 0.0

    batch = get_model_input([ds[i] for i in range(N)], device="cpu")
    input_ids, attention_mask, target, labels, sentence_id, frame_idx = batch[:6]
    out = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                labels=labels, device="cpu", for_test=False, frame_idx=frame_idx)
    loss = out["loss"] if isinstance(out, dict) else out[0]
    loss.backward()
    g = sum(1 for q in model.parameters() if q.grad is not None)
    print("[前向] %s  n=%d  序列长=%d  标签形状=%s  loss=%.4f  有梯度的参数张量 %d 个  通过"
          % (arm, N, int(input_ids.shape[1]), tuple(labels.shape), float(loss), g))
    del model, ds

if "T2" in ARMS:
    # 任务二那一格：两条链各要训一次，负例清单就是从它的预测里挖的，跑歪了整条链停在第 3 步。
    import model_task2
    from dataset_task2 import Dataset as Dataset2
    from train_task2 import get_model_input as collate2
    ds2 = Dataset2("./dataset/task3_train.json", "./dataset/frame_info.json", tokenizer)
    dv2 = Dataset2("./dataset/task3_dev.json", "./dataset/frame_info.json", tokenizer, for_test=True)
    cfg2 = BertConfig.from_json_file(cfg_path)
    cfg2.num_labels = 1
    cfg2.max_cls = ds2.max_cls
    m2 = model_task2.Model(cfg2)
    st2 = torch.load(init_path, map_location="cpu")
    msg2 = m2.load_state_dict(st2, strict=False)
    del st2
    print("[任务二] 记录 %d 条 / 验证记录 %d 条 / 角色下标上界 max_cls=%d / 装配缺失 bert. 键 %d 个"
          % (len(ds2.all_data), len(dv2.all_data), ds2.max_cls,
             len([k for k in msg2.missing_keys if k.startswith("bert.")])))
    b2 = collate2([ds2[i] for i in sorted(range(len(ds2.all_data)),
                                          key=lambda i: len(ds2.all_data[i]["text"]))[:3]])
    print("[任务二] 一个 batch 已组好：输入形状 %s，注意力 %s，标注跨度 %s，目标词类别下标 %s"
          % (tuple(b2[0].shape), tuple(b2[1].shape),
             tuple(b2[3].shape) if hasattr(b2[3], "shape") else "list",
             tuple(b2[5].shape) if hasattr(b2[5], "shape") else "list"))

print("[冒烟结论] 以上档位在当前副本内可发车。负例档位若显示'负例清单不存在则未加'，"
      "那是该资源的任务二还没跑完，属预期。")
