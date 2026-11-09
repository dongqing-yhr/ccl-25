"""核对：我方 model_task1.py 能否把 LERT-large 的预训练权重装配上。

判据（换预训练模型前必须实测，不能凭键名推断）
  1) 编码器覆盖率：model.bert 里每一个参数，是否都在过滤后的 checkpoint 里出现
     覆盖率必须是 100%，否则等于拿随机初始化的编码器去训练
  2) 形状一致：出现的键里，形状不同的必须列出来（strict=False 会静默跳过）
  3) 任务头 dense 与 lstm 属于随机初始化，缺失是预期的，只报数不判失败

用法：python3 check_lert_assembly_task1.py <config.json> <pytorch_model.bin> <frame_info.json>
"""
import json, sys, torch
import torch.nn as nn
from transformers import BertConfig

sys.path.insert(0, "/work/workspace/ccl25/baseline")
from model_task1 import Model

cfg_path, ckpt_path, frame_info = sys.argv[1], sys.argv[2], sys.argv[3]

fi = json.load(open(frame_info, encoding="utf8"))
frames = []
for line in fi:
    if line["frame_name"] not in frames:
        frames.append(line["frame_name"])
num_labels = len(frames)
print("[cfg] num_labels=%d  config=%s" % (num_labels, cfg_path), flush=True)

config = BertConfig.from_json_file(cfg_path)
config.num_labels = num_labels
model = Model(config)
print("[cfg] hidden_size=%d num_hidden_layers=%d inner_dim=%d dense.out=%d"
      % (config.hidden_size, config.num_hidden_layers, model.inner_dim, model.dense.out_features), flush=True)

state = torch.load(ckpt_path, map_location="cpu")
print("[ckpt] 原始键数 %d" % len(state), flush=True)
prefixes = {}
for k in state:
    prefixes[k.split(".")[0]] = prefixes.get(k.split(".")[0], 0) + 1
print("[ckpt] 顶层前缀分布 %s" % prefixes, flush=True)

# 照抄 train_task1.py 的过滤：丢掉 lstm.* 再 strict=False 装配
state = {k: v for k, v in state.items() if not k.startswith("lstm.")}
msg = model.load_state_dict(state, strict=False)
print("[load] missing_keys=%d  unexpected_keys=%d" % (len(msg.missing_keys), len(msg.unexpected_keys)), flush=True)

bert_params = dict(model.bert.named_parameters())
bert_buffers = dict(model.bert.named_buffers())
have = set(k[len("bert."):] for k in state if k.startswith("bert."))
pm = sum(1 for k in bert_params if k in have)
bm = sum(1 for k in bert_buffers if k in have)
print("[gate1] 编码器可学习参数覆盖率 = %d/%d = %.2f%%" % (pm, len(bert_params), 100.0 * pm / len(bert_params)), flush=True)
print("[gate1b] 编码器 buffer 覆盖率 = %d/%d（token_type_ids / position_ids 是固定常量，checkpoint 不存属正常）"
      % (bm, len(bert_buffers)), flush=True)
miss_p = [k for k in bert_params if k not in have]
if miss_p:
    print("        未覆盖的可学习参数（前 20）：%s" % miss_p[:20], flush=True)
miss_b = [k for k in bert_buffers if k not in have]
if miss_b:
    print("        未覆盖的 buffer：%s" % miss_b, flush=True)

mismatch = []
msd = model.state_dict()
for k, v in state.items():
    if k in msd and tuple(msd[k].shape) != tuple(v.shape):
        mismatch.append((k, tuple(v.shape), tuple(msd[k].shape)))
print("[gate2] 形状不一致的键 = %d" % len(mismatch), flush=True)
for k, a, b in mismatch[:20]:
    print("        %s ckpt%s vs model%s" % (k, a, b), flush=True)

head_missing = [k for k in msg.missing_keys if k.startswith(("dense.", "lstm.", "dropout"))]
other_missing = [k for k in msg.missing_keys if not k.startswith(("dense.", "lstm.", "dropout"))]
print("[head] 任务头缺失（预期）%d 个；其他缺失 %d 个 %s"
      % (len(head_missing), len(other_missing), other_missing[:20]), flush=True)

ok = (pm == len(bert_params)) and (len(mismatch) == 0) and (len(other_missing) == 0)
print("", flush=True)
print("[结论] %s" % ("可以按我方配方训练 LERT" if ok else "不可以，先解决上面的问题"), flush=True)
sys.exit(0 if ok else 2)
