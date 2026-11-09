"""LERT task1 softmax 概率导出 v2：权重路径与输出前缀由环境变量给定。

环境变量
  LERT_CKPT     要评测的 task1 权重（.bin）
  LERT_OUT      输出前缀，落盘 {prefix}_{dev,testA}.npy 与 {prefix}_sids_{dev,testA}.json
  LERT_SETS     逗号分隔，默认 dev,testA

自检（每次运行都打印，不通过就非零退出）
  1) 权重装配：missing_keys / unexpected_keys 计数必须为 0，否则说明架构不匹配
  2) 验证集按行准确率：逐行取 argmax 框架名称与该行金标框架名称比对
     （不用 sentence_id 建字典，同一 sentence_id 的多个目标词各算一行）
"""
import json, os, sys, torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer
from tqdm import tqdm

sys.path.insert(0, "/work/workspace/ccl25/baseline")
from lert_t1train import Model, collate, KMAX

BERT_DIR = "/work/workspace/ccl25/baseline/models/hfl--LERT-large"
FRAME_INFO = "/work/workspace/ccl25/baseline/dataset/frame_info.json"
DATASET = "/work/workspace/ccl25/baseline/dataset"

CKPT = os.environ["LERT_CKPT"]
OUT = os.environ["LERT_OUT"]
SETS = {
    "dev": os.path.join(DATASET, "cfn-dev.json"),
    "testA": os.path.join(DATASET, "cfn-test-A.json"),
    "testB": os.path.join(DATASET, "cfn-test-B.json"),
}
WANT = [s.strip() for s in os.environ.get("LERT_SETS", "dev,testA").split(",") if s.strip()]


class InferDS(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer):
        self.tok = tokenizer
        self.data = json.load(open(json_file, "r", encoding="utf-8"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok.encode_plus(list(d["text"]))
        tg = d["target"] if isinstance(d["target"], list) else [d["target"]]
        tg = tg[:KMAX]
        s = [int(t["start"]) + 1 for t in tg]
        e = [int(t["end"]) + 1 for t in tg]
        return (enc["input_ids"], enc["attention_mask"], s, e, 0, i, int(d["sentence_id"]), len(tg))


def main():
    device = torch.device("cuda:0")
    fi = json.load(open(FRAME_INFO, "r", encoding="utf-8"))
    frames = []
    for line in fi:
        if line["frame_name"] not in frames:
            frames.append(line["frame_name"])
    n = len(frames)
    print("[cfg] frames=%d ckpt=%s out=%s sets=%s" % (n, CKPT, OUT, ",".join(WANT)), flush=True)

    tok = BertTokenizer(vocab_file=os.path.join(BERT_DIR, "vocab.txt"), do_lower_case=True)
    cfg = BertConfig.from_json_file(os.path.join(BERT_DIR, "config.json"))
    cfg.num_labels = n
    model = Model(cfg, inner_dim=192, head_mlp=0, agg_mode="logit")

    st = torch.load(os.path.join(BERT_DIR, "pytorch_model.bin"), map_location="cpu")
    st = {k.replace("bert.", ""): v for k, v in st.items() if not k.startswith("cls.")}
    msg0 = model.bert.load_state_dict(st, strict=False)
    print("[gate1a] backbone  missing=%d unexpected=%d" % (len(msg0.missing_keys), len(msg0.unexpected_keys)), flush=True)

    ck = torch.load(CKPT, map_location="cpu")
    msg = model.load_state_dict(ck, strict=False)
    nmiss, nunexp = len(msg.missing_keys), len(msg.unexpected_keys)
    print("[gate1b] task1 权重 missing=%d unexpected=%d" % (nmiss, nunexp), flush=True)
    if nmiss or nunexp:
        print("!! 装配不完整，架构与权重不匹配，中止", flush=True)
        print("   missing 前 10:", msg.missing_keys[:10], flush=True)
        print("   unexpected 前 10:", msg.unexpected_keys[:10], flush=True)
        sys.exit(2)

    model.to(device).eval()

    for tag in WANT:
        path = SETS[tag]
        raw = json.load(open(path, "r", encoding="utf-8"))
        ds = InferDS(path, tok)
        ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)
        probs, sids = [], []
        with torch.no_grad():
            for batch in tqdm(ld, desc=tag):
                ids, am, S, E, _, _, sid = batch
                lg = model(ids.to(device), am.to(device), S.to(device), E.to(device), device)
                p = torch.softmax(lg, dim=-1).cpu().numpy()
                probs.append(p)
                sids.extend(int(x) for x in sid)
        P = np.concatenate(probs, 0)

        assert P.shape[0] == len(raw), "行数不符：%d vs %d" % (P.shape[0], len(raw))
        pred = P.argmax(1)

        if tag == "dev":
            ok = 0
            badframe = 0
            for i, d in enumerate(raw):
                g = d.get("frame", None)
                if g is None:
                    badframe += 1
                    continue
                if frames[pred[i]] == g:
                    ok += 1
            print("[gate2] 验证集按行准确率 = %d/%d = %.4f%%（金标缺失行 %d）"
                  % (ok, len(raw), 100.0 * ok / len(raw), badframe), flush=True)
            json.dump({"rows": len(raw), "correct": ok, "acc": ok / len(raw)},
                      open(OUT + "_devacc.json", "w"))

        np.save(OUT + "_%s.npy" % tag, P)
        json.dump(sids, open(OUT + "_sids_%s.json" % tag, "w"))
        np.save(OUT + "_predidx_%s.npy" % tag, pred)
        print("[out] %s rows=%d shape=%s -> %s_%s.npy" % (tag, P.shape[0], P.shape, OUT, tag), flush=True)

    json.dump(frames, open(OUT + "_frames.json", "w"), ensure_ascii=False)
    print("[done] 框架名称顺序表 -> %s_frames.json" % OUT, flush=True)


main()
