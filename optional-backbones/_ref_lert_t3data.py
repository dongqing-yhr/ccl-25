# -*- coding: utf-8 -*-
"""Task3 数据层：角色分类（target 锚点 + span 定位）。

官方基线语义（dataset_task3.py / predict_task3.py）：
  - tokenizer: BertTokenizer，字符级 (list(text)) → 每个字符一个 token；
  - 在 target 前插 id=1、后插 id=2，target 坐标 t0=start+1, t1=end+1；
  - span 在 target 前: [s+1, e+1]；在 target 后: [s+3, e+3]；
  - 与 target 重叠的 span：官方未定义 label_idx（复用上一轮的值），属 bug；
  - 多 target 行：官方只取 target[-1]；
  - 推理 data_dict 以 sentence_id 为键，重复 sid 时后行覆盖前行（官方 bug）。

本文件提供：
  build_roles(frame_info)           全量 role 列表（1009，顺序与官方一致）
  prep_rows(rows, tokenizer, ...)   逐行编码缓存
  RoleDataset                       训练集（每 gold span 一条）
  InferDataset                      推理集（外部 span 列表）
  collate(...)                      batch 拼装
参数开关用于在"严格复现官方 bug"与"修正版"之间切换：
  overlap_mode = 'prev' | 'skip'
  sid_mode     = 'last' | 'row'
  expand_targets = False | True
"""
import numpy as np
import torch


def build_roles(frame_info):
    roles = []
    for line in frame_info:
        for fe in line["fes"]:
            if fe["fe_name"] not in roles:
                roles.append(fe["fe_name"])
    return roles


def span_to_idx(s, e, t0, t1, overlap_mode="skip"):
    """返回插入锚点后的 [a,b]；重叠时按 overlap_mode 返回 None。"""
    if e + 1 < t0:
        return [s + 1, e + 1]
    if s + 1 > t1:
        return [s + 3, e + 3]
    return None


def build_ids(tokenizer, text, t0, t1, max_len):
    ids = tokenizer.encode_plus(list(text))["input_ids"]
    ids2 = ids[:t0] + [1] + ids[t0:t1 + 1] + [2] + ids[t1 + 1:]
    trunc = len(ids2) > max_len
    if trunc:
        ids2 = ids2[:max_len]
    return ids2, trunc


def prep_rows(rows, tokenizer, max_len=320):
    """返回 {row_index: (ids2, [t0,t1], sid, text)}，对每行每个 target 各一份。

    训练侧按行索引取值，不用 sid 作键，避免官方 sid 覆盖 bug。
    """
    cache = {}
    for i, r in enumerate(rows):
        text = r["text"]
        ids0 = tokenizer.encode_plus(list(text))["input_ids"]
        tgts = r["target"] if isinstance(r["target"], list) else [r["target"]]
        ent = []
        for t in tgts:
            t0 = int(t["start"]) + 1
            t1 = int(t["end"]) + 1
            ids2 = ids0[:t0] + [1] + ids0[t0:t1 + 1] + [2] + ids0[t1 + 1:]
            if len(ids2) > max_len:
                ids2 = ids2[:max_len]
            ent.append((ids2, [t0, t1]))
        cache[i] = (ent, int(r["sentence_id"]), text)
    return cache


class RoleDataset(torch.utils.data.Dataset):
    def __init__(self, rows, roles, cache, expand_targets=False, overlap_mode="skip", f2i=None):
        self.items = []
        r2i = {r: i for i, r in enumerate(roles)}
        prev_ab = [0, 0]
        for i, r in enumerate(rows):
            ent, sid, text = cache[i]
            fid = (f2i or {}).get(r.get("frame"), -1)
            tgts = ent if expand_targets else [ent[-1]]
            for ids2, (t0, t1) in tgts:
                for sp in r.get("cfn_spans") or []:
                    s, e = int(sp["start"]), int(sp["end"])
                    ab = span_to_idx(s, e, t0, t1)
                    if ab is None:
                        if overlap_mode == "skip":
                            continue
                        ab = list(prev_ab)  # 官方 bug：复用上一个 span 的坐标
                    else:
                        prev_ab = ab
                    a, b = ab
                    if b >= len(ids2):
                        continue
                    self.items.append((ids2, [t0, t1], [a, b],
                                       r2i[sp["fe_name"]], sid, s, e, text, fid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class SpanDataset(torch.utils.data.Dataset):
    """部署口径训练集：外部候选 span（T2 预测）作为输入，未命中金标 span 记为 NULL。

    spans_obj: {"spans": [ [ [[s,e],...] per target ] per row ]}
    expand_targets=False 时只用最后一个 target（与官方一致）。
    neg_boost: 负例重复倍数（NULL 极少，需加权）。
    """

    def __init__(self, rows, roles, cache, spans_obj, expand_targets=False,
                 overlap_mode="skip", neg_boost=1, f2i=None):
        self.items = []
        r2i = {r: i for i, r in enumerate(roles)}
        null_idx = len(roles)
        sp_all = spans_obj["spans"]
        for i, r in enumerate(rows):
            ent, sid, text = cache[i]
            fid = (f2i or {}).get(r.get("frame"), -1)
            per = sp_all[i] if i < len(sp_all) else []
            tgt_sel = range(len(ent)) if expand_targets else [len(ent) - 1]
            gold = set()
            for sp in r.get("cfn_spans") or []:
                gold.add((int(sp["start"]), int(sp["end"]), str(sp["fe_name"])))
            prev_ab = [0, 0]
            for ti in tgt_sel:
                ids2, (t0, t1) = ent[ti]
                for s, e in per[ti] if ti < len(per) else []:
                    s, e = int(s), int(e)
                    ab = span_to_idx(s, e, t0, t1)
                    if ab is None:
                        if overlap_mode == "skip":
                            continue
                        ab = list(prev_ab)
                    else:
                        prev_ab = ab
                    a, b = ab
                    if b >= len(ids2):
                        continue
                    hit = [r2i[fe] for (gs, ge, fe) in gold if gs == s and ge == e]
                    if hit:
                        for lab in hit:
                            self.items.append((ids2, [t0, t1], [a, b], lab, sid, s, e, text, fid))
                    else:
                        for _ in range(max(1, neg_boost)):
                            self.items.append((ids2, [t0, t1], [a, b], null_idx, sid, s, e, text, fid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class InferDataset(torch.utils.data.Dataset):
    """推理集：rows(带 target) + span 列表。

    sid_mode='last' 与 'row' 的区别见模块文档；'last' 复现官方行为。
    span 元素可为 [sid,s,e] 或 dict{start,end,sid}。
    """

    def __init__(self, rows, spans, tokenizer, max_len=320, sid_mode="row",
                 overlap_mode="skip", cache=None):
        self.items = []
        self.skipped = 0
        self.truncated = 0
        self.prev_ab = [0, 0]
        if cache is None:
            cache = prep_rows(rows, tokenizer, max_len)
        self.cache = cache
        last_of = {}
        for i, r in enumerate(rows):
            last_of[int(r["sentence_id"])] = i

        def norm(sp):
            if isinstance(sp, dict):
                return int(sp.get("sid", sp.get("sentence_id"))), int(sp["start"]), int(sp["end"])
            return int(sp[0]), int(sp[1]), int(sp[2])

        if sid_mode == "last":
            for sp in spans:
                sid, s, e = norm(sp)
                i = last_of.get(sid)
                if i is None:
                    self.skipped += 1
                    continue
                self._add(i, sid, s, e, overlap_mode)
        elif sid_mode == "near":
            by_sid = {}
            for i, r in enumerate(rows):
                by_sid.setdefault(int(r["sentence_id"]), []).append(i)
            for sp in spans:
                sid, s, e = norm(sp)
                cands = by_sid.get(sid)
                if not cands:
                    self.skipped += 1
                    continue
                best = None
                for i in cands:
                    ent, _, _t = self.cache[i]
                    ids2, (t0, t1) = ent[-1]
                    if span_to_idx(s, e, t0, t1) is None:
                        continue
                    gap = (t0 - 1 - e) if e < t0 - 1 else (s - (t1 - 1))
                    key = (-max(gap, 0), -i)
                    if best is None or key > best[0]:
                        best = (key, i)
                if best is None:
                    self.skipped += 1
                    continue
                self._add(best[1], sid, s, e, overlap_mode)
        else:
            # 逐行分配：同一 sid 的多个行按出现顺序，span 归属于"能容纳它"的行
            by_sid = {}
            for i, r in enumerate(rows):
                by_sid.setdefault(int(r["sentence_id"]), []).append(i)
            cursor = {}
            for sp in spans:
                sid, s, e = norm(sp)
                cands = by_sid.get(sid)
                if not cands:
                    self.skipped += 1
                    continue
                if len(cands) == 1:
                    self._add(cands[0], sid, s, e, overlap_mode)
                    continue
                k = cursor.get(sid, 0)
                idx = cands[min(k, len(cands) - 1)]
                if k < len(cands) - 1:
                    cursor[sid] = k + 1
                self._add(idx, sid, s, e, overlap_mode)

    def _add(self, i, sid, s, e, overlap_mode):
        ent, _, text = self.cache[i]
        ids2, (t0, t1) = ent[-1]
        ab = span_to_idx(s, e, t0, t1)
        if ab is None:
            if overlap_mode == "skip":
                self.skipped += 1
                return
            ab = list(self.prev_ab)
        else:
            self.prev_ab = ab
        a, b = ab
        if b >= len(ids2):
            self.skipped += 1
            return
        if len(ids2) >= 320:
            self.truncated += 1
        self.items.append((ids2, [t0, t1], [a, b], sid, s, e, text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(data, device=None, with_labels=True, with_frame=False):
    bs = len(data)
    max_len = max(len(x[0]) for x in data)
    input_ids = np.zeros((bs, max_len), dtype=np.int64)
    attn = np.zeros((bs, max_len), dtype=np.int64)
    tgts, span_idx, sids, os_, oe_ = [], [], [], [], []
    labels = np.zeros((bs,), dtype=np.int64) if with_labels else None
    fids = np.zeros((bs,), dtype=np.int64) if with_frame else None
    for i, d in enumerate(data):
        L = len(d[0])
        input_ids[i, :L] = d[0]
        attn[i, :L] = 1
        tgts.append(d[1])
        span_idx.append(d[2])
        if with_labels:
            labels[i] = d[3]
            sids.append(d[4])
            os_.append(d[5])
            oe_.append(d[6])
            if with_frame:
                fids[i] = max(int(d[8]), 0)
        else:
            sids.append(d[3])
            os_.append(d[4])
            oe_.append(d[5])
    input_ids = torch.from_numpy(input_ids).to(device)
    attn = torch.from_numpy(attn).to(device)
    if with_labels:
        labels = torch.from_numpy(labels).to(device)
    if with_frame:
        return input_ids, attn, tgts, span_idx, labels, sids, os_, oe_, torch.from_numpy(fids).to(device)
    return input_ids, attn, tgts, span_idx, labels, sids, os_, oe_
