# -*- coding: utf-8 -*-
"""官方 T3 口径评分 + dev 辅助。"""
import json


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def gold_quads(rows):
    s = set()
    for d in rows:
        sid = int(d["sentence_id"])
        for sp in d.get("cfn_spans") or []:
            s.add((sid, int(sp["start"]), int(sp["end"]), str(sp["fe_name"])))
    return s


def gold_span_map(rows):
    out = {}
    for d in rows:
        out.setdefault(int(d["sentence_id"]), set()).update(
            (int(sp["start"]), int(sp["end"])) for sp in d.get("cfn_spans") or [])
    return out


def span_map(raw):
    out = {}
    if isinstance(raw, dict):
        for sid, items in raw.items():
            for x in items or []:
                out.setdefault(int(sid), set()).add((int(x[0]), int(x[1])))
    else:
        for r in raw:
            out.setdefault(int(r[0]), set()).add((int(r[1]), int(r[2])))
    return out


def set_prf(ps, gs):
    inter = len(ps & gs)
    p = inter / max(len(ps), 1)
    r = inter / max(len(gs), 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def token_prf(pred, gold):
    inter = npred = ngold = 0
    for k in set(pred) | set(gold):
        gt, pt = set(), set()
        for s, e in gold.get(k, ()):
            gt.update(range(s, e + 1))
        for s, e in pred.get(k, ()):
            pt.update(range(s, e + 1))
        inter += len(gt & pt)
        npred += len(pt)
        ngold += len(gt)
    p = inter / max(npred, 1)
    r = inter / max(ngold, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def score_t3(pred_rows, dev_rows):
    ps = set((int(r[0]), int(r[1]), int(r[2]), str(r[3])) for r in pred_rows)
    return set_prf(ps, gold_quads(dev_rows))


def score_all(t3_pred, dev_rows, t2_pred=None):
    p, r, f = score_t3(t3_pred, dev_rows)
    out = {"task3_p": round(p, 5), "task3_r": round(r, 5), "task3_f1": round(f, 5)}
    if t2_pred is not None:
        p2, r2, f2 = token_prf(span_map(t2_pred), gold_span_map(dev_rows))
        out.update({"task2_p": round(p2, 5), "task2_r": round(r2, 5), "task2_f1": round(f2, 5)})
    return out
