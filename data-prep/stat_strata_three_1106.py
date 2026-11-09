#!/usr/bin/env python3
# Frequency-stratum table, identical definition for three resources.
# Strata are keyed on TRAIN-SPLIT frequency of a label; dev/test gold items are then bucketed.
import json, os, time
from collections import Counter

ROOT = r"C:/Users/Lenovo/Desktop/CCL_v2025"
BANDS = [(1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 19, "10-19"),
         (20, 49, "20-49"), (50, 99, "50-99"), (100, 10 ** 9, ">=100")]


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def band_of(n):
    if n == 0:
        return "0(未见)"
    for lo, hi, nm in BANDS:
        if lo <= n <= hi:
            return nm
    return "??"


def order():
    return [nm for _, _, nm in BANDS] + ["0(未见)"]


def cfn_units(path):
    out = []
    for rec in json.load(open(path, encoding="utf-8")):
        f = rec.get("frame", "")
        roles = [(sp.get("fe_name") or sp.get("fe_abbr") or "") for sp in rec.get("cfn_spans", [])]
        roles = [x for x in roles if x]
        out.append((f, roles))
    return out


def en_units(path):
    out = []
    block = []

    def flush(b):
        if not b:
            return None
        tags = [r[14] for r in b]
        runs = []
        prev = None
        for t in tags:
            if t in ("O", "_", ""):
                prev = None
                continue
            nm = t.split("-", 1)[1]
            if t[0] in ("S", "B") or nm != prev:
                runs.append(nm)
            prev = nm
        tgts = [r[13] for r in b if r[12] not in ("_", "")]
        return tgts, runs

    def assign(r):
        # declared rule: one FE tag column per block, so the gold runs belong to the
        # first target only; later targets of the same block carry no span gold.
        return [(f, r[1] if i == 0 else []) for i, f in enumerate(r[0])]

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                r = flush(block)
                if r:
                    out.extend(assign(r))
                block = []
                continue
            p = line.split("\t")
            if len(p) >= 15:
                block.append(p)
    r = flush(block)
    if r:
        out.extend(assign(r))
    return out


def table(title, train, evals):
    ft = Counter(f for f, _ in train if f)
    rt = Counter(x for _, rs in train for x in rs)
    log("== %s ==" % title)
    for space, tc in (("frame-name", ft), ("role-name", rt)):
        hdr = "  %-9s %-8s %-8s" % ("band", "labels", "train_n")
        for nm in evals:
            hdr += " %-16s" % nm
        log(hdr + "  <- %s" % space)
        for bn in order():
            if bn == "0(未见)":
                labels = 0
            else:
                lo, hi, _ = [b for b in BANDS if b[2] == bn][0]
                labels = sum(1 for v in tc.values() if lo <= v <= hi)
            trn = 0 if bn == "0(未见)" else sum(v for v in tc.values() if lo <= v <= hi)
            row = "  %-9s %-8d %-8d" % (bn, labels, trn)
            for nm, ev in evals.items():
                ec = Counter()
                for f, rs in ev:
                    items = [f] if space == "frame-name" else rs
                    for x in items:
                        if x:
                            ec[band_of(tc.get(x, 0))] += 1
                row += " %-16s" % ("%d (%.2f%%)" % (ec[bn], 100.0 * ec[bn] / max(1, sum(ec.values()))))
            log(row)


S = os.path.join
log("CFN 2.2")
t = cfn_units(S(ROOT, "dataset", "cfn-train.json"))
d = cfn_units(S(ROOT, "dataset", "cfn-dev.json"))
table("CFN 2.2 (train 10700 / dev 2300)", t, {"dev": d})
log("CFN 1.0")
t = cfn_units(S(ROOT, "datasets_paper", "cfn1.0", "cfn-train.json"))
d = cfn_units(S(ROOT, "datasets_paper", "cfn1.0", "cfn-dev.json"))
table("CFN 1.0 (train 10000 / dev 2000)", t, {"dev": d})
log("English fulltext")
E = S(ROOT, "datasets_paper", "framenet_splits")
t = en_units(S(E, "fn1.7.fulltext.train.syntaxnet.conll"))
dv = en_units(S(E, "fn1.7.dev.syntaxnet.conll"))
te = en_units(S(E, "fn1.7.test.syntaxnet.conll"))
log("english units train/dev/test = %d/%d/%d (self-check 20105/2382/6872)" % (len(t), len(dv), len(te)))
table("English (train 20105 / dev 2382 / test 6872)", t, {"dev": dv, "test": te})
log("done")
