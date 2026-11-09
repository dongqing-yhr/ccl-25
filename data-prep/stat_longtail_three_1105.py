#!/usr/bin/env python3
# Long-tail census on the TRAIN split of three resources, same definition for all three.
import json, os, sys, time, statistics

ROOT = r"C:/Users/Lenovo/Desktop/CCL_v2025"
BUCKETS = [100, 50, 20, 10, 5, 2, 1]


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def cfn_train(path):
    fr, ro = {}, {}
    n_units = n_spans = 0
    data = json.load(open(path, encoding="utf-8"))
    for rec in data:
        n_units += 1
        f = rec.get("frame", "")
        if f:
            fr[f] = fr.get(f, 0) + 1
        for sp in rec.get("cfn_spans", []):
            nm = sp.get("fe_name") or sp.get("fe_abbr") or ""
            if not nm:
                continue
            n_spans += 1
            ro[nm] = ro.get(nm, 0) + 1
    return n_units, n_spans, fr, ro


def en_train(path):
    fr, ro = {}, {}
    n_units = n_spans = 0
    block = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                n_units, n_spans, fr, ro = en_block(block, fr, ro, n_units, n_spans, n_spans)
                block = []
                continue
            r = line.split("\t")
            if len(r) >= 15:
                block.append(r)
    n_units, n_spans, fr, ro = en_block(block, fr, ro, n_units, n_spans, n_spans)
    return n_units, n_spans, fr, ro


def en_block(b, fr, ro, n_units, n_spans, _):
    if not b:
        return n_units, n_spans, fr, ro
    prev = None
    for r in b:
        tag = r[14]
        if tag in ("O", "_", ""):
            prev = None
            continue
        nm = tag.split("-", 1)[1]
        if tag[0] in ("S", "B") or nm != prev:
            n_spans += 1
            ro[nm] = ro.get(nm, 0) + 1
        prev = nm
    for r in b:
        if r[12] not in ("_", ""):
            n_units += 1
            f = r[13]
            if f and f not in ("_", ""):
                fr[f] = fr.get(f, 0) + 1
    return n_units, n_spans, fr, ro


def profile(name, counts, total_spans):
    vals = sorted(counts.values(), reverse=True)
    n = len(vals)
    if not n:
        return
    log("--- %s ---" % name)
    log("  distinct labels %d, occurrences %d" % (n, sum(vals)))
    for b in BUCKETS:
        k = sum(1 for v in vals if v <= b)
        log("    <=%4d : %5d labels = %6.2f %%" % (b, k, 100.0 * k / n))
    log("    median %s, max %s, singleton-occ share %.2f %%"
        % (statistics.median(vals), vals[0], 100.0 * vals[-1] / sum(vals)))
    t1 = max(1, int(round(n * 0.01)))
    t50 = max(1, int(round(n * 0.50)))
    log("    head: top 1%% (%d labels) = %.2f %%, top 50%% (%d) = %.2f %%"
        % (t1, 100.0 * sum(vals[:t1]) / sum(vals), t50, 100.0 * sum(vals[:t50]) / sum(vals)))
    top = sorted(counts.items(), key=lambda x: -x[1])[:3]
    log("    largest: %s" % ", ".join("%s %d" % (k, v) for k, v in top))


log("loading CFN 2.2 train")
u, s, f22, r22 = cfn_train(os.path.join(ROOT, "dataset", "cfn-train.json"))
log("CFN2.2 units=%d spans=%d frames=%d roles=%d" % (u, s, len(f22), len(r22)))
profile("CFN2.2 frame-name", f22, s)
profile("CFN2.2 role-name", r22, s)

log("loading CFN 1.0 train")
u, s, f10, r10 = cfn_train(os.path.join(ROOT, "datasets_paper", "cfn1.0", "cfn-train.json"))
log("CFN1.0 units=%d spans=%d frames=%d roles=%d" % (u, s, len(f10), len(r10)))
profile("CFN1.0 frame-name", f10, s)
profile("CFN1.0 role-name", r10, s)

log("loading English fulltext train")
p = os.path.join(ROOT, "datasets_paper", "framenet_splits", "fn1.7.fulltext.train.syntaxnet.conll")
u, s, fe, re_ = en_train(p)
log("ENGLISH units=%d spans=%d frames=%d roles=%d  (self-check 20105/34219/753/765)" % (u, s, len(fe), len(re_)))
profile("ENGLISH frame-name", fe, s)
profile("ENGLISH role-name", re_, s)
log("done")
