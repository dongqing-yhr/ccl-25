# -*- coding: utf-8 -*-
"""Read-only census: which splits exist locally, and which of them carry gold labels.
English column layout (SyntaxNet/Open Sesame CoNLL-like): 12=target marker, 13=frame name, 14=FE tag.
"""
import json, os, time, hashlib

ROOT = r"C:/Users/Lenovo/Desktop/CCL_v2025"
print("[start]", time.strftime("%H:%M:%S"), flush=True)


def cfn(path):
    d = json.load(open(path, encoding="utf-8"))
    sents = set()
    f_ok = spans = span_role = tgt_ok = 0
    for r in d:
        sents.add(r.get("sentence_id"))
        if r.get("frame"):
            f_ok += 1
        t = r.get("target")
        if t:
            tgt_ok += 1
        for s in r.get("cfn_spans", []) or []:
            spans += 1
            if (s.get("fe_name") or s.get("fe_abbr")):
                span_role += 1
    print("  %-24s rec=%6d uniq_sent=%6d frame=%6d span=%7d span_role=%7d"
          % (os.path.basename(path), len(d), len(sents), f_ok, spans, span_role), flush=True)


def en(path):
    units = lab_units = spans = blocks = tgts_total = 0
    frames = set()
    roles = set()
    sents = set()
    block = []

    def flush(b):
        nonlocal units, lab_units, spans, blocks, tgts_total
        if not b:
            return
        blocks += 1
        tgts = [r[13] for r in b if r[12] not in ("_", "")]
        tgts_total += len(tgts)
        units += len(tgts)
        for x in tgts:
            if x:
                frames.add(x)
        runs = 0
        prev = None
        for r in b:
            t = r[14]
            if t in ("O", "_", ""):
                prev = None
                continue
            nm = t.split("-", 1)[1]
            roles.add(nm)
            if t[0] in ("S", "B") or nm != prev:
                runs += 1
            prev = nm
        spans += runs
        if runs:
            lab_units += 1
        sents.add(hashlib.md5(" ".join(x[1] for x in b).encode("utf-8")).hexdigest())

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                flush(block)
                block = []
                continue
            p = line.split("\t")
            if len(p) >= 15:
                block.append(p)
    flush(block)
    print("  %-40s block=%6d unit=%6d block_no_arg=%6d frames=%4d roles=%4d arg_span=%7d"
          % (os.path.basename(path), blocks, units, blocks - lab_units, len(frames), len(roles), spans), flush=True)
    return sents

print("CFN 2.2 (competition resource)")
for f in ["cfn-train.json", "cfn-dev.json", "cfn-test-A.json", "cfn-test-B.json"]:
    p = os.path.join(ROOT, "dataset", f)
    if os.path.exists(p):
        cfn(p)

print("CFN 1.0")
for f in ["cfn-train.json", "cfn-dev.json", "cfn-test-A.json", "cfn-test-shutanzzz.json"]:
    p = os.path.join(ROOT, "datasets_paper/cfn1.0", f)
    if os.path.exists(p):
        cfn(p)

print("English FrameNet 1.7 (three fixed splits)")
E = os.path.join(ROOT, "datasets_paper/framenet_splits")
s = {}
for tag, f in [("train", "fn1.7.fulltext.train.syntaxnet.conll"),
               ("dev", "fn1.7.dev.syntaxnet.conll"),
               ("test", "fn1.7.test.syntaxnet.conll")]:
    s[tag] = en(os.path.join(E, f))

print("english sentence-level overlap (blocks sharing token string):")
for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
    inter = len(s[a] & s[b])
    print("  %s=%d %s=%d inter=%d  (%.2f%% of %s)"
          % (a, len(s[a]), b, len(s[b]), inter, 100.0 * inter / max(1, len(s[b])), b), flush=True)
print("[done]", time.strftime("%H:%M:%S"), flush=True)
