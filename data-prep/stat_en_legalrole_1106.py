#!/usr/bin/env python3
# Third-resource counterpart of the "legal role table tightness" comparison
# (requirement doc 7.x measures for CFN 1.0 / 2.2). Read-only.
import os, time
from collections import Counter, defaultdict

ROOT = r"C:/Users/Lenovo/Desktop/CCL_v2025"
E = os.path.join(ROOT, "datasets_paper", "framenet_splits")


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def units(path):
    out = []
    block = []

    def flush(b):
        if not b:
            return None
        tags = [r[14] for r in b]
        runs, prev = [], None
        for t in tags:
            if t in ("O", "_", ""):
                prev = None
                continue
            nm = t.split("-", 1)[1]
            if t[0] in ("S", "B") or nm != prev:
                runs.append(nm)
            prev = nm
        tg = [r[13] for r in b if r[12] not in ("_", "")]
        return [(f, runs if i == 0 else []) for i, f in enumerate(tg)]

    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line.strip():
            r = flush(block)
            if r:
                out.extend(r)
            block = []
            continue
        p = line.split("\t")
        if len(p) >= 15:
            block.append(p)
    r = flush(block)
    if r:
        out.extend(r)
    return out


tr = units(os.path.join(E, "fn1.7.fulltext.train.syntaxnet.conll"))
dv = units(os.path.join(E, "fn1.7.dev.syntaxnet.conll"))
te = units(os.path.join(E, "fn1.7.test.syntaxnet.conll"))
log("units train/dev/test = %d/%d/%d" % (len(tr), len(dv), len(te)))

pairs = set()
per_frame = defaultdict(set)
frames_in_train = set()
for f, rs in tr:
    if not f:
        continue
    frames_in_train.add(f)
    for x in rs:
        pairs.add((f, x))
        per_frame[f].add(x)

nfr = len(frames_in_train)
covered = sum(len(v) for v in per_frame.values())
log("train-derived: frames seen %d, (frame,role) pairs %d, avg allowed roles per seen frame %.2f"
    % (nfr, covered, covered / nfr))
frames_with_zero = len([f for f in frames_in_train if not per_frame[f]])
log("  of those, frames whose units carry no role at all: %d" % frames_with_zero)

for nm, ev in (("dev", dv), ("test", te)):
    tot = bad = 0
    bad_frames = 0
    for f, rs in ev:
        if not f:
            continue
        if f not in frames_in_train:
            bad_frames += 1
        for x in rs:
            tot += 1
            if (f, x) not in pairs:
                bad += 1
    print("  %s: gold spans %d, role not in that frame's train list %d = %.2f %% "
          "(units whose frame never appears in train: %d)" % (nm, tot, bad, 100.0 * bad / tot, bad_frames), flush=True)

log("distinct roles in train %d" % len(Counter(x for _, rs in tr for x in rs)))
log("done")
