"""Cross-split consistency checks for the FrameNet 1.7 SyntaxNet release.

Checks: (1) are the three splits disjoint at sentence level, (2) does each block mark exactly
one target row, (3) how many dev/test frames and role labels never occur in train (the long-tail
and unseen-label counts the paper needs), (4) does the .sents line count match the block sentences,
(5) is the "target surface form -> frame name" prior computable and how ambiguous is it.
"""
import os
from collections import Counter, defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framenet_splits")
SPL = {"train": "fn1.7.fulltext.train.syntaxnet.conll",
       "dev": "fn1.7.dev.syntaxnet.conll",
       "test": "fn1.7.test.syntaxnet.conll"}


def blocks(name):
    path = os.path.join(D, SPL[name])
    out, cur = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                if cur:
                    out.append(cur)
                    cur = []
                continue
            p = s.split("\t")
            if len(p) >= 15:
                cur.append(p)
    if cur:
        out.append(cur)
    return out


def sent_key(b):
    return " ".join(r[3] for r in b)


def main():
    B = {k: blocks(k) for k in SPL}
    print("[blocks] " + " ".join(f"{k}={len(v):,}" for k, v in B.items()), flush=True)

    for k, v in B.items():
        bad = sum(1 for b in v if sum(1 for r in b if r[12] not in ("_", "")) != 1)
        badfr = sum(1 for b in v if sum(1 for r in b if r[13] not in ("_", "")) != 1)
        print(f"[target-rows] {k}: blocks whose target count != 1 -> {bad:,}; frame count != 1 -> {badfr:,}", flush=True)

    S = {k: set(sent_key(b) for b in v) for k, v in B.items()}
    for k in SPL:
        n = len(summarize := Counter(sent_key(b) for b in B[k]))
        print(f"[sentences] {k}: distinct {n:,}", flush=True)
    print(f"[overlap] train&dev {len(S['train'] & S['dev']):,}  train&test {len(S['train'] & S['test']):,}  "
          f"dev&test {len(S['dev'] & S['test']):,}", flush=True)

    F = {k: Counter(next(r[13] for r in b if r[13] not in ("_", "")) for b in v) for k, v in B.items()}
    R = {}
    for k, v in B.items():
        c = Counter()
        for b in v:
            for r in b:
                if r[14] != "O" and r[14] not in ("_", ""):
                    c[r[14].split("-", 1)[1] if "-" in r[14] else r[14]] += 1
        R[k] = c
    for k in ("dev", "test"):
        unseen_f = [f for f in F[k] if f not in F["train"]]
        tgt_unseen = sum(F[k][f] for f in unseen_f)
        unseen_r = [r for r in R[k] if r not in R["train"]]
        tok_unseen = sum(R[k][r] for r in unseen_r)
        print(f"[unseen vs train] {k}: frames unseen {len(unseen_f):,}/{len(F[k]):,} "
              f"covering {tgt_unseen:,}/{sum(F[k].values()):,} targets = {100.0*tgt_unseen/sum(F[k].values()):.2f} %; "
              f"roles unseen {len(unseen_r):,}/{len(R[k]):,} covering {tok_unseen:,}/{sum(R[k].values()):,} "
              f"labelled tokens = {100.0*tok_unseen/sum(R[k].values()):.2f} %", flush=True)

    sp = {"train": "fn1.7.fulltext.train.syntaxnet.conll.sents",
          "dev": "fn1.7.dev.syntaxnet.conll.sents",
          "test": "fn1.7.test.syntaxnet.conll.sents"}
    for k, v in sp.items():
        with open(os.path.join(D, v), encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
        print(f"[sents] {k}: {n:,} lines vs distinct sentences {len(S[k]):,}", flush=True)

    prior = defaultdict(Counter)
    tot = 0
    for b in B["train"]:
        row = next(r for r in b if r[12] not in ("_", ""))
        prior[row[12]][row[13]] += 1
        tot += 1
    amb = Counter(len(v) for v in prior.values())
    hit = 0
    for b in B["dev"]:
        row = next(r for r in b if r[12] not in ("_", ""))
        if row[12] in prior:
            hit += 1
    print(f"[prior word->frame] train entries {len(prior):,} over {tot:,} targets; "
          f"candidates per entry: max {max(amb)*0+max(amb) if False else max(amb)} median-ish buckets {sorted(amb.items())[:6]}", flush=True)
    print(f"[prior coverage on dev] dev targets whose target-string occurs in train: {hit:,}/{len(B['dev']):,} "
          f"= {100.0*hit/len(B['dev']):.2f} %", flush=True)


if __name__ == "__main__":
    main()
