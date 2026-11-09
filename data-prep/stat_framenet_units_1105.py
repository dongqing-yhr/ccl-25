"""Final clean census on TARGET units (sentence, target) for the FrameNet 1.7 split release.

Unit = one (sentence, target) pair, i.e. one annotated frame evocation. A block may carry
several targets. Reports: units, frame-name counts, role-name counts, span counts, the
no-role-at-all pool, label-frequency buckets, and the target-string -> frame-name prior size.
Stdlib only, read-only.
"""
import os
from collections import Counter, defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framenet_splits")
SPL = {"train": "fn1.7.fulltext.train.syntaxnet.conll",
       "dev": "fn1.7.dev.syntaxnet.conll",
       "test": "fn1.7.test.syntaxnet.conll"}


def blocks(name):
    out, cur = [], []
    with open(os.path.join(D, SPL[name]), encoding="utf-8", errors="replace") as f:
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


def spans(tokens_tags):
    """Count contiguous labelled spans from the B/I/S tag column."""
    n, i, cur = 0, 0, None
    while i < len(tokens_tags):
        t = tokens_tags[i]
        if t == "O" or t in ("_", ""):
            i += 1
            continue
        if t.startswith("S-"):
            n += 1
            i += 1
        elif t.startswith("B-"):
            n += 1
            j = i + 1
            while j < len(tokens_tags) and tokens_tags[j].startswith("I-"):
                j += 1
            i = j
        else:
            i += 1
    return n


def main():
    for k in SPL:
        B = blocks(k)
        units = 0
        fr = Counter()
        ro = Counter()
        norole = 0
        nspans = 0
        span_len = Counter()
        for b in B:
            tags = [r[14] for r in b]
            nspans += spans(tags)
            lab = [x for x in tags if x not in ("O", "_", "")]
            trows = [r for r in b if r[12] not in ("_", "")]
            for tr in trows:
                units += 1
                fr[tr[13]] += 1
                if not lab:
                    norole += 1
            for x in lab:
                ro[x.split("-", 1)[1]] += 1
        fbucket = Counter(fr.values())
        rbucket = Counter(ro.values())
        prior = defaultdict(Counter)
        for b in B:
            for tr in b:
                if tr[12] not in ("_", ""):
                    prior[tr[3]][tr[13]] += 1
        pb = Counter(len(v) for v in prior.values())
        multi = sum(v for c, v in pb.items() if c >= 2)
        print(f"\n=== {k}", flush=True)
        print(f"  units(sentence+target) {units:,}  sentences {len(B):,}", flush=True)
        print(f"  frame names {len(fr):,}  top5 {fr.most_common(5)}", flush=True)
        print(f"  role names {len(ro):,}  labelled tokens {sum(ro.values()):,}  top5 {ro.most_common(5)}", flush=True)
        print(f"  role spans {nspans:,}  units with no role at all {norole:,} = {100.0*norole/units:.2f} %", flush=True)
        print(f"  frame freq: 1x {fbucket[1]:,}  2-4 {sum(v for c,v in fbucket.items() if 2<=c<=4):,}  "
              f"5-19 {sum(v for c,v in fbucket.items() if 5<=c<=19):,}  20-99 {sum(v for c,v in fbucket.items() if 20<=c<=99):,}  "
              f">=100 {sum(v for c,v in fbucket.items() if c>=100):,}", flush=True)
        print(f"  role freq: 1x {rbucket[1]:,}  2-9 {sum(v for c,v in rbucket.items() if 2<=c<=9):,}  "
              f"10-99 {sum(v for c,v in rbucket.items() if 10<=c<=99):,}  >=100 {sum(v for c,v in rbucket.items() if c>=100):,}", flush=True)
        print(f"  prior entries(target surface) {len(prior):,}  single-candidate {pb[1]:,} = {100.0*pb[1]/len(prior):.2f} %  "
              f"ambiguous(>=2) {multi:,}  max candidates {max(pb)}", flush=True)


if __name__ == "__main__":
    main()
