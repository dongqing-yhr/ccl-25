"""Census of the FrameNet 1.7 split files (SyntaxNet release) now in datasets_paper/framenet_splits/.

Counts per split: blocks (=one target), sentences, distinct frames, distinct role labels,
targets whose every token is O (the natural "no role at all" pool), plus the label
frequency buckets used for the long-tail table. Read-only, stdlib only.
"""
import glob
import os
from collections import Counter

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framenet_splits")
SPLITS = {
    "train": "fn1.7.fulltext.train.syntaxnet.conll",
    "dev": "fn1.7.dev.syntaxnet.conll",
    "test": "fn1.7.test.syntaxnet.conll",
}


def blocks(path):
    blk, cur = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                if cur:
                    blk.append(cur)
                    cur = []
                continue
            p = s.split("\t")
            if len(p) >= 15:
                cur.append(p)
    if cur:
        blk.append(cur)
    return blk


def main():
    print(f"[dir] {D}", flush=True)
    print("[files] " + ", ".join(sorted(os.path.basename(x) for x in glob.glob(os.path.join(D, "fn1.7*")))), flush=True)
    for name, fn in SPLITS.items():
        path = os.path.join(D, fn)
        bs = blocks(path)
        n_rows = sum(len(b) for b in bs)
        frames = Counter()
        roles = Counter()
        no_role = 0
        tok_per_block = []
        for b in bs:
            fr = [r[13] for r in b if r[13] not in ("_", "")]
            if fr:
                frames[fr[0]] += 1
            rl = [r[14] for r in b if r[14] not in ("_", "")]
            lab = [x for x in rl if x != "O"]
            for x in rl:
                if x != "O":
                    roles[x.split("-", 1)[1] if "-" in x else x] += 1
            if not lab:
                no_role += 1
            tok_per_block.append(len(b))
        seen = {}
        for b in bs:
            key = " ".join(r[3] for r in b)
            seen[key] = seen.get(key, 0) + 1
        print(f"\n=== {name} ({fn})", flush=True)
        print(f"  blocks(targets) {len(bs):,}  token rows {n_rows:,}  avg tokens/target {n_rows/len(bs):.2f}", flush=True)
        print(f"  distinct sentences (token-joined) {len(seen):,}", flush=True)
        print(f"  distinct frame names {len(frames):,}  top5 {frames.most_common(5)}", flush=True)
        print(f"  distinct role labels {len(roles):,}  top5 {roles.most_common(5)}", flush=True)
        print(f"  targets with NO role labelled at all (all-O) {no_role:,} = {100.0*no_role/len(bs):.2f} % of targets", flush=True)
        c = Counter(frames.values())
        print(f"  frame freq buckets: 1 time {c[1]:,} frames, 2-4 {sum(v for k,v in c.items() if 2<=k<=4):,}, "
              f"5-19 {sum(v for k,v in c.items() if 5<=k<=19):,}, 20-99 {sum(v for k,v in c.items() if 20<=k<=99):,}, "
              f">=100 {sum(v for k,v in c.items() if k>=100):,}", flush=True)
        rc = Counter(roles.values())
        print(f"  role freq buckets: 1 time {rc[1]:,} roles, 2-9 {sum(v for k,v in rc.items() if 2<=k<=9):,}, "
              f"10-99 {sum(v for k,v in rc.items() if 10<=k<=99):,}, >=100 {sum(v for k,v in rc.items() if k>=100):,}", flush=True)


if __name__ == "__main__":
    main()
