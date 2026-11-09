"""Download the community-standard FrameNet 1.7 split files (SyntaxNet format) from the HF mirror.

Read-only fetch: verifies byte count against the HF API, prints per-file progress.
"""
import json
import os
import subprocess
import sys
import urllib.request

BASE = "https://hf-mirror.com/datasets/liyucheng/FrameNet_v17/resolve/main/"
REPO_API = "https://hf-mirror.com/api/datasets/liyucheng/FrameNet_v17?blobs=true"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framenet_splits")

FILES = [
    "fn1.7/fn1.7.fulltext.train.syntaxnet.conll",
    "fn1.7/fn1.7.fulltext.train.syntaxnet.conll.sents",
    "fn1.7/fn1.7.dev.syntaxnet.conll",
    "fn1.7/fn1.7.dev.syntaxnet.conll.sents",
    "fn1.7/fn1.7.test.syntaxnet.conll",
    "fn1.7/fn1.7.test.syntaxnet.conll.sents",
]


def expected_sizes():
    with urllib.request.urlopen(REPO_API, timeout=60) as r:
        meta = json.load(r)
    return {s["rfilename"]: s.get("size") or 0 for s in meta.get("siblings", [])}


def fetch(path, want):
    dst = os.path.join(OUT, os.path.basename(path))
    url = BASE + path
    print(f"[dl] {path}  expect {want:,} bytes", flush=True)
    cmd = ["curl", "-sSL", "--fail", "--retry", "3", "--retry-delay", "2",
           "-m", "900", "-A", "Mozilla/5.0", "-o", dst, url]
    subprocess.run(cmd, check=True)
    got = os.path.getsize(dst)
    ok = (got == want) if want else (got > 0)
    print(f"[chk] {os.path.basename(dst)}: got {got:,} bytes -> {'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def main():
    os.makedirs(OUT, exist_ok=True)
    sizes = expected_sizes()
    print(f"[api] repo has {len(sizes)} files", flush=True)
    bad = []
    for f in FILES:
        if not fetch(f, sizes.get(f, 0)):
            bad.append(f)
    print(f"[done] {len(FILES) - len(bad)}/{len(FILES)} files verified, dir={OUT}", flush=True)
    if bad:
        print("[FAIL] " + ", ".join(bad), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
