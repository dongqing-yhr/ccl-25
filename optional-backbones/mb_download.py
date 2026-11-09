#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 hf-mirror 拉 ModernBERT-large 中文 wwm 底座到 baseline/models/ 下，带进度与大小校验。"""
import json
import os
import sys
import time
import urllib.request

REPO = "feynmanzhao/chinese-modernbert-large-wwm"
DST = "/work/workspace/ccl25/baseline/models/feynmanzhao--chinese-modernbert-large-wwm"
WANT = ["config.json", "model.safetensors", "pytorch_model.bin",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.txt", "vocab.json", "merges.txt"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    os.makedirs(DST, exist_ok=True)
    api = "https://hf-mirror.com/api/models/" + REPO
    d = json.load(urllib.request.urlopen(api, timeout=20))
    have = {s["rfilename"] for s in d.get("siblings", [])}
    log("repo %s 共 %d 个文件" % (REPO, len(have)))
    got = []
    for name in WANT:
        if name not in have:
            log("  跳过 %s（仓库里没有）" % name)
            continue
        url = "https://hf-mirror.com/%s/resolve/main/%s" % (REPO, name)
        out = os.path.join(DST, name)
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            log("  已有 %s（%d 字节），跳过" % (name, os.path.getsize(out)))
            got.append(name)
            continue
        ok = False
        for attempt in range(1, 6):
            try:
                log("  下载 %s（第 %d 次）..." % (name, attempt))
                t0 = time.time()
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) huggingface-hub/0.24.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(out + ".tmp", "wb") as f:
                    n = 0
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        n += len(chunk)
                        if n % (100 << 20) < (1 << 22):
                            log("    %.0f MB / %.0f s" % (n / 1e6, time.time() - t0))
                os.replace(out + ".tmp", out)
                log("  完成 %s（%d 字节，%.0f s）" % (name, os.path.getsize(out), time.time() - t0))
                ok = True
                break
            except Exception as e:
                log("  第 %d 次失败：%s: %s；10 秒后重试" % (attempt, type(e).__name__, str(e)[:120]))
                time.sleep(10)
        if not ok:
            sys.exit("[FATAL] %s 连试 5 次都失败" % name)
        got.append(name)
    for must in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if must not in got:
            sys.exit("[FATAL] 缺关键文件 %s" % must)
    if "model.safetensors" not in got and "pytorch_model.bin" not in got:
        sys.exit("[FATAL] 权重文件一个都没拿到")
    log("MB_DOWNLOAD_OK " + " ".join(sorted(got)))


main()
