import json, statistics, hashlib, os
from collections import Counter

D = 'datasets_paper/cfn1.0/'
FILES = ['cfn-train.json', 'cfn-dev.json', 'cfn-test-A.json', 'cfn-test-shutanzzz.json']

for f in FILES:
    d = json.load(open(D + f, encoding='utf-8'))
    sid = set(x['sentence_id'] for x in d)
    nspan = sum(len(x.get('cfn_spans') or []) for x in d)
    print('%-26s records %6d | distinct sentence_id %6d | spans %6d | md5 %s'
          % (f, len(d), len(sid), nspan, hashlib.md5(open(D + f, 'rb').read()).hexdigest()[:12]))

tr = json.load(open(D + 'cfn-train.json', encoding='utf-8'))
fi = json.load(open(D + 'frame_info.json', encoding='utf-8'))
print('frame_info:', type(fi).__name__, len(fi))

fc = Counter(x['frame'] for x in tr)
tot = sum(fc.values())
vals = sorted(fc.values(), reverse=True)
print('distinct frames in train: %d (of %d defined) | annotations %d' % (len(fc), len(fi), tot))
print('per-frame: max %d min %d mean %.1f median %s' % (vals[0], vals[-1], tot / len(vals), statistics.median(vals)))
for th in (100, 50, 20, 10, 5, 1):
    n = sum(1 for v in vals if v <= th)
    print('  frames <=%3d: %d (%.1f%%)' % (th, n, 100.0 * n / len(vals)))
for fr_ in (0.01, 0.50):
    k = max(1, int(round(fr_ * len(vals))))
    print('  top %d frames hold %.1f%%' % (k, 100 * sum(vals[:k]) / tot))
print('top frames:', fc.most_common(8))

rc = Counter()
nsp = 0
for x in tr:
    for s in (x.get('cfn_spans') or []):
        rc[s['fe_name']] += 1
        nsp += 1
rv = sorted(rc.values(), reverse=True)
rt = sum(rv)
print('spans %d | distinct fe_name %d' % (nsp, len(rc)))
print('per-role: max %d min %d mean %.1f median %s' % (rv[0], rv[-1], rt / len(rv), statistics.median(rv)))
for th in (1000, 100, 50, 20, 10, 5, 1):
    n = sum(1 for v in rv if v <= th)
    print('  roles <=%5d: %d (%.1f%%)' % (th, n, 100.0 * n / len(rv)))
for fr_ in (0.01, 0.50):
    k = max(1, int(round(fr_ * len(rv))))
    print('  top %d roles hold %.1f%%' % (k, 100 * sum(rv[:k]) / rt))
print('top roles:', rc.most_common(8))
print('NULL present:', rc.get('NULL'))
