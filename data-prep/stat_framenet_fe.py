import re, glob, statistics
from collections import Counter

re_ann = re.compile(r'<annotationSet\b(?:(?!</annotationSet>).)*?</annotationSet>', re.I | re.S)
re_fe_layer = re.compile(r'<layer\b[^>]*name="FE"[^>]*>(.*?)</layer>', re.I | re.S)
re_label = re.compile(r'<label\b[^>]*?name="([^"]+)"', re.I)

fe = Counter()
manual = 0
lur = sorted(glob.glob('framenet_v17/lu/lu*.xml'))
for i, fp in enumerate(lur):
    txt = open(fp, encoding='utf-8', errors='ignore').read()
    for blk in re_ann.findall(txt):
        if 'status="MANUAL"' not in blk:
            continue
        manual += 1
        for lay in re_fe_layer.findall(blk):
            for nm in re_label.findall(lay):
                fe[nm] += 1
    if i % 3000 == 0:
        print('   ...lu %d/%d' % (i, len(lur)), flush=True)

rv = sorted(fe.values(), reverse=True)
rt = sum(rv)
print('MANUAL annotationSets:', manual)
print('distinct FE names tagged:', len(rv), '| FE tag occurrences:', rt)
print('max %d min %d mean %.1f median %s' % (rv[0], rv[-1], rt / len(rv), statistics.median(rv)))
for th in (1000, 100, 50, 20, 10, 5, 1):
    n = sum(1 for v in rv if v <= th)
    print('  FE <=%5d occurrences: %d (%.1f%%)' % (th, n, 100.0 * n / len(rv)))
for f in (0.01, 0.50):
    k = max(1, int(round(f * len(rv))))
    print('  top %d FE hold %.1f%%' % (k, 100 * sum(rv[:k]) / rt))
print('top FE:', fe.most_common(8))
print('tail FE:', list(fe.items())[-6:])
