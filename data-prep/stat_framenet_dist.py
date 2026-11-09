import re, glob, statistics
from collections import Counter

root = 'framenet_v17'

# --- frame-level: annotations per frame, from LU headers ---
re_lu = re.compile(r'<lexUnit\b[^>]*?frame="([^"]+)"[^>]*?totalAnnotated="(\d+)"', re.I)
frame_ann = Counter()
n_lu = 0
for fp in glob.glob(root + '/lu/lu*.xml'):
    txt = open(fp, encoding='utf-8', errors='ignore').read(3000)
    m = re_lu.search(txt)
    if not m:
        continue
    n_lu += 1
    frame_ann[m.group(1)] += int(m.group(2))

vals = sorted(frame_ann.values(), reverse=True)
tot = sum(vals)
print('lexical units parsed:', n_lu)
print('frames with >=1 annotation:', len(vals), '| defined frames:', len(glob.glob(root + '/frame/*.xml')))
print('annotated sentences (frame-annot pairs):', tot)
print('per-frame: max %d min %d mean %.1f median %s' % (vals[0], vals[-1], tot / len(vals), statistics.median(vals)))
for th in (100, 50, 20, 10, 5, 1):
    n = sum(1 for v in vals if v <= th)
    print('  frames <=%3d annotations: %d (%.1f%%)' % (th, n, 100.0 * n / len(vals)))
k = max(1, int(round(0.01 * len(vals))))
print('  top %d frames hold %.1f%%' % (k, 100 * sum(vals[:k]) / tot))
k = max(1, int(round(0.50 * len(vals))))
print('  top %d frames hold %.1f%%' % (k, 100 * sum(vals[:k]) / tot))
print('  top:', frame_ann.most_common(6))

# --- FE-level: core FE tag occurrences across all LU annotation files ---
re_fe = re.compile(r'<feTag\b[^>]*?name="([^"]+)"', re.I)
re_note = re.compile(r'<FE\b[^>]*?name="([^"]+)"', re.I)
fe_defined = Counter()
for fp in glob.glob(root + '/frame/*.xml'):
    for nm in re_note.findall(open(fp, encoding='utf-8', errors='ignore').read()):
        fe_defined[nm] += 1

re_tag = re.compile(r'<tagset\b[^>]*?name="([^"]+)"', re.I)
re_annot = re.compile(r'<annotationSet\b.*?</annotationSet>', re.I | re.S)
fe_ann = Counter()
n_sent = 0
lur = sorted(glob.glob(root + '/lu/lu*.xml'))
for i, fp in enumerate(lur):
    txt = open(fp, encoding='utf-8', errors='ignore').read()
    for blk in re_annot.findall(txt):
        n_sent += 1
        for nm in re_tag.findall(blk):
            if nm not in ('Target', 'SelfEvaluator', 'Passive', '-', 'Positive', 'Negative'):
                fe_ann[nm] += 1
    if i % 3000 == 0:
        print('   ...lu %d/%d' % (i, len(lur)), flush=True)

rv = sorted(fe_ann.values(), reverse=True)
rt = sum(rv)
print('annotationSets counted:', n_sent)
print('FE names defined:', len(fe_defined), '| FE names tagged:', len(rv))
print('FE tag occurrences:', rt, '| max %d min %d median %s' % (rv[0], rv[-1], statistics.median(rv)))
for th in (1000, 100, 50, 20, 10, 5, 1):
    n = sum(1 for v in rv if v <= th)
    print('  FE <=%5d occurrences: %d (%.1f%%)' % (th, n, 100.0 * n / len(rv)))
k = max(1, int(round(0.01 * len(rv))))
print('  top %d FE hold %.1f%%' % (k, 100 * sum(rv[:k]) / rt))
k = max(1, int(round(0.50 * len(rv))))
print('  top %d FE hold %.1f%%' % (k, 100 * sum(rv[:k]) / rt))
print('  top FE:', fe_ann.most_common(8))
