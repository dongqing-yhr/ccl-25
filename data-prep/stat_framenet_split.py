import re, glob
from collections import Counter

root = 'framenet_v17'
re_open = re.compile(r'<annotationSet\b', re.I)
re_close = '</annotationSet>'
re_status = re.compile(r'<annotationSet\b[^>]*?status="([^"]+)"', re.I)
re_tagset = re.compile(r'<tagset\b[^>]*?name="([^"]+)"', re.I)
SKIP = {'Target', 'SelfEvaluator', 'Passive', '-', 'Positive', 'Negative'}

n_block = Counter(); n_fe = Counter(); n_sentref = Counter()
files = sorted(glob.glob(root + '/lu/lu*.xml'))
for i, fp in enumerate(files):
    txt = open(fp, encoding='utf-8', errors='ignore').read()
    for m in re_open.finditer(txt):
        end = txt.find(re_close, m.end())
        blk = txt[m.start():end] if end != -1 else txt[m.start():m.start() + 4000]
        sm = re_status.search(blk)
        status = sm.group(1) if sm else 'NO_STATUS'
        n_block[status] += 1
        if any(nm not in SKIP for nm in re_tagset.findall(blk)):
            n_fe[status] += 1
        if '<sentence ' in blk:
            n_sentref[status] += 1
    if i % 3000 == 0:
        print('  ...lu %d/%d' % (i, len(files)), flush=True)

tot = sum(n_block.values())
print('lu files parsed:', len(files), '| annotationSets:', tot)
print('%-16s %9s %8s %9s %10s' % ('status', 'blocks', 'share%', 'withFE', 'withSent'))
for k, v in n_block.most_common():
    print('%-16s %9d %8.2f %9d %10d' % (k, v, 100.0 * v / tot, n_fe[k], n_sentref[k]))
