#!/usr/bin/env python3
# Parameterized version of runbash/fuse3_AB.py's arithmetic, for building submissions on top of a
# swapped task2 span file. Kept intentionally identical to fuse3_AB.py (same label table from
# frame_info.json, same p_null > tau abstain, same argmax, same frame-legal-role veto, same output
# ordering) so that re-running it on the ORIGINAL B sidecars must reproduce
# dataset/B/B_task3_fn3_t0.08_veto.json byte-for-byte -- the driver asserts exactly that before
# spending any GPU time.
#
# Sidecar row = [sentence_id, span_start, span_end, p_role_0 ... p_role_1008, p_null].
import json, codecs, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--oof', required=True)
ap.add_argument('--la', required=True)
ap.add_argument('--fn', required=True)
ap.add_argument('--weights', default='0.25,0.25,0.50', help='oof,la,fn probability-average weights')
ap.add_argument('--tau', type=float, default=0.08)
ap.add_argument('--gate', default='veto', choices=['veto', 'plain'])
ap.add_argument('--veto_conf', type=float, default=None,
                help='when set, a span whose role is outside the frame-legal list is still emitted if that role probability is >= this value; unset keeps the original hard veto')
ap.add_argument('--task1', required=True)
ap.add_argument('--frame_info', default='./dataset/frame_info.json')
ap.add_argument('--frame_roles', default='./dataset/frame_roles.json')
ap.add_argument('--out', required=True)
ap.add_argument('--label', default='')
A = ap.parse_args()

W = [float(x) for x in A.weights.split(',')]
assert len(W) == 3 and abs(sum(W) - 1.0) < 1e-9, 'weights must be 3 values summing to 1, got %s' % W
idx2label = []
for line in json.load(codecs.open(A.frame_info, 'r', encoding='utf8')):
    for fes in line['fes']:
        if fes['fe_name'] not in idx2label:
            idx2label.append(fes['fe_name'])

P, K = {}, {}
for nm, w, path in zip(('oof', 'la', 'fn'), W, (A.oof, A.la, A.fn)):
    rows = json.load(codecs.open(path, 'r', encoding='utf8'))
    K[nm] = [(r[0], r[1], r[2]) for r in rows]
    P[nm] = np.array([r[3:] for r in rows], dtype=np.float64)
    print('[load] %-4s w=%.2f  %s  rows=%d cols=%d' % (nm, w, path, len(rows), P[nm].shape[1]))
if not (K['oof'] == K['la'] == K['fn']):
    raise SystemExit('[FATAL] the three sidecars do not line up row-by-row -> they were not decoded '
                     'from the same task2 span file; refusing to average')
NL = P['oof'].shape[1] - 1
if NL != len(idx2label):
    raise SystemExit('[FATAL] prob dim %d != role table %d' % (NL, len(idx2label)))

allowed = {}
for sid, frame in json.load(codecs.open(A.task1, 'r', encoding='utf8')):
    allowed.setdefault(int(sid), set()).update(json.load(codecs.open(A.frame_roles, 'r', encoding='utf8')).get(frame, []))

Pm = W[0] * P['oof'] + W[1] * P['la'] + W[2] * P['fn']
out = []
n_null = n_veto = n_resc = 0
for i, key in enumerate(K['oof']):
    row = Pm[i]
    if row[NL] > A.tau:
        n_null += 1
        continue
    conf = float(np.max(row[:NL]))
    name = idx2label[int(np.argmax(row[:NL]))]
    if A.gate == 'veto' and (key[0] in allowed) and (name not in allowed[key[0]]):
        if A.veto_conf is not None and conf >= A.veto_conf:
            n_resc += 1
        else:
            n_veto += 1
            continue
    out.append([key[0], key[1], key[2], name])
with open(A.out, 'w', encoding='utf8', newline='\n') as fh:
    fh.write(json.dumps(out, indent=1, ensure_ascii=False))
print('[fuse3] %s tau=%.2f gate=%s candidates=%d abstain=%d vetoed=%d kept=%d -> %s'
      % (A.label, A.tau, A.gate, len(K['oof']), n_null, n_veto, len(out), A.out))
print('[fuse3] veto_conf=%s rescued_over_illegal=%d' % (A.veto_conf, n_resc))
print('FUSE3_GENERIC_DONE')
