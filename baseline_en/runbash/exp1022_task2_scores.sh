#!/usr/bin/env bash
# exp1022_task2_scores.sh -- decode BOTH task2 checkpoints once at a low logit floor and dump the
# span-head score for every emitted span, so the whole fusion parameter space (each checkpoint's own
# threshold, union/intersection/add rules, a probability-weighted head) can be swept on CPU afterwards.
#
# Nothing here scores or fuses anything, and nothing touches a production slot: every output goes to
# dataset/<split>/cmp1022/ via --out_name. task3 stays at the champion recipe and is run separately.
#
#   cd /work/workspace/ccl25/baseline && bash runbash/exp1022_task2_scores.sh dev      # ~6 min GPU
#   bash runbash/exp1022_task2_scores.sh A                                             # ~12 min GPU
#   bash runbash/exp1022_task2_scores.sh B                                             # ~12 min GPU
#   FLOOR=-3.5 bash runbash/exp1022_task2_scores.sh dev
#
# Proofs built in, each one aborts the run:
#  - NEUTRAL: for each checkpoint the decode at the production threshold -0.5, with the patch loaded,
#    must be byte-identical to the file the incumbent recipe was built from. A non-neutral patch
#    would silently move AI, so this is checked before the floor decodes are believed.
#  - CONTAINMENT: the floor sidecar filtered back at -0.5 must contain exactly the -0.5 span set.
set -uo pipefail
cd /work/workspace/ccl25/baseline || exit 1

SPLIT=${1:-}
case "$SPLIT" in
    dev) TEST=./dataset/cfn-dev.json
         OUTDIR=dataset/dev/cmp1022
         BASE2_PRIM=dataset/dev/dev_task2_test.json
         BASE2_AUX=dataset/dev/cmp1021/new_m05_r0.json ;;
    A)   TEST=./dataset/cfn-test-A.json
         OUTDIR=dataset/A/cmp1022
         BASE2_PRIM=dataset/A/A_task2_test_union_all_bak.json
         BASE2_AUX=dataset/A/A_task2_aux_test.json ;;
    B)   TEST=./dataset/cfn-test-B.json
         OUTDIR=dataset/B/cmp1022
         BASE2_PRIM=dataset/B/B_task2_test.json
         BASE2_AUX=dataset/B/B_task2_aux_test.json ;;
    *)   echo "[FATAL] usage: bash runbash/exp1022_task2_scores.sh <dev|A|B>" >&2; exit 1 ;;
esac

LF=logfile/exp1022_task2
MC=./chinese-macbert-large/config.json
MV=./chinese-macbert-large/vocab.txt
RC=./chinese-roberta-wwm-ext-large/config.json
RV=./chinese-roberta-wwm-ext-large/vocab.txt
PRIM=saves/model_task2_roberta_target.bin
AUX=saves/model_task2_anchoraug_best.bin
PROD_T=-0.5
FLOOR=${FLOOR:--3.0}
START=$(date +%s)

log() { echo ">>> $(date -u +%FT%TZ) $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }
need() { [ -f "$1" ] || die "missing $1"; }
md5() { md5sum "$1" | cut -d' ' -f1; }
rows() { python3 -c "import json,sys;print(len(json.load(open(sys.argv[1],encoding='utf8'))))" "$1"; }

for f in "$TEST" "$PRIM" "$AUX" "$RC" "$RV" "$BASE2_PRIM" "$BASE2_AUX" \
         runbash/task2_fuse_grid.py; do need "$f"; done
grep -q 'span_dump' predict_task2.py || die "predict_task2.py has no span_dump patch -- push the patched file first"
grep -q 'span_dump' params.py        || die "params.py has no --span_dump argument -- push the patched params.py first"
if pgrep -f 'predict_task2.py|predict_task3.py|train_task2.py' >/dev/null; then
    die "another task2/task3 train or decode process is alive, refusing to collide"
fi
if [ -d "$OUTDIR" ] && [ -n "$(ls -A "$OUTDIR" 2>/dev/null)" ] && [ "${FORCE:-0}" != 1 ]; then
    die "$OUTDIR is not empty (set FORCE=1 to reuse it)"
fi
mkdir -p "$OUTDIR" "$LF"
log "===== config actually used ====="
log "split=$SPLIT test=$TEST out=$OUTDIR prod_thresh=$PROD_T floor=$FLOOR"
log "primary=$PRIM  auxiliary=$AUX  backbone both = chinese-roberta-wwm-ext-large"
log "with_nms=off refine_boundary=0 (defaults, not passed) -> emission is exactly logit >= threshold"

decode() {   # $1 ckpt $2 threshold $3 tag  -> sets R_FILE R_SIDE
    local CK=$1 THR=$2 TAG=$3
    R_FILE="$OUTDIR/${TAG}_thr${THR}.json"
    R_SIDE="$OUTDIR/scores_${TAG}.json"
    log "[decode] $TAG thr=$THR -> $R_FILE (+ sidecar $R_SIDE)"
    python3 -u predict_task2.py --test_file "$TEST" --config_file "$RC" --vocab_file "$RV" \
        --model_path "$CK" --logit_threshold "$THR" --batch_size 16 \
        --out_name "$R_FILE" --span_dump "$R_SIDE" \
        > "$LF/${TAG}_${THR}.log" 2>&1 \
        || { tail -20 "$LF/${TAG}_${THR}.log"; die "decode failed: $TAG @ $THR"; }
    need "$R_FILE"; need "$R_SIDE"
    log "[decode] $TAG spans=$(rows "$R_FILE") sidecar_rows=$(rows "$R_SIDE")"
}

log "[1/4] neutrality: patched decode at $PROD_T must equal $BASE2_PRIM (the file this split's shipped package was built from)"
decode "$PRIM" "$PROD_T" prim
[ "$(md5 "$R_FILE")" = "$(md5 "$BASE2_PRIM")" ] \
    || die "prim@$PROD_T ($(md5 "$R_FILE")) != $BASE2_PRIM ($(md5 "$BASE2_PRIM")) -> patch is NOT neutral"
log "  prim@$PROD_T == $BASE2_PRIM (md5 $(md5 "$BASE2_PRIM")) OK"
decode "$AUX" "$PROD_T" aux
[ "$(md5 "$R_FILE")" = "$(md5 "$BASE2_AUX")" ] \
    || die "aux@$PROD_T ($(md5 "$R_FILE")) != $BASE2_AUX ($(md5 "$BASE2_AUX")) -> patch is NOT neutral"
log "  aux@$PROD_T == $BASE2_AUX (md5 $(md5 "$BASE2_AUX")) OK"

log "[2/4] floor decodes, the sidecars the grid actually reads"
# decode() already writes the sidecar at scores_<tag>.json, and the floor run overwrites the -0.5 run's
# file at the same path, so the floor sidecar is what remains -- no copy needed
decode "$PRIM" "$FLOOR" prim
mv "$R_FILE" "$OUTDIR/task2_prim_floor.json"
decode "$AUX" "$FLOOR" aux
mv "$R_FILE" "$OUTDIR/task2_aux_floor.json"

log "[3/4] containment: sidecar filtered at $PROD_T must equal the $PROD_T span set exactly"
python3 - "$OUTDIR/scores_prim.json" "$OUTDIR/prim_thr$PROD_T.json" "$PROD_T" "$OUTDIR/scores_aux.json" <<'PY' || die "containment check failed"
import json, sys
side_p, f_p, thr, side_a = sys.argv[1:5]
thr = float(thr)
def keys(p):
    return sorted({(int(r[0]), int(r[1]), int(r[2])) for r in json.load(open(p, encoding='utf8'))})
def filt(p):
    return sorted({(int(r[0]), int(r[1]), int(r[2])) for r in json.load(open(p, encoding='utf8')) if float(r[3]) >= thr})
a, b = filt(side_p), keys(f_p)
assert a == b, 'primary containment differs: sidecar>=thr %d rows, file %d rows' % (len(a), len(b))
print('[check] prim: %d spans at %.2f recovered from the %d-row floor sidecar' % (len(a), thr, len(keys(side_p))))
print('[info ] prim floor: %d rows, logit %.3f..%.3f'
      % (len(json.load(open(side_p, encoding='utf8'))),
         min(float(r[3]) for r in json.load(open(side_p, encoding='utf8'))),
         max(float(r[3]) for r in json.load(open(side_p, encoding='utf8')))))
sa = json.load(open(side_a, encoding='utf8'))
print('[info ] aux floor: %d rows, logit %.3f..%.3f' % (len(sa), min(float(r[3]) for r in sa), max(float(r[3]) for r in sa)))
PY

log "[4/4] artifacts"
ls -l "$OUTDIR" | sed 's/^/    /'
log "next (CPU, no GPU): python3 runbash/task2_fuse_grid.py --prim $OUTDIR/scores_prim.json --aux $OUTDIR/scores_aux.json"
log "elapsed $(( ($(date +%s) - START) / 60 )) min"
echo "TASK2_SCORES_DONE $SPLIT $(date -u +%FT%TZ)"
