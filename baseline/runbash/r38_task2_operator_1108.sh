#!/usr/bin/env bash
# R38 task2 fusion OPERATOR comparison with the two production members fixed (CPU, no GPU).
#   members: prim + aux (the pair the current recipe combines), dev split, gold = dataset/cfn-dev.json
#   operator 1: each member thresholded on its own, then union  (the current recipe: -0.25 / -0.75)
#   operator 2: scores averaged as they come, one threshold      (no scale alignment)
#   operator 3: scores averaged after each member is shifted by its own median, one threshold
#   plus: each member alone at its own threshold, score-scale quantiles, changed-span accounting.
# Everything below is recomputation over sidecars already on disk; nothing retrains, nothing writes
# into saves/ or into cells/. Outputs go to dataset/dev/cmp1108/r38/ only.
set -euo pipefail
cd /work/workspace/ccl25/baseline
stamp() { date -u "+[%F %T UTC] "; }
ln -f /opt/conda/bin/python3.11 /opt/conda/bin/ccl25_r38_t2op 2>/dev/null || true
PY=/opt/conda/bin/ccl25_r38_t2op

CELL=dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json
P2="prim=dataset/dev/cmp1022/scores_prim.json,aux=dataset/dev/cmp1022/scores_aux.json"
OUT=dataset/dev/cmp1108/r38
NS="$OUT/mean_noshift"; AL="$OUT/mean_align"; NS3="$OUT/mean_noshift_fill3"; AL3="$OUT/mean_align_fill3"
mkdir -p "$NS" "$AL" "$NS3" "$AL3"
for f in dataset/dev/cmp1022/scores_prim.json dataset/dev/cmp1022/scores_aux.json "$CELL" \
         runbash/task2_shift_mean.py runbash/r38_scale_quantiles_1108.py; do
    [ -s "$f" ] || { echo "[FATAL] missing $f"; exit 1; }
done

THR=-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0

echo "===== 1/6 gate: union at -0.25/-0.75 must be byte-identical to the production cell ====="
stamp
PYTHONPATH=. $PY -u runbash/task2_shift_mean.py --lists "$P2" \
    --rule union --per_thr prim=-0.25,aux=-0.75 \
    --out_dir "$OUT" --check_file "$CELL" --expect_ai 88.9700

echo "===== 2/6 member score-scale quantiles and the aligning shifts ====="
stamp
PYTHONPATH=. $PY -u runbash/r38_scale_quantiles_1108.py --lists "$P2" --ref_member prim \
    --emit_offset "$OUT/offsets.txt"
OFF=$(cat "$OUT/offsets.txt")
echo "[offsets in use] $OFF"

echo "===== 3/6 each member alone at its own threshold (single-member readouts) ====="
stamp
for kv in "prim=-0.25" "aux=-0.75"; do
    n=${kv%%=*}; t=${kv##*=}
    PYTHONPATH=. $PY -u runbash/task2_shift_mean.py --lists "$n=$(echo "$P2" | tr ',' '\n' | grep "^$n=" | cut -d= -f2-)" \
        --rule mean --mean_thr_csv="$t" --ref_file "$CELL"
done

echo "===== 4/6 operator 2: average as-is (no alignment), threshold sweep ====="
stamp
PYTHONPATH=. $PY -u runbash/task2_shift_mean.py --lists "$P2" \
    --rule mean --mean_thr_csv="$THR" --fill present --ref_file "$CELL" --out_dir "$NS"

echo "===== 5/6 operator 3: average after median-alignment shift, threshold sweep ====="
stamp
PYTHONPATH=. $PY -u runbash/task2_shift_mean.py --lists "$P2" --offset "$OFF" \
    --rule mean --mean_thr_csv="$THR" --fill present --ref_file "$CELL" --out_dir "$AL"

echo "===== 5b/6 sensitivity: the same two operators with absent spans patched at -3.0 ====="
stamp
PYTHONPATH=. $PY -u runbash/task2_shift_mean.py --lists "$P2" \
    --rule mean --mean_thr_csv="$THR" --fill=-3.0 --ref_file "$CELL" --out_dir "$NS3"
PYTHONPATH=. $PY -u runbash/task2_shift_mean.py --lists "$P2" --offset "$OFF" \
    --rule mean --mean_thr_csv="$THR" --fill=-3.0 --ref_file "$CELL" --out_dir "$AL3"

echo "===== 6/6 changed-span accounting versus the production cell ====="
stamp
ARGS=""
for d in "$NS" "$AL" "$NS3" "$AL3"; do
    for f in "$d"/task2_mean*.json; do
        [ -e "$f" ] || continue
        ARGS="$ARGS --cell $(basename "$d")/$(basename "$f" .json)=$f"
    done
done
[ -n "$ARGS" ] || { echo "[FATAL] no mean cells were written"; exit 1; }
eval PYTHONPATH=. $PY -u runbash/r38_scale_quantiles_1108.py --lists "$P2" --ref_member prim \
    --incumbent "$CELL" $ARGS
echo "R38_ALL_DONE"
