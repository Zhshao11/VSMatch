#!/usr/bin/env bash
# Evaluate the UBCv2-trained GDROS and re-evaluate M4 across the three seeds,
# then print the in-domain vs cross-dataset comparison table.
#
# Runs on zzvr.  Assumes:
#   * training finished and wrote checkpoints/gdros-ubcv2-512.pth
#     (plus 20000_/40000_/... intermediates from the VAL_FREQ=20000 save)
#   * generated/seed901|902|903 test splits exist
#   * .codex_tmp/ubcv2_official_20261115.json holds the original cross-dataset run
#
# GPU: 3 by default (GPU 5 is the training card).
set -euo pipefail

ROOT=${ROOT:-/mnt/sda/zhushihao/VS_Match}
REPO=${REPO:-$ROOT/external_repos/E068A_GDROS_official}
WORK=${WORK:-$ROOT/image_matching_part2_migrated}
PY=${PY:-/home/zzvr/miniconda3/envs/compared/bin/python}
GPU=${GPU:-3}
GEN=${GEN:-$ROOT/datasets/ubcv2/generated}
OUT=${OUT:-$WORK/.codex_tmp}
CKPT=${CKPT:-$REPO/checkpoints/gdros-ubcv2-512.pth}

cd "$WORK"
export CUDA_VISIBLE_DEVICES=$GPU

run_eval () {  # $1=which  $2=seed  $3=outfile  [$4=ckpt]
  local which=$1 seed=$2 out=$3 ckpt=${4:-}
  if [ -s "$out" ]; then echo "SKIP existing $out"; return; fi
  echo "=== eval which=$which seed=$seed ckpt=${ckpt:-released} -> $out"
  local extra=()
  [ -n "$ckpt" ] && extra=(--baseline-ckpt "$ckpt")
  "$PY" .codex_tmp/eval_ubcv2_official_protocol.py \
    --test-root "$GEN/seed$seed" --which "$which" --gpu 0 --max-pairs 1625 \
    --tag "ubcv2_seed$seed" "${extra[@]}" --out "$out"
}

# 1) convergence curve: every 20K checkpoint on seed901 only (~2.5 min each)
for ck in "$REPO"/checkpoints/[0-9]*_gdros-ubcv2-512.pth; do
  [ -e "$ck" ] || continue
  step=$(basename "$ck" | cut -d_ -f1)
  run_eval baseline 901 "$OUT/ubcv2_indomain_ck${step}_s901.json" "$ck"
done

# 2) final checkpoint on all three seeds
for s in 901 902 903; do
  run_eval baseline "$s" "$OUT/ubcv2_indomain_s$s.json" "$CKPT"
done

# 3) released OS weights on the remaining seeds (901 already done on 11-15)
for s in 902 903; do
  run_eval baseline "$s" "$OUT/ubcv2_os_s$s.json"
done

# 4) M4 on the remaining seeds (901 already done on 11-15, ~21 min each)
for s in 902 903; do
  run_eval m4 "$s" "$OUT/ubcv2_m4_s$s.json"
done

# 5) final table
"$PY" research/eval/compare_ubcv2_indomain.py \
  --test-root "$GEN/seed901" \
  --baseline-from "$GEN/seed902,$GEN/seed903" \
  --row "GDROS UBCv2-trained|UBCv2 in-domain|baseline|$OUT/ubcv2_indomain_s901.json" \
  --row "GDROS UBCv2-trained|UBCv2 in-domain|baseline|$OUT/ubcv2_indomain_s902.json" \
  --row "GDROS UBCv2-trained|UBCv2 in-domain|baseline|$OUT/ubcv2_indomain_s903.json" \
  --row "GDROS OS-trained|OS cross-dataset|baseline|$OUT/ubcv2_official_20261115.json" \
  --row "GDROS OS-trained|OS cross-dataset|baseline|$OUT/ubcv2_os_s902.json" \
  --row "GDROS OS-trained|OS cross-dataset|baseline|$OUT/ubcv2_os_s903.json" \
  --row "M4@80K|OS cross-dataset|m4|$OUT/ubcv2_official_20261115.json" \
  --row "M4@80K|OS cross-dataset|m4|$OUT/ubcv2_m4_s902.json" \
  --row "M4@80K|OS cross-dataset|m4|$OUT/ubcv2_m4_s903.json" \
  2>&1 | tee "$OUT/ubcv2_indomain_table.txt"
echo "TABLE DONE -> $OUT/ubcv2_indomain_table.txt"
