#!/usr/bin/env bash
# Prepare the UBCv2 in-domain training data for the GDROS official recipe.
#
# Steps (idempotent):
#   1. unzip  raw/train_images.zip  -> raw/train_extracted/{train,val}/{rgb,sar}
#   2. build GDROS-layout training/ (4008 pairs) and validation/ (1681 pairs)
#      reusing the *published test-set* SAR LUT so all three splits share one
#      identical radiometric mapping.
#
# Run on zzvr.  The affine generator and flow convention are identical to
# research/eval/build_ubcv2_testset.py (which produced the published 1625-pair
# test set), therefore train/val/test are transform-compatible.
set -euo pipefail

ROOT=${ROOT:-/mnt/sda/zhushihao/VS_Match}
UBC=${UBC:-$ROOT/datasets/ubcv2}
PY=${PY:-/home/zzvr/miniconda3/envs/compared/bin/python}
GEN=${GEN:-$ROOT/image_matching_part2_migrated/.codex_tmp/build_ubcv2_split.py}

# Published test-set LUT (seed901/protocol_metadata/manifest.json).  Reused on
# purpose: a per-split LUT would give train and test slightly different SAR
# histograms, which is exactly the kind of spurious domain gap we want to avoid.
LUT=${LUT:--28.307898330688477,10.545851440429704}

mkdir -p "$UBC/raw/train_extracted"
if [ ! -d "$UBC/raw/train_extracted/train/rgb" ]; then
  echo "[1/2] unzip train_images.zip"
  unzip -q -o "$UBC/raw/train_images.zip" -d "$UBC/raw/train_extracted"
else
  echo "[1/2] already extracted, skip"
fi
ls "$UBC/raw/train_extracted/train/rgb" | wc -l
ls "$UBC/raw/train_extracted/train/sar" | wc -l
ls "$UBC/raw/train_extracted/val/rgb"   | wc -l
ls "$UBC/raw/train_extracted/val/sar"   | wc -l

echo "[2/2] build splits (LUT=$LUT)"
if [ ! -f "$UBC/generated/train_seed701/training/truth_flow/000000.flo" ]; then
  "$PY" "$GEN" --raw-root "$UBC/raw/train_extracted/train" \
    --output-root "$UBC/generated/train_seed701" \
    --split training --sar-lut="$LUT" --seed 701 --workers 8
else
  echo "training split exists, skip"
fi
if [ ! -f "$UBC/generated/val_seed702/validation/truth_flow/000000.flo" ]; then
  "$PY" "$GEN" --raw-root "$UBC/raw/train_extracted/val" \
    --output-root "$UBC/generated/val_seed702" \
    --split validation --sar-lut="$LUT" --seed 702 --workers 8
else
  echo "validation split exists, skip"
fi

echo "=== counts ==="
for s in "$UBC/generated/train_seed701/training" "$UBC/generated/val_seed702/validation"; do
  for sub in opt sar_warped truth_flow; do
    printf "%s/%s: " "$s" "$sub"; ls "$s/$sub" | wc -l
  done
done
du -sh "$UBC/generated/train_seed701" "$UBC/generated/val_seed702"
echo "PREPARE DONE"
