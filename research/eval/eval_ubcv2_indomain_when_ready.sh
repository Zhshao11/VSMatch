#!/usr/bin/env bash
# Wait for the UBCv2 in-domain GDROS training to finish, then evaluate every
# saved checkpoint on the 1625-pair test set with the official protocol.
#
# Chaining it here (instead of polling by hand) means the numbers are ready the
# moment training stops, and the eval cannot accidentally run on a checkpoint
# that is still being written.
set -uo pipefail
EXIT_MARK=${EXIT_MARK:-/tmp/ubcv2_train.exit}
SEEN_MARK=${SEEN_MARK:-/tmp/ubcv2_indomain_eval.exit}
LOG=${LOG:-/tmp/ubcv2_indomain_eval.log}
OUT=${OUT:-.codex_tmp/ubcv2_indomain_curve.json}
CKPT_DIR=${CKPT_DIR:-/mnt/sda/zhushihao/VS_Match/external_repos/E068A_GDROS_official/checkpoints}
PY=${PY:-/home/zzvr/miniconda3/envs/compared/bin/python}
GPU=${GPU:-5}

rm -f "$SEEN_MARK"
echo "[wait] $(date) waiting for $EXIT_MARK" 
while [ ! -f "$EXIT_MARK" ]; do
  sleep 60
done
echo "[wait] $(date) train finished, exit=$(cat "$EXIT_MARK")"

ls -la "$CKPT_DIR"

CUDA_VISIBLE_DEVICES=$GPU "$PY" .codex_tmp/eval_ubcv2_indomain.py \
  --gpu 0 --max-pairs 1625 \
  --ckpt-glob "$CKPT_DIR/*_gdros-ubcv2-512.pth" \
  --ckpts "$CKPT_DIR/gdros-ubcv2-512.pth" \
  --out "$OUT"
rc=$?
echo "[wait] $(date) eval rc=$rc"
echo $rc > "$SEEN_MARK"
