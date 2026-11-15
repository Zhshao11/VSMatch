#!/usr/bin/env bash
# Launch the *official* GDROS recipe on UBCv2 (in-domain) on zzvr.
#
# Recipe is untouched: configs/default.py -> num_steps=120000, iters=12,
# batch_size=2, image_size=[400,400], OneCycleLR, canonical_lr=12.5e-5,
# adamw_decay=1e-5, clip=1.0, add_noise=False, RAFT-Sintel init.
#
# Two hardcoded paths in the official repo have to be satisfied:
#   * train.py            -> torch.load('./checkpoints/raft-sintel.pth')
#                            (cfg.restore_ckpt is only used as a truthy flag; the
#                             value itself is never loaded, so no GDROS resume)
#   * core/datasets.py    -> opt_sar_train(root='./datasets/os_dataset')
# Both are satisfied with symlinks so the upstream code stays byte-identical.
set -euo pipefail

ROOT=${ROOT:-/mnt/sda/zhushihao/VS_Match}
REPO=${REPO:-$ROOT/external_repos/E068A_GDROS_official}
DATA=${DATA:-$ROOT/datasets/ubcv2/generated/train_seed701}
REF=${REF:-$ROOT/GDROS/official_models_reference}
PY=${PY:-/home/zzvr/miniconda3/envs/compared/bin/python}
GPU=${GPU:-5}
NAME=${NAME:-gdros-ubcv2-512}

cd "$REPO"
mkdir -p checkpoints datasets
[ -e checkpoints/raft-sintel.pth ] || ln -sfn "$REF/raft-sintel.pth" checkpoints/raft-sintel.pth
ln -sfn "$DATA" datasets/os_dataset

echo "REPO=$REPO"
echo "DATA=$DATA -> datasets/os_dataset"
echo "GPU=$GPU  NAME=$NAME"
echo "train.py sha256: $(sha256sum train.py | cut -d' ' -f1)"
echo "default.py sha256: $(sha256sum configs/default.py | cut -d' ' -f1)"

CUDA_VISIBLE_DEVICES=$GPU exec "$PY" train.py \
  --name "$NAME" --stage OSdata --dstype training --mixed_precision
