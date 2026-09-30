# SparseGeo-GDROS Image Matching Research

This repository records the reproducible source, protocols, and compact result
cards for the OS and WHU optical-SAR image-matching experiments. Large datasets,
model weights, prediction tensors, scheduler logs, and generated plots are not
versioned.

## Current method state

- OS protected best: E075A-M4 at step 80K. Its main limitation is model-unseen
  generalization and the pair-level error tail beyond 5 px.
- WHU protected best: E081A-M4 at step 120K. Its main limitation is fine
  localization: CMR@5 is high while CMR@1/2 remains low.
- LuckyR1 was closed as `NO_GAIN`.
- E090D proved that structured evidence can drive a late recurrent residual
  path on 1/4/8 training pairs.
- E091A tested that path on 1600 training pairs and a disjoint 800-pair dev
  split. It improved CMR@1 from 22.50% to 24.75%, but failed the preregistered
  held-out gate because AEPE improvement was only 0.0148 px and shuffled
  evidence performed equivalently. WHU 700-test remained sealed.

The latest method decision context is in
[`research/HANDOFF_SPARSEGEO_METHOD_DECISION_20260919.md`](research/HANDOFF_SPARSEGEO_METHOD_DECISION_20260919.md).
The canonical current-best and latest-experiment page is
[`research/CURRENT_METHOD_STATE.md`](research/CURRENT_METHOD_STATE.md).
The latest held-out result is in
[`research/results/E091A_WHU_HELDOUT_GENERALIZATION_20260930/RESULT_CARD.md`](research/results/E091A_WHU_HELDOUT_GENERALIZATION_20260930/RESULT_CARD.md).

## Repository layout

- `e081a_whu_our_native_train.py`: WHU E081A-M4 training and evaluation entry.
- `research/remote_stage/`: frozen experiment implementations for E081A
  diagnostics through E091A held-out validation and the watchdog DAG.
- `research/remote_stage/os/`: the exact E072-E075 source chain required by
  the protected OS E075A-M4 method, with original remote hashes recorded.
- `research/experiments/`: experiment specifications, contracts, and focused
  implementation tests from the earlier method search.
- `research/results/`: compact result cards and selected audit summaries.
- `tools/`: dataset protocol, validation, transfer, and metric utilities.
- `REPRODUCTION_STATUS.md`: official baseline reproduction status.

## Reproduction boundary

The repository does not vendor GDROS, datasets, or checkpoints. The current
experiment scripts import the upstream GDROS code from a deployment workspace.
Set `SPARSEGEO_ROOT` to that workspace; the historical zzvr layout remains the
default so frozen runs retain their original behavior.

See [`research/REPRODUCIBILITY.md`](research/REPRODUCIBILITY.md) for the data
contract, environment variables, commands, metrics, and exact limitations.

## Local validation

Install the direct research dependencies, then run syntax and CPU unit checks:

```bash
python -m pip install -r requirements-research.txt
python -m compileall -q tools research/remote_stage e081a_whu_our_native_train.py
python -m unittest discover -s tools -p 'test_*.py'
python scripts/preflight.py --scope all
```

Tests under `research/experiments/` additionally require the upstream GDROS
source and experiment-specific fixtures described by their run manifests.
