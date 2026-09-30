# E-045A implementation map

Base family:
- `tmp/e028b_staging/experiments/sparsegeo/e028b/model.py`
- `tmp/e028b_staging/experiments/sparsegeo/e028b/train.py`
- `tmp/e028b_staging/experiments/sparsegeo/e028b/features.py`
- `tmp/e028b_staging/experiments/sparsegeo/e028b/run_all.py`

## Branch 1: `teacher_consistency`

Likely edits:
- strengthen optical teacher consistency term in `train.py`
- add a stricter drift probe / penalty
- widen or deepen the optical-to-teacher projection path in `model.py`

Goal:
- keep optical prior stable while making SAR latent closer to a more consistent shared space.

## Branch 2: `sar_structure_stem`

Likely edits:
- extend SAR structure channels in `model.py`
- optionally add a stronger SAR residual stem / conv stack
- keep output descriptor shape unchanged for downstream compatibility

Goal:
- improve SAR-side structural selectivity before matcher consumption.

## Branch 3: `common_unique_split`

Likely edits:
- add a new wrapper model class in `model.py`
- produce shared/common latent plus modality-specific residual latent
- add orthogonality / redundancy reduction losses in `train.py`

Goal:
- decouple what is common across OPT/SAR from what is modality-private.

## Branch 4: `masked_cross_modal_pretrain`

Likely edits:
- add a pretext head and mask generator in `train.py`
- optionally add a decoder/reconstruction target in `model.py`
- log cross-modal reconstruction/consistency metrics in `run_all.py`

Goal:
- force the frontend to learn recoverable cross-modal structure instead of only pairwise CE.

## Non-negotiables

- freeze E-044A matcher-side reference
- no GDROS baseline mutation
- no dev-threshold tuning
- no truth_flow leakage into feature construction
- keep feature materialization before GT join
