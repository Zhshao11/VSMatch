# Proposed A D1 Corrected R3

Status: `COMPLETED / CAPACITY_SUPPORTED / POOL_NO_GAIN / HELDOUT_NOT_TESTABLE`

No GPU inference or training was run in R3. R3 reuses the frozen R2 OS/WHU 32-pair evidence bundles and recomputes D1 on CPU. Existing E085/E086/E087 jobs and result cards were not modified.

## Corrections

The R2 interpretation `PARAMETER_FAMILY_CAPACITY_FAIL_WITH_PIPELINE_ANOMALY` is superseded.

R3 fixes four definition errors:

1. Translation and affine capacity both fit `GT - coarse` residuals.
2. Generated translation and affine modes are residual corrections added to the same coarse flow.
3. Both pools contain the zero-residual/base-coarse mode.
4. Both pools have the same fixed candidate budget: 12 modes.

R3 reports both raw and GDROS-compatible post-LSR output operators.

## Post-LSR Main Results

| Dataset/track | AEPE | CMR@1 | CMR@2 | CMR@5 | >=5px failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| OS coarse | 1.517 | 37.50% | 78.13% | 100.00% | 0/32 |
| OS best translation capacity | 0.929 | 65.63% | 90.63% | 100.00% | 0/32 |
| OS best affine-family capacity | 0.634 | 84.38% | 96.88% | 100.00% | 0/32 |
| OS generated translation oracle | 1.517 | 37.50% | 78.13% | 100.00% | 0/32 |
| OS generated affine oracle | 1.517 | 37.50% | 78.13% | 100.00% | 0/32 |
| WHU coarse | 1.261 | 40.63% | 90.63% | 100.00% | 0/32 |
| WHU best translation capacity | 0.654 | 84.38% | 100.00% | 100.00% | 0/32 |
| WHU best affine-family capacity | 0.528 | 87.50% | 100.00% | 100.00% | 0/32 |
| WHU generated translation oracle | 1.261 | 40.63% | 90.63% | 100.00% | 0/32 |
| WHU generated affine oracle | 1.261 | 40.63% | 90.63% | 100.00% | 0/32 |

`best affine-family capacity` is the per-pair envelope of translation and fitted affine residuals, because the affine family contains the translation subfamily.

## Edge-Policy Control

At the same 16,384-edge budget, replacing exported source-top4 edges with reciprocal-top16-selected edges did not improve translation or affine pool oracle. Both policies select the base coarse mode as the oracle.

## Interpretation

- `PARAMETER_FAMILY_CAPACITY_SUPPORTED`: residual affine capacity is real, especially on OS.
- `POOL_NO_GAIN`: current no-GT global cosine edges and clustering do not generate a residual mode that improves over base coarse.
- `HELDOUT_NOT_TESTABLE`: because the fixed generated pool contains no beneficial alternative mode, held-out-region scoring cannot demonstrate selection gain. It often falls back to base and can regress on the translation pool.
- `ROBUST_SCORE_UNSAFE`: the current affine robust score selects catastrophic hypotheses; it is not a valid positive control.
- `TRAINING_NOT_AUTHORIZED`: do not train a selector or geometry posterior from this pool.

This 32-pair diagnostic uses released GDROS checkpoints on training-overlap samples. It is mechanism evidence only. It does not represent E075's OS long-tail failure population and cannot support a generalization claim.

## Evidence Paths

- Remote root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/PROPOSED-A-D1-CORRECTED-R3`
- OS corrected result: `d1_os_final.json`
- WHU corrected result: `d1_whu_final.json`
- Reciprocal edge controls: `d1_os_reciprocal_top16.json`, `d1_whu_reciprocal_top16.json`
- Frozen input bundles remain under R2 and were not modified.

## Verification

- Static compilation: PASS.
- Synthetic CPU geometry tests: PASS.
- Synthetic evidence/D1/LSR regression tests: PASS, 6 tests.
- GPU use in R3: none.
- Training/backward/optimizer: none.
- Testing.zip: not used.
- Attempt 01 packaging failure has its own remote RESULT_CARD; it produced no method metrics.
