# E-001 Public Error-Metric Discovery

Date: `2026-08-04`

## Finding

GDROS `test.py` receives an LSR prediction shaped `[1, 2, H, W]` and GT shaped
`[2, H, W]`, then computes squared-error reduction on `dim=0`. At batch 1 this is the
batch axis, so the published AEPE/CMR implementation uses mean absolute component error:

```python
(prediction - target).abs().flatten(1).mean(dim=1)
```

It is not the standard two-dimensional vector EPE, which reduces the flow-channel axis.

On official pair `000000_opt`, using the same checkpoint, input and dense prediction:

- Public component error: `1.248353 px`
- Standard vector EPE: `1.882520 px`
- Repeated model forwards: bitwise identical
- E-001 generalized 200k LSR versus original dense flow: max absolute difference
  `6.1035e-05 px`

## E-001 Treatment

- The primary AEPE, CMR, P95/P99, failure rate and E-002 decision gate use the exact
  public component-error convention so they remain comparable with E-000 and the paper.
- Standard vector EPE is reported for every reference, sparse strategy and oracle under
  an explicitly diagnostic field.
- Both metrics evaluate all pixels without a valid mask, including black borders.
- The metric correction is applied uniformly to reference, sparse candidates and GT
  oracle; GT is never used to select indices.

## Validation Artifacts

- Two-pair regression:
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/validation/E-001/smoke-06-two-pairs`
- Full one-pair result/benchmark structure:
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/validation/E-001/smoke-05-public-metric`
- Pair 0/1 public-error deltas versus E-000:
  `3.58e-07` and `1.19e-07 px`
- Current unit and invariant suite: `20/20` passed on an RTX 4090 using the GDROS conda
  environment. This includes formal E-000 provenance/content binding, mutated-data,
  split-escape, loader-path and real-parser-default tests.
- Independently approved implementation SHA256 values:
  `9628c826...` (`evaluate_sparse_lsr.py`), `3dbcb88f...`
  (`benchmark_sparse_lsr.py`) and `081aca6b...` (`test_sparse_lsr.py`).
- Latest two-pair regression:
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/validation/E-001/smoke-10-parser-default`
