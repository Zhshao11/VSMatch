# E-008A Evaluator Review R2

Verdict: `CHANGES_REQUIRED`

Reviewed source manifest SHA256:

- `415bde6e2572a441971f3e4f4847b8922a77645b98f82128beeafb8d4114c4bc`

Reported automatic tests: `13/13 PASS`.

## Blocking Finding

### F1 - Exact endpoint labels are reconstructed from FP16 errors

Severity: high

`build_labels.make_labels()` computes the `<=1/2/5 px` booleans from FP32 endpoint errors, but stores
`endpoint_error_px` as FP16. `evaluate.generated_errors()` gathers that FP16 tensor and P0/P1 subsequently
recompute every threshold label using `error <= 1/2/5`.

This can change labels close to a strict threshold, for example an FP32 error just above 2 px can round to exactly
2 px in FP16. The reconstructed values currently feed:

- P0 correct-proposal and exact 1/2/5 candidate counts/base rates;
- hard-2px oracle membership;
- P1 AUROC/AP truth, P@K precision/recall and recoverable counts;
- P1 selected precision.

The nearest-per-proposal oracle also obtains its argmin from the quantized FP16 errors, so close candidates may tie
or change order.

Required fix:

- gather `label_1px`, `label_2px` and `label_5px` through `pool_slots` for all threshold membership and exact counts;
- preserve FP32 endpoint errors for nearest-candidate selection, or recompute the FP32 errors from
  `endpoint_pixel_xy` and the generated hard target coordinates;
- add a regression case with an FP32 endpoint error just above a threshold that rounds onto the threshold in FP16.

## Confirmed Repairs

- Endpoint correctness is no longer conjoined with `in_bounds`; in-bounds remains separately reported.
- Hard-2px-inlier and nearest-per-proposal affine oracles are separate.
- Aggregate 1/2/5 candidate counts and base rates are present.
- P1 fail-closed identity checks bind run mode, formal 404 pairs, split, source, GDROS checkpoint, selector
  checkpoint, baseline identities and the exact P0 check-key set and values.
- Smoke status is distinct from formal PASS/FAIL.
- Candidate traces are persisted before labels are loaded.
- Affine target coordinates use hard token coordinates in pixel units; full-image no-mask AEPE/CMR/PLR/valid
  paths remain intact.
