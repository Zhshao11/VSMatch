# E-008A Evaluator Review R3

Verdict: `APPROVE`

This review is limited to the R2 F1 repair for exact endpoint labels and nearest-candidate ordering. No experiment
source was modified, and previously approved evaluator paths were not reopened.

## Reviewed Identity

- Source manifest SHA256: `b8e4dd9f4fce7bf55f2f8ff5a875052e214248c2052ac0de5cb3fb6398ba04a5`
- Reported automatic tests: `13/13 PASS`

## Finding Closure

### R2 F1 CLOSED: endpoint precision at metric boundaries

- `build_labels.make_labels()` persists both `endpoint_error_px` and `endpoint_pixel_xy` as `torch.float32`.
- `label_1px`, `label_2px` and `label_5px` are derived from the same FP32 endpoint-error tensor, so stored
  threshold membership cannot diverge through FP16 rounding.
- `validate_label()` rejects endpoint tensors that are not FP32, making the precision contract fail closed for
  cached or newly generated label artifacts.
- The tests assert FP32 dtype for both endpoint tensors and cover the repaired label contract.
- `evaluate.generated_errors()` gathers `endpoint_error_px` directly. P0/P1 threshold comparisons and the
  nearest-candidate argmin therefore retain FP32 precision rather than reconstructing decisions from FP16 data.

These changes close the R2 counterexample in which an error just above 1, 2 or 5 px could round onto the threshold,
and also prevent FP16 quantization from changing nearest-candidate ordering.

## Decision

R2 F1 is closed. The evaluator is approved for the frozen E-008A gate sequence. This verdict confirms evaluator
correctness only; it does not predict or waive any P0/P1 method gate.
