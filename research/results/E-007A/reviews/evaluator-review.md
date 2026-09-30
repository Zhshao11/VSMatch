# E-007A Evaluator Review

Conclusion: `APPROVE`

- All required evidence and ablation tracks, candidate metrics, topK 16/32/64, uncapped threshold, and region-balanced selection are present.
- Affine evaluation uses ordinary FP32 fitting and full 512x512 no-mask metrics after method output completion.
- Region counts are grouped into fixed `0-5`, `6-15`, `16-31`, and `32-64` bins with AEPE, CMR, PLR, valid, precision, and recall.
- Resources separate extraction, frozen selector, joint scoring, end-to-end peak, and each learned P/S/D/fusion route.
- Formal evaluation requires exactly 404 dev records and formal checkpoints; boundary and interpretation-precedence tests pass.

Reviewer: independent evaluator reviewer.
