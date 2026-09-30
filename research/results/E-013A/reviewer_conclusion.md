# E-013A Reviewer Conclusion

Status: APPROVE_WITH_RESULT_FAIL.

Reviewed scope: E-013A incremental learned inlier/basin scorer only.

Findings:

- Frozen identity and E-011A top128 parity pass.
- E-013A reads E-011A checkpoint/formal metrics and E-008A cache as read-only inputs, and writes only under independent `e013a` / `runs/E-013A` paths.
- Candidate outputs are generated before label join; raw truth-flow labels are used only for train supervision and dev evaluation joins.
- Training report shows endpoint, basin, affine, and precision losses are nonzero; gradients reach both scorer and basin heads.
- Formal inference records no GT/datum/OS424/confirm use, no hidden 4096x4096, no RANSAC/131k hypotheses, one-source-one-target, dynamic active size, and residual-feedback trace change.
- G1 failure is a substantive result for run-01: the train-calibrated threshold over-abstains on dev, producing almost no active matches, so candidate purification does not meet the approved gate.

Conclusion: implementation is valid for E-013A run-01 semantics; result should be interpreted as learned purification failure via over-abstention/generalization failure, not protocol contamination.
