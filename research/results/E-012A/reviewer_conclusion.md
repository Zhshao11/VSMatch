# E-012A Reviewer Conclusion

Status: APPROVE_WITH_RESULT_FAIL.

Reviewed scope: E-012A incremental geometry-aware top128 dynamic compression only.

Findings:

- E-011A/GDROS isolation is respected: E-012A reads E-011A checkpoint/formal pair metrics and E-008A cache as inputs, and writes only under independent `e012a` / `runs/E-012A` paths.
- Candidate outputs are materialized before label join in `load_e011a_top128_candidates_readonly()` and `evaluate.py`; truth-flow labels are loaded only after compression output exists.
- The implementation has multiple affine basin slots, iterative assignment -> FP32 weighted affine solve -> residual feedback, one-source-one-target final suppression, and active/pending/rejected/dustbin state reporting.
- G0 machine checks pass: top128 parity, no GT inference trace, one-source-one-target, dynamic active size, no hidden 4096x4096, no RANSAC/131k hypotheses, coordinate/weighted-affine tests.
- No implementation evidence suggests the G1 failure is caused by fixed K64/K128 semantics or baseline fallback.
- Result failure is substantive for this deterministic no-training compressor: it selects large wrong self-consistent basins with very low GT precision.

Conclusion: implementation is valid for the approved E-012A run-01 semantics; negative G1 result should be interpreted as failure of this deterministic geometry-compression design, not as invalid execution.
