# E-013A-CALIB-DIAG Targeted Reviewer Conclusion

Verdict: **APPROVE_WITH_RESULT_FAIL**

Scope reviewed:

- `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e013a_diag/evaluate.py`
- `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e013a_diag/calibration.py`
- `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e013a_diag/sweeps.py`
- `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e013a_diag/diagnose.py`
- `research/experiments/E-013A-CALIB-DIAG.md`
- `research/REVIEW_PACKET.md`
- `research/results/E-013A/RESULT_CARD.md`

Checks:

- Frozen SHA/provenance: PASS.
- No training, finetune, checkpoint write, E-013A formal-result overwrite, E-013B/rescue launch: PASS.
- Candidate outputs and all score tracks are materialized before diagnostic label join: PASS.
- Train/dev split separation: PASS.
- E-013A G1 parity: PASS.
- Threshold sweep/topK/quota sweep are diagnostic only and do not modify formal E-013A status: PASS.
- Score source ablation includes endpoint/pre-feedback, basin-only, post-feedback/final, and combined inlier-basin tracks: PASS.
- Diagnosis rule trace is consistent with experiment text after applying the “substantial unreachable fraction” interpretation for LABEL_OR_BASE_RATE_LIMITED: PASS.

Finding:

- No implementation finding that would invalidate the diagnostic result. The result fails because final score ranking is not usable, not because of a protocol violation.
