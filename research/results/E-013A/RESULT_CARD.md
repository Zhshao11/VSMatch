# E-013A RESULT_CARD

Status: **COMPLETED / LEARNED_PURIFICATION_FAIL**

Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-013A/run-01`

Watchdog:

- Job: `20260810_161937_ea528dc2`
- Status: `succeeded`, return code `0`
- Log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260810_161937_ea528dc2.log`
- Report JSON: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_161937_ea528dc2.json`
- Report text: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_161937_ea528dc2.txt`

Training:

- Train records: `1607`
- Epochs: `1`
- Parameters: `10,630`
- Threshold calibrated on train: `0.9699785920`
- Train seconds: `642.985`
- Loss terms nonzero: endpoint `0.0961`, basin `0.0966`, affine `0.7284`, precision `0.1506`
- Basin valid / affine-loss enabled pairs: `99.50% / 99.50%`
- Gradients: scorer `true`, basin head `true`

G0:

- Verdict: **PASS**
- E-011A top128 parity: PASS
- E-012A baseline parity: PASS
- Label construction audit: PASS
- Candidate-before-label join: PASS
- No GT inference / datum / OS424 / confirm: PASS
- Loss/gradient participation: PASS
- residual feedback changed: PASS
- one-source-one-target: PASS
- dynamic active size: PASS
- no hidden `4096x4096`, no RANSAC, no 131k hypotheses: PASS
- Reviewer: `APPROVE_WITH_RESULT_FAIL`

G1:

- Verdict: **FAIL**
- active count mean: `0.658` / gate `>=32` → FAIL
- active count P10/P50/P90: `0 / 0 / 2`; P10 gate `>=12` → FAIL
- correct@2px mean: `0.017` / gate `>=24` → FAIL
- correct@2px P10/P50/P90: `0 / 0 / 0`; P10/P50 gates `>=8/20` → FAIL
- precision mean: `1.246%` / gate `>=20%` → FAIL
- precision P10/P50/P90: `0 / 0 / 0`; P10/P50 gates `>=5%/15%` → FAIL
- coverage pass: `0%` / gate `>=90%` → FAIL

G2:

- Not run because G1 failed.

Resource:

- eval p50/p95: `598.45 / 684.16 ms`
- max live score elements: `204,544`
- peak allocated bytes: `42,146,816`
- hidden N2: false

Interpretation:

- E-013A run-01 avoids E-012A’s large wrong-basin failure, but collapses into over-abstention after train-calibrated thresholding.
- The learned scorer did receive endpoint/basin/affine supervision and gradients, but the no-GT dev active set is nearly empty, so it does not solve top128 purification.
- Per protocol, G2 and all follow-up rescue/new-version experiments were not launched.

Artifacts:

- Local G0: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A/g0.json`
- Local G1: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A/g1.json`
- Local summary: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A/summary.json`
- Local train report: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A/train.json`
- Local reviewer: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A/reviewer_conclusion.md`
- Local manifest: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A/RUN_MANIFEST.json`
- Remote per-pair metrics: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-013A/run-01/formal/per_pair.jsonl`
