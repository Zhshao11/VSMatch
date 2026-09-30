# E-012A RESULT_CARD

Status: **COMPLETED / DYNAMIC_COMPRESSION_FAIL**

Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-012A/run-01`

Watchdog:

- Job: `20260810_152000_ee789fb1`
- Status: `succeeded`, return code `0`
- Log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260810_152000_ee789fb1.log`
- Report JSON: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_152000_ee789fb1.json`
- Report text: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_152000_ee789fb1.txt`

Config:

- Basins: `4`
- Rounds: `3`
- Final residual threshold: `10 px`
- Main path: deterministic no-training geometry-aware top128 compression

G0:

- Verdict: **PASS**
- E-011A top128 parity: PASS
- Reproduced top128 mean/P10/P50/P90: `68.710 / 34 / 66 / 106`
- Reproduced E-011A top128 oracle AEPE: `14.249`
- Candidate-before-label join: PASS
- One-source-one-target: PASS
- Dynamic active size not constant 64/128: PASS
- No GT inference / OS424 / confirm / datum: PASS
- No hidden `4096x4096`, no RANSAC, no 131k hypotheses: PASS
- Unit tests: `5 passed`
- Reviewer: `APPROVE_WITH_RESULT_FAIL`

G1:

- Verdict: **FAIL**
- active matched source count mean: `327.723` / gate `>=64` → PASS
- active matched source count P10/P50/P90: `218.6 / 294 / 497.7`; P10 gate `>=32` → PASS
- active 2px-correct count mean: `2.861` / gate `>=60` → FAIL
- active 2px-correct count P10/P50/P90: `0 / 0 / 7`; P10/P50 gates `>=31/56` → FAIL
- active 2px precision mean: `0.673%` / gate `>=35%` → FAIL
- active 2px precision P10/P50/P90: `0 / 0 / 1.827%`; P10 gate `>=20%` → FAIL
- coverage pass: `100%` / gate `>=95%` → PASS

G2:

- Not run because G1 failed.

Resource:

- eval p50/p95: `613.24 / 775.55 ms`
- max live score elements: `204,544`
- peak allocated bytes: `42,146,816`
- hidden N2: false

Interpretation:

- E-012A run-01 does not fail because active set is too small or spatial coverage is missing; it fails because the no-GT deterministic basin assignment locks onto wrong but self-consistent geometry.
- This sharply separates the bottleneck from E-011A-TOP128-GEOM-DIAG: top128 contains compressible GT inliers, but naive geometry-only residual feedback cannot identify them without stronger cross-modal/inlier evidence.
- Per protocol, no G2, E-012B, descriptor retraining, sparse affine feedback, OS424/confirm, or rescue was launched.

Artifacts:

- Local G0: `/Users/shihao/Documents/图像匹配之二/research/results/E-012A/g0.json`
- Local G1: `/Users/shihao/Documents/图像匹配之二/research/results/E-012A/g1.json`
- Local summary: `/Users/shihao/Documents/图像匹配之二/research/results/E-012A/summary.json`
- Local reviewer: `/Users/shihao/Documents/图像匹配之二/research/results/E-012A/reviewer_conclusion.md`
- Local manifest: `/Users/shihao/Documents/图像匹配之二/research/results/E-012A/RUN_MANIFEST.json`
- Remote per-pair metrics: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-012A/run-01/formal/per_pair.jsonl`
