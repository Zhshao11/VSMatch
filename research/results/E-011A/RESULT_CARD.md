# E-011A RESULT_CARD

Status: **COMPLETED / RAW_PATCH_RETRIEVER_FAIL**

Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-011A/run-01`

Watchdog:

- Job: `20260810_101308_3273a602`
- Status: `succeeded`, return code `0`
- Log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260810_101308_3273a602.log`
- Report JSON: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_101308_3273a602.json`
- Report text: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_101308_3273a602.txt`

Frozen identity:

- METHOD_DELTA: `925b2bda09f49d730fb736fc42bd8ee79c9d5bdf8c03ef0d4cbac0e1bceb8634`
- INNOVATION_CONTRACT: `e63a369f2cfa23bddbd581e65ee0adf284332b4c502c6ee26868ce5bc5c704ce`
- METHOD_SPEC: `1eb5907a6db2790c2e394907e4a4f89ed5eeb0b891313beae9d992dd891c9e9a`
- E-011A spec: `dcc64457fc853b09626300728f7903dcc72365a2c43ddd0f463de6a8f9c97a1e`
- REVIEW_PACKET: `ca334084a5b8097531bd130a565dac797048a1942a00c022ac0255800c0bfde2`
- CURRENT_STATE: `e4ce1408ca3afa32330eef9016b07ec1d1342b5f3862cbd297b61eefb275fab1`
- E-011A source SHA: `dfd7fe1b13da5beb875aac878c439c1f7e8ad25b50faa136af53ccba88ccaca1`

Implementation / isolation gates:

- Frozen SHA validation: PASS
- `py_compile`: PASS
- Unit tests: `5 passed`
- Smoke train/eval: PASS
- Baseline isolation audit: PASS
- Targeted review: APPROVE

Training:

- Train split records: `1607`
- Epochs: `4`
- Parameters: `146,402`
- Final epoch loss: `6.955725`
- Checkpoint: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-011A/run-01/models/seed-20260820.pt`
- Checkpoint SHA256: `44131806c81881843e4b315cd76fe52f1cadcdc8cbea756ea5a77896d216aacd`
- Train seconds: `2220.319`
- Train peak allocated: `165,992,960` bytes
- Raw-image descriptor path: confirmed
- E-008A cached tail as main input: false

Formal dev404 top64 main result:

- mean correct@2px proposals: `45.8787` / gate `>=60.0` → FAIL
- P10/P50/P90: `20 / 44 / 73.7` / gate P10/P50 `>=31/56` → FAIL
- candidate base rate@2px: `0.04734%` / gate `>=0.0619%` → FAIL
- nearest P50/P90: `19.4599 / 82.9717` / frozen target `<15.490 / <75.309` → FAIL
- oracle affine AEPE: `24.4642`
- PLR/valid: `95.297% / 95.297%`

G1 verdict: **FAIL**. G2 was not evaluated because G1 failed.

TopR diagnostic:

| R | mean correct@2px | P10/P50/P90 | base@2px | nearest P50/P90 | oracle AEPE | PLR/valid |
|---:|---:|---|---:|---|---:|---|
| 4 | 8.399 | 2 / 8 / 15 | 0.13866% | 81.120 / 220.901 | 393.021 | 23.267% / 23.267% |
| 8 | 14.067 | 4 / 13 / 24.7 | 0.11612% | 57.189 / 165.557 | 233.494 | 54.455% / 54.455% |
| 16 | 21.384 | 7 / 20.5 / 36 | 0.08826% | 40.183 / 126.669 | 132.194 | 74.257% / 74.257% |
| 32 | 31.092 | 12 / 29.5 / 52.7 | 0.06417% | 28.074 / 100.504 | 67.583 | 86.881% / 86.881% |
| 64 | 45.879 | 20 / 44 / 73.7 | 0.04734% | 19.460 / 82.972 | 24.464 | 95.297% / 95.297% |
| 128 | 68.710 | 34 / 66 / 106 | 0.03545% | 13.129 / 70.895 | 14.249 | 97.277% / 97.277% |

Interpretation:

- E-011A fails the pre-registered top64 gate and is worse than frozen S2 at the official top64 decision boundary.
- The top128 diagnostic improves substantially and exceeds the E-006A mean/P50 candidate-count thresholds, but top128 is explicitly diagnostic and cannot rescue top64 failure.
- This suggests raw patch descriptors do add recoverable evidence deeper in the ranked list, but the learned retriever is not ranking enough correct candidates into top64.

Artifacts:

- TopR summary local: `/Users/shihao/Documents/图像匹配之二/research/results/E-011A/topr_summary.json`
- Train report local: `/Users/shihao/Documents/图像匹配之二/research/results/E-011A/train.json`
- Run manifest local: `/Users/shihao/Documents/图像匹配之二/research/results/E-011A/RUN_MANIFEST.json`
- Baseline isolation audit local: `/Users/shihao/Documents/图像匹配之二/research/results/E-011A/baseline_isolation_audit.md`
- Reviewer conclusion local: `/Users/shihao/Documents/图像匹配之二/research/results/E-011A/reviewer_conclusion.md`
- Per-pair metrics remote: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-011A/run-01/formal/topr_pair_metrics.jsonl`
