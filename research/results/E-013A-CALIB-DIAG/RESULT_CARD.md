# E-013A-CALIB-DIAG RESULT_CARD

Status: **COMPLETED / NO_USABLE_RANKING_SIGNAL**

Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-013A-CALIB-DIAG/run-01`

Watchdog:

- Job: `20260810_203734_0d8a1a4b`
- Status: `succeeded`, return code `0`
- Log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260810_203734_0d8a1a4b.log`
- Report JSON: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_203734_0d8a1a4b.json`
- Report text: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_203734_0d8a1a4b.txt`

P0 parity/provenance:

- Verdict: **PASS**
- E-013A G1 parity reproduced:
  - active mean/P10/P50/P90: `0.656 / 0 / 0 / 2`
  - correct@2px mean/P10/P50/P90: `0.017 / 0 / 0 / 0`
  - precision mean/P10/P50/P90: `1.246% / 0 / 0 / 0`
  - coverage pass: `0%`
- Candidate-before-label join: PASS
- No training / finetune / checkpoint write / formal-result overwrite: PASS

Main diagnostic:

- Primary: **NO_USABLE_RANKING_SIGNAL**
- Secondary: none
- Final score train/dev AUROC: `0.5036 / 0.5024`
- Final score train/dev AP: `0.000477 / 0.000471`
- Final score train/dev positive rate: `0.0354% / 0.0355%`
- Best dev threshold sweep: threshold `0.0`, active `615.20`, correct `2.01`, precision `0.298%`, coverage `100%`
- Best dev topK/quota sweep: global topK-256, active `235.91`, correct `0.98`, precision `0.395%`, coverage `99.01%`
- Train best also fails: threshold `0.0`, active `629.11`, correct `2.29`, precision `0.325%`, coverage `100%`

Ablation signal:

- Pre-feedback endpoint score has weak ranking signal:
  - train/dev AUROC: `0.7026 / 0.7019`
  - train/dev AP: `0.002624 / 0.002560`
- Post-feedback/final score collapses near random:
  - train/dev AUROC: `0.5036 / 0.5024`
  - train/dev AP: `0.000477 / 0.000471`
- Basin-only is anti-informative: dev AUROC `0.0587`.

Interpretation:

- E-013A failure is not just threshold calibration: threshold-free and topK/quota sweeps do not approach G1.
- It is not primarily train-dev shift: train and dev fail similarly.
- It is not candidate-availability limited at the proposal level: dev positive-proposal P10 is `34`, though `3/404` pairs have zero positive proposals.
- The clearest bottleneck is that the final feedback/scoring path destroys or ignores the weak endpoint rank signal.

Artifacts:

- Local diagnostic summary: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A-CALIB-DIAG/diagnostic_summary.json`
- Local score separability: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A-CALIB-DIAG/score_separability.json`
- Local threshold sweep: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A-CALIB-DIAG/threshold_sweep.json`
- Local topK/quota sweep: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A-CALIB-DIAG/topk_quota_sweep.json`
- Local reviewer: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A-CALIB-DIAG/reviewer_conclusion.md`
- Local manifest: `/Users/shihao/Documents/图像匹配之二/research/results/E-013A-CALIB-DIAG/RUN_MANIFEST.json`
