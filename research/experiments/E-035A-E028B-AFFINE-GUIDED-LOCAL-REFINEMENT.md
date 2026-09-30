# E-035A-E028B-AFFINE-GUIDED-LOCAL-REFINEMENT：16px Basin to Fine Correspondence

方法版本：`M001 v3.13 design`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-034C best affine/CMR 已达到 CMR@16 `87.942%`，但 CMR@5 只有 `38.783%`、CMR@2
只有 `8.996%`。E-034D/CMR@16 补算说明当前粗几何多数已进入 16px basin，而不是完全错配。

E-035A 固定 E-034C best coarse selection 与 E028B feature，设计 local refinement：

> 把 16px token-level coarse correspondences 推向 5px/2px 的 continuous endpoint，同时用 confidence/uncertainty
> 抑制不可细化或 outlier matches。

分段 16→5→2 只作为诊断 scaffold；最终方法形态目标是 direct continuous refinement head。

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不训练/finetune E028B descriptor；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- 允许训练 local refinement head / confidence head；
- primary coarse input 固定为 E-034C best affine/CMR variant `rank_growth_dual_256_irls`；
- 可把 E-034C best inlier variant 作为 diagnostic input，但不能替代 primary；
- 不使用 official testing.zip / confirm / OS424；
- selected/refined matches 必须先落盘，再读取 truth-flow 做 metric join；
- 完成后停止，不自动启动 E035B、descriptor finetune、full sparse feedback 或 rescue。

## 3. 参考设计

借鉴 LoFTR/Efficient LoFTR/RoMa/SURE 类 coarse-to-fine refinement：

- coarse match 后在 fine/local feature 或 raw/structure patch 上开局部窗口；
- 输出 local heatmap / 1D heatmaps / soft-argmax；
- 预测 uncertainty/confidence；
- refined correspondences 再做 weighted affine / IRLS。

## 4. Tracks

### A. `COARSE_INPUT_REPLAY`

复现 E-034C primary coarse input：

- AEPE `8.550`；
- CMR@2 `8.996%`；
- CMR@5 `38.783%`；
- CMR@16 `87.942%`。

### B. `LOCAL_WINDOW_ORACLE`

对 selected coarse matches 做局部窗口理论上限诊断：

- window center：coarse target token center、E-034C affine-predicted target、二者融合；
- radius：`±8 / ±16 / ±24` px；
- 统计 window contains GT endpoint；
- oracle local endpoint bound 的 AEPE / CMR@2/5/16。

### C. `DIAGNOSTIC_TWO_STAGE_REFINEMENT`

诊断 scaffold：

1. Stage1 `16px -> 5px`
   - affine-guided local window；
   - heatmap CE / focal 或 distribution loss；
   - output local endpoint + confidence。

2. Stage2 `5px -> 2px`
   - tiny-window soft-argmax 或 x/y 1D heatmaps；
   - output sub-pixel offset + uncertainty。

必须分别报告 stage1-only 和 stage1+stage2，判断哪一步更难。

### D. `DIRECT_CONTINUOUS_REFINEMENT_HEAD`

最终形态候选：

- 输入 coarse match、E028B local feature context、raw/log-SAR、gradient/edge/census-like structure、current affine residual；
- 单个 head 直接输出 continuous endpoint offset、confidence、uncertainty；
- 与 two-stage scaffold 对比。

### E. `CONFIDENCE_AND_OUTLIER_CLEANUP`

refined matches 进入 affine 前，使用：

- refinement confidence；
- uncertainty；
- local heatmap entropy；
- affine residual；
- deterministic weighted affine / IRLS。

报告 confidence 是否能剔除 bad coarse/refinement outlier。

## 5. Gates

### P0：setup/parity

必须 PASS：

- official training/testing archive SHA 与 pair count verified；
- E028B checkpoint SHA、feature hook、feature shape recorded；
- E-034C best coarse selection artifacts located/replayed；
- E-034D CMR@16 and oracle references recorded；
- no E028B descriptor mutation；
- no GDROS backend adapter / dense GRU reuse；
- baseline artifacts readonly；
- testing.zip / confirm / OS424 not used。

P0 fail：`E035A_SETUP_BLOCKED`。

### G0：training/execution sanity

必须 PASS：

- no-refine replay complete；
- local window oracle complete；
- two-stage and direct-head training/execution complete or typed failure；
- losses finite/nonzero and gradients reach refinement/confidence heads where enabled；
- selected/refined outputs written before truth-flow metric join；
- reviewer approves no leakage, coordinate mapping, checkpoint isolation, and interpretation。

G0 fail：`E035A_EXECUTION_BLOCKED`。

### G1：local refinement diagnosis

判读：

- `LOCAL_REFINEMENT_GAIN_FOUND`：refined output improves over E-034C best affine CMR@5 `38.783%` and AEPE
  `8.550` without CMR@16 collapse；
- `TWO_STAGE_DIAG_ONLY_DIRECT_HEAD_NEEDED`：two-stage diagnostic works but direct-head underperforms；
- `WINDOW_ORACLE_LIMITED`：selected coarse matches often do not contain GT in local windows；
- `REFINER_LEARNING_LIMITED`：window oracle is good but learned refiner fails；
- `CONFIDENCE_OUTLIER_LIMITED`：refined points improve individually but final affine remains outlier-contaminated；
- `NO_LOCAL_REFINEMENT_GAIN`；
- `E035A_RESOURCE_BLOCKED`。

## 6. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `coarse_replay_metrics.json`；
- `local_window_oracle.json`；
- `two_stage_refinement_metrics.json`；
- `direct_head_metrics.json`；
- `confidence_outlier_report.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。

完成后只回传精简结果和核心路径，等待方法任务和用户决策。不得自动启动 E035B、descriptor finetune、
full sparse feedback、OS424/confirm 或 rescue。
