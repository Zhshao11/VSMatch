# E-035B-E028B-LOCAL-REFINEMENT-SIGNAL-DIAG：Does Local Refinement Have Observable Signal?

方法版本：`M001 v3.14 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-035A 的 best window oracle CMR@5 为 `100.0%`，但 learned two-stage/direct refiner 均劣化。
因此需要先诊断：

> 在 E-034C coarse 结果给出的 local window 内，是否存在 no-GT 可观测信号能够把 GT endpoint 排到前面？

这回答用户关心的问题：是否真的存在一个 refinement 方法能将 16px basin 推到更细粒度，还是 window oracle
只是“GT 坐标可达”但无可学习/可观测依据。

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不训练/finetune E028B descriptor；
- 不训练新 refiner/head；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- primary coarse input 固定为 E-034C best affine/CMR variant `rank_growth_dual_256_irls`；
- 优先只读 E-035A/E-034C artifacts；必要时可重算 local signal maps；
- no-GT local signal maps 必须先落盘，再读取 truth-flow 做 rank/metric join；
- 不使用 official testing.zip / confirm / OS424；
- 完成后停止，不自动启动 E035C、training、descriptor finetune、full sparse feedback 或 rescue。

## 3. Tracks

### A. `LOCAL_WINDOW_REPLAY`

复现：

- E-034C coarse replay AEPE `8.550`、CMR@2 `8.996%`、CMR@5 `38.783%`、CMR@16 `87.942%`；
- E-035A best window oracle CMR@5 `100.0%`。

### B. `NO_GT_LOCAL_SIGNAL_RANKING`

在每个 selected coarse match 的 local window 内计算 no-GT signal heatmaps：

- E028B local feature correlation；
- raw OPT vs log-SAR NCC / ZNCC；
- gradient magnitude / edge correlation；
- census / rank transform correlation；
- phase-like correlation if cheap；
- local mutual/cycle consistency；
- heatmap entropy、margin、peak sharpness。

对每个 signal 输出：

- GT endpoint rank；
- top1 / top4 / top8 capture；
- AP / AUROC；
- local CMR@2/5 proxy；
- per-pair refined affine AEPE / CMR@2/5 if using that signal top1。

### C. `WINDOW_CENTER_AND_RADIUS_ATTRIBUTION`

比较：

- center = coarse target token center；
- center = E-034C affine-predicted target；
- center = fused center；
- radius = `±8 / ±16 / ±24` px。

报告 contains-GT、best no-GT signal、oracle/noGT gap。

### D. `OUTLIER_CONFIDENCE_DIAG`

判断 local signals 是否能区分：

- 可 refine 的 good coarse matches；
- 不可 refine/outlier coarse matches；
- high-confidence wrong local peaks。

报告 confidence/error calibration。

## 4. Gates

### P0：setup/parity

必须 PASS：

- E-035A RESULT_CARD / local_window_oracle / coarse replay artifacts located；
- E-034C primary coarse reference recorded；
- E028B checkpoint SHA recorded；
- no E028B descriptor mutation；
- no GDROS backend adapter / dense GRU reuse；
- no new training；
- baseline artifacts readonly；
- testing.zip / confirm not used。

P0 fail：`E035B_SETUP_BLOCKED`。

### G0：diagnostic execution

必须 PASS：

- local windows and no-GT local signal maps materialized before truth-flow rank join；
- all specified signal families complete or typed unavailable；
- GT local rank/topK capture/AP/AUROC computed；
- center/radius attribution complete；
- reviewer approves no leakage, coordinate mapping, and interpretation。

G0 fail：`E035B_EXECUTION_BLOCKED`。

### G1：refinement feasibility diagnosis

判读：

- `SIGNAL_EXISTS_REFINER_DESIGN_BAD`：no-GT local signal clearly ranks GT, so E-035A refiner design/training is bad；
- `LOCAL_SIGNAL_WEAK`：window oracle strong but no-GT local signals cannot rank GT；
- `CENTERING_LIMITED`：signal exists only with better center/radius；
- `OUTLIER_CONFIDENCE_LIMITED`：good matches can be refined but bad coarse/outliers cannot be rejected；
- `MIXED_REFINEMENT_SIGNAL`；
- `E035B_RESOURCE_BLOCKED`。

## 5. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `local_signal_table.json`；
- `center_radius_table.json`；
- `signal_top1_affine_metrics.json`；
- `outlier_confidence_report.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。

完成后只回传精简结果和核心路径，等待方法任务和用户决策。不得自动启动 E035C、training、descriptor
finetune、full sparse feedback、OS424/confirm 或 rescue。
