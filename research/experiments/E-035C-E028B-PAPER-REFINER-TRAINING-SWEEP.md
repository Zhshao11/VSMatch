# E-035C-E028B-PAPER-REFINER-TRAINING-SWEEP：Trainable Paper-Style Local Refinement

方法版本：`M001 v3.15 design`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-035A 的 local window oracle CMR@5 为 `100.0%`，但 learned refiner 劣化；E-035B 显示传统
no-train local signals weak。此前没有正式实现/训练 LoFTR、Efficient LoFTR、RoMa、SURE 风格的可训练
fine refinement。

E-035C 固定 E-034C/E-035A coarse input 与 E028B coarse feature，训练多种 paper-inspired local
refinement heads，回答：

> 是否存在一个可训练 refinement 方法，能把当前 16px basin 稳定推向 5px/2px？

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不 finetune E028B descriptor；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- 允许训练 local fine encoder / local refinement head / confidence or uncertainty head；
- 允许多卡训练，但必须记录 GPU 数、batch、seed、训练时长、显存；
- primary coarse input 固定为 E-034C best affine/CMR variant `rank_growth_dual_256_irls`；
- E-034C best inlier 只能作为 diagnostic secondary；
- 不使用 official testing.zip / confirm / OS424；
- selected/refined outputs 必须先落盘，再读取 truth-flow 做 metric join；
- 完成后停止，不自动启动 E035D、descriptor finetune、full sparse feedback 或 rescue。

## 3. 论文参考路线

实现不是逐字复刻，但要覆盖以下设计思想：

- LoFTR：coarse match 后，在 fine-level feature local window 内做 fine matching / soft-argmax；
- Efficient LoFTR：two-stage fine correlation，先粗局部再 tiny-window subpixel；
- RoMa：frozen foundation coarse feature + specialized ConvNet fine feature + match decoder + regression-by-classification；
- SURE：coordinate regression 同时输出 uncertainty/confidence，避免 overconfident wrong refinement。

## 4. Tracks

### A. `COARSE_AND_ORACLE_REPLAY`

复现：

- E-034C/E-035A primary coarse replay；
- E-035A window oracle；
- E-035B best no-train local signal。

### B. `LOFTR_STYLE_FINE_WINDOW`

- 训练 lightweight fine encoder，输出 source/target fine local features；
- 对 selected coarse match 裁局部窗口；
- local correlation heatmap + soft-argmax；
- heatmap CE/focal + offset Huber；
- 输出 refined endpoint + confidence。

### C. `EFFICIENT_LOFTR_STYLE_TWO_STAGE`

- Stage1：large local window `16px -> 5px`；
- Stage2：tiny window `5px -> 2px`，可用 x/y 1D heatmap 或 small 2D heatmap；
- 分段只作为诊断 scaffold，不作为最终方法形态 claim；
- 必须报告 stage1-only 与 stage1+stage2。

### D. `ROMA_STYLE_FINE_CONVNET_RBC`

- E028B/DINO-like coarse feature 保持冻结；
- 训练 specialized ConvNet fine feature from OPT/SAR raw/structure inputs；
- local decoder 输出 discrete bins + robust continuous residual；
- 使用 regression-by-classification + robust regression；
- 输出 uncertainty/confidence。

### E. `SURE_STYLE_UNCERTAINTY_HEAD`

- 在 B/C/D 的最佳 track 上加 uncertainty/evidential or calibrated confidence head；
- 用 confidence/uncertainty 过滤或降权 refined matches；
- refined endpoints 再做 weighted affine / IRLS。

## 5. Gates

### P0：setup/parity

必须 PASS：

- official training/testing archive SHA 与 pair count verified；
- E028B checkpoint SHA、feature hook、feature shape recorded；
- E-034C primary coarse artifacts located/replayed；
- E-035A window oracle and E-035B signal diagnostic references recorded；
- no E028B descriptor mutation；
- no GDROS backend adapter / dense GRU reuse；
- baseline artifacts readonly；
- testing.zip / confirm / OS424 not used。

P0 fail：`E035C_SETUP_BLOCKED`。

### G0：training sanity

必须 PASS：

- each trainable track has finite nonzero losses；
- gradients reach fine encoder/refinement/confidence heads where enabled；
- train-small overfit sanity improves local endpoint error or heatmap GT mass；
- train-full completes or typed resource failure；
- strict checkpoint load verified；
- refined outputs written before truth-flow metric join；
- reviewer approves no leakage, coordinate mapping, training protocol, and baseline isolation。

G0 fail：`E035C_TRAINING_BLOCKED`。

### G1：dev404 refinement diagnosis

主对照：E-034C/E-035A coarse replay AEPE `8.550`、CMR@2 `8.996%`、CMR@5 `38.783%`、
CMR@16 `87.942%`。

判读：

- `PAPER_REFINER_GAIN_FOUND`：best refined track improves CMR@5 over `38.783%` and AEPE under `8.550`
  without CMR@16 collapse；
- `CMR5_GAIN_ONLY`：CMR@5 improves but AEPE/CMR@2 remain weak；
- `REFINER_OBSERVATION_STILL_WEAK`：trained paper-style refiners cannot beat coarse replay；
- `UNCERTAINTY_CLEANUP_LIMITED`：local endpoints improve but confidence/outlier cleanup fails final affine；
- `E035C_RESOURCE_BLOCKED`。

完成后只写结果卡并等待用户；不得自动启动 E035D、descriptor finetune、full sparse feedback、OS424/confirm 或 rescue。

## 6. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `coarse_oracle_replay.json`；
- `loftr_style_metrics.json`；
- `efficientloftr_style_metrics.json`；
- `roma_style_metrics.json`；
- `uncertainty_cleanup_report.json`；
- `training_resource_report.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。
