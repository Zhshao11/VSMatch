# E-034B-E028B-RAW-TOP1-COARSE-SELECTION-DIAG：No-Train Top1 Coarse Selection

方法版本：`M001 v3.10 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-033A 显示 E028B raw correlation 的 GT-neighborhood top1 capture 为 `46.54%`，但 E-034A learned
decoder 最终 selected coarse token in GT-neighborhood 只有 `17.62%`。这意味着 learned decoder 可能把
raw correlation 的强信号用坏。

E-034B 只做一个窄诊断：

> 把 coarse 选择策略改成每个 source token 直接选 raw top1，正确粗点是否明显变多？正确粗点变多后，最终
> affine 是否改善？

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不训练、不 finetune descriptor；
- 不训练 matcher/decoder/head；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- 不使用 official testing.zip / confirm / OS424；
- raw top1/topK selected matches 必须先落盘，再读取 truth-flow 做 metric join；
- 不使用 random RANSAC、three-point、131k sampling；
- 完成后停止，不自动启动后续实验或 rescue。

## 3. Tracks

### A. `RAW_TOP1_SELECTION`

每个 source token 直接选择 raw correlation top1 target token。

报告：

- selected coarse token in GT-neighborhood；
- correct@2 / precision@2；
- selected count；
- region/quadrant coverage；
- coarse-only affine AEPE / CMR@1/2/5 / PLR / valid。

### B. `MINIMAL_ROBUST_VARIANTS`

在 raw top1 基础上测试最小 no-train 变体：

- top1 + mutual；
- top1 + margin / entropy filter；
- top1 + region-balanced quota；
- top1 + score weighted LS；
- top1 + deterministic residual trimming / IRLS。

不得使用 hidden RANSAC / random minimal solver。

### C. `TOPK_SIMPLE_SELECTION_SANITY`

对 top4/top8 做简单非学习选择：

- raw score top1 within topK；
- margin-ranked；
- dual-softmax-ranked；
- mutual-rank-ranked。

目标不是找最终方法，而是确认 E-034A learned decoder 是否低于简单 raw correlation baseline。

### D. `COARSE_TO_AFFINE_ATTRIBUTION`

输出每个变体：

- coarse GT-neighborhood capture；
- active count；
- outlier rate；
- affine AEPE/CMR；
- 与 E-034A selected `17.62%` / AEPE `99.877`；
- 与 E-032A best consensus AEPE `15.3632` / CMR@2 `4.0633%`。

## 4. Gates

### P0：setup/parity

必须 PASS：

- official training/testing archive SHA 与 pair count verified；
- E028B checkpoint SHA、feature hook、feature shape recorded；
- E-032A best consensus reference recorded；
- E-034A learned decoder reference recorded；
- E-033A raw top1/topK capture parity reproduced；
- no GDROS backend adapter / dense GRU reuse；
- raw top1/topK selected matches persisted before truth-flow metric join；
- baseline artifacts readonly；
- testing.zip / confirm / OS424 not used。

P0 fail：`E034B_SETUP_BLOCKED`。

### G0：execution

必须 PASS：

- raw top1 complete；
- mutual / margin-filter / region-balanced / weighted or trimmed variants complete or typed failure；
- selected matches written before labels/flow are read；
- dev404 coarse GT-neighborhood capture and affine AEPE/CMR computed；
- reviewer approves no leakage, no hidden RANSAC, and result interpretation。

G0 fail：`E034B_EXECUTION_BLOCKED`。

### G1：diagnosis

判读：

- `RAW_TOP1_SELECTION_GAIN`：raw top1 capture substantially exceeds E-034A selected `17.62%` and affine improves over E-034A；
- `RAW_TOP1_AFFINE_GAIN`：raw top1/robust variant improves over E-032A best consensus AEPE `15.3632` or CMR@2 `4.0633%`；
- `CONSENSUS_OUTLIER_LIMITED`：raw top1 capture high but affine remains poor due to outliers；
- `RAW_TOP1_NO_GAIN`：top1 capture/affine do not improve over E-034A/E-032A；
- `E034B_RESOURCE_BLOCKED`：resource/time prevents completion。

## 5. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `top1_selection_metrics.json`；
- `variant_table.json`；
- `coarse_to_affine_attribution.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。

完成后只回传精简结果和核心路径，等待方法任务和用户决策。不得自动启动 E034C/E035、training、descriptor
finetune、full sparse feedback、OS424/confirm 或 rescue。
