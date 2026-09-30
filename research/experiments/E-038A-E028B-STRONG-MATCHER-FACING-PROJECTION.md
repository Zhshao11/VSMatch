# E-038A-E028B-STRONG-MATCHER-FACING-PROJECTION

方法版本：`M001 v3.20`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-037A 的成功点是 `NO_INTERACTION_TRAINED_PROJECTION_lr125e4`，不是 plain cross-attention。E-038A
系统增强 matcher-facing projection/norm 适配层，测试：

> 不引入复杂 feature interaction、保持 E-034C 后半段不变时，projection/norm 最多能把 coarse matching
> 与 final affine 指标拉到哪里？

## 2. 主对照

E-034C `rank_growth_dual_256_irls`：

- AEPE `8.550`；
- CMR@2 `8.996%`；
- CMR@5 `38.783%`；
- CMR@16 `87.942%`；
- inlier `60.491%`。

E-037A best projection/norm：

- AEPE `7.247`；
- CMR@2 `10.720%`；
- CMR@5 `45.939%`；
- CMR@16 `92.337%`。

## 3. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不 finetune E028B backbone/adapter 本体；
- 不做 cross-attention / cross-image message passing；
- 只训练新增 matcher-facing projection / normalization / scoring-temperature head；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- E-034C downstream coarse selection / rank-growth / IRLS 必须保持不变；
- 不使用 official testing.zip / confirm / OS424；
- projected features / selected matches 必须先落盘，再读取 truth-flow 做 metric join；
- 完成后停止，不自动启动 E039、rescue、descriptor finetune、feature interaction、full sparse feedback。

## 4. 必做 tracks

### `BASELINE_REPLAY`

- replay E-034C；
- replay E-037A best projection/norm；
- 记录 top1/top4/top8/top16/top32 GT-neighborhood capture、rank P50/P90、entropy、top1-top2
  margin、mutuality、selected inlier ratio、AEPE、CMR@2/5/16。

### `PROJECTION_DEPTH_WIDTH_SWEEP`

- linear projection；
- 2-layer MLP；
- 4-layer residual MLP；
- widths `{256,512,1024}` 或资源等价；
- shared projection vs modality-specific stem + shared projection；
- all outputs L2 normalized。

### `NORM_AND_WHITENING_SWEEP`

- L2 norm；
- LayerNorm + L2；
- GroupNorm/BatchNorm variants if stable；
- learnable temperature；
- optional train-set whitening/decorrelation head。

Whitening/decorrelation statistics 只能来自 train split，禁止 dev/test leakage。

### `LOSS_AND_NEGATIVE_SWEEP`

- bidirectional coarse token CE；
- soft GT-neighborhood gaussian target；
- supervised contrastive / InfoNCE；
- hard-negative mining：same-row/nearby/visually confusing negatives；
- rank margin loss：提升 GT-neighborhood 相对 top wrong 的 score margin；
- entropy sharpening / confidence calibration。

### `SCHEDULE_AND_LR_SWEEP`

- LR 至少 `6.25e-5`、`1.25e-4`、`2.5e-4`；
- 保存并 formal eval best train、best dev、final/EMA checkpoint；
- 短 schedule 可以用于筛配置，但正式 claim 必须来自充分训练配置。

## 5. 训练充分 hard gate

G0 必须满足：

- CUDA available true；
- 可用多卡；
- 使用既有 `1607` train / `404` dev split；
- 不使用 official testing.zip；
- train-small overfit：至少 `16` pairs，loss 下降且 top1/top4 capture 或 GT mass 上升；
- formal candidate train-full 至少满足：
  - `>=60k` optimizer steps；
  - 或 `>=40` full-train epochs；
  - 或 reviewer 认可等价充分训练量；
- 主结果至少包含 LR `6.25e-5`、`1.25e-4`、`2.5e-4`，除非 typed resource stop；
- 训练曲线记录 loss、train/dev topK capture、rank、entropy、margin、final E-034C metric；
- 如果资源不能满足，typed stop `E038A_RESOURCE_BLOCKED`，不得判方法失败。

## 6. 监督

训练只允许用 train raw truth_flow 构建监督：

- bidirectional coarse token CE；
- soft GT-neighborhood target；
- optional supervised contrastive / InfoNCE；
- optional rank-margin/hard-negative loss；
- optional entropy/margin calibration；
- optional affine utility auxiliary loss。

dev/formal 必须 no-GT inference：先落盘 projected features / selected matches / affine，再 join truth-flow 计算指标。

## 7. Gates

### P0

必须 PASS：

- E-034C reference located/replayed；
- E-037A success and projection-only interpretation recorded；
- E028B checkpoint SHA recorded and read-only；
- no E028B finetune；
- no cross-attention / cross-image interaction；
- no GDROS backend/dense GRU reuse；
- no testing.zip / confirm / OS424；
- output-before-GT-join verified。

### G0

必须 PASS：

- CUDA available true；
- train-small overfit positive；
- train-full meets sufficient-training gate or typed resource stop；
- losses finite/nonzero；
- gradients reach projection/norm/temperature heads；
- strict checkpoint load verified；
- reviewer approves label construction, coordinate mapping, no leakage, training sufficiency, and baseline isolation。

### G1

判读：

- `PROJECTION_STRONG_GAIN_FOUND`：AEPE < `7.247` 且 CMR@5 > `45.939%`，CMR@16 不明显崩溃；
- `PROJECTION_MAJOR_GAIN_FOUND`：AEPE <= `6.5` 或 CMR@5 >= `52%`，且另一项不低于 E-037A；
- `RANKING_GAIN_FINAL_WEAK`：top1/top4 capture 或 margin/entropy 明显改善，但 final affine 不过 E-037A；
- `PROJECTION_TUNING_PLATEAU`：充分调参后不超过 E-037A；
- `IMPLEMENTATION_OR_LABEL_SUSPECT`；
- `E038A_RESOURCE_BLOCKED`。

## 8. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `training_curve.json`；
- `ranking_diagnostics.json`；
- `coarse_pipeline_metrics.json`；
- `projection_ablation_summary.json`；
- `per_pair.jsonl`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。
