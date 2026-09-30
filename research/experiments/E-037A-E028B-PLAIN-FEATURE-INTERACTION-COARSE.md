# E-037A-E028B-PLAIN-FEATURE-INTERACTION-COARSE

方法版本：`M001 v3.19`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

在当前最好非 GDROS 后端路线 E-034C coarse reference 上，只在特征匹配前加入普通跨模态 feature
interaction，检验：

> interaction 后的特征是否能让 coarse candidate ranking / inlier ratio / affine CMR 变好？

本实验不换 E-034C 后半段，不做新 refinement，不接 GDROS backend。

## 2. 背景对照

主对照 E-034C `rank_growth_dual_256_irls`：

- AEPE `8.550`；
- CMR@2 `8.996%`；
- CMR@5 `38.783%`；
- CMR@16 `87.942%`；
- inlier `60.491%`。

E-036A/B/C 替代 refinement 全部失败，说明下一步应前移到 correlation 前的 feature interaction。

## 3. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不 finetune E028B backbone/adapter 本体；
- 只训练新增 feature interaction / projection / norm head；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- E-034C downstream coarse selection / rank-growth / IRLS 尽量保持不变；
- 不使用 official testing.zip / confirm / OS424；
- interacted features / selected matches 必须先落盘，再读取 truth-flow 做 metric join；
- 完成后停止，不自动启动 E038、rescue、descriptor finetune、full sparse feedback。

## 4. 方法 tracks

### `BASELINE_REPLAY`

复现 E-034C，并记录：

- top1/top4/top8/top16/top32 GT-neighborhood capture；
- rank P50/P90；
- correlation entropy；
- top1-top2 margin；
- mutuality；
- selected inlier ratio；
- AEPE、CMR@2/5/16。

### `PLAIN_CROSS_ATTENTION_2L`

- E028B OPT/SAR feature projection；
- 2 层 alternating self-attention + cross-attention；
- 2D positional encoding；
- residual update + L2 normalization；
- 输出 interacted feature；
- 复用 E-034C correlation/selection/IRLS。

### `PLAIN_CROSS_ATTENTION_4L`

同上，但 4 层，用于判断容量收益。

### `COST_VOLUME_CONTEXT_INTERACTION`

- 构建 full 32x32 token correlation 或 topK cost summary；
- 将 cost context 注入 token feature；
- 不直接输出匹配点；
- downstream 仍走 E-034C。

### `NO_INTERACTION_TRAINED_PROJECTION`

- 只训练 projection/norm；
- 不做 cross-image attention；
- 区分收益来自 interaction 还是简单特征适配。

## 5. 训练协议：充分训练 hard gate

G0 必须满足：

- CUDA available true；
- 可用多卡；
- 使用既有 `1607` train / `404` dev split；
- 不使用 official testing.zip；
- train-small overfit：至少 `16` pairs，要求 loss 下降且 top1/top4 capture 或 GT mass 上升；
- train-full 至少满足以下之一：
  - `>=60k` optimizer steps；
  - 或 `>=40` full-train epochs；
  - 或 reviewer 认可等价充分训练量；
- 至少两个 LR：`1.25e-4` 与半量级或相邻量级；资源允许加 `2.5e-4`；
- formal eval 至少包含 best train、best dev、final checkpoint；
- 训练曲线记录 loss、train/dev topK capture、entropy、margin、final E-034C metric；
- 如果资源不能满足，typed stop `E037A_RESOURCE_BLOCKED`，不得判方法失败。

## 6. 监督

训练只允许用 train raw truth_flow 构建监督：

- bidirectional coarse token CE；
- soft GT-neighborhood target，允许 1-token gaussian/soft label；
- optional mutual consistency loss；
- optional entropy/margin loss；
- optional affine utility auxiliary loss。

dev/formal 必须 no-GT inference：先落盘 interacted features / selected matches / affine，再 join truth-flow 计算指标。

## 7. Gates

### P0

必须 PASS：

- E-034C reference located/replayed；
- E-036A/B/C failed-result recorded；
- E028B checkpoint SHA recorded and read-only；
- no E028B finetune；
- no GDROS backend/dense GRU reuse；
- no testing.zip / confirm / OS424；
- output-before-GT-join verified。

### G0

必须 PASS：

- CUDA available true；
- train-small overfit positive；
- train-full meets sufficient-training gate or typed resource stop；
- losses finite/nonzero；
- gradients reach interaction/projection heads；
- strict checkpoint load verified；
- reviewer approves label construction, coordinate mapping, no leakage, training sufficiency, and baseline isolation。

### G1

主对照：E-034C AEPE `8.550`、CMR@2 `8.996%`、CMR@5 `38.783%`、CMR@16 `87.942%`。

判读：

- `FEATURE_INTERACTION_GAIN_FOUND`：AEPE < `8.550` 且 CMR@5 > `38.783%`，CMR@16 不明显崩溃；
- `RANKING_GAIN_FINAL_WEAK`：top1/top4 capture 或 margin/entropy 明显改善，但 final affine 不过门；
- `PROJECTION_ONLY_GAIN`：no-interaction projection 与 cross-attention 相当或更好；
- `PLAIN_INTERACTION_FAIL`：充分训练后 ranking/final 均无有效提升；
- `IMPLEMENTATION_OR_LABEL_SUSPECT`；
- `E037A_RESOURCE_BLOCKED`。

## 8. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `training_curve.json`；
- `ranking_diagnostics.json`；
- `coarse_pipeline_metrics.json`；
- `ablation_summary.json`；
- `per_pair.jsonl`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。
