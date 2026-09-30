# E-029A-GDROS-OFFICIAL-RETRAIN：Official GDROS Recipe Retrain From Scratch

方法版本：`M001 v3.4 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

用户要求按官方训练协议重新训练 GDROS，并且不使用官方已训练 GDROS 权重，检查能否达到或接近原 baseline 指标。
本实验建立完整官方 recipe 重训审计，区别于 E-029A-GDROS-RETRAIN-DIAG 的 same-budget short retrain。

核心问题：

> official GDROS training recipe 在 no official trained weight initialization 条件下，能否复现 GDROS full baseline？

## 2. 术语澄清

- `COPIED_GDROS_REPLAY_SANITY`：加载已有 GDROS checkpoint，不训练，只 replay/evaluate。
- `COPIED_GDROS_SAME_BUDGET_RETRAIN`：GDROS frontend/features + E-029A-like 短训 harness，不是 E-028B 替换特征。
- `OFFICIAL_RECIPE_GDROS_RETRAIN_FROM_SCRATCH`：本实验 primary；按官方 recipe 训练，不加载官方已训练 GDROS 权重。

## 3. Tracks

### A. `OFFICIAL_RECIPE_DISCOVERY`

必须读取并记录：

- official README / train scripts / config；
- optimizer、lr schedule、epoch/steps、batch size；
- augmentation、loss、unroll / recurrence、mixed precision；
- checkpoint selection / validation 规则；
- seed、hardware assumptions；
- 所有初始化权重来源。

### B. `GDROS_OFFICIAL_REPLAY_REFERENCE`

- 只读 replay official/frozen GDROS checkpoint；
- 记录 checkpoint/config/source SHA；
- 确认 formal dev evaluator 与 reference。

### C. `OFFICIAL_RECIPE_GDROS_RETRAIN_FROM_SCRATCH`

- 不加载官方已训练 GDROS checkpoint；
- 按 official recipe 训练 GDROS frontend+backend；
- 训练输出只写实验目录；
- formal dev404 只在最终 checkpoint/official selection 后评价。

若 official recipe 依赖非 GDROS 通用预训练，必须单独记录来源；若与“不使用官方权重”语义冲突，primary 仍以
不加载官方已训练 GDROS 权重为硬条件，并在 RESULT_CARD 中说明。

## 4. 边界

- 原始官方 GDROS checkpoint 只读，不得作为训练初始化；
- 原始 baseline code/cache/run 不写入；
- 不使用 OS424/confirm/datum；
- dev404 不用于调参、early stop 或 checkpoint selection；
- 若 official recipe 需要 validation，必须从 train 内部划分；
- 长训练必须交给 watchdog，不持续轮询。

## 5. Metrics

必须报告：

- AEPE；
- CMR@1/2/5；
- PLR；
- finite / in-bounds；
- train/dev gap；
- runtime / peak memory / wall time；
- 与 official replay reference、E-029A best、E-028B fair reference 对比；
- 是否达到 official baseline reproduction criterion。

## 6. Gate

### P0：official recipe and no-weight audit

必须 PASS：

- official training recipe source/config/SHA recorded；
- official trained GDROS weights not loaded for training；
- all initialization sources recorded；
- original baseline readonly/no writes；
- dataset split/archive SHA verified；
- no OS424/confirm/datum；
- train/dev isolation and no dev checkpoint selection；
- evaluator same as replay reference。

P0 fail：`OFFICIAL_GDROS_RETRAIN_SETUP_BLOCKED`。

### G0：training execution sanity

必须 PASS：

- official-recipe training launches and completes to planned stop or documented resource stop；
- train loss finite/decreases；
- checkpoints strict-load；
- train no-GT inference finite；
- formal dev output materialized before truth_flow metric join；
- reviewer approves no-official-weight and recipe equivalence audit。

G0 fail：`OFFICIAL_GDROS_RETRAIN_EXECUTION_BLOCKED`。

### G1：retrain result interpretation

判读：

- `OFFICIAL_RETRAIN_REPRODUCES_BASELINE`：AEPE `<=3 px` or within `2x` replay reference；
- `OFFICIAL_RETRAIN_REASONABLE_BUT_BELOW_BASELINE`：AEPE `<=8 px` but not near replay；
- `OFFICIAL_RETRAIN_FAILS_TO_REPRODUCE`：AEPE `>8 px` after completed official recipe；
- `OFFICIAL_RECIPE_RESOURCE_BLOCKED`：official recipe cannot complete due resource/time, with partial progress card；
- `OFFICIAL_RECIPE_UNCLEAR_OR_UNAVAILABLE`：cannot identify/audit official recipe enough for claim。

## 7. 审阅

需要两路 reviewer：

1. official-recipe/model reviewer：
   - recipe discovery 是否完整；
   - no official trained GDROS weight 是否成立；
   - 初始化来源与训练 schedule 是否准确；
   - training run 是否按 official recipe。
2. evaluator/protocol reviewer：
   - replay reference/evaluator 是否一致；
   - no dev selection/no OS424/datum；
   - metric aggregation 与 comparison 是否正确。

blocking finding：`IMPLEMENTATION_BLOCKED`。

## 8. 输出

只回传：

- `RESULT_CARD.md`
- `official_recipe_manifest.json`
- `initialization_manifest.json`
- `baseline_reference_manifest.json`
- `train_report.json`
- `dev_metrics.json`
- `resource_report.json`
- `comparison_table.json`
- `isolation_audit.md`
- reviewer conclusions
- `RUN_MANIFEST.json`
- watchdog log/report

完成后停止，等待方法决策；不得自动启动 E-029B/E-030A、E-028B integration rerun、轻量 matcher、OS424/confirm 或 rescue。
