# E-029A-GDROS-PAPER-PROTOCOL-RETRAIN：GDROS Paper Train/Test Protocol Retrain From Scratch

方法版本：`M001 v3.4 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

按用户要求，严格使用论文/官方 train-test 协议重训 GDROS：

- official `training.zip` 全 `2011` 对训练；
- official `testing.zip` 全 `424` 对只做最终测试；
- 不加载官方已训练 GDROS 权重作为训练初始化。

核心问题：

> 当 split 与论文协议一致时，从头按 official recipe 训练的 GDROS 能否接近 paper / released checkpoint 指标？

## 2. 必须先核对的官方协议

P0 必须先确认并记录：

- training.zip SHA `2887e7de8cd72290df688de0ce4071f42d7bfdec8507a24716ae2ea15c3f3cf5`，count `2011`；
- testing.zip SHA `e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218`，count `424`；
- official README / train command / test command / config；
- optimizer、LR schedule、steps/epochs、batch size、augmentation、loss、iters、checkpoint selection；
- 所有初始化来源；
- official testing.zip 未用于训练、validation、early stop、checkpoint/hparam selection。

P0 不通过不得启动正式训练。

## 3. Tracks

### A. `GDROS_RELEASED_CHECKPOINT_TEST_REFERENCE`

- 只读 replay official released checkpoint on official testing.zip 424；
- 记录 paper/reproduced AEPE、CMR@1/2/5、public-code variance；
- 不得作为训练初始化。

### B. `PAPER_PROTOCOL_GDROS_RETRAIN_FROM_SCRATCH`

- full training.zip 2011 训练；
- 不加载官方已训练 GDROS checkpoint；
- 若 official recipe 允许非 GDROS generic pretrain，如 RAFT/Sintel，必须记录；
- final checkpoint 或 official selection rule 评估 official testing.zip 424。

## 4. 边界

- 原始 GDROS checkpoint/code/cache/run 只读；
- 训练只写本实验目录；
- testing.zip 只用于 final evaluation；
- 若需要 validation，只能从 training.zip 内部划分；
- 不使用 confirm 或额外测试数据；
- 长训交给 watchdog，不持续轮询。

## 5. Metrics

必须报告：

- AEPE；
- CMR@1/2/5；
- public-code variance if evaluator reports；
- PLR/finite/in-bounds；
- train loss、runtime、peak memory、wall time；
- 与 paper/released checkpoint reference、train1607/dev404 official retrain、same-budget short retrain 对比。

## 6. Gate

### P0：paper protocol alignment

必须 PASS：

- training/testing SHA 与 pair count verified；
- official train/test commands/config/SHA recorded；
- official trained GDROS weights not loaded for training；
- all initialization sources recorded；
- original baseline readonly/no writes；
- testing.zip not used for selection；
- evaluator matches released checkpoint reference。

P0 fail：`PAPER_PROTOCOL_SETUP_BLOCKED`。

### G0：training execution

必须 PASS：

- training completes planned stop or documented resource stop；
- train loss finite/decreases；
- checkpoints strict-load；
- train no-GT inference finite；
- formal testing output materialized before truth_flow metric join；
- reviewer approves protocol/no-weight audit。

G0 fail：`PAPER_PROTOCOL_RETRAIN_EXECUTION_BLOCKED`。

### G1：paper split reproduction

判读：

- `PAPER_PROTOCOL_RETRAIN_REPRODUCES_BASELINE`：AEPE `<=3 px` or within `2x` released checkpoint reference；
- `PAPER_PROTOCOL_RETRAIN_REASONABLE_BUT_BELOW_BASELINE`：AEPE `<=8 px` but not near reference；
- `PAPER_PROTOCOL_RETRAIN_FAILS_TO_REPRODUCE`：AEPE `>8 px` after completed recipe；
- `PAPER_PROTOCOL_RESOURCE_BLOCKED`：resource/time prevents completion；
- `PAPER_PROTOCOL_UNCLEAR_OR_UNAVAILABLE`：official train/test protocol cannot be audited enough。

## 7. 审阅

需要两路 reviewer：

1. paper-protocol/model reviewer：
   - official train/test split 是否匹配论文；
   - no official trained GDROS weight 是否成立；
   - recipe / initialization / schedule 是否准确。
2. evaluator/protocol reviewer：
   - testing.zip 是否只用于 final；
   - released checkpoint reference/evaluator 是否一致；
   - metric aggregation 与 comparison 是否正确。

blocking finding：`IMPLEMENTATION_BLOCKED`。

## 8. 输出

只回传：

- `RESULT_CARD.md`
- `paper_protocol_manifest.json`
- `initialization_manifest.json`
- `baseline_reference_manifest.json`
- `train_report.json`
- `official_testing_metrics.json`
- `resource_report.json`
- `comparison_table.json`
- `isolation_audit.md`
- reviewer conclusions
- `RUN_MANIFEST.json`
- watchdog log/report

完成后停止，等待方法决策；不得自动启动 E-028B integration rerun、轻量 matcher、E-029B/E-030A、confirm 或 rescue。
