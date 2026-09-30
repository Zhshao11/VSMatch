# E-029A-GDROS-RETRAIN-DIAG：Copied GDROS Retrain Parity Audit

方法版本：`M001 v3.4 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-029A 的接入实验训练正常但没有超过 E-028B simple fair matcher。用户质疑：如果接入 GDROS-style 后端后
还更差，是否说明后端接入/训练本身有问题？本诊断用同一套 E-029A copied backend/trainer/evaluator 换回
GDROS baseline frontend/features，重新训练或 replay baseline，看结果是否接近 GDROS full system。

核心问题：

> copied GDROS-style pipeline 能否 replay/retrain 出强 baseline 结果？

## 2. 已知参考

- E-029A best full-system：AEPE `21.916`；
- E-028B fair reference：AEPE `21.418`；
- GDROS full baseline / baseline-init diagnostic reference：AEPE 约 `1.47`，但 exact replay 状态需重新标注。

## 3. Tracks

### A. `GDROS_OFFICIAL_OR_FROZEN_REFERENCE`

- 只读引用 official/frozen GDROS full baseline；
- 记录 checkpoint/config/source SHA；
- 标注 exact replay、formal reference 或 diagnostic reference。

### B. `COPIED_GDROS_REPLAY_SANITY`

```text
GDROS checkpoint / weights
-> isolated copied GDROS pipeline
-> no training
-> dev404 evaluation
```

验证 code copy、normalization、evaluator、data path 是否破坏 baseline。

### C. `COPIED_GDROS_SAME_BUDGET_RETRAIN`

```text
GDROS frontend + copied GDROS-style backend
-> same or comparable E-029A train budget/harness
-> dev404 evaluation
```

判断同一训练管线是否能把 baseline 训练到合理性能。

可选：

- `COPIED_GDROS_OFFICIAL_RECIPE_RETRAIN`：官方 recipe 可审计且资源允许时运行；
- `GDROS_FEATURE_CACHED_BACKEND_RETRAIN`：full frontend 重训过重时，用 frozen/cached GDROS features 验证 backend trainer。

## 4. 边界

- 原始 GDROS baseline code/checkpoint/cache/run 只读；
- 训练只发生在诊断副本；
- 不写 baseline 目录；
- 不使用 OS424/confirm/datum；
- dev 不用于调参或 checkpoint selection；
- replay/retrain 结果必须标注 exact/diagnostic；
- 不自动启动轻量 matcher、E-029B/E-030A 或 rescue。

## 5. Metrics

必须报告：

- AEPE；
- CMR@1/2/5；
- PLR；
- finite / in-bounds；
- train/dev gap；
- runtime / peak memory；
- 与 GDROS reference、E-029A best、E-028B fair reference 对比；
- replay 是否 exact parity。

## 6. Gate

### P0：reference and isolation

必须 PASS：

- GDROS reference source/SHA/config recorded；
- copied pipeline isolated from original baseline；
- original baseline readonly/no writes；
- same dev404/evaluator as E-029A；
- no OS424/confirm/datum；
- replay/retrain outputs materialized before truth_flow metric join；
- train/dev split isolation；
- no dev checkpoint/hparam selection。

P0 fail：`GDROS_RETRAIN_AUDIT_IMPLEMENTATION_BLOCKED`。

### G0：replay/training sanity

必须 PASS：

- replay or documented replay-unavailable path complete；
- same-budget retrain finite；
- strict checkpoint load；
- train loss finite/decreases；
- train no-GT inference finite；
- reviewer approves isolation and metric path。

G0 fail：`GDROS_RETRAIN_AUDIT_BLOCKED`。

### G1：diagnostic interpretation

判读：

- `COPIED_GDROS_PARITY_PASS`：replay or retrain reaches AEPE `<=3 px` or within `2x` exact GDROS reference；
- `COPIED_GDROS_REASONABLE_BUT_WEAK`：AEPE `<=8 px` but not near full reference；
- `COPIED_GDROS_RETRAIN_WEAK`：copied GDROS retrain AEPE remains near E-029A/E-028B (`>8 px`)；
- `REFERENCE_REPLAY_BLOCKED`：无法获得足够可信的 baseline replay/retrain reference；
- `E029A_NEGATIVE_TRUSTWORTHY`：copied GDROS parity passes, so E-029A failure likely reflects E-028B/backend mismatch；
- `E029A_NEGATIVE_NOT_TRUSTWORTHY`：copied GDROS also weak, so E-029A negative cannot be used for method conclusion。

## 7. 审阅

需要两路 reviewer：

1. retrain-audit/model reviewer：
   - copied GDROS pipeline 是否真实使用 baseline frontend/backend；
   - baseline 是否只读；
   - replay/retrain 是否与 E-029A harness 可比；
   - checkpoint/replay exactness 标注是否准确。
2. evaluator/protocol reviewer：
   - dev404/evaluator 是否同 E-029A；
   - no OS424/datum/dev selection；
   - metric aggregation 与 comparison 是否正确。

blocking finding：`IMPLEMENTATION_BLOCKED`。

## 8. 输出

只回传：

- `RESULT_CARD.md`
- `baseline_reference_manifest.json`
- `replay_manifest.json`
- `train_report.json`
- `dev_metrics.json`
- `comparison_table.json`
- `isolation_audit.md`
- reviewer conclusions
- `RUN_MANIFEST.json`
- watchdog log/report

完成后停止，等待方法决策；不得自动启动轻量 matcher、E-029B/E-030A、OS424/confirm、DINO/GDROS baseline finetune 或 rescue。
