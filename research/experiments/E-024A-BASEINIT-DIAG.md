# E-024A-BASEINIT-DIAG：Baseline-Init Backend Isolation

方法版本：`M001 v2.8 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-024A formal 失败类别为 `COARSE_BASIN_FAIL`：R1 window coverage 与 R2 update 已有一定作用，但 R0
coarse basin 在 dev 上严重失败。本诊断用 frozen baseline 的 no-GT coarse/init 结果替换 R0，隔离检查
E-024A 的 R1/R2/R3 后半段是否有效。

该诊断不证明完整方法成功，只回答：

```text
given a good/frozen baseline init,
does E-024A backend preserve, improve, or degrade it?
```

## 2. 输入与边界

只读：

- E-024A formal checkpoint/source/result；
- frozen GDROS/baseline coarse matching/intermediate output，如存在；
- 若不存在，可用 frozen GDROS/baseline predicted final flow 拟合 affine init，但必须标注为
  `GDROS_FLOW_INIT_DIAGNOSTIC`；
- 官方 train/dev split 与 raw truth_flow，仅用于 metrics/oracle reference。

禁止：

- 不训练新模型；
- 不修改 E-024A checkpoint/formal result；
- 不修改或 finetune DINO/GDROS/baseline checkpoint；
- 不写 baseline 目录；
- 不使用 OS424/confirm/datum；
- 不把 baseline-init 诊断成绩称为我们方法 no-GT 主成绩；
- 不启动 E-024B/rescue。

## 3. Baseline init source protocol

必须先冻结 source：

1. 查找 baseline 是否有正式 coarse output：coarse matching、coarse flow、intermediate affine 或等价低分辨率 init；
2. 若使用 final baseline flow 拟合 affine：
   - 从 frozen predicted flow 生成 source-target correspondences；
   - region-balanced 采样；
   - FP32 least squares 拟合 affine；
   - 输出 source path/checkpoint/config SHA；
   - 标记 `init_source_type = GDROS_FLOW_INIT_DIAGNOSTIC`。
3. baseline init 必须在 truth_flow label/metric join 之前物化并 hash。

若无法找到或构造可审计 baseline init：停止为 `BASELINE_INIT_SOURCE_UNAVAILABLE`。

## 4. Tracks

### T0：E-024A original reference

读取 E-024A formal metrics：

- R0 selected/oracle-best AEPE；
- R1 coverage；
- R2 improvement；
- R3 final AEPE/CMR/PLR。

### T1：baseline-init-only

只评估 baseline init，不运行 E-024A backend update：

- initial affine AEPE/CMR/PLR；
- R1 window coverage if windows centered by baseline init；
- region/quadrant coverage；
- comparison to E-024A R0 selected/oracle-best。

### T2：baseline-init + E-024A R1/R2/R3

固定 E-024A checkpoint 与 R1/R2/R3，实现只替换 R0 selected state：

- R1 window coverage；
- R2 predicted update improvement or degradation；
- R3 final AEPE/CMR/PLR；
- final relative delta vs T1 baseline-init-only。

### T3：baseline-init + oracle local update reference

仅诊断：

- 用 truth_flow 构造 oracle local target/residual/update；
- 报告 oracle update AEPE/CMR/PLR；
- 必须标注 diagnostic-only。

## 5. 判读

- `BASELINE_INIT_SOURCE_UNAVAILABLE`
- `BACKEND_DEGRADES_GOOD_INIT`：T2 AEPE 比 T1 变差 `>10%`
- `BACKEND_NEUTRAL_WITH_GOOD_INIT`：T2 相对 T1 变化 `<10%`
- `BACKEND_IMPROVES_GOOD_INIT`：T2 相对 T1 改善 `>=10%`
- `BACKEND_EFFECTIVE_WITH_GOOD_INIT`：T2 final AEPE <= `12 px` 且相对 T1 改善 `>=20%`
- `BACKEND_STRONG_WITH_GOOD_INIT`：T2 final AEPE <= `5 px` 且 CMR@2 >= `50%`
- `BACKEND_ORACLE_ONLY`：T3 好但 T2 不好

如果 T1 已经很好而 T2 只是保持，不得写成后半段有效，只能写 neutral / preserve。

## 6. 审阅

需要一名定向 reviewer：

- baseline init source 是否 no-GT、只读、可审计；
- 若使用 final baseline flow，是否明确标注 diagnostic upper-bound；
- E-024A checkpoint/source 是否未修改；
- truth_flow 是否只用于 metric/oracle reference；
- T1/T2/T3 比较是否定义正确；
- 结论是否没有把 baseline-init 当成完整方法成绩。

blocking finding：`IMPLEMENTATION_BLOCKED`，不得用结果解释方法。

## 7. 输出

只回传：

- `RESULT_CARD.md`
- `baseline_init_source.json`
- `backend_isolation_summary.json`
- `t1_baseline_init_only.json`
- `t2_baseline_init_backend.json`
- `t3_oracle_update_reference.json`
- `comparison_to_e024a.json`
- reviewer conclusion
- `RUN_MANIFEST.json`
- watchdog log/report

完成后停止，等待方法决策。
