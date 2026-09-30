# E-011A-TOP128-GEOM-DIAG：top128 几何可用性诊断

## 目的

E-011A top64 正式失败，但 top128 暴露明显可恢复信号：mean/P10/P50/P90 为
`68.710/34/66/106`，超过 E-006A 的 mean/P10/P50 数量线；nearest P50/P90 也优于 frozen S2。
但 top128 oracle AEPE 仍为 `14.249 px`、PLR `97.28%`。

本诊断只回答：

> top128 中正确点数量已经上来，为什么不能形成几何可用的仿射候选集？

本实验只读、无训练、不改 E-011A 输出、不启动 E-011B/rescue。

## 冻结输入

- 复用 E-011A formal dev404 topR outputs，重点分析 `R=128`。
- 复用 E-011A run identity、split、S0/S1 trace 和 raw truth_flow labels。
- raw truth_flow 只允许用于诊断标签、oracle 选择和后置评估。
- 不访问 OS424/confirm/datum label。
- 不修改 GDROS baseline、E-011A checkpoint/cache/output。

## 必须分析的对象

### A0：E-011A top128 原始候选

对每个 pair/proposal/candidate 记录：

- source pixel/token；
- target pixel/token；
- rank；
- score；
- endpoint error to raw truth_flow；
- 1/2/5 px 标签；
- in-bounds；
- candidate source proposal 所属 source region/象限。

### A1：正确点数量与空间分布

对每个 pair 汇总：

- `<=2 px` correct proposal count；
- source 四象限覆盖数；
- source 8x8 region 覆盖数；
- convex hull area 或等价 normalized source coverage；
- source coordinate condition number；
- duplicated target token ratio；
- correct candidates 的 rank 分布：top32/top64/top96/top128。

### A2：仿射支持性

对每个 pair 计算以下 oracle affine：

1. `all-hard2px-LS`：使用 top128 中所有 `<=2 px` candidates 普通 LS；
2. `rank64-hard2px-LS`：仅使用 rank<=64 且 `<=2 px` candidates；
3. `spatial-balanced-K64-GT-oracle`：从 top128 的 `<=2 px` candidates 中，按 source 8x8 region
   或四象限均衡选最多 64 个点后拟合；
4. `condition-filtered-GT-oracle`：从 top128 的 `<=2 px` candidates 中，按改善 source coverage 和
   condition 的确定性规则选点，最多 64 个；
5. `GT-endpoint-affine-oracle`：使用同一 source 点的 raw truth endpoint 而不是 candidate target，
   只隔离 candidate quantization/target error 与 source 分布影响。

所有 oracle 都只能用于诊断，不得作为方法结果。

### A3：失败归因

将 dev404 pair 分桶：

- `COUNT_LIMITED`：top128 `<=2 px` correct count `<33` 或 rank64 correct count `<12`；
- `COVERAGE_LIMITED`：correct count 足够，但象限覆盖 `<3`、8x8 region 覆盖低，或 condition 太差；
- `RANKING_TAIL_LIMITED`：top128 足够但 top64/rank64 不足，说明正确点主要在后排；
- `CANDIDATE_NOISE_LIMITED`：GT-endpoint oracle 好，但 candidate target oracle 差；
- `NON_AFFINE_OR_SOURCE_SET_LIMITED`：GT-endpoint oracle 仍差，说明当前 source set 对全局 affine 不友好；
- `COMPRESSION_ORACLE_READY`：spatial/condition GT oracle 能达到 AEPE `<=0.5 px` 且 PLR/valid `>=99%`。

允许多标签，但 `RESULT_CARD.md` 需列出主导比例最高的 2 个原因。

## 汇总指标

必须报告：

- per-bucket pair counts / percentages；
- all-hard2px、rank64、spatial-balanced、condition-filtered、GT-endpoint 的 AEPE、PLR、valid；
- 各 oracle 的 P50/P90/P95 AEPE；
- correct count 与 AEPE 的相关性；
- coverage/condition 与 AEPE 的相关性；
- rank distribution：2px correct 在 top32/top64/top96/top128 的累计比例；
- 若 `COMPRESSION_ORACLE_READY` 存在，报告其 pair 比例和失败 pair 特征。

## 判读状态

最终状态只能是：

- `COMPLETED / COUNT_AND_RANK_LIMITED`：主要是正确点数量或 rank64 不足；
- `COMPLETED / GEOMETRY_DISTRIBUTION_LIMITED`：数量够但空间覆盖/condition 差；
- `COMPLETED / CANDIDATE_NOISE_LIMITED`：GT endpoint oracle 明显好于 candidate target oracle；
- `COMPLETED / SOURCE_SET_LIMITED`：GT endpoint oracle 也差；
- `COMPLETED / COMPRESSION_ORACLE_READY`：GT 几何压缩 oracle 达到 AEPE `<=0.5`、PLR/valid `>=99%`；
- `COMPLETED / MIXED_GEOMETRY_FAILURE`：无单一主因；
- `BLOCKED / PARITY_OR_PROVENANCE_FAIL`：E-011A top128 parity 或 provenance 不一致。

## 自动检查

必须通过：

1. E-011A top128 parity：
   - mean correct `68.710 ± 0.05`；
   - P10/P50/P90 `34/66/106`；
   - oracle AEPE `14.249 ± 0.05` 或说明数值差异来源。
2. candidate outputs 必须先于 label/oracle join。
3. 不修改 GDROS baseline 或 E-011A output。
4. 坐标方向：optical source → SAR target，source/target pixel 均沿 E-011A 定义。
5. 至少一个 synthetic test 覆盖：
   - 数量足够但象限不足；
   - 数量不足；
   - top128 足够但 top64 不足；
   - candidate target noisy、GT endpoint oracle good。

## 审阅

低 token 定向审阅一名即可，重点检查：

- E-011A parity；
- GT/oracle 只在诊断阶段使用；
- oracle 选择规则是否确定性；
- bucket 归因是否由指标支持；
- baseline/E-011A 只读隔离。

## 产物

- `RESULT_CARD.md`，不超过 100 行；
- `geom_summary.json`；
- `geom_pair_metrics.jsonl`；
- `geom_bucket_counts.json`；
- reviewer 结论；
- watchdog report/log。

完成后只回传结果卡、summary、bucket counts、reviewer 和 run root。不得自动启动 E-011B 或任何救援。
