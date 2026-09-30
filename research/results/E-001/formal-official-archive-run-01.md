# E-001 Formal Official Archive Run 01

日期：`2026-08-04`

方法规范：`M001 v0.2`

结论：`COMPLETED / PASS`

## 运行身份

- 正式数据：`/mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official_e000_archive`
- Provenance SHA256：`bff6f565e9a8a12bb2675d54b97fccaeb3a88d39b49ae56a1571c57d8489aad9`
- Checkpoint SHA256：`80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- Evaluator：`a16fe7586053ea8fb6ea33a9607fdff4b75eee215b264af13d4ebcfbca8a226e`
- Benchmark：`3dbcb88f2a443018fdc7c7312b81051f6d522173140c5900cebd8dc9ba3f0473`
- Tests：`39e00d837376cd74e701e549cc1715d9751985ce9ec01a2e46168dcf91d9b3d0`
- Accuracy watchdog：`20260804_040321_a1ab77de`，GPU 2，`succeeded`
- Benchmark watchdog：`20260804_040325_fa3cd863`，GPU 3，`succeeded`

初始与运行后数据 manifest 都为
`c129ae7147a7d9ad1c1a7ead35bb6da2d8c85ad8da73eef4723cf277e5ea7e96`。Loader 严格绑定
000000 至 000423 的 424 对。完整结果包含 91 个条件和 38,584 条唯一 pair/result 记录；
45 个非 baseline 采样设计全部 rank 3。两次独立结果审计均为 `APPROVE`。

## Baseline 与 Parity

- Baseline AEPE：`1.461230482`
- CMR@1/2/5：`33.726415 / 81.603774 / 99.528302%`
- P95/P99：`3.247386 / 4.597962 px`
- Failure：`2/424 = 0.471698%`
- 新 generalized 200k 拟合相对原 LSR 的最大 dense-flow 误差：`1.220703e-4 px`
- 最大 pair public-error 差：`1.907349e-6 px`

## 压缩效应：Exact Public Pool

正式决策只使用 row-major `[10000:210000]` 内的非 GT 子采样。K64/K128 的 14 个候选均
满足 `delta AEPE <= 0.05 px` 和 `delta CMR@1 >= -1 pp`。

| 策略 | Delta AEPE | Delta CMR@1 | P95 / P99 | Failure |
| --- | ---: | ---: | ---: | ---: |
| K64 stratified | `+0.002259 px` | `-0.471698 pp` | `3.294463 / 4.603612` | `0.471698%` |
| K128 stratified | `+0.001100 px` | `+0.235849 pp` | `3.283239 / 4.605558` | `0.471698%` |

五个固定随机种子的稳定性：

| K | Delta AEPE mean +/- std | AEPE range | CMR@1 range |
| --- | ---: | ---: | ---: |
| 64 | `+0.005654 +/- 0.004379 px` | `0.011542 px` | `1.886792 pp` |
| 128 | `+0.002345 +/- 0.002227 px` | `0.006758 px` | `0.707547 pp` |

K128 比 K64 更稳定，建议 E-002A 的默认全局 anchor 预算从 `K=128` 起步，同时保留
K64/K256 作为 Pareto 对照。

## 覆盖效应：Full-image Diagnostic

Coverage 项从未进入正式门槛。K128 相对 exact stratified 的聚合差异为：

- full-grid-edge：`+0.001044 px / 0 pp CMR@1`
- full-stratified-center：`-0.000005 px / -0.943396 pp CMR@1`

聚合均值掩盖了明显逐对抵消：

| K128 结果 | Delta `> +0.05` | Delta `< -0.05` | Pair-delta 范围 |
| --- | ---: | ---: | ---: |
| exact stratified | 1 | 0 | `[-0.046490, +0.059488]` |
| full-grid-edge | 130 | 138 | `[-0.432719, +0.782699]` |
| full-stratified-center | 99 | 107 | `[-0.277309, +0.599556]` |

因此当前证据支持“点数压缩可行”，不支持“改成全图覆盖天然更优”。后续 anchor 必须保持
空间可辨识性，但 coverage 设计要作为独立消融，不能把覆盖变化归因成点数收益。

## Oracle 与资源结果

45 个 GT oracle 的 AEPE 为 `1.5981e-5` 至 `3.5416e-5 px`，failure 为 0，证明拟合器、
坐标和标签协议正常；它们仅为 diagnostic，从未进入门槛。

RTX 4090、FP32、batch 1、5 次 warm-up、30 次重复、每次前后 CUDA synchronize：

| 路径 | Generalized 200k | K64 | K128 |
| --- | ---: | ---: | ---: |
| Postprocess mean | `0.4206 ms` | `0.3991 ms` | `0.3995 ms` |
| Postprocess incremental peak | `7.061 MiB` | `7.061 MiB` | `7.061 MiB` |
| End-to-end mean | `84.181 ms` | `82.671 ms` | `82.633 ms` |
| End-to-end peak allocated | `316.298 MiB` | `316.298 MiB` | `316.298 MiB` |

端到端时延的小差异处于单进程顺序、GPU 时钟和负载噪声范围，且显存完全相同。E-001
保留完整 CorrBlock 和 32 次 GRU，只能证明最终仿射拟合点数高度冗余，不能宣称已经实现
端到端轻量化或低显存。

## 绝对产物路径与哈希

- Accuracy：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-001/formal-official-archive/run-01/accuracy`
- Benchmark：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-001/formal-official-archive/run-01/benchmark`
- `metrics.json`：`f6e0c0afab6f6665a9fb0ecf883f88c216a130c7e3643133b46fa8b63ab86217`
- `per_pair.csv`：`fe4220fe97b25a57c84dadd5a8117336016a855fe79d02791fbf1106bc6a3e28`
- `conditions.csv`：`04621d56c85fa5230f2e3a4d9b29738456195d3488c700a013914cb88a4e5e16`
- `benchmark.json`：`77c0df510699da2a9058c8b4050a848e2f3a2ae7c0d3cba27c1934050c7faab2`
- Accuracy report：`/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_040321_a1ab77de.json`
- Benchmark report：`/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_040325_fa3cd863.json`
- 独立正式结果审计：`research/results/E-001/formal-result-review.md`

限制：benchmark 没有保存 30 次原始 duration，且各 case 共享预驻留 design。后续部署级
benchmark 应保存 raw timings，并使用隔离进程复核微小延迟差异。
