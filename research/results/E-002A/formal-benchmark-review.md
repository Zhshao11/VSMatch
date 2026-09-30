# E-002A Formal Resource Independent Review

日期：`2026-08-04`

方法版本：`M001 v0.3`

结论：资源产物可信，但预注册 `G5 = FAIL`，实验状态必须为 `EFFICIENCY_HOLD`。

## 完整性

- Watchdog：`20260804_102002_f4e2dea6`，固定 GPU 4，`succeeded`
- 20 个唯一 PID，严格为 2 contracts x 2 methods x 5 replicates
- 每 worker：20 warm-up、200 measured、FP32、batch 1、RTX 4090、eval/inference mode
- Primary raw 4,000 行；baseline segment 6,000 行；slim segment 12,000 行；slim
  K-scaling 6,000 行
- 218 个 artifact SHA256/size 全部重验通过；raw 重算与 summary 最大差 `9.95e-14`
- Dense 输出均为 `[1,2,512,512]` FP32 finite；affine 输出均为 `[1,3,3]`
- Truth flow 未加载；baseline official parity 10/10，最大差 `0`

## 资源结果

| 指标 | GDROS baseline | SparseGeo slim |
| --- | ---: | ---: |
| Dense-contract p50 | `78.951272 ms` | `8.638904 ms` |
| Dense-contract p95 | `97.157804 ms` | `9.157010 ms` |
| Cold/warm peak allocated | `315.472656 MiB` | `276.454590 MiB` |
| Registered parameters | `12,847,408` | `4,804,448` |
| Active parameters | `8,992,256` | `4,804,448` |

Median latency speedup 为 `9.139x`，峰值显存降低 `12.368%`。K128 matcher 增量峰值为
`5.628418 MiB`。Slim 没有注册或执行 context、CorrBlock、GRU、upsampler 或
cost-perceiver，且从未构造 cross-image `[4096,4096]` correlation。

## G5 失败项

Slim dense 五进程 p50 为
`[8.628909, 8.659851, 11.433246, 8.638904, 8.549575] ms`，CV 为
`13.712815% > 5%`。异常完全来自 r2；其 200 次迭代中有 109 次大于 10 ms、65 次大于
12 ms。Wall/CUDA Event 相关为 `0.999991`，同一 worker 后续分段恢复正常，代码、shape、
显存和 provenance 与其余 worker 完全一致。快照显示 warm-up 后 SM clock 为 2520 MHz，
结束时恢复 2670 MHz，最符合瞬态 DVFS/GPU 状态，但快照不足以证明唯一原因。剔除 r2 后
CV `0.5597%` 只能作为诊断，不能用于追认通过。

K64/128/256 matcher p50 为 `0.586000/0.589447/0.585670 ms`；拟合斜率
`-5.69085e-6 ms/K`、R2 `0.070739`，不满足预注册 timing 线性条件。显存为
`5,048,832 / 5,901,824 / 7,608,832 bytes`，线性 R2 `0.999999978`；score 数严格为
`K x 4096`，K128 最大 live score 仅 `[1,128,1024] = 0.5 MiB`。

Timing 平坦来自每次固定的 token `.contiguous()` 全量拷贝，以及三个 K 都固定执行四组
chunk/BMM/max/where launch；小 GEMM 未饱和时固定开销淹没 K 项。它不是旧 GDROS 路径
或隐藏 N2，但仍然未满足已锁定的 timing 门槛。

因此 G5 只有 `process_p50_cv_le_5_percent` 和 `k_scaling_approximately_linear` 两项失败，
其他检查全部通过。不能删除异常进程、修改门槛或把内存/理论复杂度线性替代 timing 线性。

另外，FeatureTransformer 仍保留 `[8,1024,1024]` split-window attention。结果只能声称
移除了完整 cross-image CorrBlock/N2 matcher，不能声称整网不存在二次复杂度。

正式路径：
`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002A/run-01/benchmark`
