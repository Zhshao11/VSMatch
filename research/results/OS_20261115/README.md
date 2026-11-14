# 2026-11-15 一手结果：C 分支 teacher 剪枝

| 文件 | 内容 |
|---|---|
| `verify_c_teacher_pruning.json` | 同进程 A/B bit-exact + 424 对官方口径 sanity check |
| `probe_c_prune_timing.json` | 同进程逐对计时：推理 `iters=32` / 训练前向 `iters=12` |

两次运行都在 zzvr 空闲 GPU 5 上，`CUDA_VISIBLE_DEVICES=5` 映射为进程内
`cuda:0`，脚本 `--gpu 0`；权重为保护版本 `M4_step80000.pt`。

## `verify_c_teacher_pruning.json`

由 `research/eval/verify_c_teacher_pruning.py` 产出，**GDROS 官方 `test.py` 口径**：
OS 测试集全量 424 对、`iters=32`、`LSR NPOINT=200000`。

顶层字段：

- `ab` — 8 对 × 32 个迭代点 + post-LSR 的 `max|Δ|`，全为 `0`（`bit_exact: true`）；
  `elapsed_sec: 111.8` 含两次模型构建，**不是**纯前向时间
- `pruned.official` — 剪枝后 424 对聚合指标，`AEPE=1.473827834446231`、
  CMR@1/2/5 `72.41/84.91/94.10`、`RMSE_official_var=8.3350`
- `pruned.pair_EPE` — 424 项逐对官方 EPE
- `pruned.elapsed_sec: 317.2`
- `archived_legacy` — 11-13/11-14 历史值（AEPE 精确值取自
  `research/results/OS_20261113/os_official_protocol_20261113.json`；
  `elapsed_sec: 422.4` 是 11-14 消融同进程 `both` 的空闲 GPU 记录）
- `abs_delta_vs_archived` — 5 项全为 `0.0`
- `speedup_vs_idle_gpu_legacy: 1.3317` — **跨运行对比**，只作参考；严格加速比看计时文件
- `call_counts` — 剪枝前 `5×32=160` 次 GRU update、剪枝后 `3×32=96`；LSR 两侧都是 `4×32+1=129`
- `sanity_pass: true`

## `probe_c_prune_timing.json`

由 `research/eval/probe_c_prune_timing.py` 产出。`same_process_ab: true`：
同一进程构建 legacy / pruned 两个模型，在**同一批 33 对**（3 对 warmup +
30 对计时）上分别计时，避免跨运行的 GPU / 页缓存状态差异。

| 路径 | `iters` | legacy | pruned | 加速 |
|---|---|---|---|---|
| `eval`（`PairFusion.forward`） | 32 | 0.9638 s/对 | 0.7197 s/对 | **1.3392×** |
| `train`（手工 `b` + `c` + `gate_features`） | 12 | 0.6006 s/对 | 0.4128 s/对 | **1.4549×** |

`caveat` 字段写明：**只测前向**。完整训练 step 还含 `LSR_list(fused)`、loss、
只过 L4 gate 的反向、数据加载与周期性 dev eval，不能把 `0.741 s/step` /
`16.5 h` 直接除以 1.455。11-14 单次 probe 报的 `1.796×` 偏高，以这里的
30 对均值为准。

结论与边界见 [`docs/GRU_ITER_SWEEP.md`](../../../docs/GRU_ITER_SWEEP.md) §6.1，
脚本说明见 [`research/eval/README.md`](../../eval/README.md)。
