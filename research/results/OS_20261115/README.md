# 2026-11-15 一手结果：C 分支剪枝 + 单路消融（旁路 / 短训）

| 文件 | 内容 |
|---|---|
| `verify_c_teacher_pruning.json` | 同进程 A/B bit-exact + 424 对官方口径 sanity check |
| `probe_c_prune_timing.json` | 同进程逐对计时：推理 `iters=32` / 训练前向 `iters=12` |
| `ablate_v1only_pruned.json` | 对称单路旁路 `v1only`（只留 GDROS 前端），GPU 5 |
| `ablate_v2only_pruned.json` | 对称单路旁路 `v2only`（门控推向含 DINO 那路），GPU 1 |
| `backbone_ablation_merged.json` | 合并主表 + 三组逐对 ΔEPE 统计 |
| `short_ft_v1only.json` | `v1only` 2000 步适配性短训的训练曲线 + 官方口径评测，GPU 5 |
| `short_ft_v2only.json` | `v2only` 同预算短训，GPU 7 |
| `short_finetune_merged.json` | 短训 vs 零训练旁路 vs `both` 的合并对比 |

剪枝验证与计时在 zzvr 空闲 GPU 5 上；对称单路旁路分别在 **GPU 5 与 GPU 1**
并行，适配性短训在 **GPU 5 与 GPU 7** 并行。`CUDA_VISIBLE_DEVICES=<物理卡>` 映射为进程内 `cuda:0`，脚本 `--gpu 0`；
权重统一为保护版本 `M4_step80000.pt`。

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

## 对称单路旁路（剪枝后 M4，双 GPU）

三份文件见顶部表格：两份 `ablate_*_pruned.json` 是单配置原始结果，
`backbone_ablation_merged.json` 是本地合并产物。
两份单路 JSON 均含 `ablation_class: "diagnostic_bypass_not_module_ablation"`、
`structural_caveat`（V2 不是独立 backbone）、`pruned_model.c_skip_teachers: true`、
5 个 `E073FusionMatcher` 的门控快照、1 对 probe 的 `max|Δpost|`、
`result.official` 与 424 项 `pair_EPE`，以及 `sanity_checks` / `sanity_pass`。

`v1only` 的 5 项官方指标与 11-14 历史 `v2off` **Δ=0**；`v2only` 无历史值，
sanity 检查的是「门控确实改变输出」与「AEPE 与 both 有显著差异」。
`v2only` 的 `elapsed_sec: 950.7` 受 GPU 1 被另一任务占满影响，
**不能**与 `v1only` 的 `350.2` 直接比较；指标不受影响。

结论、结构边界（「只留 DINO」不可严格实现）与推论限制见
[`docs/ABLATION_BACKBONE_SYMMETRIC.md`](../../../docs/ABLATION_BACKBONE_SYMMETRIC.md)。

## 适配性短训（方案 A，2000 步）

`short_ft_*.json` 顶层含 `experiment_class`、`train`（`steps` / `seed` /
`train_iters` / `lr` / `trainable_params=4401475` / `trainable_groups` /
`frozen` / `gradient_note` / `elapsed_sec` / `sec_per_step` / `curve`）、
`gate_lock`、`historical`、`official_after`（聚合指标 + 424 项 `pair_EPE`）
与 `checkpoint_delta`（zzvr 上的可训练参数增量路径，**完整权重不入库**）。

`short_finetune_merged.json` 给出主表、每个配置的 `AEPE_improvement`、
`AEPE_gap_vs_both`、`recovery_fraction_of_zeroshot_gap` 和
`pair_delta_after_minus_zero`（逐对 mean/median、改善比例、>10px 灾难失败变化）。

| 配置 | 零训练 AEPE | 短训 2000 步后 | 恢复落差 | >10px 失败 |
|---|---|---|---|---|
| `v1only` | 7.7457 | 1.7127 | 96.2% | 104 → 17 |
| `v2only` | 41.3298 | 3.3824 | 95.2% | 416 → 17 |

`both` 参照 `1.4738`、>10px 失败 12 对。**短训定量证明零训练旁路的大幅掉点
主要是下游 OOD**；但 2000 步预算与受限的解冻范围意味着残余差距
（+0.24 / +1.91 AEPE）**仍不是**任一路的边际贡献。完整边界见
[`docs/ABLATION_SHORT_FINETUNE.md`](../../../docs/ABLATION_SHORT_FINETUNE.md)。
