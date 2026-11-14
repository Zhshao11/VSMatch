# 2026-11-14 一手结果

| 文件 | 内容 |
|---|---|
| `ablate_v2_off_result_official.json` | V2（DINO 辅助）分支零训练旁路原始结果 |
| `sweep_gru_iters_official.json` | GRU 迭代次数 `[4,6,8,12,16,24,32]` 官方口径 sweep 原始结果 |

## `ablate_v2_off_result_official.json`

在 zzvr 上由 `research/eval/ablate_v2_off.py` 产出，
**GDROS 官方 `test.py` 口径**，OS 测试集全量 424 对、`iters=32`、`LSR NPOINT=200000`。

顶层字段：

- `configs.gdros` — GDROS 官方权重 `GDROS-os512.pth` 参照行（`run_one("baseline")`）
- `configs.both` — `E075A-M4@80K` 原始两路
- `configs.v2off` — 同一权重旁路 V2 后的单路
- `configs.{both,v2off}.pair_EPE` — 424 项逐对官方 EPE 数列，用于配对差值分析
- `configs.*.official.AEPE_vector_EPE_reference` — 仅作旧口径交叉验证，不可与官方行并列
- `structure` — 5 个 `E073FusionMatcher` 的 variant 与训练后 `corr_gate_logit`
- `sanity_pass: true` — `gdros` 与 `both` 共 10 项指标全部复现 2026-11-13 官方口径历史值

`gdros` 行不含 `pair_EPE`（上游 `run_one` 只返回聚合结果）；逐对差值分析只在
`both` / `v2off` 之间做。

结论、逐对统计、以及「这是诊断性旁路而非模块消融」的定位见
[`docs/ABLATION_V2.md`](../../../docs/ABLATION_V2.md)。

## `sweep_gru_iters_official.json`

由 `research/eval/sweep_gru_iters.py` 产出，**GDROS 官方 `test.py` 口径**，
424 对全量；一次 `iters=32` 前向，`results.<k>` 给出每个候选迭代点的
`official` 聚合指标、424 项 `pair_EPE`、`lsr_calls_per_pair` 与
`relative_inference_cost`。顶层 `train_iters_used_by_checkpoint: 12`，
`sanity_pass: true`（`k=32` 逐项复现 `AEPE 1.4738` 等历史值），
`elapsed_sec: 451.1`。

**只有 `k=32` 与论文 Table II / 本目录另一份 JSON 同口径**；
其他 `k` 是推理超参的敏感性曲线，不可与 GDROS baseline 行并列。
分析见 [`docs/GRU_ITER_SWEEP.md`](../../../docs/GRU_ITER_SWEEP.md)。
