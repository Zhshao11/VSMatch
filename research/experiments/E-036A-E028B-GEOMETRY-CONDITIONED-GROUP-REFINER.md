# E-036A-E028B-GEOMETRY-CONDITIONED-GROUP-REFINER

方法版本：`M001 v3.18A`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-035E 判定 isolated local refiner 应更换。E-036A 测试第一条替代路线：

> 不再让每个 coarse match 独立从局部窗口回归 offset，而是让一组 coarse matches 在共享低维 affine state 下联合细化。

核心思想：当前 coarse affine CMR@16 高但 CMR@5 低，说明很多点落在正确 basin 但局部定位不准。若单点 local
evidence 弱，就把 refinement 变成“群体几何约束下的局部候选重排/偏移更新”。

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不训练/finetune E028B descriptor；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- primary coarse input 固定为 E-034C best affine/CMR variant `rank_growth_dual_256_irls`；
- 允许训练 group refiner / geometry message passing / confidence head；
- 不使用 official testing.zip / confirm / OS424；
- selected/refined outputs 必须先落盘，再读取 truth-flow 做 metric join；
- 完成后停止，不自动启动 E037、rescue、descriptor finetune 或 full sparse feedback。

## 3. 方法

### A. 输入

- source coarse matches from E-034C `rank_growth_dual_256_irls`；
- E028B feature maps and optional raw/gradient fine crops；
- initial affine from robust fit of selected coarse matches；
- per-match local target candidates in a radius around affine-predicted endpoint and original coarse endpoint。

### B. Group refinement module

必须至少实现两个 track：

1. `GROUP_ATTENTION_REFINER`
   - 每个 match token 包含 source xy、coarse target xy、affine-pred target xy、local correlation summary、residual；
   - self-attention/message passing across selected matches；
   - cross-attention or local cost aggregation over candidate offsets；
   - 输出 offset distribution、confidence、optional inlier weight。

2. `AFFINE_STATE_FEEDBACK_REFINER`
   - 迭代 `2-4` rounds；
   - 每轮：根据当前 affine 生成局部候选 → group module 更新 offsets/weights → weighted affine/IRLS 更新 state；
   - 记录每轮 AEPE/CMR@2/5/16、inlier ratio、active count。

### C. 对照

- E-034C/E-035A coarse reference；
- E-035D/E-035E best refined track；
- no-message-passing ablation；
- no-affine-feedback ablation。

## 4. Gates

### P0

必须 PASS：

- E-035E `REFINEMENT_METHOD_SHOULD_CHANGE` recorded；
- E-034C primary coarse artifacts located/replayed；
- E028B checkpoint SHA recorded and read-only；
- no GDROS backend/dense GRU reuse；
- no testing.zip / confirm / OS424；
- candidate/refined outputs materialized before GT metric join。

### G0

必须 PASS：

- losses finite/nonzero；
- gradients reach group attention/refiner/confidence heads；
- train-small overfit improves local endpoint CMR@5 or GT mass；
- strict checkpoint load verified；
- reviewer approves coordinate mapping, no leakage, and group/affine update implementation。

### G1

主对照：E-034C/E-035A coarse reference AEPE `8.550`、CMR@2 `8.996%`、CMR@5 `38.783%`、
CMR@16 `87.942%`。

判读：

- `GROUP_REFINER_GAIN_FOUND`：CMR@5 > `38.783%` 且 AEPE < `8.550`，CMR@16 不明显崩溃；
- `GEOMETRY_FEEDBACK_GAIN_ONLY`：CMR@5 上升但 AEPE/CMR@2 仍弱；
- `GROUP_REFINER_FAIL`；
- `IMPLEMENTATION_OR_LABEL_SUSPECT`；
- `E036A_RESOURCE_BLOCKED`。

## 5. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `round_metrics.json`；
- `ablation_summary.json`；
- `per_pair.jsonl`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。
