# E-036C-E028B-UNCERTAINTY-ADAPTIVE-SPAN-REFINER

方法版本：`M001 v3.18C`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-036C 测试第三条替代路线：

> 不使用固定局部窗口和固定解码器，而是根据 coarse uncertainty / local ambiguity 自适应选择搜索 span，并在多尺度候选分布中细化。

动机来自 ASpanFormer 类思路：局部 attention/search span 不应固定，而应由中间 flow/uncertainty 决定。

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不训练/finetune E028B descriptor；
- 允许训练 uncertainty/span head 与 local candidate scorer；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- primary coarse input 固定为 E-034C `rank_growth_dual_256_irls`；
- 不使用 official testing.zip / confirm / OS424；
- outputs before GT metric join；
- 完成后停止，不自动启动 E037、rescue、descriptor finetune 或 full sparse feedback。

## 3. 方法

### A. Adaptive span

对每个 coarse match 预测或计算：

- local uncertainty / entropy；
- candidate search radius：例如 `{4, 8, 16, 32}` px；
- candidate density：coarse-to-fine grid 或 top-M local candidates；
- reject/confidence weight。

### B. Tracks

必须至少实现：

1. `UNCERTAINTY_SPAN_NO_TRAIN`
   - 根据 correlation entropy、rank gap、mutuality、local texture 等确定 span；
   - 作为可解释 baseline。

2. `LEARNED_SPAN_SCORER`
   - 训练 span/radius + local candidate scorer；
   - 输出多尺度 candidate distribution；
   - final endpoint 可 hard select、soft expectation、RBC 三种汇报。

3. `SPAN_PLUS_GEOMETRY_FILTER`
   - adaptive span 后做 lightweight weighted affine/IRLS；
   - 不做 GDROS dense recurrence。

## 4. Gates

### P0

必须 PASS：

- E-034C coarse replay verified；
- E028B checkpoint SHA recorded and read-only；
- no GDROS backend/dense GRU reuse；
- no testing.zip / confirm / OS424；
- selected/refined outputs before GT metric join。

### G0

必须 PASS：

- span statistics non-degenerate；
- learned span/scorer losses finite；
- gradients reach span/scorer head；
- strict checkpoint load；
- reviewer approves coordinate mapping, no leakage, and adaptive span implementation。

### G1

主对照：E-034C/E-035A coarse reference AEPE `8.550`、CMR@2 `8.996%`、CMR@5 `38.783%`、
CMR@16 `87.942%`。

判读：

- `ADAPTIVE_SPAN_GAIN_FOUND`：CMR@5 > `38.783%` 且 AEPE < `8.550`；
- `SPAN_SIGNAL_FOUND_FINAL_WEAK`：local CMR@5 上升但 final affine 未升；
- `ADAPTIVE_SPAN_FAIL`；
- `IMPLEMENTATION_OR_LABEL_SUSPECT`；
- `E036C_RESOURCE_BLOCKED`。

## 5. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `span_statistics.json`；
- `local_candidate_report.json`；
- `ablation_summary.json`；
- `per_pair.jsonl`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。
