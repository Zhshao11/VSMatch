# E-036B-E028B-ROMA-STYLE-FINE-FEATURE-PYRAMID

方法版本：`M001 v3.18B`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-036B 测试第二条替代路线：

> 保留 E028B/DINO-style global robustness，但新增专门跨模态 fine feature pyramid，让局部 refinement 有真正可定位的细粒度观测。

动机来自 RoMa 类路线：foundation features 鲁棒但 coarse，通常需要 specialized ConvNet fine features、match decoder、
anchor probability / RBC 和 robust regression 才能精确定位。

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 允许训练新增 fine feature pyramid / fine matching head；
- 不 finetune E028B backbone/adapter 本体；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- primary coarse input 固定为 E-034C `rank_growth_dual_256_irls`；
- 不使用 official testing.zip / confirm / OS424；
- outputs before GT metric join；
- 完成后停止，不自动启动 E037、rescue、full sparse feedback。

## 3. 方法

### A. Fine feature pyramid

输入 raw OPT/SAR crops + E028B coarse feature context，训练轻量但专门的跨模态 fine encoder：

- pixel/patch resolutions：至少包含 stride `4` 或更细的 crop-level features；
- modality-specific stem + shared fine trunk 或 dual stem + shared projection；
- optional gradient/edge/log-SAR side channels；
- 输出 local descriptors 和 uncertainty/confidence。

### B. Matching / loss

必须至少实现：

1. `ROMA_STYLE_RBC_FINE`
   - local offset 使用 regression-by-classification；
   - subsequent robust regression / Huber；
   - anchor probability or confidence head。

2. `FINE_FEATURE_CORR_REFINER`
   - local cost volume over fine pyramid；
   - soft-argmax 与 hard-argmax 均报告；
   - uncertainty filtering/weighting。

### C. 对照

- E-034C coarse reference；
- E-035D/E-035E best refined；
- no-fine-feature track：只用旧 E028B/local signals；
- raw-only fine feature ablation。

## 4. Gates

### P0

必须 PASS：

- E028B checkpoint SHA recorded and read-only；
- new fine feature module initialized/trained only in experiment copy；
- no GDROS backend/dense GRU reuse；
- no testing.zip / confirm / OS424；
- E-034C coarse replay verified。

### G0

必须 PASS：

- train-small overfit local CMR@5 / GT mass improves clearly；
- train-full losses finite and gradients reach fine encoder and matcher head；
- strict checkpoint load；
- reviewer approves label construction, crop coordinate mapping, no leakage, and baseline isolation。

### G1

主对照：E-034C/E-035A coarse reference AEPE `8.550`、CMR@2 `8.996%`、CMR@5 `38.783%`、
CMR@16 `87.942%`。

判读：

- `FINE_FEATURE_REFINER_GAIN_FOUND`：CMR@5 > `38.783%` 且 AEPE < `8.550`；
- `FINE_FEATURE_LOCAL_SIGNAL_FOUND`：local CMR@5/GT mass 明显升，但 final affine 未升；
- `FINE_FEATURE_REFINER_FAIL`；
- `IMPLEMENTATION_OR_LABEL_SUSPECT`；
- `E036B_RESOURCE_BLOCKED`。

## 5. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `fine_feature_signal_report.json`；
- `ablation_summary.json`；
- `per_pair.jsonl`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。
