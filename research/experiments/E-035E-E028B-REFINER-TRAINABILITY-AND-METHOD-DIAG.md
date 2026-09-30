# E-035E-E028B-REFINER-TRAINABILITY-AND-METHOD-DIAG：Should We Change Refinement Method?

方法版本：`M001 v3.17 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-035D 已在 CUDA 4090 上用 8 epochs、64 tokens/pair 训练 paper-style refiner，但仍低于 coarse
reference。现在需要判断：

> refiner 失败是训练量还不够、模型容量不够、监督形式不对，还是当前 local refinement 方法/观测本身不适合？

本实验不是再做一个正式改进方法，而是做 trainability/method-choice 诊断，决定是否继续加训或换 refinement
路线。

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不训练/finetune E028B descriptor；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- primary coarse input 固定为 E-034C best affine/CMR variant `rank_growth_dual_256_irls`；
- 允许训练 local fine/refinement/confidence heads；
- CUDA available 是 hard gate；
- 不使用 official testing.zip / confirm / OS424；
- selected/refined outputs 必须先落盘，再读取 truth-flow 做 metric join；
- 完成后停止，不自动启动 E036、descriptor finetune、full sparse feedback 或 rescue。

## 3. 诊断矩阵

### A. `TRAINING_SCALE_CURVE`

在 E-035D best track 上做 step/epoch 曲线：

- checkpoint at epochs `2/4/8/16/24` 或等价 steps；
- tokens/pair `64` 与 `128`；
- 记录 train/dev local endpoint error、heatmap GT mass、CMR@5、CMR@16。

判读：

- train/dev 仍不升：方法/观测弱；
- train 升 dev 不升：泛化/采样/augment 问题；
- train/dev 都升但 affine 不升：confidence/outlier cleanup 问题。

### B. `CAPACITY_OVERFIT_TEST`

比较 small/medium 两个 refiner capacity：

- train-small overfit：固定少量 pairs 和大量 tokens，要求 local CMR@5 / GT heatmap mass 显著上升；
- train-full sanity：看 medium 是否明显优于 small。

若 medium 仍无法在 train-small overfit，则强烈指向方法/输入不可学或实现/label 问题。

### C. `SUPERVISION_FORM_ABLATION`

在同一输入下比较：

- heatmap CE/focal；
- regression-by-classification；
- direct offset Huber；
- combined heatmap + offset；
- confidence/reject loss only for window-contained samples vs with outlier negatives。

输出每种监督在 train/dev 的 local endpoint rank/error 和 final affine。

### D. `METHOD_CHANGE_DECISION`

根据 A/B/C 给出结论：

- `MORE_TRAINING_PROMISING`：训练曲线未饱和且 dev 同步提升；
- `CAPACITY_LIMITED`：medium/long training 明显改善；
- `SUPERVISION_LIMITED`：换 loss 后显著改善；
- `REFINEMENT_METHOD_SHOULD_CHANGE`：train-small/long/medium/loss 均不能超过 coarse reference；
- `IMPLEMENTATION_OR_LABEL_SUSPECT`：train-small 无法 overfit 且 reviewer 发现 label/坐标风险。

## 4. Gates

### P0：setup/resource

必须 PASS：

- CUDA available true；
- training/testing SHA 与 pair count verified；
- E028B checkpoint SHA recorded；
- E-034C primary coarse artifacts located/replayed；
- E-035D result and resource report recorded；
- no E028B descriptor mutation；
- no GDROS backend adapter / dense GRU reuse；
- baseline artifacts readonly；
- testing.zip / confirm / OS424 not used。

P0 fail：`E035E_RESOURCE_BLOCKED` or `E035E_SETUP_BLOCKED`。

### G0：diagnostic execution

必须 PASS：

- training scale curve complete or typed resource failure；
- capacity overfit test complete；
- supervision ablation complete；
- checkpoints strict-loaded where evaluated；
- refined outputs written before truth-flow metric join；
- reviewer approves no leakage, coordinate mapping, label construction, and interpretation。

G0 fail：`E035E_EXECUTION_BLOCKED`。

### G1：trainability/method diagnosis

判读：

- `MORE_TRAINING_PROMISING`；
- `CAPACITY_LIMITED`；
- `SUPERVISION_LIMITED`；
- `REFINEMENT_METHOD_SHOULD_CHANGE`；
- `IMPLEMENTATION_OR_LABEL_SUSPECT`；
- `E035E_RESOURCE_BLOCKED`。

完成后只写结果卡并等待用户；不得自动启动 E036、new refinement method、descriptor finetune、full sparse feedback、
OS424/confirm 或 rescue。

## 5. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `training_scale_curve.json`；
- `capacity_overfit_report.json`；
- `supervision_ablation_report.json`；
- `method_change_decision.json`；
- `training_resource_report.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。
