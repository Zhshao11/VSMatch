# E-030A-GDROS-PROTOCOL-SWEEP：GDROS Baseline Protocol Sweep

方法版本：`M001 v3.5 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-029A-GDROS-PAPER-PROTOCOL-RETRAIN 在 official OS `training.zip` 2011 -> `testing.zip` 424 上从头重训
GDROS，未加载官方已训练 GDROS 权重，得到 AEPE `1.7014`、CMR@1 `23.5849%`、CMR@2 `71.2264%`、
CMR@5 `99.2925%`。released checkpoint replay 为 AEPE `1.4612`、CMR@1 `33.726%`、CMR@2
`81.604%`、CMR@5 `99.528%`。

本实验只查 baseline 训练协议：CMR@1/2 低约 10pp 是否由公开仓库默认协议、论文文字协议、LR 解释或
checkpoint 曲线造成。如果简单协议/超参能追回 CMR@1/2，后续 E-028B 系统接入必须使用该训练 recipe。

## 2. 边界

- 只训练 GDROS baseline 副本；
- official `training.zip` 全 `2011` 对用于训练；
- official `testing.zip` 全 `424` 对只用于 final evaluation 与离线曲线诊断；
- 禁止用 `testing.zip` 做训练、validation、early stop、checkpoint selection、hparam selection 或成功追认；
- 禁止加载 official trained `GDROS-os512.pth` 作为训练初始化；
- 允许 RAFT/Sintel generic init，并必须记录 SHA；
- 原 GDROS repo、released checkpoint、E-028B checkpoint、DINO checkpoint 只读；
- 不启动 E-028B integration rerun、轻量 matcher、E-030B、OS424/confirm 或 rescue。

## 3. 必须先核对

P0 必须记录：

- `training.zip` SHA/count：`2887e7de8cd72290df688de0ce4071f42d7bfdec8507a24716ae2ea15c3f3cf5` / `2011`；
- `testing.zip` SHA/count：`e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218` / `424`；
- released checkpoint replay reference；
- official GitHub default config：batch `2`、canonical LR `1.25e-4`、120k steps、400 crop、public augment/loss；
- paper text config：AdamW、initial LR `1.2e-5`、batch `12`、120k steps、train/test iters `12/32`、400 center crop；
- E-029A actual config：batch `12`、OneCycle max LR `1.2e-5`、public augment/loss；
- no official trained GDROS weight init；
- baseline readonly/no writes；
- testing.zip not used for selection.

## 4. Tracks

### A. `REPO_DEFAULT_RETRAIN`

- 从 official training.zip 2011 训练；
- 使用公开仓库默认核心超参：batch `2`、canonical LR `1.25e-4`、120k steps、public augmentor、public
  `flow_loss + mat_flow_loss`；
- final checkpoint 在 official testing.zip 424 上评估。

### B. `BATCH12_HIGH_LR_RETRAIN`

- 从 official training.zip 2011 训练；
- 保留 batch `12`，把 OneCycle max LR 改为 `1.25e-4`；
- 其余尽量同 E-029A paper-protocol；
- final checkpoint 在 official testing.zip 424 上评估。

### C. `CHECKPOINT_CURVE_DIAG`

- 对 E-029A paper-protocol、Track A、Track B 的 saved checkpoints（如 20k/40k/.../final）在 official
  testing.zip 424 上离线评估；
- 输出 AEPE、CMR@1/2/5 曲线；
- 该曲线只解释 training dynamics，不允许用于 checkpoint selection 或成功追认。

## 5. Metrics

必须报告：

- AEPE；
- CMR@1/2/5；
- public-code variance / paper RMSE if official evaluator reports；
- PLR/finite；
- train loss、runtime、peak memory、wall time；
- 每个 track 的 actual command、config、init SHA、checkpoint SHA；
- 与 E-029A paper-protocol 和 released checkpoint reference 的差值。

## 6. Gates

### P0：protocol/isolation

必须 PASS：

- archive SHA/count verified；
- protocol diff manifest written；
- no official trained GDROS weight loaded for training；
- all init sources recorded；
- original baseline readonly/no writes；
- testing.zip excluded from train/val/early-stop/checkpoint/hparam selection；
- evaluator matches released checkpoint reference。

P0 fail：`PROTOCOL_SWEEP_SETUP_BLOCKED`。

### G0：training execution

必须 PASS：

- Track A/B 完成 planned stop，或资源阻塞被记录；
- train loss finite/decreases；
- checkpoints strict-load；
- final testing outputs materialized before truth_flow metric join；
- reviewer approves protocol/no-weight/evaluator audit。

G0 fail：`PROTOCOL_SWEEP_EXECUTION_BLOCKED`。

### G1：baseline CMR recovery

判读：

- `PROTOCOL_SWEEP_RECOVERS_RELEASED_CMR`：任一 final track（非 testing-selected checkpoint）达到 AEPE
  `<=2.0`、CMR@2 `>=78%`、CMR@5 `>=99%`；
- `PROTOCOL_SWEEP_PARTIAL_CMR_GAIN`：best final track 相对 E-029A paper-protocol CMR@2 提升 `>=5pp`
  但未达到 released range；
- `PROTOCOL_SWEEP_NO_SIMPLE_GAIN`：无 final track CMR@2 提升 `>=5pp`；
- `PROTOCOL_SWEEP_RESOURCE_BLOCKED`：资源/时间阻塞。

## 7. 审阅

需要两路 reviewer：

1. protocol/model reviewer：协议 diff、初始化、训练命令、loss/LR/batch/augment 是否绑定清楚；
2. evaluator/isolation reviewer：testing.zip 不参与选择、released checkpoint reference/evaluator 一致、metric aggregation 正确。

## 8. 输出

只回传：

- `RESULT_CARD.md`
- `protocol_diff_manifest.json`
- `initialization_manifest.json`
- `track_train_reports.json`
- `official_testing_metrics.json`
- `checkpoint_curve.json`
- `comparison_table.json`
- `resource_report.json`
- reviewer conclusions
- `RUN_MANIFEST.json`
- watchdog log/report

完成后停止等待用户。
