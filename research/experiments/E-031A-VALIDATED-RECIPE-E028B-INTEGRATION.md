# E-031A-VALIDATED-RECIPE-E028B-INTEGRATION：E028B Frontend + GDROS Backend Under Validated Recipe

方法版本：`M001 v3.6 design`

状态：`PROPOSED / WAITING_USER_APPROVAL`

## 1. 目标

E-030A 已证明 GDROS baseline 在 validated recipe 下可达到 AEPE `1.4347`、CMR@2 `83.4906%`。此前
E-029A 的 E028B 接入负结果使用了不可信训练 harness，因此不能否定 E-028B。

本实验目标：在 validated GDROS recipe 下，重新测试 E-028B safe dual adapter frontend 接入 GDROS backend
是否能带来系统级增益，或明确失败来自 frontend/backend interface mismatch。

## 2. 后续训练参考 recipe

默认使用 E-030A Track B recipe：

- official OS `training.zip` 全 `2011` 对训练；
- official OS `testing.zip` 全 `424` 对 final evaluation only；
- batch `12`；
- OneCycle max LR `1.25e-4`；
- 120k steps；
- train/test iters `12/32`；
- public augmentor；
- public loss：`flow_loss + mat_flow_loss`；
- 不加载 official trained `GDROS-os512.pth` 作为训练初始化；
- 允许 RAFT/Sintel generic init，并记录 SHA。

## 3. Tracks

### A. `BASELINE_PARITY_REFERENCE`

- 只读 E-030A Track B final checkpoint；
- replay official testing.zip 424，确认 evaluator/metrics 与 E-030A 一致；
- 作为系统比较 baseline。

### B. `E028B_FRONTEND_VALIDATED_RECIPE_FULL_TRAIN`

- 复制 E-028B safe dual adapter frontend 到实验副本；
- 原 E-028B formal checkpoint、DINO checkpoint、GDROS baseline checkpoint 均只读；
- 用 bridge/adapter 将 E-028B frontend 接到 GDROS backend；
- 在 validated recipe 下训练到 planned stop；
- final checkpoint 在 official testing.zip 424 上评估。

### C. `INTERFACE_DIAGNOSTICS`

必须记录：

- E-028B feature-only fair matcher reference；
- frontend copy drift；
- bridge 输入/输出统计；
- backend loss 与梯度是否有效；
- feature distribution 与 GDROS baseline feature distribution 的距离；
- full-system 失败时，区分 frontend degradation、bridge mismatch、backend limitation。

## 4. Gates

### P0

必须 PASS：

- E-030A Track B recipe 与 checkpoint SHA 记录；
- official training/testing SHA/count verified；
- E-028B checkpoint copied before training，原 checkpoint 只读；
- no official trained GDROS checkpoint loaded for training；
- testing.zip not used for train/val/early-stop/checkpoint/hparam selection；
- evaluator matches E-030A/released reference。

P0 fail：`E031A_SETUP_BLOCKED`。

### G0

必须 PASS：

- training completes planned stop or documented resource stop；
- losses finite/decrease；
- gradients reach frontend copy、bridge、backend；
- checkpoints strict-load；
- final testing outputs materialized before truth_flow metric join；
- reviewers approve recipe/isolation/interface audit。

G0 fail：`E031A_EXECUTION_BLOCKED`。

### G1

判读：

- `E028B_SYSTEM_GAIN`：相对 E-030A Track B validated baseline 获得预注册系统级增益；
- `E028B_SYSTEM_PARITY_ONLY`：接近 baseline 但无增益；
- `E028B_BACKEND_MISMATCH`：E028B 诊断仍强但 full-system 弱，指向 interface/backend mismatch；
- `E028B_FRONTEND_DEGRADES_UNDER_FULL_TRAIN`：joint training 中 E028B 前端优势崩塌；
- `E031A_RESOURCE_BLOCKED`：资源/时间阻塞。

## 5. 输出

只回传：

- `RESULT_CARD.md`
- baseline parity metrics
- full-system metrics
- interface diagnostics
- train report
- resource report
- reviewer conclusions
- `RUN_MANIFEST.json`

完成后停止等待用户，不自动启动轻量 matcher、sparse feedback、E-031B、confirm 或 rescue。
