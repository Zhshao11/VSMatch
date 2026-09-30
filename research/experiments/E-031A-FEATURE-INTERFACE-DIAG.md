# E-031A-FEATURE-INTERFACE-DIAG：E028B ↔ GDROS Backend Feature Interface Diagnostic

方法版本：`M001 v3.6 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

在不启动 120k full integration 的前提下，先回答：

> 把 GDROS native frontend feature 换成 E-028B safe dual adapter feature 后，GDROS 后端接口到底差多少？

本实验是 E-031A full integration 的前置诊断。它不做系统成功 claim，只判断是否值得进入 full training。

## 2. 边界

- 只使用 official `training.zip` 固定 probe / small-overfit subset；
- official `testing.zip` 不用于本诊断的选择、阈值、early stop 或成功判读；
- E-030A Track B checkpoint、released GDROS checkpoint、E-028B checkpoint、DINO checkpoint 全部只读；
- 允许复制 E-028B frontend 到实验目录用于诊断；
- 不启动 120k full integration、轻量 matcher、sparse feedback、E-031B、OS424/confirm 或 rescue。

## 3. Tracks

### A. `GDROS_NATIVE_INTERFACE_REFERENCE`

- 用 E-030A Track B / released reference 提取 GDROS native backend-interface feature；
- 在相同 training probe records 上记录 native backend output、flow/affine loss、AEPE/CMR；
- 作为后续替换退化倍数的基准。

### B. `E028B_FEATURE_DELTA`

- 用 E-028B safe dual adapter frontend 提取相同 records 的 feature；
- 报告：
  - shape / token count / channel count；
  - norm、mean/std、channel covariance；
  - FID-like distance；
  - token cosine / nearest-neighbor mutuality；
  - feature-only fair matcher reference。

### C. `BACKEND_COMPATIBILITY_PROBE`

在 training probe 上测试三种接入：

1. direct replacement（若维度/语义允许）；
2. simple normalization / scale alignment；
3. trainable linear bridge。

报告 forward 是否可运行、loss/AEPE/CMR 相对 GDROS native 的退化倍数、失败类型。

### D. `SMALL_OVERFIT_ADAPTABILITY`

- 在固定 small-overfit train subset 上短训 bridge/backend；
- 验证 loss 是否显著下降、是否能 overfit、梯度是否到达 bridge/backend；
- 不使用 testing.zip，不作为方法成功 claim。

## 4. Gates

### P0

必须 PASS：

- training/testing archive SHA/count recorded；
- E-030A Track B recipe/checkpoint SHA recorded；
- E-028B checkpoint SHA recorded and readonly；
- original GDROS/E028B/DINO artifacts readonly；
- fixed probe/small-overfit subsets frozen before metrics；
- testing.zip not used for selection。

P0 fail：`E031A_DIAG_SETUP_BLOCKED`。

### G0

必须 PASS：

- native/E028B features extracted on identical records；
- feature distribution report written；
- backend compatibility probe completed or typed failure recorded；
- small-overfit gradients/loss finite if enabled；
- reviewers approve isolation/interface/evaluator.

G0 fail：`E031A_DIAG_EXECUTION_BLOCKED`。

### G1

判读：

- `INTERFACE_COMPATIBLE_FULL_TRAIN_WORTHY`：bridge/normalization 后 train-probe 退化可控，small-overfit 成功；
- `BRIDGE_REQUIRED_BUT_ADAPTABLE`：direct replacement 明显失败，但 trainable bridge 可 overfit；
- `FEATURE_DISTRIBUTION_MISMATCH`：feature 统计/后端损失差距很大且 small-overfit 失败；
- `BACKEND_INTERFACE_UNAVAILABLE`：无法无歧义访问或替换后端接口；
- `DIAGNOSTIC_RESOURCE_BLOCKED`：资源/时间阻塞。

## 5. 输出

只回传：

- `RESULT_CARD.md`
- `validated_recipe_manifest.json`
- `probe_set_manifest.json`
- `feature_distribution_report.json`
- `backend_compatibility_report.json`
- `small_overfit_report.json`
- reviewer conclusions
- `RUN_MANIFEST.json`
- watchdog log/report

完成后停止等待用户。
