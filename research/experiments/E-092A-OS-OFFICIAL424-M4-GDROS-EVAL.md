# E092A：M4 与 GDROS 的 OS 官方口径重评估

日期：2026-11-15。用户已授权启动评测。只评估固定模型，不训练，不选择新 checkpoint。

## 固定比较

- 主表：GDROS 官方发布权重与 OS E075A-M4 step80K，均 32 次 recurrent iteration，最后执行官方 LSR。
- 补充表：同一 M4 step80K 的 iter24 post-LSR，保留历史部署点；提前声明，不根据结果在 24/32 间挑选主表。
- 完整官方 OS testing.zip 的 424 对，512×512，全图统计，包含黑边，不使用 valid mask。
- 官方 LSR 固定 `Npoint=200000`，保留 row-major `[10000:210000]` 取点方式和原始求解逻辑。
- batch=1、eval、inference_mode、FP32，seed=1234。记录设备、库版本、实际初始化和全部代码哈希。

## 模型身份

- M4：`GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt`
- M4 SHA256：`55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b`。
- 核对 B/C checkpoint 及其上游依赖；哈希见 `research/CHECKPOINTS.json`。不覆盖权重。
- GDROS：官方 OS 发布权重，历史已审 SHA256 `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`。若实际不符，先定位原因并留卡，不能悄悄换版本。
- 对参数加载记录 missing/unexpected keys；不以未审 `strict=False` 掩盖有效路径参数缺失。

## 公式

主指标严格复现官方 test.py 在 batch=1 下的公开口径：

`pair_public_error = mean_pixels((abs(dx) + abs(dy)) / 2)`。

须以真实 `[1,2,H,W]` prediction 和 `[2,H,W]` GT 回归验证，它与官方 `sum(dim=0).sqrt().mean()` 的结果一致。

同时独立输出标准向量 EPE：

`pair_vector_epe = mean_pixels(sqrt(dx**2 + dy**2))`。

每个口径先统计逐对误差，再统计均值、CMR@1/2/3/4/5/10/16（严格 `<t`）、失败数、P95/P99。公开代码名为 RMSE 的字段实际是方差，按 variance 标注，真正标准差另列。不得把两种误差混表或用固定比例换算。

## 产物与完整性

1. CPU preflight：验证所有权重、424 个唯一 pair ID、数据来源、尺寸、加载路径和代码版本。
2. 独立预测阶段：仅加载图像，分别为 GDROS32、M4-32、M4-24 保存完整 post-LSR 预测张量和 manifest；不得在该阶段读取 GT 或写入含标签的门控特征。可以每对保存一个含所有输出的 NPZ，控制文件数量。
3. 所有 424 对 manifest、预测文件及哈希齐全之后，单独 CPU metric join 加载 GT，生成两套指标和逐对结果。
4. 原始预测留 zzvr，不提交 Git；同步紧凑 RESULT_CARD.md/json、协议、运行入口、依赖及逐对紧凑摘要。
5. 每次失败、OOM、取消、SIGTERM 均留 attempt 卡；不得覆盖历史实验。

## 回归与报告

- GDROS 官方32轮主表应接近 E068A 回放：公开误差约1.461486、CMR@1/2/5约33.73/81.60/99.53%。同时解释与 E000 的微小浮点差异，不能重调模型以追数值。
- M4-24 前404对的标准向量结果应复现 AEPE2.361299、CMR@1/2/5=56.683168/78.960396/88.861386%。偏差必须调查，不能直接覆盖旧记录。
- 完整424对为主表；前404对只能作为回归附表。
- 不把测试结果用于 checkpoint、轮数、阈值或新架构选择。

## 调度

- 实验由原工程对话 `01a0b9a5-4074-7ec1-903b-4acb62361f69` 实施，在 zzvr 运行。
- 使用远端 watchdog/orchestrator，`candidate_gpus=null`、独占 GPU、进程内逻辑 `cuda:0`。不直接抢占或终止其他人的进程。
- 2026-11-15 查询发现原 orchestrator state heartbeat 停在 2026-10-11，旧计划还有 waiting 项。先判断进程状态；新任务使用隔离计划/状态，或明确隔离旧 waiting 项，防止启动旧 finalizer。
- 不创建 Codex heartbeat/Goal；由远端调度独立完成。主任务通过读取工程对话及远端卡跟进。
