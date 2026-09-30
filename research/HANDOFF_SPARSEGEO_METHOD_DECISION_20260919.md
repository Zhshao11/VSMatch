# SparseGeo-GDROS 方法决策交接（2026-09-19）

## 1. 交接目的

本文件是新方法决策对话的主入口。它总结 OS 与 WHU 两条研究线、当前最佳方法 E075A-M4、历史有效模块、LuckyR1 设计与实验结果，以及远端实验基础设施。

新对话应基于本文件和链接的正式卡继续，不需要重读完整旧聊天。任何新结论必须以当前文件、正式 `RESULT_CARD`、源码和远端运行状态为准。

## 2. 协作与执行规则

- 与用户讨论方法、解释结果和汇报时使用中文；内部方法设计和实验指令可用英文。
- 方法决策对话负责诊断、理论判断、实验设计和验收，不直接承担长期 GPU 执行。
- 所有新实验移交给普通实验工程对话：
  - 标题：`GDROS SparseGeo 实验工程_5.6sol`
  - Thread ID：`01a09ae2-c972-7ef1-93a4-62448d2fd11a`
  - 模型：`gpt-5.6-sol`
  - 推理强度：`high`
- 禁止在实验工程中调用 `create_goal`；禁止 Codex heartbeat 轮询。长期等待和依赖推进由远端 watchdog/orchestrator 负责。
- 实验派发采用低 token 协议：只发送增量目标、事实路径、动作、门槛和回传项，通常 300–600 字，不重复历史背景。
- 实验工程负责 model-facing 实现、watchdog 提交、数据/环境/OOM/调度/reducer 等非方法修复，无需用户逐次确认。
- 每次 attempt 无论成功、失败、OOM、SIGTERM、kill、超时或协议阻塞，都必须生成 `RESULT_CARD.md/json`；被 kill 时尽量核查执行者和系统证据，不可无证据归因。
- GPU 任务使用 `candidate_gpus=null`，watchdog 自动选择真正独占空卡；进程内仅使用 `CUDA_VISIBLE_DEVICES` 映射后的逻辑 `cuda:0`。
- 不覆盖既有 checkpoint 和正式结果；预测必须在读取 GT 计算指标前落盘。
- 正式主指标默认是 pair-level AEPE 与 CMR@1/2/5/10/16。Pixel/token 指标必须明确标为诊断，不能与论文 pair-level CMR 混用。

## 3. 数据集与比较边界

### OS dataset

- 原始分辨率约 1 m。
- 官方训练来源：2011 pairs。
- 官方测试对象原始数据集可见 424 items；历史 E075A 正式 evaluator 使用有序前 404 pairs。LuckyR1 已审计 `Subset(range(404))` 与历史 E075A materialized pair ID 一致。
- 不能用 OS 与 WHU 的绝对指标横向解释方法优劣；只在各自数据集内比较本方法和 baseline。

### WHU-OPT-SAR dataset

- 原始分辨率约 5 m。
- 官方协议已核对为 5600 train / 700 test。
- 架构应与 OS 统一，但损失、训练日程和数据增强可按数据集分别训练。

## 4. 基准

### OS GDROS released replay

正式复现参考：

| AEPE ↓ | CMR@1 ↑ | CMR@2 ↑ | CMR@5 ↑ |
|---:|---:|---:|---:|
| 1.4615 | 33.7264% | 81.6038% | 99.5283% |

来源：`research/results/E-001/formal-official-archive-run-01.md`。

### WHU 论文 GDROS baseline

用户提供的论文表：

| 指标 | 论文 baseline |
|---|---:|
| CMR@1 | 72.05 +/- 1.06% |
| AEPE@1 | 0.60 +/- 0.01 |
| CMR@2 | 96.86 +/- 0.65% |
| AEPE@2 | 0.78 +/- 0.01 |
| CMR@5 | 99.57 +/- 0.11% |
| AEPE@5 | 0.83 +/- 0.01 |
| All AEPE | 0.90 +/- 0.04 |
| RMSE | 0.62 +/- 0.15 |

E080A 已通过官方数据协议审计；其小样本 replay sanity 不是完整 700-pair 最终复现，不能代替论文表。

## 5. 当前最佳 OS 方法：E075A-M4

### 定位

截至本交接，OS 当前最佳主方法仍是：

**E075A-M4 Region Gate，`M4_step80000.pt`，`iter24_post_lsr`。**

正式状态：`PRESERVE_RECOVER_SUCCESS`。

| AEPE ↓ | CMR@1 ↑ | CMR@2 ↑ | CMR@5 ↑ | CMR@10 ↑ | CMR@16 ↑ |
|---:|---:|---:|---:|---:|---:|
| 2.3613 | 56.6832% | 78.9604% | 88.8614% | 94.8020% | 97.0297% |

M4 相对 released GDROS：CMR@1 高 `22.95 pp`，但 CMR@2 低 `2.64 pp`、CMR@5 低 `10.67 pp`，AEPE 高约 `0.90 px`。优势是 1 px 精定位，主要缺陷是 5 px 外长尾。

### 结构

```text
OPT / SAR
  ├─ E074A-B
  │   ├─ E073-F2 matcher
  │   ├─ E073-F3 matcher
  │   └─ B 内部空间 gate 融合 F2/F3 flow
  ├─ E074A-C
  │   └─ F3 初始化 student + teacher-average distillation
  └─ E075A-M4 region gate
      ├─ B flow, C flow, B-C, |B-C|
      ├─ LSR(B), LSR(C), |LSR(B)-B|, |LSR(C)-C|
      ├─ 16 -> 32 -> 16 -> 1 Conv gate
      ├─ 4x4 average-pool region gate
      ├─ fused = (1-g)B + gC，g 为 C 权重
      └─ iter24 后 LSR
```

训练时冻结 B/C，只训练 M4 region gate。40K/80K/120K 都评估过；80K 在 AEPE/CMR@5 综合上最好，不能简单用最长训练。

### 关键路径

- 快照：`research/results/CURRENT_BEST_OS_METHOD_20260913.md`
- OS 诊断：`research/results/E075D_OS_DIAGNOSTIC_REPORT.md`
- checkpoint：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt`
- checkpoint SHA256：`55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b`
- 源码：`/mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated/.codex_tmp/e075a_preserve_high_precision_recover_c2c5.py`
- 源码 SHA256：`1b74c437306e236538e24d110c609d349c56bd7f5d26435c6cc166ae86ec95c3`

### 已确认局限性

- 2011 个训练来源 pair 上 pair-level CMR@5=100%，但这些样本被模型见过；历史测试 404 中有 45 个 AEPE>=5 px 失败，说明主要是泛化落差。
- 45 个失败中，41 个在 B/C 精确 iter24 post-LSR 上都失败；M4 gate 仅可能直接恢复剩余 4 个。
- 没有普遍的 iter12 成功后 iter24 失败；没有系统性 pre-LSR 成功后 post-LSR 失败，因此迭代数与最终 LSR 不是首要瓶颈。
- GT 通常在 recurrent query 支持内；扩大窗口缺少针对性。
- 失败样本的 raw correlation 和正确 basin token 覆盖较弱，但不完全缺失；未严格因果区分“证据表达不足”和“recurrent update 未利用证据”。

## 6. 当前最佳 WHU 自研方法：E081A-M4

E081A-M4 为 WHU 当前已出的最佳自研方法，独立在 WHU 原生训练，初始化只使用 RAFT-Sintel architecture pretrain，没有 OS/E075/E077 task checkpoint。

| AEPE ↓ | CMR@1 ↑ | CMR@2 ↑ | CMR@5 ↑ | CMR@10 ↑ | CMR@16 ↑ |
|---:|---:|---:|---:|---:|---:|
| 1.9054 | 28.00% | 72.00% | 98.1429% | 99.5714% | 99.5714% |

- checkpoint：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-081A-ZZVR-WHU-OUR-METHOD-NATIVE-TRAIN/run-m4-native-r2/checkpoints/M4_step120000.pt`
- SHA256：`4122c27a7a0a1d0cf6488b9207277b0024d2fe0a9f07f6fcd8eec8ef09ca745a`

WHU 局限：CMR@5 已接近 baseline，但 CMR@1/2 与论文 baseline 差约 44.05/24.86 pp，说明 WHU 主要是局部精化/细粒度对应问题；OS 则主要是 5 px 外泛化长尾。统一架构不等于两个数据集使用相同损失或完全相同病因。

## 7. 历史有效技术链

### 7.1 E015A：DINOv2-L expanded candidate generation

- DINOv2-L full4096 candidate generation 证明强预训练特征能显著增加正确候选。
- top128 correct@2 mean/P50 约 `115.94/115`，超过早期 E011A 的 `68.71/66`。
- 但 no-GT 排序和几何压缩失败，oracle affine AEPE 仍约 `12.90 px`。
- 价值：把瓶颈从“候选里没有正确点”推进到“如何组织、解码和几何利用候选”。它不是最终匹配方法。

### 7.2 E028B：safe dual adapter / projection-only feature adaptation

- 冻结 DINO 主干，训练 OPT/SAR adapter；projection-only safe checkpoint 保留了 optical teacher，SAR residual scale 0.1、OPT residual scale 0.001。
- 训练损失和双 adapter 梯度有效，形成后续 E037–E046 特征实验的基础 checkpoint。
- checkpoint：`.../E-028B-ZZVR-RECOVERY/run-01/checkpoints/projection_only_safe.pt`
- SHA256：`76d70a1e2ef0677571590ad1dba6e68bad772f883b59da05c519129beb9c6459`
- 局限：它本身不是正式最终 matcher 结果；接入旧 GDROS backend 曾失败，证明特征改善不能脱离 matcher 分布共同训练。

### 7.3 E046A-R2：masked-pretrain joint frontend + matcher

相对 E044A frozen reference，在相同公平 affine evaluator 上产生小但真实提升：

| 方法 | AEPE ↓ | CMR@1 ↑ | CMR@2 ↑ | CMR@5 ↑ | CMR@16 ↑ | top1 capture |
|---|---:|---:|---:|---:|---:|---:|
| E044A frozen | 6.6377 | 2.8787% | 11.9514% | 49.5706% | 93.9990% | 未列 |
| E046A-R2 masked joint | 6.4743 | 3.3974% | 12.3450% | 50.1211% | 94.4388% | 75.6362% |

价值：证明 frontend 与 matcher 联合训练是必要方向，且 masked-pretrain joint 优于同预算 reference。局限是绝对指标离 dense GDROS 很远，不是最终主线结果。

### 7.4 E073A：F2/F3 fusion foundation

| 分支 | AEPE ↓ | CMR@1 | CMR@2 | CMR@5 |
|---|---:|---:|---:|---:|
| F2 120K | 3.0074 | 22.2772% | 60.8911% | 89.1089% |
| F3 120K | 2.8286 | 35.6436% | 62.6238% | 88.3663% |

F2 提供较好的 CMR@5，F3 提供较好的 AEPE/CMR@1，形成后续互补融合基础。

### 7.5 E074A：两阶段 F2/F3 融合

| 分支 | AEPE ↓ | CMR@1 | CMR@2 | CMR@5 | 定位 |
|---|---:|---:|---:|---:|---|
| B 180K | 2.5053 | 44.8020% | 72.7723% | 89.3564% | balanced |
| C 180K | 2.5697 | 48.7624% | 76.9802% | 87.8713% | high precision |
| A no-GT chooser | 2.6361 | 36.8812% | 67.8218% | 90.0990% | CMR@5 diagnostic |

E074 B/C 是 E075 M4 的两个冻结输入分支。来源：`research/results/E074A_BEST_FREEZE.md`。

### 7.6 E075A-M4

M4 将 E074 B/C 的互补优势推进到当前 OS 最佳：CMR@1 从 C 的 48.76% 提高到 56.68%，AEPE 降到 2.3613；但 CMR@5 仍只有 88.86%，未超过 B/A no-GT，更远低于 released GDROS 99.53%。

## 8. 已淘汰或不应回退的方向

- 早期“从候选池挑少量正确点”路线：E017/E018/E021/E022 等虽然学到部分 enrichment/ranking 信号，但 precision、正确点数量和 affine utility 不足。问题不应继续定义为独立 selector。
- 只修 topK/threshold：K512 比 K256 更受 outlier contamination，已停止阈值修补。
- 小窗口 fine refinement：大量实验显示窗口 oracle、hypothesis coverage 或 fine feature 都受前级 basin 限制；简单扩大窗口只会提高 oracle 并加重选择难度。
- 只改 final LSR 或迭代次数：OS 诊断不支持其为主瓶颈。
- Proposed A 全图 affine candidate pool：参数族有容量，但现有 frozen evidence 和 candidate generator 没产生优于 base 的候选；不能用 released GDROS 的 R3/R4 直接判定当前 M4 特征无效。
- 仅替换 feature 后直接接旧 backend：GDROS backend 对原 feature distribution 有强适配，替换后不联合训练会显著退化。

## 9. LuckyR1 方法

设计源：`/Users/shihao/Documents/文献库/knowledge/reports/E075D_OS_对应信念与关系相关性_方法设计_20260910.md`。

### LuckyR1-A：几何耦合对应信念

- 在真实 CorrBlock query 与 BasicUpdateBlock 之间维护每个 source 的多候选分布 `q` 和 unmatched mass。
- 候选以绝对 target 坐标跨轮迁移，不能把 offset index 当作相同对应。
- source 邻居候选使用局部 Jacobian 几何兼容性传递支持。
- 目标是让弱但空间一致的正确峰进入 recurrent update；分离模式不直接平均。

### LuckyR1-B：几何对齐的模态内关系相关性

- 原始点积保留细节。
- 在 OPT 与 SAR 内分别构造局部结构关系，按当前 Jacobian 对齐 target 邻域。
- 关系证据作为 raw correlation 的互补通道进入原 updater，不产生第二个 final flow head。

### LuckyR1-C：联合 belief + relation

- relation score 进入候选 belief logits。
- 几何消息在 source 邻域耦合候选分布。
- 共同驱动原 BasicUpdateBlock。

新颖性边界：GMA、RoMa、DASC、GOCor、CHM、CRFT 已覆盖全局聚合、多峰、自相关、相关体优化、几何投票和 recurrent cross-modal refinement 的部分机制。潜在贡献只能落在“显式绝对坐标对应分布 + 候选间几何耦合 + 跨轮迁移 + 与当前 OPT-SAR recurrent matcher 的联合方式”。

## 10. LuckyR1 实验结论

### 10.1 Current-model exploratory（历史 M4，404 pairs）

| Arm | AEPE ↓ | CMR@1 | CMR@2 | CMR@5 | CMR@10 | CMR@16 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| M4 control | 2.3613 | 56.6832% | 78.9604% | 88.8614% | 94.8020% | 97.0297% | baseline |
| A step500 | 6.1013 | 25.7426% | 54.7030% | 71.2871% | 79.9505% | 86.8812% | 强退化 |
| B step500 | 2.36174 | 56.6832% | 78.9604% | 88.8614% | 94.8020% | 97.0297% | 等同 control，无信号 |
| C step500 | 2.36151 | 56.4356% | 78.9604% | 88.8614% | 94.8020% | 97.0297% | 无信号，CMR@1 略降 |

三条 current-model 路线均已 cost-gate stop。404 prediction、manifest、GT join 均完整，prediction 先落盘；这些是探索性 test evidence，不是 model-unseen 证明。

### 10.2 Clean 1810/201 lineage

固定 split：1810 train / 201 model-unseen dev。由于历史 V1/V2 lineage 有数据暴露问题，clean 路径采用 split-safe 重构，必须只和自己的 raw control 比较，不能与历史 E075A 数值直接作 model-unseen 比较。

选定 B/C：

| 模块 | checkpoint | AEPE | CMR@1 | CMR@2 | CMR@5 |
|---|---:|---:|---:|---:|---:|
| clean B | 80K | 3.6655 | 18.9055% | 51.2438% | 77.1144% |
| clean C | 40K | 3.8171 | 18.4080% | 39.3035% | 77.1144% |

Clean M4 raw control 120K：

| AEPE | CMR@1 | CMR@2 | CMR@5 | CMR@10 | CMR@16 |
|---:|---:|---:|---:|---:|---:|
| 3.35894 | 22.8856% | 55.2239% | 79.1045% | 94.0299% | 98.5075% |

Clean LuckyR1-A 在 2K/5K/10K/20K 都与 raw M4 几乎完全一致，最佳 2K AEPE `3.35879`，所有 CMR 不变；说明新增模块在 clean 路径也近似 no-op。

Clean LuckyR1-B/C 曾多次受外部 GPU collision/SIGTERM；截至 2026-09-19：

- B 最新任务达到最大重试次数后失败，没有可信 clean endpoint。
- C 已有至少 10K checkpoint，但最新任务在 10,040 step 被 SIGTERM；正式 clean endpoint 尚未收口。
- `lucky_reduce` / finalizer 尚未完成。
- 专用 orchestrator PID `2324644` 仍在运行，heartbeat 正常，但专用队列当前为空；需由实验工程决定是否为 B/C 增加有界恢复 attempt，不能无限重试。

### 10.3 当前判断

- A：明确失败或 no-op，不应继续当前形态。
- B/C：current-model 结果表明几乎 no-op；clean A 同样 no-op。B/C clean 尚未完整，但当前证据不支持大规模继续长训。
- LuckyR1 顶层假设尚不能被数学上完全否定，但当前“仅训练少量 evidence adapter、冻结 base matcher/update”没有足够作用能力。
- 如果继续此类方法，优先转为 LuckyR2：保留原 BasicUpdateBlock 作为 identity 主路，增加零初始化、置信门控的 structured residual update，并联合训练 evidence adapter、motion encoder 和部分 update block；先做 8–16 pair 过拟合和 action-path delta 诊断，再上完整训练。

## 11. 当前远端基础设施

- 专用 watchdog root：`/mnt/sda/zhushihao/experiment_watchdog_luckyr1`
- 专用 daemon：由该 root 的 heartbeat 验证。
- orchestrator：`/mnt/sda/zhushihao/experiment_watchdog_luckyr1/luckyr1_orchestrator.py`
- orchestrator state：`/mnt/sda/zhushihao/experiment_watchdog_luckyr1/lucky_orchestrator_state.json`
- DAG：`/mnt/sda/zhushihao/experiment_watchdog_luckyr1/lucky_plan.json`
- 父实验：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN`
- 当前普通实验工程对话：`01a09ae2-c972-7ef1-93a4-62448d2fd11a`
- 禁止再使用已卡死的共享 watchdog 处理 LuckyR1。

## 12. 新对话的建议起点

新方法决策对话应先决定以下三项之一，不要直接继续无界长训：

1. **收口 LuckyR1**：允许实验工程为 clean B/C 各做一次有界恢复，拿到 endpoint 后正式判定 NO_GAIN；不再增加新 loss 或长训。
2. **设计 LuckyR2**：先做 actionability/overfit 验证，确认 structured residual branch 能改变错误 basin 且关闭时严格还原 M4，再做完整 OS 实验。
3. **先做 WHU 同构诊断**：在 WHU E081A-M4 上分离 feature/correlation、recurrent update 和 fine localization，再决定一个统一模块是否合理。

推荐顺序：先低成本收口 clean LuckyR1-B/C；若仍 no gain，停止 LuckyR1，做 WHU 诊断后再决定 LuckyR2。不要因为模块能训练或有非零梯度就继续长训。

## 13. 必读文件

- `research/results/CURRENT_BEST_OS_METHOD_20260913.md`
- `research/results/E075D_OS_DIAGNOSTIC_REPORT.md`
- `research/results/E074A_BEST_FREEZE.md`
- `research/results/LUCKYR1_RECOVERY_20260915_RESULT_CARD.md`
- `research/results/LUCKYR1_CURRENT_MODEL_EXPLORATORY_A_STEP500_RESULT.md`
- `research/results/LUCKYR1_CURRENT_MODEL_EXPLORATORY_B_STEP500_RESULT.md`
- `research/results/LUCKYR1_CURRENT_MODEL_EXPLORATORY_C_STEP500_RESULT.md`
- `research/results/LUCKYR1_5_6SOL_TAKEOVER_RESULT_CARD.md`
- `/Users/shihao/Documents/文献库/knowledge/reports/E075D_OS_对应信念与关系相关性_方法设计_20260910.md`

## 14. 不应丢失的最终事实

- OS 当前最佳仍是 E075A-M4 80K，而不是 LuckyR1。
- WHU 当前最佳自研是 E081A-M4，但 CMR@1/2 距论文 baseline 很远。
- M4 的主要 OS 瓶颈是测试泛化和 5 px 外长尾，非简单窗口、迭代或 LSR。
- LuckyR1 当前实现没有正信号；不能继续用“训练不够”解释所有失败。
- B/C、M4、LuckyR1 的 clean 数值只在重构 lineage 内比较，不可与历史 M4 直接当作公平 model-unseen 对照。
- 所有进一步实验必须通过普通 `5.6sol/high` 实验工程对话移交，低 token，实验工程自行解决非方法阻塞。
