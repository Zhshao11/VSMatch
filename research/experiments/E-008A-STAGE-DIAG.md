# E-008A-STAGE-DIAG：Region-to-Point 候选生成逐阶段 GT recall 诊断

## 目的

本实验只诊断 `E-008A / M001 v1.1` 的候选生成失败位置，不改变方法、不训练新模型、不启动
`E-008B` 或 rescue。

要回答的问题：

> E-008A 正确候选覆盖下降，是发生在 region top-L 粗对应、matched-region 内 top64 预筛、
> descriptor/patch/structure top4 排序，还是 final union/dedup？

## 冻结输入

- 复用 `E-008A` 正式实现、split、checkpoint、cache 和 dev 404 pairs。
- 复用 `E-008A` 正式候选生成路径；不得改变任何 candidate 输出语义。
- raw `truth_flow` 只允许在诊断/eval 阶段读取，不得影响候选生成、排序、top-k 或 dedup。
- 不读取 OS424、confirm 或 datum label。

## 必须记录的阶段

对每个 dev pair、每个 source proposal，记录以下逐阶段状态。

### S0：source proposal

- source proposal 数量、source token 坐标、pixel 坐标。
- 该 source 的 raw truth endpoint 是否 in-bounds。
- endpoint 标签阈值固定为 `<=1/2/5 px`，距离为 candidate discrete target pixel 到 raw truth endpoint 的
  Euclidean distance。

### S1：region top-L containment

- 4/8/16 三个尺度各自的 matched target regions 是否包含 raw truth endpoint 所在 target token/区域。
- combined matched-region union 是否包含 raw truth endpoint。
- union target token 数量。

### S2：within-region top64 pool

- 在 matched-region union 内进行 E-008A 原始 top64 预筛后，记录：
  - GT-nearest candidate distance；
  - GT-nearest candidate rank；
  - 是否存在 `<=1/2/5 px` candidate；
  - top64 pool size。

### S3：各 top4 分支

分别对以下分支记录 top4 中是否存在 `<=1/2/5 px` candidate，以及 GT-nearest rank/distance：

- region-score top4；
- descriptor-score top4；
- patch/structure-score top4；
- full-score top4。

若 E-008A 代码中分支命名略有不同，需在 `RESULT_CARD.md` 说明一一映射，不能合并成一个黑盒分支。

### S4：pre-dedup union

- 上述 top4 分支 union 后、dedup 前的候选数。
- 是否存在 `<=1/2/5 px` candidate。

### S5：post-dedup final candidates

- dedup 后最终候选数。
- 是否存在 `<=1/2/5 px` candidate。
- final stage 指标必须与 E-008A P0 的 mean correct、P10/P50、base rate 在预注册容差内一致。

## 汇总指标

输出 pair-level 与 aggregate 两级统计。

Aggregate 至少包含：

- 每阶段 mean/P10/P50/P90 的 `2px-correct proposals`；
- 每阶段 source-proposal-level `1/2/5px` recall；
- 每阶段 candidate-level `1/2/5px` base rate；
- 条件保留率：
  - S1/S0；
  - S2/S1；
  - S3/S2，分别按四个 top4 分支；
  - S4/S2；
  - S5/S4。
- 每阶段 GT-nearest distance 的 mean/P50/P90/P95。
- 每阶段 in-bounds 与 invalid endpoint 数量。

## 判读规则

本实验是诊断，不设方法成功 gate。最终状态只能是：

- `COMPLETED / REGION_LOSS_DIAGNOSED`：S1 combined region containment 是最大主要损失；
- `COMPLETED / TOP64_POOL_LOSS_DIAGNOSED`：S1 仍保留较多 GT，但 S2 top64 大幅丢失；
- `COMPLETED / TOP4_RANKING_LOSS_DIAGNOSED`：S2 保留较多 GT，但 S3/S4 大幅丢失；
- `COMPLETED / DEDUP_LOSS_DIAGNOSED`：S4 保留较多 GT，但 S5 大幅丢失；
- `COMPLETED / MIXED_LOSS_DIAGNOSED`：不存在单一主损失阶段；
- `BLOCKED / PROVENANCE_OR_PARITY_FAIL`：final stage 无法复现 E-008A P0 或 provenance 不一致。

“最大主要损失”按 `2px-correct proposals` 的绝对下降量和条件保留率共同判读；若两个阶段差距不足
`10 pp`，标为 `MIXED_LOSS_DIAGNOSED`，不得强行归因。

## 自动检查

必须通过：

1. E-008A final stage parity：
   - mean 2px-correct proposals 与 `17.022` 差异 `<=0.05`；
   - P10/P50 与 `7/15` 一致或说明 ties/rounding；
   - candidate 2px base rate 与 `0.09828%` 相对误差 `<=1%`。
2. GT 隔离：candidate trace 先物化，truth_flow 只在诊断 join/eval 使用。
3. 坐标检查：source/target 均为 stride-8 `token_xy*8`，方向 optical source → SAR target。
4. 无 hidden `4096x4096`、无 full-image blind search、无 baseline fallback。
5. Synthetic/unit test 覆盖：
   - GT 在 region 内但不在 top64；
   - GT 在 top64 但不在 top4；
   - GT 在 pre-dedup union 但被 dedup 丢弃；
   - GT in/out-of-bounds。

## 审阅

低 token 定向审阅即可。至少一名 reviewer 检查：

- 阶段定义是否对应 E-008A 真实代码路径；
- final parity 是否通过；
- GT 是否只用于诊断；
- loss attribution 是否由统计支持。

若 reviewer 发现 final parity 或 GT 隔离问题，停止并返回方法任务。

## 产物

本地和远端都需提供：

- `RESULT_CARD.md`：不超过 100 行，含最终判读、核心漏斗表、结果路径；
- `stage_summary.json`；
- `stage_pair_metrics.jsonl`；
- `stage_proposal_metrics.jsonl` 或等价压缩表；
- `RUN_MANIFEST.json`；
- reviewer 结论路径。

实验完成后只回传 `RESULT_CARD.md`、`stage_summary.json`、reviewer 结论和远端 run root。不得自动启动后续实验。
