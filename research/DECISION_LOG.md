# 方法决策日志

本文件只追加，不重写历史。详细证据见对应实验结果目录。

## 2026-08-04 / D-001

E-001 证明 K128 足以拟合 GDROS 已预测的仿射，但没有端到端资源收益。批准 E-002A 检查
冻结特征 K-to-all，对应结果见 `results/E-001/`。

## 2026-08-04 / D-002

E-002A 的 direct top1 失败，但 top-k oracle 仍高；不进入 B1/B3/learned descriptor，批准一次
B2-bridge 候选几何救援。结果见 `results/E-002A/`。

## 2026-08-04 / D-003

E-002B 固定 Python 后端资源 gate 失败，accuracy 未测。只否定该执行后端，不否定 selector；
批准唯一预注册救援 E-002B-R，保持全部方法语义与预算不变。结果见 `results/E-002B/`。

## 2026-08-05 / D-004

用户暂停 E-002B-R，并启用低 Token 工作流 v2.0。冻结规范 SHA 不变；实验保持
`PAUSED_BY_USER / P0_EARLY_INCOMPLETE`。恢复需用户明确批准，完成后不得自动派生下一版本。

## 2026-08-05 / D-005

用户明确批准恢复原 E-002B-R 实验任务。按低 Token 工作流从 pair 47 的单一实现等价修复
继续；不得重选 block、改变 gate 或自动创建新版本。

## 2026-08-05 / D-006

E-002B-R 的 P0-early PASS，但 P1 以 `205.055582 ms` 和 `4.365844x` 未达到
`<=80 ms`、`>=8x`，按冻结顺序停止。全预算精确等价后端的预注册救援额度已用尽；不运行
review、G5-R2、P0-full 或 accuracy，不写 `FAILED_METHODS.md`，不自动建立预算缩减或其他
新分支。等待用户决定是否结束 B2 方向，或明确批准一个语义发生变化的新研究分支。

## 2026-08-05 / D-007

用户批准一次独立 E-002B-ACC-DIAG。该实验不改变方法，使用已审阅的只读 v0.4 reference，
跳过资源资格只诊断 selector accuracy；结果不能解除 EFFICIENCY_HOLD，完成后必须等待用户。

## 2026-08-05 / D-008

E-002B-ACC-DIAG 在 provenance、finite、GT 隔离、完整 424 对和实现审阅均通过的条件下，
C8-main PairLocalReady `75.71%`、grid mean `17.233 px`，accuracy gate FAIL；同时 cosine top8
GT oracle PairLocalReady `99.53%`。因此停止固定全预算 B2 六键共识/重排与置信拒绝器，失败不外推到
top-k candidate set、冻结 GDROS 特征或 SparseGeo 方法族。资源 hold 保持；不自动启动下一实验。

## 2026-08-05 / D-009

用户批准 learned selector 训练，以精度和效率双目标推进，并重新建立独立验证/测试协议。
建立 M001 v0.6 / E-003A：冻结 GDROS post-Transformer 特征、K128 anchors 和 cosine top8，
只训练两轮 `O(TNK)` 几何反馈候选选择器与失败置信度。官方 training.zip 用于 group-isolated
train/dev；OS 424 因已参与方法决策仅保留为 secondary legacy benchmark。E-003A 通过后仍须
用户批准，才可使用冻结源码/配置/checkpoint 启动绑定外部未见光学-SAR数据的 E-003B。

## 2026-08-05 / D-010

E-003A 在 G0 和两轮审阅均通过后，三个 main seed 的 dev grid error 稳定为
`11.10/11.70/11.57 px`；主结果 AEPE `10.6808 px`、PairLocalReady `67.08%`、coverage
`0.74%`，G1 FAIL。corrected top8 oracle 仍为 AEPE `2.8410 px`、PairLocalReady `100%`。
因此停止 v0.6 的独立候选 MLP、dustbin 联合九分类和单一仿射反馈组合；不否定冻结候选前端或
learned selector 方法族。下一候选为 recoverability 分解加稀疏跨-anchor 仿射一致性图，但在
用户批准前不建立 v0.7、实验单或派工。

## 2026-08-05 / D-011

用户批准将 source anchor acquisition 前置：每个区域保底、易匹配区域动态增配，并为低质量点
保留候补替换。建立 M001 v0.7 / E-004A，只验证 M256 proposal、覆盖约束 K128 和 R64 单轮
replacement 的可行性；先做 GT oracle headroom gate，再训练无 GT pre/post recoverability heads。
本实验不加入 RG-SAC 或新 candidate selector，完成后必须等待用户。

## 2026-08-05 / D-012

E-004A 的 `uniform-region-K128` 唯一绑定为每个区域固定选择 offset `(2,2)` 与 `(6,6)`，以保持
区域内二维最大分离。该对照不得改用另一对角、proposal_id 前两个或按结果选择。

## 2026-08-06 / D-013

E-004A 的 P0 oracle PASS，但三 seed 的正式 G1 均失败。区域动态分配相对 uniform 稳定提高约
`9.68 pp`，oracle affine AEPE `2.4522 px` 通过；pre-head 仅达到 `41.80%` recoverable，post-head
replacement 仅提高 `0.0522 pp`。因此停止当前手工统计量 pre/post heads 和阈值式同区域替换，
不否定 M256 proposal 与覆盖约束。候选下一步为 M256 全 fine-probe 后的 setwise anchor ranking，
但用户批准前不建立新版本或实验。

## 2026-08-06 / D-014

用户批准 M001 v0.8 / E-005A，以“稀疏跨模态对应观测和低维仿射状态之间的递归反馈”替代
GDROS dense flow state、dense GRU 和完整 `4096x4096` correlation。主结构固定为 M256 exact
top8、稀疏 local correlation、16 global consensus tokens、四轮 shared recurrent update、每轮
coverage dynamic K128 和 normalized FP32 weighted affine。实现借鉴 DPVO/IHN/CoTracker/SGMNet，
但禁止其不符合预算的 dense correlation、BA/SE(3)、full Sinkhorn 等模块。E-005A 同时设精度和
资源 gate；完成后无论成败均等待用户，不自动创建 E-005B 或救援版本。

## 2026-08-06 / D-015

E-005A 的 P0、G0、两轮 reviewer 和双 seed 训练均有效，但 G1 稳定失败：main/review AEPE
`4.6343/4.5593 px`，远差于 dense GDROS dev `1.4597 px`。递归使 candidate accuracy 提升到约
`90%`，且去除 global tokens、affine feedback、local correlation 或 dynamic active set 都明显退化，
说明核心模块确实工作；但 final active K128 平均只有 `78.77` 个 recoverable，ESS 仍为 `86.70`，
错误 anchors 没有被充分拒绝。M256-all 与 hard K128 AEPE 几乎相同，K 截断不是主因。停止 v0.8
当前 candidate/dustbin objective、soft target expectation、hard active K128 与 ordinary weighted LS
组合；不外推否定所有 sparse recurrent affine consensus。任何 oracle 分解或语义不同的新后端均须
用户另行批准，G2、E-005B 和 rescue 不自动运行。

## 2026-08-06 / D-016

用户批准一次只读、无训练的 E-005A-INLIER-DIAG。该诊断分别测量 GDROS predicted flow 相对 GT
endpoint 的 point inlier ratio、相对自身最终 affine 的 self-consensus ratio，并以统一 pixel 阈值
与 E-005A final coordinates/weights 比较；历史 `recoverable<=16 px` 不得冒充最终内点率。同时
以冻结 M256 pool 生成 K x controlled-rho 二维 AEPE/CMR 曲线，并完成 final active K128 的
coordinate/weight oracle decomposition。该批准不恢复 E-005A G2，不授权训练或 E-005B。

## 2026-08-06 / D-017

E-005A-INLIER-DIAG 在 404 dev、11 tests 和两轮 APPROVE 下完成。GDROS matched-M256 的 2px
GT/self inlier 为 `72.40/96.75%`，E-005A active K128 raw 为 `14.70/47.41%`、weighted 为
`22.20/73.33%`，证明当前 reliability 主要强化错误 affine basin 的内部自洽。K x rho 曲线只有
`rho=100%` 通过 dense AEPE/CMR 双门槛，ordinary LS 下增加 K 不能抵消少量高影响 outlier。
GT coordinates + predicted weights 在 404/404 上近乎精确；predicted coordinates + GT binary weights
只在 156/404 有足够空间内点，但条件子集精度恢复。因此下一方法若获批准，必须先提升多候选到
GT coordinate 的恢复并避免单 H/soft expectation 过早锁定，其次加入显式 robust/hard rejection；
不得只增加 K、T 或继续 ordinary weighted LS。

## 2026-08-06 / D-018

用户批准围绕中心创新继续实验：用稀疏跨模态对应观测和低维仿射状态之间的递归反馈替代 GDROS
dense flow/dense GRU，同时保留全局纠偏能力。建立 M001 v0.9 / E-006A：区域结构匹配先初始化
B=8 个 affine basins，再以 global top8、region top8 和 affine-guided local candidates 递归执行
置信点增长；proposal 在 active/unresolved/rejected 间迁移，最终 active 点数不固定 K128 且不设
人工 Kmax；通过 leave-one-region-out validation、region-balanced robust solve 和 hard rejection
防止错误 basin 自证。P0 capacity/budget 通过后才训练；G1/G2 任一失败即写结果卡并停止，不自动
建立 E-006B、descriptor、扩大 proposal grid 或救援实验。

## 2026-08-06 / D-019

E-006A 按冻结 gate 完成并停止，P0 FAIL，未启动训练/G1/G2。corrected formal gate 和两路 reviewer
有效；旧 `p0-gate.json` 无效且不解释。P0 oracle AEPE `0.190768 px`、PairLocalReady/valid `100%`
均通过，但 mean 2px-correct proposals 仅 `66.210/256`，P10/P50/P90 为 `33/64/98`，未达到冻结
`140/256`。这说明 E-006A 候选空间中正确点分布足以 oracle 拟合极准 affine，但达不到“接近 GDROS
稠密点内点率”的正样本密度假设。因此停止冻结 E-006A 的 candidate-regeneration coverage 版本；
不否定稀疏观测和低维仿射反馈主张。下一步若用户批准，应把问题改写为从约 30-100 个正确候选
中学习高纯度 hard inlier selection，而不是继续要求暴露 >=140 个 2px 正确 proposal。

## 2026-08-06 / D-020

用户指出“从候选里挑点很难”，并批准先尝试验证该判断。建立只读/轻训练诊断
`E-006A-SELECT-DIAG`：完全复用 E-006A corrected candidate generator，不扩大 proposal grid、不训练
GDROS descriptor/backbone/Transformer、不启动 E-006B。诊断分 D1 hand-crafted no-GT selectors 和
D2 小型 no-GT MLP/logistic selector，检查候选正确性 AUROC/AP/ECE、topK precision/recall 和 affine
AEPE/CMR。如果 D2 仍无法从现有特征中选出高纯度 inliers，则下一步应转向更强 observation/patch
evidence，而不是继续几何 selector。

## 2026-08-06 / D-021

绑定 E-006A-SELECT-DIAG 中 `PARTIALLY_SELECTABLE` 的“明显优于”口径：唯一比较 track 为 dev
`full-main global topK-32`；D1 是该 track 上 affine AEPE 最低的 hand-crafted selector；E-006A
no-GT score 代理固定为 `D1 cosine/topK-32`。D2 必须相对最佳 D1 同时满足 candidate AUROC 或 AP
至少一项提升 `>=0.03`，且 affine AEPE 降低 `>=0.25 px`，否则不得判为 `PARTIALLY_SELECTABLE`。
若 D2 达到 `SELECTABLE`，优先判为 `SELECTABLE`。

## 2026-08-06 / D-022

绑定 E-006A-SELECT-DIAG 的标签语义：endpoint `<=1/2/5 px` 标签只按离散 candidate 到 raw-flow
endpoint 的距离计算，不与 in-bounds 做 conjunction；in-bounds 作为独立 secondary label 和分组报告。
geometry condition proxy 绑定为独立第 26 维 feature，不得把 determinant 或 condition 信息混入
basin_score。

## 2026-08-07 / D-023

E-006A-SELECT-DIAG 完成并停止，判读为 `NOT_SELECTABLE_WITH_CURRENT_FEATURES`。19/19 tests PASS，
feature/evaluator review 和两次增量复核均 `APPROVE`。main D2 dev AUROC/AP 为 `0.9171/0.01386`，
global topK-32 precision/recall 仅 `2.259/7.778%`，affine AEPE `63.516 px`、CMR@2 `0%`；
相对 best D1 候选指标提升 `0.0794`，但 affine AEPE 反而差 `1.883 px`，因此
`PARTIALLY_SELECTABLE` 也不成立。D0 hard 2px inliers 在有效 pairs 上 AEPE `0.653 px`，证明
solver 可用，瓶颈是极稀疏候选在当前 26 维无 GT 特征下无法高纯选择。停止当前 E-006A 候选空间
上的几何 selector 路线；下一步若用户批准，应转向更强 observation/patch evidence 或
descriptor-level 改造，而不是继续在同一候选特征上调 selector。

## 2026-08-07 / D-024

用户批准同时尝试三类更强观测证据：Patch evidence、Structure evidence 和 Descriptor evidence。
建立 M001 v1.0 / E-007A：第一版仍复用 E-006A corrected candidates 和 M256 proposals，不扩大候选
空间；对每个候选增加 raw/GDROS feature patch、跨模态结构通道和轻量 descriptor adapter 分数。
GDROS backbone/FeatureTransformer 保持冻结，只允许训练 patch/structure verifier 与 local descriptor
adapter。主对照为 E-006A-SELECT-DIAG D2 topK-32 precision `2.259%`、AEPE `63.516 px`、CMR@2
`0%`。E-007A 只判断更强 observation 是否带来 useful/strong gain，完成后等待用户；不得自动启动
E-007B、E-006B、proposal expansion、OS424/confirm 或 dense fallback。

## 2026-08-07 / D-025

E-007A 完成并停止，判读为 `NO_USEFUL_OBSERVATION_GAIN`。14/14 tests PASS，三名定向 reviewer
均 `APPROVE`。冻结 E-006A baseline parity PASS：topK-32 precision `2.2587%`、AEPE `63.5157 px`、
CMR@2 `0%`。all-evidence 主 track topK-32 precision `1.7868%`，AEPE `36.0761 px`，CMR@2 `0%`；
AEPE 虽下降 `27.4396 px`，但 precision 与 AP 均低于 baseline，且 CMR@2 仍为 0。raw patch、
structure、descriptor-only 及 fusion 均未达到 weak/useful/strong gate。停止固定 E-006A candidate
set 上继续堆 patch/structure/descriptor verifier 的路线；该结果不否定改变 candidate generator、
更强跨模态 backbone/descriptor 或不同 observation space。

## 2026-08-07 / D-026

冻结当前候选生成基准为 `E-006A corrected candidate generator`，记录于
`research/CANDIDATE_GENERATION_BASELINE.md`。当前候选生成方案为冻结 GDROS `64x64` 特征、M256
source proposals、global top8、region top8 和 B8 affine-local `9x9` candidates。核心基准为：
oracle AEPE `0.190768 px`、mean 2px-correct proposals `66.210/256`、P10/P50/P90 `33/64/98`、
candidate 2px base rate `0.1343%`、D2 P@32 `2.259%`、D2 AEPE `63.516 px`、E-007A all-evidence
best AEPE `36.0761 px` 且 CMR@2 `0%`。后续 candidate generator 必须让正确候选更多且更可分：
至少超过 `66.210/256`、P@32 `2.259%` 和 AEPE `36.0761 px`，建议目标为 mean correct `>=100/256`、
P@32 `>=10%`、AP `>=0.05`、oracle AEPE `<=0.5 px`。

## 2026-08-07 / D-027

用户批准按新的候选生成设想执行，并与旧方案比较候选生成结果。建立 M001 v1.1 / E-008A：
Region-to-Point Cross-modal Candidate Generator。新方法必须改变 candidate generator，而不是继续在
E-006A fixed candidates 上堆 verifier。主结构为多尺度 region matching、动态 source proposal
generation、matched target regions 内 dense-lite descriptor/patch search，并保存候选来源 trace。
P0 比较候选覆盖：mean correct、P10/P50/P90、2px base rate、oracle AEPE、PLR/valid；P1 比较
可选择性：P@32、AP、no-GT topK-32 affine AEPE、CMR@2。完成后只写结果卡，不自动启动 E-008B、
完整 sparse affine feedback、OS424/confirm 或救援版本。

## 2026-08-07 / D-028

E-008A 按冻结协议完成并停止于 P0 FAIL，P1 未运行。13 tests PASS，candidate/evaluator/protocol/
result-interpretation reviewers 均 `APPROVE`。相对 E-006A baseline，mean 2px-correct proposals
从 `66.210` 降到 `17.022`，P10/P50 从 `33/64` 降到 `7/15`，2px base rate 从 `0.1343%` 降到
`0.09828%`，hard-2px oracle AEPE `189.229 px`，PLR/valid `63.119%`；仅 no-N2/no-blind-search
通过。动态生成器产生更多 source proposals 和 candidates，但最终正确覆盖更差。因 P0 未记录
stage-wise GT recall，不能定位损失发生在 region selection、within-region top64 retrieval 或 final
top4 ranking/dedup。停止 E-008A region-to-point v1；下一步若用户批准，应先做 stage-wise recall
诊断或重新设计候选生成器，不得直接 rescue E-008A。

## 2026-08-07 / D-029

用户批准执行 `E-008A-STAGE-DIAG`。该实验只诊断 E-008A 正确候选在哪个候选生成阶段丢失，不改
方法、不训练、不启动 E-008B/rescue。诊断漏斗固定为：S0 source proposal，S1 多尺度 region top-L
containment，S2 matched-region 内 top64 pool，S3 region/descriptor/patch/full 各 top4 分支，S4
pre-dedup union，S5 post-dedup final candidates。final stage 必须复现 E-008A P0 的 mean correct
`17.022`、P10/P50 `7/15` 和 2px base rate `0.09828%`，否则判为 provenance/parity blocked。
诊断完成后由方法任务决定是重设计 region matching、top64 pool、top4 ranking/dedup，还是停止该方向。

## 2026-08-07 / D-030

E-008A-STAGE-DIAG 完成并停止，判读 `COMPLETED / MIXED_LOSS_DIAGNOSED`。final P0 parity、GT
隔离、no-N2/no-blind-search、review R2 均 PASS/APPROVE。漏斗显示 S0/S1/S2/S4/S5 的 mean
2px-correct proposals 为 `239.351/196.686/50.015/17.022/17.022`。region containment 保留
`82.17%`，不是主要瓶颈；top64 预筛保留 `25.43%`，是最大绝对损失；top4 union 相对 S2 仅保留
`34.03%`，条件损失与 top64 只差 `8.61 pp`，所以不能单点归因。dedup 不丢 2px proposal。
S3 retention 为 region `17.16%`、descriptor `12.33%`、patch/structure `6.62%`、diagnostic
full-score `9.89%`，说明第一版 learned descriptor/patch 没有提供强于简单 region score 的局部
候选生成证据。下一步若用户批准，应设计同时解决 within-region pool 与 top4 ranking 的新候选生成器，
而不是 rescue E-008A v1。

## 2026-08-07 / D-031

用户批准直接改进 frozen cosine / region score 预筛。建立 `M001 v1.2 / E-009A`：
Learned Region-Conditioned Top64 Retriever。固定 E-008A 的 source proposals 和 matched-region
union，只替换 S2 top64 ranker；允许训练新的双模态 stride-8 descriptor/retrieval module，使 learned
local evidence 在 top64 之前参与检索。learned retriever 不读取 GDROS frozen features，frozen
features 只用于旧 S2 baseline parity。主对照为 E-008A-STAGE-DIAG frozen S2：mean correct
`50.015`、P10/P50 `26/48`、base rate `0.05161%`；候选生成资格再对比 E-006A baseline：
mean correct `66.210`、P10/P50 `33/64`、oracle AEPE `0.190768 px`。完成后只写结果卡，不自动
启动 E-009B、E-008B、完整 sparse affine feedback、OS424/confirm 或 rescue。

## 2026-08-07 / D-032

E-009A 完成并停止，判读 `COMPLETED / LEARNED_TOP64_FAIL`。Reviewer R1 为
`APPROVE_WITH_RESULT_FAIL`，未发现实现偏移：learned retriever 只使用 raw/structure tail，确实在
top64 前排序，不是 frozen top64 后 verifier；candidate outputs 在 GT label 前物化，无 hidden N2
或 full-image blind search。G1 全部关键项 FAIL：learned top64 mean/P10/P50/P90 为
`34.745/15/33/55.7`，低于 frozen S2 `50.015/26/48/78`；2px base rate `0.03585%` 低于
`0.05161%`，nearest P50/P90 `27.634/102.911 px` 差于 `15.490/75.309 px`。top128 诊断
mean `55.394`、P50 `53.5` 超过 frozen top64，但 base rate 降到 `0.02858%`，oracle AEPE
`20.625 px`、PLR/valid `96.04%`，不能作为候选生成成功证据。停止 E-009A v1.2；若继续，
必须改变训练目标/负样本挖掘或底层跨模态表示，不得只把 `R=128` 作为 `R=64` 失败后的追认。

## 2026-08-07 / D-033

用户同意建立 `M001 v1.3 / E-010A`：Strong Cross-modal Region-local Retriever。明确 frozen S2
只是很差的最低对照，不是学习目标；E-010A 不是 E-009A rescue，不做门控/调参/追认 top128。
实验继续固定 E-008A S0 source proposals 与 S1 matched-region union 以隔离区域内检索能力，但更换
底层表示和训练目标：modality-specific raw/structure/patch encoder、stride-8 descriptor、full-union
listwise retrieval、hard negative mining。GDROS frozen features 不得进入 learned retriever，也不得
作为 teacher，只用于 frozen S2 parity。G1 要求显著超过 frozen S2：top64 mean `>=60.0`、P10/P50
`>=31/56`、base2px `>=0.0619%` 且 nearest P50/P90 优于 frozen；G2 再判断是否超过 E-006A
candidate baseline。完成后只写结果卡，不自动启动 E-010B、E-009B、完整 sparse affine feedback、
OS424/confirm 或 rescue。

## 2026-08-07 / D-034

E-010A 完成并停止，判读 `COMPLETED / STRONG_RETRIEVER_FAIL`。Reviewer R1 为
`APPROVE_WITH_RESULT_FAIL`，未发现实现偏移：score 在 S1 union 内、top64 前排序；full-union
Gaussian listwise loss 与 hard-negative auxiliary 已执行；不读取 GDROS frozen prefix，不使用 frozen
score teacher；无 hidden N2 或 full-image blind search。G1 全部关键项 FAIL：strong top64
mean/P10/P50/P90 为 `37.077/16/36/60`，低于 frozen S2 `50.015/26/48/78`；base2px
`0.03826%` 低于 `0.05161%`；nearest P50/P90 `25.331/99.603 px` 差于 `15.490/75.309 px`；
oracle AEPE/PLR 为 `48.541 px/90.59%`。top128 mean `58.131`、P50 `55` 略强于 E-009A top128，
但 base2px `0.02999%`、oracle AEPE `23.138 px`、PLR `95.54%`，不能追认 top64 失败。
解释边界：E-010A 仍使用 E-008A cached raw/structure/local tail，而不是从 512 原图重新提取更丰富
patch；因此只否定 cached-evidence 强检索器，不否定真正的 raw-image patch/descriptor backbone。

## 2026-08-10 / D-035

用户批准按方法侧建议执行 `M001 v1.4 / E-011A`：Raw-image Cross-modal Patch Descriptor Retriever。
核心变化是真正改变 observation source：不再使用 E-008A cached raw/structure/local tail，而是从
official training archive 的原始 `512x512` optical/SAR 图像重新提取多尺度 patch descriptor，在固定
E-008A S0 source proposals 与 S1 matched-region union 内进行 full-union retrieval。特别绑定：
不得污染 GDROS baseline，GDROS 源码、checkpoint、baseline cache、baseline run/results 全部只读；
E-011A 的代码、cache、checkpoint、日志、结果全部写入独立 `e011a` / `E-011A` 路径，不得训练或微调
GDROS backbone/FeatureTransformer。G1 仍要求显著超过 frozen S2：top64 mean `>=60.0`、P10/P50
`>=31/56`、base2px `>=0.0619%` 且 nearest P50/P90 优于 frozen；G2 再判断是否超过 E-006A
candidate baseline。完成后只写结果卡，不自动启动 E-011B、完整 sparse affine feedback、OS424/confirm
或 rescue。

## 2026-08-10 / D-036

E-011A 完成并停止，判读 `COMPLETED / RAW_PATCH_RETRIEVER_FAIL`。G0、baseline isolation、review
均 PASS，确认 GDROS baseline 未污染；retriever 从 512 原图/SAR 提取 descriptor，未使用 E-008A
cached tail 主输入。G1 top64 全部关键项 FAIL：mean/P10/P50/P90 为 `45.879/20/44/73.7`，低于
frozen S2 `50.015/26/48/78`；base2px `0.04734%` 低于 `0.05161%`；nearest P50/P90
`19.460/82.972 px` 差于 `15.490/75.309 px`；oracle AEPE/PLR `24.464 px/95.30%`。但 top128
诊断出现明确正信号：mean/P10/P50/P90 为 `68.710/34/66/106`，超过 E-006A 的 mean/P10/P50
数量线，nearest P50/P90 `13.129/70.895 px` 优于 frozen S2；不过 oracle AEPE/PLR 仍为
`14.249 px/97.28%`，不足以追认。解释：raw-image patch evidence 确实带来后排可恢复信息，
但 top64 排序、空间分布或几何质量不足。下一步若用户批准，应先诊断 top128 正确点为何不能
oracle 拟合好，或设计 geometry-aware compression，而不是继续简单调 topR。

## 2026-08-10 / D-037

用户批准先做 `E-011A-TOP128-GEOM-DIAG`，不直接设计 E-011B。该诊断只读 E-011A formal dev404
top128 outputs，复现 top128 mean `68.710`、P10/P50/P90 `34/66/106` 和 oracle AEPE `14.249`，
分析正确点数量已上来但仿射仍差的原因。诊断包括 correct count/rank 分布、source 四象限/8x8
region 覆盖、condition、target duplicate、all-hard2px LS、rank64 LS、spatial-balanced K64
GT oracle、condition-filtered GT oracle、GT-endpoint affine oracle，并按 COUNT/RANK、GEOMETRY
DISTRIBUTION、CANDIDATE NOISE、SOURCE SET、COMPRESSION ORACLE READY 等原因分桶。完成后只写
结果卡，不自动启动 E-011B、geometry-aware compression 或 rescue。

## 2026-08-10 / D-038

`E-011A-TOP128-GEOM-DIAG` 完成并停止，判读 `COMPLETED / COMPRESSION_ORACLE_READY`。parity、
GT/oracle 隔离、只读隔离与 reviewer 均 PASS。top128 mean/P10/P50/P90 复现为
`68.710/34/66/106`，E-011A top128 oracle AEPE 复现为 `14.249 px`。分桶显示
`COMPRESSION_ORACLE_READY` 为 `401/404`，`RANKING_TAIL_LIMITED` 为 `77/404`，
`CANDIDATE_NOISE_LIMITED` 为 `47/404`，`NON_AFFINE_OR_SOURCE_SET_LIMITED` 仅 `3/404`。
rank 分布显示 top64 只覆盖 top128 正确 proposals 的 `66.77%`，top96 覆盖 `84.39%`。在 GT-guided
选择下，all-hard2px LS 的 P50/P90/P95 AEPE 为 `0.299/0.533/0.617 px`，GT-endpoint oracle
P50/P90/P95 约 `0.00004/0.00006/0.00007 px`。方法解释：raw-image patch top128 evidence
并未死亡，source set 几何基本可用；下一步若用户批准，应优先设计无 GT 的 geometry-aware
top128 compression/reranking，而不是继续简单调 top64 ranker、盲目扩大 proposal grid 或否定
raw patch 路线。本诊断不证明已有 no-GT 压缩器可用，也不解除 E-011A top64 失败结论。

## 2026-08-10 / D-039

用户批准启动 `M001 v1.5 / E-012A Geometry-aware Top128 Dynamic Compression`。方法绑定：
`top128` 是每个 source proposal 的 target candidate depth，不是最终点数上限；最终 active set
动态决定，每个 source 最多一个 target，不可靠 source 进入 dustbin/pending/rejected。E-012A
复用 E-011A raw-image top128 candidates 作为只读输入，不重训 E-011A descriptor，不修改 GDROS
baseline。新增模块维护多个 affine basin slots，执行 assignment、FP32 weighted affine solve、
residual feedback、coverage/condition/duplicate 抑制和 active set 更新，目标是在无 GT inference
下把 top128 evidence 压缩成高纯稀疏观测集。G1 要求 active correct count 与 precision 显著超过
frozen S2/E-011A top64，G2 要求 no-GT affine AEPE `<=2.0 px`、CMR@2 `>=60%`、CMR@5 `>=90%`。
禁止固定最终 K128/K64、三点 RANSAC、131,072 hypotheses、dense flow/GRU、hidden `4096x4096`、
OS424/confirm/datum 泄漏、baseline fallback 或自动启动 E-012B/完整 sparse affine feedback。

## 2026-08-10 / D-040

E-012A 完成并停止，判读 `COMPLETED / DYNAMIC_COMPRESSION_FAIL`。G0 PASS，reviewer 为
`APPROVE_WITH_RESULT_FAIL`；E-011A top128 parity、candidate-before-label、one-source-one-target、
dynamic active size、no GT inference、no hidden N2/no RANSAC/131k 和 unit tests 均通过。G1 中
active 数量与覆盖通过：active mean/P10/P50/P90 为 `327.723/218.6/294/497.7`，coverage pass
`100%`；但纯度灾难性失败：correct@2px mean/P10/P50/P90 仅 `2.861/0/0/7`，precision
mean/P10/P50/P90 仅 `0.673%/0/0/1.827%`。G2 按协议未运行。方法解释：run-01 的 deterministic
no-training geometry compression 会锁到大规模错误但自洽的 basin；在 optical-SAR 候选里，仅靠
几何残差、coverage、condition、duplicate 和 basin 自洽无法区分跨模态错误一致簇。该结果不否定
E-011A top128 中存在 GT-guided 可压缩 inliers，也不否定 learned/appearance-conditioned inlier
机制；但停止 deterministic no-training geometry-only compression rescue。

## 2026-08-10 / D-041

用户批准启动 `M001 v1.6 / E-013A Geometry-supervised Learned Inlier-Basin Scorer`。方法绑定：
E-012A 失败说明 geometry-only 自洽会被错误 optical-SAR basin 欺骗；E-013A 将几何改为训练监督和
推理辅助特征。训练时允许用 raw truth_flow 构造 endpoint hard/soft labels、oracle affine/basin labels
和 affine EPE loss；正式 inference、阈值选择、basin 初始化和 candidate compression 不得读取 GT、
datum、OS424/confirm 或 oracle labels。E-013A 复用 E-011A top128 candidates 作为只读输入，不重训
E-011A descriptor，不修改 GDROS baseline。G1 要求 learned purification 显著超过 E-012A：
precision mean/P10/P50 `>=20/5/15%`，correct@2px mean/P10/P50 `>=24/8/20`，active count
mean/P10 `>=32/12`，coverage `>=90%`；G2 再要求 no-GT affine AEPE `<=2.0 px`、CMR@2
`>=60%`、CMR@5 `>=90%`。禁止 deterministic E-012A rule fallback、固定 K64/K128、RANSAC、
131,072 hypotheses、dense flow/GRU、hidden `4096x4096`、baseline fallback 或自动启动 E-013B。

## 2026-08-10 / D-042

E-013A 完成并停止，判读 `COMPLETED / LEARNED_PURIFICATION_FAIL`。G0 PASS，reviewer 为
`APPROVE_WITH_RESULT_FAIL`；E-011A top128 parity、E-012A baseline parity、label construction audit、
candidate-before-label、no GT inference、loss terms nonzero、gradient to scorer/basin heads、
feedback_changed、one-source-one-target、dynamic active size、no hidden N2/no RANSAC/131k 均通过。
训练链路参数：`1607` train pairs、`1` epoch、`10,630` params、train-calibrated threshold
`0.9699785920`，endpoint/basin/affine/precision loss 为 `0.0961/0.0966/0.7284/0.1506`。G1 FAIL：
active mean/P10/P50/P90 为 `0.658/0/0/2`，correct@2px 为 `0.017/0/0/0`，precision 为
`1.246%/0/0/0`，coverage pass `0%`；G2 未运行。方法解释：E-013A 避免了 E-012A 的大规模错误
自洽 basin，但退化为 train-calibrated high-threshold 下的 dev over-abstention/generalization failure。
该结果说明当前轻量 learned candidate/basin scorer 的训练/推理链路干净但不够可用；下一步若继续，
需先诊断阈值校准、类别极不平衡、特征/模型容量或训练轮数，而不是直接启动 E-013B rescue。

## 2026-08-10 / D-043

用户批准先做 `E-013A-CALIB-DIAG`，不直接启动 E-013B。该诊断只读 E-013A checkpoint、outputs、
logits、labels 和 train/dev split，复现 E-013A G1 over-abstention 后，进行 threshold-free PR/ROC、
score quantile、threshold sweep、topK/quota sweep、train-vs-dev calibration、endpoint/basin/final score
ablation 和 per-pair positive base-rate 分析。判读类别固定为 `CALIBRATION_ONLY`、
`RANKING_SIGNAL_PRESENT`、`TRAIN_DEV_SHIFT`、`NO_USABLE_RANKING_SIGNAL` 或
`LABEL_OR_BASE_RATE_LIMITED`。禁止训练、finetune、改 checkpoint、用 dev threshold 追认正式成功、
启动 E-013B/rescue/descriptor retraining/完整 sparse affine feedback/OS424/confirm。

## 2026-08-10 / D-044

E-013A-CALIB-DIAG 完成并停止，判读 `COMPLETED / NO_USABLE_RANKING_SIGNAL`。P0 parity/provenance
PASS，reviewer 为 `APPROVE_WITH_RESULT_FAIL`；未训练、未 finetune、未改 checkpoint/formal results，
未启动 E-013B/rescue。final score train/dev AUROC `0.5036/0.5024`，AP `0.000477/0.000471`，
近随机。best dev threshold sweep 仍只有 active `615.20`、correct `2.01`、precision `0.298%`；
best dev topK/quota 为 global topK-256，active `235.91`、correct `0.98`、precision `0.395%`，
远低于 E-013A G1。train best 也失败，说明不是主要 train-dev shift；dev positive-proposal P10 为
`34` 且仅 `3/404` zero-positive pairs，说明不是 proposal-level label/base-rate 不可达。唯一正信号是
pre-feedback endpoint score：dev AUROC/AP `0.7019/0.00256`；post-feedback/final score 近随机，
basin-only dev AUROC `0.0587`，反向/无用。方法解释：当前 final feedback/scoring path 毁掉或忽略
弱 endpoint rank signal；不能靠降低阈值或简单 calibration 救 E-013A，若继续必须重新设计几何反馈
接入方式或加强 endpoint/appearance evidence。

## 2026-08-10 / D-045

用户批准启动 `M001 v1.7 diagnostic / E-014A Cross-modal Signal Ranking Diagnostic`。本实验不训练、
不 finetune、不接几何 feedback，只在 E-011A top128 candidates 上比较 raw patch、log-SAR/contrast
normalized patch、gradient/edge、phase-congruency/RIFT-style、census/rank/self-similarity、structure
tensor/cornerness、E-011A raw score 和 E-013A pre-feedback endpoint/final score 等信号对 endpoint
correctness 的排序能力。为避免实现误否认方法，P0 必须先通过 E-011A top128 parity、candidate-before-label
join、source->target 坐标方向、patch center/radius/padding、identity/known-shift/gradient-orientation/
phase-congruency/structure-tensor synthetic sanity、shuffled/swapped/random negative controls，以及
feature implementation reviewer 和 evaluator/protocol reviewer 两路审阅。P0 或 reviewer 未通过时，
结果只能判 `IMPLEMENTATION_BLOCKED`，不得解释负面信号排序。禁止启动 E-014B、训练、descriptor
retraining、完整 sparse affine feedback、OS424/confirm 或 rescue。

## 2026-08-11 / D-046

E-014A 完成并停止，判读 `COMPLETED / NO_SIGNAL_FOUND`。P0 PASS，feature implementation reviewer
与 evaluator/protocol reviewer 均 `APPROVE`；E-011A top128 parity、candidate-before-label、
optical source -> SAR target 坐标方向、stride-8 token/pixel 转换、patch center/radius/padding、
same-image identity、known shift、gradient/RIFT/structure tensor toys、shuffled target、swapped direction
和 random score controls 均通过。best non-reference signal 为
`census_rank_self_similarity/local_self_similarity`，dev AUROC/AP `0.531545/0.000443655`，best
region_quota=8 correct mean `0.212871`、precision `0.000866667`。reference 中 E-013A pre-feedback
endpoint score 仍最好，AUROC/AP `0.776095/0.00274588`；E-011A raw score AP `0.00125910`，
random AP `0.000353054`。方法解释：测试的 raw patch、log-SAR/normalized patch、gradient/edge、
RIFT/HOPC-style、census/rank/self-similarity、structure tensor/cornerness、E-011A raw score 和
E-013A final score tracks 均不能提供可用 endpoint-correctness ranking signal；因 P0 和两路 reviewer
通过，本次不归因为实现/协议阻塞。下一步不应继续在现有 E-011A top128 上堆类似手工 signal scorer；
若继续需要改变 observation source 或学习方式。

## 2026-08-11 / D-047

用户批准继续 SparseGeo-GDROS 主线并启动 `M001 v1.8 diagnostic / E-015A Strong Cross-modal Feature
Stack Diagnostic`。方法定位：E-015A 已退到网络最前端 observation/descriptor 阶段，前面只剩图像
读取、预处理、尺度/patch 提取和特征编码，没有更早的匹配阶段。文献启发包括 PromptMID 的
diffusion/VFM/text prompt modal-invariant descriptors、MatchAnything 的大规模跨模态预训练 matcher、
XoFTR 的 detector-free cross-modal transformer，以及 SOMA 的梯度增强深特征/DINOv2 coarse alignment。
E-015A 允许不计成本尝试 DINOv2、OpenCLIP/RemoteCLIP、SAM/SAM2/MAE、MatchAnything、XoFTR、
RoMa/DKM/LoFTR-style、PromptMID/SOMA official code/checkpoint（若可用）等强特征或 matcher，
评估 E-011A top128 reranking 与 S1/full/直接 matcher candidate generation。高成本计算必须标为
diagnostic/resource-unqualified，不得宣称轻量化。禁止自动启动 E-015B、训练 descriptor、完整 sparse
affine feedback、OS424/confirm 或 rescue。

## 2026-08-11 / D-048

E-015A 完成并停止，判读 `COMPLETED / MATCHER_CANDIDATE_FOUND`。P0 PASS，feature/model reviewer
与 evaluator/protocol reviewer 均 `APPROVE`，watchdog succeeded。top128 reranking 仍未超过
E-013A pre-feedback：best non-reference `fusion/zscore_available_pretrained` 的 AUROC/AP 为
`0.801631/0.00166453`，best global topK-256 correct mean `0.9901`、precision `0.4030%`；
E-013A pre-feedback AP 为 `0.00274588`、best correct mean `1.2203`。但 expanded/full4096
candidate generation 找到强候选来源：`foundation_dense/dinov2_vitl14_local/full4096_top128`
的 correct@2px proposals mean/P10/P50/P90 为 `115.943/65.6/115/164`，nearest P50/P90 为
`7.283/54.623 px`，PLR/valid `97.525%/97.525%`，明显超过 E-011A top128 与 E-006A quantity line。
该 track 为 full-image blind search、resource-unqualified diagnostic，p50/p95 time `98.8/177.3 ms`，
peak p50/p95 `2.842/2.884 GB`，不能宣称轻量化。oracle affine AEPE 仍 `12.903 px`，所以当前不是
完整配准成功；下一瓶颈转为 DINO full4096 候选的几何感知压缩、空间分布和 outlier 控制。未启动
E-016A、E-015B、descriptor training、完整 sparse affine feedback、OS424/confirm 或 rescue。

## 2026-08-11 / D-049

用户批准启动 `M001 v1.9 diagnostic / E-016A DINO Candidate Geometry / Distribution Diagnostic`。
方法定位：E-015A 已证明 DINOv2-L full4096 retrieval 能显著提高 correct candidate coverage，但
top128 oracle affine AEPE 仍 `12.903 px`。因此 E-016A 不训练、不 finetune、不改变 baseline，也不启动
完整 sparse affine feedback；只诊断 DINO full4096 topK candidates 的 rank depth、source/target
分布、重复/outlier、no-GT score selection 和 GT-guided compression oracle。判读类别冻结为
`DINO_GEOMETRY_COMPRESSIBLE`、`RANK_TAIL_LIMITED`、`SOURCE_TARGET_DIVERSITY_LIMITED`、
`NO_GT_COMPRESSION_LIMITED`、`OBSERVATION_STILL_LIMITED` 或 `IMPLEMENTATION_BLOCKED`。
full4096 继续标记为 resource-unqualified diagnostic，不得宣称轻量化。完成后只回传 RESULT_CARD
和核心 JSON/reviewer/RUN_MANIFEST 路径，等待方法任务决策。

## 2026-08-11 / D-050

E-016A formal dev404 完成并停止，判读 `COMPLETED / RANK_TAIL_LIMITED`。P0 PASS，两路 reviewer
通过，watchdog `20260811_120822_17a24530` succeeded。K128 correct@2 mean/P10/P50/P90 为
`115.943/65.6/115.0/164.0`，nearest P50/P90 `7.283/54.623`；K256 correct@2 mean/P10/P50/P90 为
`152.389/95.3/150.5/207.1`，nearest P50/P90 `4.725/44.973`。K256-K128 correct mean gain
`36.446`，触发 rank-tail bottleneck。Best no-GT selection 仍失败：region_quota:2 AEPE `54.991`、
CMR@2 `0.199%`。GT-guided oracle 非常强：all_hard:K256:thr5 AEPE `0.294`、CMR@2 `99.169%`、
PLR `100%`；region-balanced K256 thr5 B128 AEPE `0.364`、CMR@2 `99.252%`、PLR `100%`。
方法解释：DINO full4096 候选已经在几何上可压缩，K128 太浅，K256 更合适；当前主要瓶颈是
no-GT DINO score/ranking compression 不会挑点。下一步若获用户批准，应以 DINO K256 为 observation
source，训练 geometry-aware compression/ranking，而不是继续依赖 DINO 原始分数或手工 no-GT region
quota。实验任务未启动 E-017A、E-016B、descriptor training、完整 sparse affine feedback、OS424/confirm
或 rescue。

## 2026-08-11 / D-051

用户赞同进入 learned compression，并提出是否需要尝试超过 256 个候选。方法判断：E-016A 只测到
K256，且 K256 相比 K128 correct@2 mean 增加 `36.446`，因此 rank tail 仍有价值；但 K256 GT oracle
已经足够强，不能让 K512 变成新的无边界诊断。正式启动 `M001 v2.0 prototype / E-017A Supervised
DINO Geometry-Aware Compression`：K256 为 main track，K512 为 secondary headroom track，同步报告
两者 learned dev accuracy 和资源。E-017A 冻结 DINOv2-L full4096 retrieval，不微调 DINO/GDROS，
训练 candidate scorer、affine basin assignment 和 residual feedback，从 train truth_flow 学 endpoint
correctness、source-level candidate/dustbin、affine/grid loss 与 coverage/diversity。dev inference 禁止
读取 GT。G1 以 no-GT dev affine accuracy 为准：strong 门槛 AEPE `<=1.8 px`、CMR@2 `>=70%`、
PairLocalReady `>=95%`；weak 门槛 AEPE `<=3.0 px`、CMR@2 `>=50%`、PairLocalReady `>=90%`。
full4096 仍为 resource-unqualified effectiveness prototype，不宣称轻量化。完成后等待方法任务决策，
不得自动启动 E-017B、DINO finetune、lightweight distillation、完整 sparse affine feedback、OS424/confirm
或 rescue。

## 2026-08-11 / D-052

E-017A formal-r4 dev404 完成并停止，判读 `COMPLETED / OUTLIER_CONTAMINATED`。P0 PASS：冻结 SHA、
split `1607/404`、DINO checkpoint SHA、K256/E-016A dev parity、K256/K512 nestedness、
candidate-before-label、no datum/OS424/confirm、coordinate mapping 和 baseline write audit 均通过。
G0 PASS：loss finite/nonzero、梯度到 scorer/basin/residual feedback、toy overfit、strict checkpoint load
和真实 no-GT dev inference 均通过；两路 reviewer 通过。K256 dev AEPE `45.906`，CMR@2 `0.3508%`，
CMR@5 `2.1558%`，PLR `3.47%`，valid `100%`，active mean `115.2`，correct@2 mean `1.047`，
precision@2 mean `0.0070`，AP/AUROC `0.002845/0.878941`。K512 更差：AEPE `49.970`，CMR@2
`0.2097%`，PLR `0.99%`，active mean `104.7`，correct@2 mean `0.562`，AP/AUROC
`0.002247/0.906077`。解释：当前 supervised compression 有弱 ranking/AUROC，但 AP 极低，无法形成
高纯 active set；大量 outlier 进入 LS，K512 未带来可用改善。该结果停止 v2.0 compression head，但
不否定 E-016A 的 DINO candidate GT-guided geometry-compressible 上限。下一步若获用户批准，建议先
做只读 PR/selection diagnostic，判断是否存在可通过 threshold/topM/region quota 挖出的高精度前缀；
若没有，则需换成更强监督目标或 pairwise/listwise inlier set prediction，而非继续调当前 head。

## 2026-08-12 / D-053

用户同意停止 E-017A threshold/PR/K512 修补，进入新的 `M001 v2.1 prototype / E-018A DINO-K256
Affine-State-First Consensus`。方法改变：不再走 “candidate independent score -> active set -> affine
solve”，而是固定 DINO K256 candidates，让 source/region/candidate set 先聚合出 B=8 个 affine state
tokens，直接预测初始 affine，再用 T=4 轮 affine residual、candidate-to-basin assignment、
region-normalized FP32 weighted solve 和 affine token update 递归解释候选。训练使用 raw truth_flow 的
affine/grid、state-conditioned inlier、assignment/dustbin、final active affine 和 coverage/calibration loss；
dev inference 禁止读取 GT。K512 禁止，因 E-017A 已显示 K512 在当前选择器下更差，且 K256 的
GT-guided oracle 上限足够。该实验是结构替换，不是 threshold rescue；不得微调 DINO/GDROS，不得恢复
dense flow/dense GRU、hidden 4096x4096、RANSAC/131k，full4096 仍不宣称轻量。完成后等待方法任务
决策，不自动启动 E-018B、DINO finetune、distillation、完整 sparse affine feedback、OS424/confirm 或 rescue。

## 2026-08-12 / D-054

E-018A formal 完成并停止，判读 `COMPLETED / OUTLIER_CONTAMINATED`。P0 PASS，G0 PASS，两路 reviewer
均 APPROVE，formal watchdog `20260812_155606_948ee169` succeeded。Direct affine token AEPE `29.806`，
CMR@2 `1.75%`，CMR@5 `8.55%`，PLR `25.50%`；final active LS AEPE `29.580`，CMR@2 `1.96%`，
CMR@5 `9.31%`，PLR `28.96%`，active mean `292.1`，precision@2 `0.0114`。hidden 4096x4096 为
false，K512 未使用。解释：affine-state-first primary path 和 residual feedback 已实现并通过审阅，但
direct affine state 本身没有学到正确 basin，final selection 只是略微改善；失败不是 threshold/K512 或
final LS 单点问题，而是当前 DINO-K256 learned consensus 表示/监督仍不足以从大量 outlier 中发现 GT
affine basin。E-016A 的 GT-guided oracle 上限仍然成立，因此不写失败方法总否定，但当前 E-017A/E-018A
两版 learned compression head 应停止连续救援。下一步应先做方法复盘，若继续需要改变监督/表示层级，
例如显式 teacher affine/inlier-set distillation、pairwise/set transformer correspondence grouping、或回到
更强跨模态 matcher 端到端候选，而不是继续调 threshold、K512 或当前 affine-token head。未启动 E-018B、
threshold rescue、DINO finetune、distillation、完整 sparse affine feedback、OS424/confirm 或 rescue。

## 2026-08-12 / D-055

用户批准启动 `M001 v2.2 diagnostic / E-019A Oracle Feature Separability / Geometry Consensus Diagnostic`。
问题定位：DINO-K256 中有足够高质量候选，E-016A GT oracle 很强，但 E-017A/E-018A 的 learned heads
都挑不出正确点。E-019A 不训练模型、不写 checkpoint、不启动 E-019B；它只做上帝视角诊断：所有
no-GT features 先于 truth_flow label join 物化并 hash，然后用 raw truth_flow 构造 endpoint labels、
E-016A oracle inlier-set labels 和 high-value inlier labels。重点比较 DINO rank/score、局部 displacement
consensus、region/neighbor support、global coarse affine/Hough-style support、target collision、coverage
contribution 及其 fusion/probe，对 endpoint@2 和 oracle-inlier 的 AP/AUROC、topM precision、region-quota
selection 和 selected-set affine 的解释力。目标是验证用户假设：高质量点是否有局部到全局的几何共识
特征。若存在强 no-GT geometry proxy，下一步围绕显式 inlier-set / correspondence grouping 设计；若不存在，
则需换更强 observation/matcher，而不是继续训练当前 DINO-K256 head。禁止 OS424/confirm/datum、RANSAC/
三点/131k、GT proxy 泄漏、DINO finetune、distillation、完整 sparse affine feedback 或 rescue。

## 2026-08-13 / D-056

E-019A formal-r10 完成并停止，判读 `COMPLETED / NO_USABLE_PROXY_SIGNAL`。P0、watchdog
`20260812_201427_dd52316e` 与两路 reviewer 均通过；no-GT features 在 truth_flow label join 前物化，
train-only probe 未写 checkpoint，baseline write audit unchanged。结果显示当前 DINO-K256 正确候选虽有
弱排序信号，但不足以形成高纯选择：train-only linear probe endpoint@2 AP/AUROC `0.003938/0.888130`，
geometry_fusion `0.003748/0.851422`，global_consensus `0.002611/0.847179`；region_quota128 最好
correct@2 mean `0.649`、P50 `0`、precision `0.5067%`，selected-set PLR `9.41%`。因此停止继续依赖
DINO score/rank、手工 local/regional/global geometry consensus、collision/coverage proxy 或浅层 fusion
来捞点；不否定 E-016A 的 DINO-K256 GT-guided oracle 上限。

用户批准进入 `M001 v2.3 diagnostic / E-020A High-Capacity Candidate Verifier Learnability Diagnostic`。
E-020A 训练一个新的强候选验证网络，必须同时具备 raw optical/SAR multi-scale patch evidence、DINO
local feature crop、candidate token、within-source K256 competition、source-neighbor graph message passing
和 `B>=8` global/affine basin tokens。训练标签来自 train truth_flow 的 endpoint/source-choice/basin/affine
构造，不称为 teacher distillation；dev inference 禁止读取 GT。实验目的不是直接完成 sparse feedback，而是
预注册区分 `INPUT_NOT_LEARNABLE_TRAIN_FAIL`、`TRAIN_MEMORIZATION_ONLY_DEV_FAIL`、
`LEARNABLE_SIGNAL_FOUND_AFFINE_FAIL`、`USEFUL/STRONG_LEARNABLE_SIGNAL_FOUND`。不得微调 DINO/GDROS、
使用 OS424/confirm/datum、RANSAC/三点/131k 或 hidden new full `4096x4096`，完成后等待方法决策。

## 2026-08-13 / D-057

E-020A formal 完成并停止，判读 `COMPLETED / IMPLEMENTATION_BLOCKED_OR_MODEL_TOO_WEAK`。P0 PASS：
冻结 SHA、split、DINO checkpoint、K256 parity、no OS424/confirm/datum、candidate/evidence before
truth_flow label join、baseline write audit、coordinate/stride/DINO sampling、no RANSAC/131k/hidden new
4096x4096 均通过。强网络主通道存在并被调用：patch encoder、DINO local evidence、within-source K256
competition、source-neighbor graph、B=8 affine/global tokens、dustbin/head；gradients 到达 required modules。
但旧 G0 的 4-pair train-small overfit 阈值失败：held-in endpoint@2 AP `0.039667 < 0.05`，best
top256 correct@2 mean/P50 `17/15`，AEPE `19.957 > 14.8`，因此 train-full/dev404 未运行。该结果不能
解释为“输入不可分”，只能说明旧 G0/监督设计未完成可学习性验证。

用户批准按新要求启动 `M001 v2.4 diagnostic / E-020B Train-Full Candidate Verifier Learnability`。E-020B
保留 E-020A 强网络与隔离边界，但将 1/4/8-pair overfit 降为 sanity，不再作为硬停止；G0 主门改为
1607 train pairs 的 train-full sanity。允许改变 loss/监督为 source-choice listwise/sampled softmax、
hard-negative focal、supervised contrastive/group metric、affine-aware set loss 与 coverage/diversity。
dev404 仅在 train-full G0 通过后 no-GT 运行；阈值只能 train calibration，不得用 dev 后验追认。不得微调
DINO/GDROS、使用 OS424/confirm/datum、RANSAC/三点/131k 或 hidden new full 4096x4096；完成后等待
方法决策，不自动启动 E-020C 或完整 sparse feedback。

## 2026-08-13 / D-058

E-020B formal 完成并停止，判读 `COMPLETED / TRAIN_FULL_SANITY_FAIL`。P0 与两路 reviewer APPROVE；
G0 主门已按新规范在 1607 train pairs 上执行，dev404 因 G0 fail 未运行。train-full endpoint@2 AP/AUROC
`0.006255/0.916803`，best top256 AEPE `35.686 px`，correct@2 mean/P50 `2.163/1`，precision
`1.29%`，PLR `16.05%`。这说明失败不再是 4-pair 小样本 gate 过严；在 hard candidate classification /
ranking / source-choice / contrastive / affine-aware set loss 目标下，正确候选仍被 hard negatives 淹没。

用户指出最终目标是挑点，不必让每个 candidate 拟合 GT；只要正确点比错误点更高概率被选中，并让
selected set 支撑 affine 即可。方法任务接受该修正，批准 `M001 v2.5 prototype / E-021A Probabilistic
Set Selection for Inlier Enrichment`：保留 frozen DINO-K256 与强观测通道，但训练 selection distribution，
优化 soft inlier enrichment、set-level affine/grid utility、coverage/diversity 和 train-only budget calibration。
主指标改为 selected set 的 `P(inlier|selected)` 富集倍率、correct count、coverage 和 selected-set
AEPE/CMR/PLR，而非 candidate AP。不得微调 DINO/GDROS、使用 OS424/confirm/datum、RANSAC/三点/131k
或 hidden new full 4096x4096；完成后等待方法决策，不自动启动 E-021B 或完整 sparse feedback。

## 2026-08-14 / D-059

E-021A formal 完成并停止，判读 `COMPLETED / SET_ENRICHMENT_TRAIN_FAIL`。P0 PASS；G0 failed checks =
`train_full_enrichment_thresholds`；G1 未运行。best track `expected_top256` 的 enrichment@2
`32.2557x` 证明 selected-set 富集信号存在，但 precision@2 `1.3979%`、correct@2 mean/P50
`1.919/1`、AEPE `32.405 px`、PLR `22.34%` 均不足。该结果说明当前特征/表示可以把正确点概率
提高到随机池的几十倍，但仍不足以在 hard negatives 中形成高纯、足量、可拟合 affine 的稀疏点集。

用户要求按输入到输出链路修复，并追问候选区分阶段是否需要换特征。方法任务决定启动
`M001 v2.6 diagnostic / E-022A Candidate Feature Discriminability`：冻结 DINO-K256 candidate pool，
系统比较 DINO rank/margin、raw/log-SAR patch、gradient/census/structure、DINO local patch、learned
patch verifier、neighborhood consistency、region context、pair/source adaptive normalization 和 fusion
tracks。主输出是 feature scorecard、per-source hit@1/4/8、selected precision/correct count、enrichment、
coverage 和 selected-set affine，用于判断哪些特征值得接入下一版 selector。E-022A 不启动完整 sparse
affine feedback，不微调 DINO/GDROS，不使用 OS424/confirm/datum、RANSAC/三点/131k 或 hidden new
4096x4096，完成后等待方法决策。

## 2026-08-15 / D-060

E-022A formal-r2 完成并停止，判读 `COMPLETED / NO_TRAIN_FEATURE_SEPARABILITY`。P0、train-small、
资源恢复、4/4 shards 聚合与 reviewer 均通过，G0 因 `train_full_enrichment_thresholds` 失败，
dev404 按协议未运行。best learned/fusion gate track 为 `learned_patch_verifier_fusion_top256`：
precision@2 mean `1.3201%`、correct@2 mean/P50 `1.805/1`、enrichment@2 `30.28x`、
selected-set AEPE mean/P50 `34.001/26.505 px`、PLR `19.73%`。该结果确认 E-021A 的判断：
在当前 frozen DINO-K256 后验候选空间里，DINO rank/margin、raw/log-SAR patch、structure、
DINO local patch、neighborhood consistency、region context 与 fusion verifier 都不足以把 hard
negatives 中的正确点分离成高纯、足量、可拟合 affine 的点集。

用户指出“挑点”问题本身不正常，每对图的地物/散射/纹理分布不同，应回到主流 coarse-to-fine /
state-conditioned matching 的方向。方法任务据此批准 `M001 v2.7 prototype / E-023A Sparse
Affine-State Recurrent Matcher`：不再使用 fixed DINO-K256 selector 作为主路径，而是先从图像产生
coarse affine/region state，再用当前 state 限制 target local search window，产生稀疏/半稠密跨模态
correspondence observations，并用这些观测递归更新低维 affine state。E-023A 是有效性优先原型，
资源只诊断记录；禁止 dense flow state、dense GRU、RANSAC/三点/131k、hidden new full `4096x4096`、
OS424/confirm/datum、DINO/GDROS finetune 与 baseline 写入。完成后等待用户，不自动启动 E-023B、
加速版或 rescue。

## 2026-08-15 / D-061

E-023A formal 完成并停止，判读 `COMPLETED / NO_STATE_FEEDBACK_SIGNAL`。P0 PASS、G0 PASS，model/state
与 evaluator/protocol reviewer 均 APPROVE，watchdog `20260815_133255_730fbcc3` succeeded。train final
AEPE `5.502 px`，但 dev404 final AEPE `50.578 px`、CMR@2 `0.302%`。因此 E-023A 的递归状态机制
已经实现并能在 train 上改善，但 dev 上没有形成泛化的 state feedback signal；该结果不是 implementation
blocked，也不能直接否定 sparse/semi-dense observation + low-dimensional affine state recurrent feedback
主线。

用户批准只读诊断 `E-023A-DIAG Stage Decomposition`。该诊断不训练、不修改 E-023A checkpoint、不追认
E-023A、不启动 E-023B/rescue。它只分解 failure source：coarse state generation、basin selection、
state-conditioned window coverage、local matcher ranking、oracle local residual/update 和 train/dev
memorization gap。truth_flow 只用于 diagnostic labels、oracle 和 metrics；禁止 OS424/confirm/datum、
baseline 写入、DINO/GDROS finetune 或 dev oracle 后验调参。诊断完成后等待方法决策。

## 2026-08-15 / D-062

E-023A-DIAG formal 完成并停止，判读 `COMPLETED / COARSE_STATE_GENERATION_FAIL`。D0 artifact/replay
sanity PASS，reviewer APPROVE，watchdog `20260815_151208_37ddc243` succeeded。关键数值：dev selected
initial AEPE mean/P50 `50.546/41.432 px`，dev oracle-best initial `23.057/21.194 px`，dev final
AEPE mean/P50 `50.578/41.456 px`，round0 selected window coverage mean/P50 `0.498/0.435`，
local top1/top5 mean `0.018/0.083`，oracle endpoint best-update AEPE mean/P50 `6.206/5.705 px`。
failure buckets：coarse state `52.7%`、basin selection `32.4%`、local ranking `14.9%`。解释：
E-023A 的最早瓶颈是初始 coarse state/稀疏粗匹配，没有进入正确 basin，导致 local window 经常罩不住
真值；但 oracle local observation/update 表明后端 affine update 仍有可恢复空间。

用户批准按“每轮解决一个问题”的思路建立 `M001 v2.8 prototype / E-024A Role-Separated Recurrent
Coarse-to-Fine Affine Matcher`。E-024A 将递归拆为 R0 coarse basin generation、R1 wide local capture、
R2 geometric consensus/outlier rejection、R3 narrow refinement。该方法与 LoFTR 类一次性 coarse-to-fine
的区别是：fine observations 会反向更新 low-dimensional affine state，更新后的 state 再定义下一轮
local search。E-024A 允许训练本实验专属 coarse/fine/consensus modules，但禁止修改或 finetune
DINO/GDROS/baseline checkpoint；禁止 dense flow state、dense GRU、fixed DINO-K256 selector primary path、
RANSAC/三点/131k、hidden full stride-8 4096x4096、OS424/confirm/datum。完成后等待用户，不自动启动
E-024B、加速版或 rescue。

## 2026-08-15 / D-063

E-024A formal 完成并停止，判读 `COMPLETED / COARSE_BASIN_FAIL`。P0 PASS、G0 PASS，两路 reviewer
APPROVE，watchdog `20260815_183333_7919cccb` succeeded。train R3 final AEPE `7.513 px`，但 dev
R3 final AEPE `50.937 px`、CMR@2 `0.166%`。dev R0 selected AEPE mean/P50 `64.741/60.525 px`，
oracle-best `22.998/21.194 px`；R1 window coverage mean/P50 `0.960/1.000`，R2 improvement
mean/P50 `0.285/0.312`。解释：E-024A 成功把 window coverage 拉高，R2 也有一定更新收益，但 R0
coarse basin 仍没有泛化生成/选择正确初始状态，导致最终仍失败。

用户提出用 baseline 的粗匹配结果作为初始，隔离检查后半段是否有效。方法任务批准只读诊断
`E-024A-BASEINIT-DIAG Baseline-Init Backend Isolation`：优先使用 frozen baseline 的 no-GT coarse
matching/intermediate affine；若不存在，可从 frozen baseline predicted final flow 拟合 low-dimensional
affine init，但必须标注 `GDROS_FLOW_INIT_DIAGNOSTIC`，只能作为 diagnostic upper-bound，不能作为我们
完整方法成绩。诊断固定 E-024A R1/R2/R3 checkpoint 与实现，只替换 R0 selected state，比较
baseline-init-only、baseline-init+backend 与 oracle local update reference。禁止训练、修改 E-024A
formal/checkpoint、写 baseline、OS424/confirm/datum、DINO/GDROS finetune 或启动 E-024B/rescue。

## 2026-08-15 / D-064

E-024A-BASEINIT-DIAG 完成并停止，判读 `COMPLETED / BACKEND_ORACLE_ONLY`。baseline init source 为
`GDROS_FLOW_INIT_DIAGNOSTIC`，dev `404/404`；T1 baseline-init-only AEPE mean/P50 `1.474/1.278 px`，
T2 baseline-init + E-024A R1/R2/R3 退化到 `3.626/3.584 px`，相对 T1 为 `-146.0%`，CMR@2
`21.579%`；T3 oracle local update AEPE mean/P50 `0.442/0.384 px`。reviewer APPROVE。解释：
当前 E-024A 后半段不能作为 refinement 使用；在好初始化下，predicted local observation/consensus 会污染
affine state，但 oracle local observation 上限极强。

用户指出图像匹配本质是特征建模与特征相似度匹配，多尺度/coarse-to-fine 主要用于效率和缩小搜索范围；
因此下一步应寻找和训练更好的 optical-SAR feature。方法任务同意，并批准 `M001 v2.9 diagnostic /
E-025A Feature Frontend Fair Matching Benchmark`。E-025A 不比较完整 GDROS detector-free 系统作为 primary，
而是把 `GDROS_FEATURE_FAIR` 定为 primary feature-only baseline：所有 feature 经过同一 scale、
similarity、topK/NMS/region quota 和 affine evaluator。实验同时比较 frozen/prior features、结构/频域先验、
truth_flow-supervised descriptor、GDROS teacher/hybrid distilled descriptor；训练可使用 differentiable
fair matcher，但正式 dev 比较必须回到统一 hard matcher。完整 GDROS system 只作 secondary reference。
完成后等待用户，不自动启动 E-025B、完整系统、OS424/confirm 或 rescue。

## 2026-08-15 / D-065

E-025A 完成并停止，方法层判读 `COMPLETED / FROZEN_PRIOR_FEATURE_GAIN_FOUND`。统一 fair matcher 下，
`DINO_FOUNDATION_FAIR` AEPE `79.246 px` 优于 `GDROS_FEATURE_FAIR` AEPE `96.430 px`，说明 DINO /
foundation prior 对全局 affine geometry 有帮助；但 `DINO_FOUNDATION_FAIR` recall@2 / correct@2
`0.198/0.163` 低于 `GDROS_FEATURE_FAIR` 的 `0.729/0.762`，且 E-025A 的小型 supervised 与
teacher/hybrid descriptor 均失败。解释：frozen DINO 不是最终答案，它提供了有用的视觉先验，但没有
针对 optical-SAR correspondence 和 SAR speckle/structure 做适配。

用户批准下一步“参考 DINOv2 设计我们的网络，并使网络有 opt-SAR 输入、对 SAR 进行适配”。方法任务
据此建立 `M001 v3.0 prototype / E-026A DINOv2-Inspired Opt-SAR Adaptive Feature Network`。E-026A
训练我们自己的双模态特征前端：optical RGB stem、SAR log/structure adaptive stem、ViT-style patch-token
encoder、descriptor projection，并用 train truth_flow 的 token correspondence、hard-negative contrastive、
cycle/mutual consistency、SAR augmentation 与 soft affine/grid loss 训练。DINOv2 checkpoint 若使用，只能
作为只读 teacher、初始化副本或 reference；GDROS/DINO baseline 与历史产物只读。正式 dev 仍使用 E-025A
primary fair matcher `topK256_nms2_quota4_mutual0` 比较 `GDROS_FEATURE_FAIR` 与 `DINO_FOUNDATION_FAIR`。
完成后等待用户，不自动启动 E-026B、完整 sparse feedback、OS424/confirm、DINO/GDROS finetune 或 rescue。

## 2026-08-15 / D-066

E-026A formal 完成并停止，判读 `COMPLETED / NO_OPT_SAR_FEATURE_GAIN`。P0 PASS、G0 PASS，reviewer
APPROVE，watchdog `20260815_220750_823be023` succeeded。dev primary fair matcher
`topK256_nms2_quota4_mutual0` 下，`OPT_SAR_DINOSTYLE_NET` AEPE `149.866 px`、recall@2 `0.0358`、
correct@2 `0.0099`，弱于 `DINO_FOUNDATION_FAIR` AEPE `79.246`、recall@2 `0.198`、correct@2
`0.163`，也弱于 `GDROS_FEATURE_FAIR`。训练 sanity 显示 loss `13.819 -> 12.900`、probe CE
`17.86 -> 12.64`，optical stem、SAR stem/adapter、ViT encoder 和 projection 梯度均非零；DINO
checkpoint 只读引用，未作为 teacher/init 使用。解释：E-026A 不是优先归因为实现错误，而是证明“只参考
DINOv2 结构、从零训练 opt-SAR 小网络”不足。

用户确认下一步应做强预训练迁移。方法任务据此建立 `M001 v3.1 prototype / E-027A Strong DINOv2
Pretraining Transfer for Opt-SAR Features`：primary track 固定为 frozen optical DINOv2 teacher/reference
+ trainable SAR log/structure/speckle-adaptive stem 与 SAR-to-DINO adapter/projection；用 train
`truth_flow` 对齐 SAR token 和 optical DINO token，并加入 correspondence contrastive/listwise、hard
negative、cycle/mutual、SAR augmentation 与 soft affine/grid loss。可选复制 DINOv2 到 E-027A checkpoint
后训练 adapters/LoRA，但原始 DINO/GDROS checkpoint 只读。正式 dev 仍使用 E-025A primary fair matcher。
完成后等待用户，不自动启动 E-027B、完整 sparse feedback、OS424/confirm、DINO/GDROS baseline finetune
或 rescue。

## 2026-08-15 / D-067

E-027A formal 完成并停止，判读 `COMPLETED / TRANSFER_FRONTEND_STRONG_GAIN`。P0 PASS、G0 PASS、
transfer/model 与 evaluator/protocol reviewer 均 APPROVE，watchdog `20260815_223503_9fc718b9`
succeeded。dev primary fair matcher `topK256_nms2_quota4_mutual0` 下，
`FROZEN_DINO_OPTICAL__SAR_ADAPTER` AEPE `26.246 px`、recall@2 `0.569`、correct@2 `0.438`，
显著优于 E-025A `DINO_FOUNDATION_FAIR` `79.246/0.198/0.163`、`GDROS_FEATURE_FAIR`
`96.430/0.729/0.762` 的 AEPE 维度，也显著优于 E-026A scratch `149.866/0.0358/0.0099`。
DINOv2 checkpoint SHA verified `d5383ea8...cbf428`，DINO frozen params require_grad tensors `0`，
trainable SAR adapter params `475,648`，loss `11.312 -> 9.210`，dev 无 GT 调参。解释：强 DINOv2
预训练迁移是当前第一条真正有效的跨模态 feature frontend 路线。

用户提出继续改进：E-027A 只适配 SAR，可以给 optical 也加适配，并加入辅助监督/损失，让网络更能看到
模态不变特征。方法任务据此建立 `M001 v3.2 prototype / E-028A Dual-Adapter DINO Transfer with
Modality-Invariant Supervision`。E-028A 保留 frozen DINOv2 teacher/reference 与 E-027A SAR adapter
路径，新增 trainable optical residual adapter/projection 或实验 checkpoint 副本 LoRA，并用 optical teacher
consistency 防止遗忘 DINO prior。训练加入 paired-token contrastive、modality-invariant alignment、
variance/covariance regularization、domain-adversarial/modality-confusion、local-neighborhood structure
consistency、cycle/mutual、soft affine/grid 与 SAR augmentation。正式 dev 仍使用 E-025A primary fair matcher。
完成后等待用户，不自动启动 E-028B、完整 sparse feedback、OS424/confirm、DINO/GDROS baseline finetune
或 rescue。

## 2026-08-15 / D-068

E-028A formal 完成并停止，判读 `COMPLETED / DUAL_ADAPTER_TRAINING_BLOCKED`。P0 PASS，G0 唯一失败项
为 `optical_teacher_drift_bounded`：optical drift `0.178630`，超过 bound `0.020000`；因此 G1 按 gate
未运行，reviewer `CHANGES_REQUIRED`，watchdog `20260815_231120_3ff00d14` succeeded。训练 sanity
显示双 adapter 有潜力：loss `8.6373 -> 6.5134`，train sanity primary AEPE `16.8356`，
recall@2/correct@2 `1.0440/0.7188`。解释：E-028A 不是方向失败，而是 optical adapter 自由度过大，
需要强约束或调参后再评估。

用户明确批准继续双 adapter 调参，目标是看它在安全约束下能比 E-027A 单 SAR adapter 提升到什么程度。
方法任务据此建立 `M001 v3.3 prototype / E-028B Drift-Controlled Dual-Adapter Tuning Sweep`。E-028B
保留 E-027A parity 和 E-028A drift-blocked 证据，预注册 bounded tuning sweep：optical adapter residual
scale、teacher consistency weight、optical trainable subset、modality-invariant loss 与 local-structure
consistency。配置只能由 train 内部 `train_fit/train_calib` 选择，dev 不得参与配置/阈值/early-stop。
primary hard gate 仍要求 optical drift `<=0.02`；drift `>0.02` tracks 只可作为 unsafe diagnostic，不支持
成功 claim。正式 dev 仍使用 E-025A primary fair matcher。完成后等待用户，不自动启动 E-028C、完整
sparse feedback、OS424/confirm、DINO/GDROS baseline finetune 或 rescue。

## 2026-08-16 / D-069

E-028B formal 完成并停止，判读 `COMPLETED / SAFE_DUAL_ADAPTER_STRONG_GAIN`。P0 PASS、G0 PASS、
G1 `SAFE_DUAL_ADAPTER_STRONG_GAIN`，reviewer APPROVE，watchdog `20260815_235342_e9f1de97` succeeded。
selected primary 为 `PROJECTION_ONLY_SAFE`。dev optical drift `4.38e-7 <= 0.02`，AEPE `21.418 px`、
recall@2 `0.669`、correct@2 `0.520`。相对 E-027A 单 SAR adapter：AEPE 改善 `18.4%`，recall@2
提升 `17.6%`，correct@2 提升 `18.6%`。解释：safe dual adapter 已在 feature-only fair matcher 下
形成稳健增益，且最有效的是克制的 optical projection-only，而非更自由的 optical token 修改。

用户提出两条后续路线：第一，把原 baseline 的 feature extraction 替换为 E-028B，并将替换后的 frontend
与 GDROS 后半段一起训练，以适应我们的 feature distribution，再与 baseline 全流程比较；第二，既然我们的
点质量更好，后续应设计更轻的匹配器，在不损失精度的前提下降低匹配成本。方法任务决定先执行第一条系统
接入实验，建立强后端适配后的系统级上限，再用该结果定义轻量匹配器的压缩目标。用户批准 E-029A：
`M001 v3.4 prototype / E-029A E-028B Frontend + GDROS-Style Backend Joint Adaptation`。E-029A 将复制/
隔离 GDROS-style backend 到实验目录，用 E-028B safe dual-adapter frontend 替换特征输入，训练 bridge
adapter、backend copy，并可端到端训练 E-028B frontend 副本；原始 GDROS baseline 与 E-028B formal
checkpoint 只读。完成后等待用户，不自动启动轻量 matcher、E-029B/E-030A、OS424/confirm、DINO/GDROS
baseline finetune 或 rescue。

## 2026-08-16 / D-070

E-029A formal 完成并停止，判读 `COMPLETED / NO_SYSTEM_INTEGRATION_GAIN`。P0 PASS、G0 PASS、reviewer
APPROVE，watchdog `20260816_085337_ee042264` succeeded。best full-system track 为
`TRAINABLE_E028B_FRONTEND__TRAINED_GDROS_BACKEND`，dev AEPE `21.916 px`、CMR@2 `1.160`、CMR@5
`6.932`、PLR `100.0`，没有超过 E-028B fair reference AEPE `21.418`；frozen frontend track AEPE
`23.013`。训练 sanity 正常：frozen track loss `6.006 -> 4.061`，trainable track loss `6.210 -> 4.066`，
梯度到达 bridge/backend/frontend copy。baseline reference 标注为 `GDROS_FLOW_INIT_DIAGNOSTIC`，
exact replay false。

用户质疑该负结果：如果后端接入并联合训练后比简单 matcher 还差，需要重新训练 baseline，看 copied
GDROS pipeline 自己能得到什么结果。方法任务同意这是红旗审计，而不是方法分支扩展。用户批准
`M001 v3.4 diagnostic / E-029A-GDROS-RETRAIN-DIAG Copied GDROS Retrain Parity Audit`：用 E-029A 同一
copied backend/training harness/evaluator 换回 GDROS baseline frontend/features，执行 no-train replay
sanity、same-budget copied GDROS retrain，必要时 official recipe retrain 或 cached-feature backend retrain。
若 copied GDROS 能接近 full baseline，E-029A 负结果才可信归因为 E-028B/backend mismatch；若 copied
GDROS 也很弱，则 E-029A 的 `NO_SYSTEM_INTEGRATION_GAIN` 不能用于否定方法。完成后等待用户，不自动启动
轻量 matcher、E-029B/E-030A、OS424/confirm、DINO/GDROS baseline finetune 或 rescue。

## 2026-08-16 / D-071

E-029A-GDROS-RETRAIN-DIAG 完成并停止，判读 `COMPLETED / E029A_NEGATIVE_NOT_TRUSTWORTHY`。P0 PASS、
G0 PASS、G1 `E029A_NEGATIVE_NOT_TRUSTWORTHY`，diagnostic `COPIED_GDROS_RETRAIN_WEAK`。关键结果：
GDROS official archive reference AEPE `1.461`；`COPIED_GDROS_REPLAY_SANITY` dev404 anchor AEPE `1.426`、
dense AEPE `1.460`、CMR@2 `78.244%`、CMR@5 `99.040%`，说明 replay/evaluator/data path 强；
但 `COPIED_GDROS_SAME_BUDGET_RETRAIN` dev404 AEPE `53.779`、CMR@2 `0.758%`、CMR@5 `4.461%`。
解释：E-029A-like same-budget 短训 harness 即使用 GDROS 自己的 frontend/features 也复现不了 GDROS，
因此 E-029A 的负结果不能用于否定 E-028B 接入路线。

用户进一步澄清两点：1）需要按官方训练协议重训 GDROS，不使用官方已训练权重，看能否达到相同指标；
2）询问 `COPIED_GDROS_SAME_BUDGET_RETRAIN` 是否是 E-028B feature 替换 GDROS feature。方法任务明确：
`COPIED_GDROS_SAME_BUDGET_RETRAIN` 不是 E-028B 替换特征，而是 GDROS frontend/features + E-029A-like
短训 harness；它弱说明短训 recipe 不可信。用户批准新的官方协议重训审计
`M001 v3.4 diagnostic / E-029A-GDROS-OFFICIAL-RETRAIN Official GDROS Recipe Retrain From Scratch`：
先审计官方 README/train scripts/config，绑定 optimizer、schedule、epoch/steps、augmentation、loss、
checkpoint selection 与所有初始化来源；primary track 禁止加载官方已训练 GDROS checkpoint，只按 official
recipe 在实验副本中训练，并用 formal dev404 最终评价。完成后等待用户，不自动启动 E-029B/E-030A、
E-028B integration rerun、轻量 matcher、OS424/confirm 或 rescue。

## 2026-08-17 / D-072

E-029A-GDROS-OFFICIAL-RETRAIN 完成并停止，判读 `OFFICIAL_RETRAIN_REASONABLE_BUT_BELOW_BASELINE`。
该实验不加载官方已训练 GDROS 权重，使用可审计 official recipe 训练 120000 steps；允许的非 GDROS
初始化为 `raft-sintel.pth`。结果：AEPE `3.5769`、CMR@2 `35.7612%`、CMR@5 `83.2822%`。但方法任务
随后指出一个协议差异：该实验使用研究内部 `train1607/dev404` split，不是论文官方 train/test protocol。
官方 published/released checkpoint 指标来自 official training/testing protocol：`training.zip` 与
`testing.zip`。

用户明确要求：既然官方有 `training.zip` 和 `testing.zip`，就按这个协议重跑；跑之前核对训练测试协议是否
和官方对得上。方法任务据此建立
`M001 v3.4 diagnostic / E-029A-GDROS-PAPER-PROTOCOL-RETRAIN`。该实验 P0 必须先核对 official
`training.zip` SHA/count `2887e7... / 2011`、official `testing.zip` SHA/count `e9c29d... / 424`、
官方 README/train/test command/config、初始化来源与 evaluator；P0 通过后，用 full training.zip 2011
训练，不加载官方已训练 GDROS 权重，testing.zip 424 只做 final evaluation，禁止用于训练、validation、
early stop 或 checkpoint/hparam selection。完成后等待用户，不自动启动 E-028B integration rerun、轻量
matcher、E-029B/E-030A、confirm 或 rescue。

## 2026-08-18 / D-073

E-029A-GDROS-PAPER-PROTOCOL-RETRAIN 完成并停止，判读
`COMPLETED / PAPER_PROTOCOL_RETRAIN_REPRODUCES_BASELINE`。P0/G0 PASS，未加载官方已训练 GDROS 权重，
允许 RAFT/Sintel generic init。official testing.zip 424 结果：AEPE `1.7014`、CMR@1 `23.5849%`、
CMR@2 `71.2264%`、CMR@5 `99.2925%`、PLR `100%`。released checkpoint replay reference：
AEPE `1.4612`、CMR@1 `33.726%`、CMR@2 `81.604%`、CMR@5 `99.528%`。该结果排除主要数据/测试污染，
但 CMR@1/2 仍低约 10pp。

用户要求先看 baseline：比较当前训练协议、官方仓库默认协议与论文文字协议，如果简单调参即可追回十几个点，
则优先修 baseline recipe。方法任务确认：公开 GitHub 默认为 batch `2`、LR `1.25e-4`、120k steps；
论文文字为 initial LR `1.2e-5`、batch `12`、120k steps；E-029A paper-protocol 实际把 `1.2e-5`
作为 OneCycleLR max LR，可能导致真实初始 LR 约 `4.8e-7`，影响 CMR@1/2 精细收敛。用户批准
`M001 v3.5 diagnostic / E-030A-GDROS-PROTOCOL-SWEEP`，包含三条轨道：
`REPO_DEFAULT_RETRAIN`（batch2/LR1.25e-4）、`BATCH12_HIGH_LR_RETRAIN`（batch12/LR1.25e-4）和
`CHECKPOINT_CURVE_DIAG`（E-029A/A/B checkpoints 离线曲线，仅解释不做 testing selection）。完成后等待用户，
不自动启动 E-028B integration rerun、轻量 matcher、E-030B、confirm 或 rescue。

## 2026-08-20 / D-074

E-030A-GDROS-PROTOCOL-SWEEP 完成并停止，判读
`COMPLETED / PROTOCOL_SWEEP_RECOVERS_RELEASED_CMR`。P0/G0 PASS，watchdog succeeded。Track A
`REPO_DEFAULT_RETRAIN` final：AEPE `1.5798`、CMR@1 `23.8208%`、CMR@2 `77.8302%`、CMR@5
`99.7642%`。Track B `BATCH12_HIGH_LR_RETRAIN` final：AEPE `1.4347`、CMR@1 `37.0283%`、
CMR@2 `83.4906%`、CMR@5 `99.5283%`，超过 released checkpoint replay 的 AEPE/CMR@1/2。checkpoint
curve 显示 Track B 100k CMR@2 `84.434%` 高于 final，但该曲线为 diagnostic-only，testing curve
不得用于 checkpoint selection 或成功追认。

用户要求将该参数作为后续训练参考。方法任务将 E-030A Track B 固化为后续 GDROS-style full training
默认参考：official training.zip 2011 训练、testing.zip 424 final only、batch `12`、OneCycle max LR
`1.25e-4`、120k steps、train/test iters `12/32`、public augmentor、public `flow_loss + mat_flow_loss`。
下一步方法设计回到 E-028B 主线：`M001 v3.6 design / E-031A-VALIDATED-RECIPE-E028B-INTEGRATION`。
该实验若获批准，将用 validated recipe 重新做 E-028B safe dual adapter frontend + GDROS backend 系统级接入，
并用 interface diagnostics 区分 frontend 分布不适配、joint training 退化与 backend 无法利用更好特征。

用户指出：之前 E-028B frontend + GDROS backend 已经失败过，不能假设换训练协议就一定成功；应先诊断
换成 E-028B 特征后和 GDROS 后端接口差多少。方法任务据此将 full 120k integration 暂缓，建立
`E-031A-FEATURE-INTERFACE-DIAG`。该诊断只使用 official training.zip 固定 probe/small-overfit subset，
比较 GDROS native backend-interface feature 与 E-028B feature 的分布差、direct replacement / normalization /
linear bridge 的后端兼容性，以及小样本 bridge/backend overfit 能否成立。testing.zip 不参与选择或成功判读。
完成后停止等待用户，不自动启动 full integration、轻量 matcher、sparse feedback 或 rescue。

E-031A-FEATURE-INTERFACE-DIAG 完成，判读 `BRIDGE_REQUIRED_BUT_ADAPTABLE`。关键证据：GDROS backend
interface 为 `[B,256,64,64]`，E-028B feature 为 `[B,1024,32,32]`；direct replacement 因 shape/channel/grid
mismatch 失败；simple bridge AEPE `18.6692`、CMR@2 `1.1656%`；overfit bridge AEPE `44.7482`、
CMR@2 `0.0302%`；但 E-028B feature-only retrieval top32 `85.93%` 明显强于 native `23.77%`。用户据此
明确要求：不要继续设计 adapter，若 GDROS backend 不适配，就针对 E-028B 特征设计后续 matcher。
方法任务停止 E028B-to-GDROS-backend adapter 路线，提出 `E-032A-E028B-NATIVE-MATCHER-DIAG`：固定 E-028B
特征，设计/诊断 native matcher，包括 mutual/dual-softmax/region-balanced matching、local refinement 与
affine consensus solver；不训练新 descriptor，不复用 GDROS dense GRU/backend。
