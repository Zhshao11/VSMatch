# 失败方法档案

本文件只记录经过实现复核、数据复核和合理补救后仍失败的方法。一次训练失败、单个随机
种子下降或未经审阅的实现结果不能进入本档案。

### F-001：固定全预算 B2 六键共识/重排与置信拒绝器

- 对应方法版本：`M001 v0.5 / E-002B / BIND-001`
- 对应实验编号：`E-002B`、`E-002B-R`、`E-002B-ACC-DIAG`
- 日期：`2026-08-05`
- 原始假设：冻结 GDROS 特征的 top-k 候选可由固定几何共识、六键排序和置信拒绝恢复可靠仿射，同时避免全对全相似度。
- 实现版本和审阅报告：冻结 reference manifest `3dd002c8f92c4ac58571ba04b92cbda6830843bce9662d93dd5d5ab67f35e58c`；`results/E-002B/review-core.md` 为 `APPROVE`，诊断无 evaluator/glue diff。
- 数据与 baseline 复核：官方 424 对、全图无 mask；1,696 行全部 finite，provenance、GT isolation、完整性和 evaluator parity PASS。
- 主要负面结果：C8-main PairLocalReady `75.71%`（门槛 `90%`），grid mean `17.233 px`（门槛 `8 px`），AEPE `10.4349 px`；non-abstention `100%`，未识别重尾失败。
- 失败集中在哪些样本或条件：103/424 对未达到 PairLocalReady，156/424 对误差不小于 `5 px`；P95/P99 为 `43.989/118.151 px`。
- 已尝试的补救措施：保持方法语义的等价后端优化未通过资源门槛；C8-C16 诊断仅改善到 AEPE `8.2733 px`、PairLocalReady `79.72%`，仍失败。
- Oracle 或上限实验：cosine top8 GT oracle AEPE `1.4792 px`、PairLocalReady `99.53%`，说明候选集上限充足。
- 判断为方法失败而非实现失败的证据：28/28 自动测试 PASS、独立核心审阅 APPROVE、正式数据和协议检查 PASS；失败位于固定 selector 与拒绝机制。
- 失败边界：只否定当前固定全预算 B2 六键共识/重排与置信拒绝器；不否定 top-k candidate set、冻结 GDROS 特征、其他预算化选择器或 SparseGeo 方法族。
- 未来可以重启该方向的条件：必须提出语义不同且可检验的新选择/置信机制，并由用户明确批准；不得作为当前实现的连续救援版本自动启动。
- 完整日志和结果路径：`results/E-002B/`、`results/E-002B-R/`、`results/E-002B-ACC-DIAG/RESULT_CARD.md`。

### F-002：v0.6 独立候选 MLP 与单一仿射反馈 selector

- 对应方法版本：`M001 v0.6`
- 对应实验编号：`E-003A`
- 日期：`2026-08-05`
- 原始假设：共享候选 MLP、top8+dustbin 联合分类和两轮单一加权仿射残差反馈，足以从冻结 top8 候选恢复可靠对应并校准失败置信度。
- 实现版本和审阅报告：source closure `daaf3fea395180560109d74154bb1078e50958cbaacb337738fe9c1c86518482`；16/16 自动测试及两轮独立审阅 `APPROVE/APPROVE`。
- 数据与 baseline 复核：group-isolated train/dev `1,607/404`，group overlap 与 OS testing exact overlap 均为 0；缓存/在线 parity 和冻结 checkpoint provenance PASS。
- 主要负面结果：主 seed AEPE `10.6808 px`、PairLocalReady `67.08%`、grid mean `11.0970 px`、coverage `0.7426%`、90%-coverage risk reduction `8.72%`，G1 FAIL。
- 失败集中在哪些样本或条件：平均仅 `34.92%` anchors 可恢复；大位移、旋转和黑边组更差。模型趋向中等误差的全局仿射，未可靠识别稀疏 inlier/dustbin。
- 已尝试的补救措施：三个固定 main seed 的 best grid 为 `11.10/11.70/11.57 px`；T2 明显优于 unary，但仍未过门槛。按冻结停止规则未事后改结构、阈值或预算。
- Oracle 或上限实验：corrected top8 GT oracle AEPE `2.8410 px`、PairLocalReady `100%`。
- 判断为方法失败而非实现失败的证据：G0、数据隔离、梯度、toy overfit、cache parity、两轮审阅均通过，三 seed 负面结果稳定，oracle 容量充分。
- 失败边界：只否定 v0.6 的局部候选编码、联合 dustbin 分类及单一全局 residual feedback 组合；不否定冻结 GDROS/top8、显式 recoverability 分解、跨-anchor learned consensus 或 learned descriptor。
- 未来可以重启该方向的条件：必须显式解决不可恢复 anchor 的类别失衡和跨-anchor 共识，并由用户批准为语义不同的新分支；不得把调阈值或增加 T 当作 v0.6 救援。
- 完整日志和结果路径：`results/E-003A/RESULT_CARD.md`；远端 `experiments/sparsegeo/runs/E-003A/run-01`。

### F-003：v0.7 手工统计量 Pre/Post Heads 与阈值式候补替换

- 对应方法版本：`M001 v0.7`
- 对应实验编号：`E-004A`
- 日期：`2026-08-06`
- 原始假设：coarse scalar statistics 可在 fine search 前可靠预测 anchor recoverability，post-search scalar verifier 可通过同区域阈值式 reserve replacement 显著修复低质量 primary。
- 实现版本和审阅报告：source SHA `76c735d1fbe34543903148bd69c1f8f43a0f9ed8f855d44334565256a92b9453`；23 tests PASS，两轮独立审阅 `APPROVE/APPROVE`。
- 数据与 baseline 复核：复用 group-isolated `1,607/404` train/dev，无 OS424/confirm；2,011-pair cache 完整，checkpoint/provenance PASS。
- 主要负面结果：主 seed recoverable `41.7969%`、median `53/128`，未达到 `50%/64`；相对 uniform `+9.6767 pp` 未达到 `10 pp`；replacement 只增加 `0.0522 pp`。
- 失败集中在哪些样本或条件：pre-head AUROC `0.6978`、AP `0.5282`，只恢复约三分之二 oracle headroom；post-head 仅接受 `519/11153` 次 reserve 尝试。
- 已尝试的补救措施：三个冻结 seed 失败项相同且方向一致；按预注册规则未事后改变阈值、预算、特征或替换范围。
- Oracle 或上限实验：P0 coverage-constrained GT oracle recoverable `61.7942%`、median `78/128`、AEPE `2.0848 px`、PairLocalReady `100%`。
- 判断为方法失败而非实现失败的证据：P0/G0、自动测试、两轮审阅、三 seed、结构不变量和 provenance 均通过；负面结果稳定。
- 失败边界：只否定当前 scalar pre/post representation 与 `p_post<0.5`、改善 `>=0.05` 的逐点同区域 replacement；不否定 M256 proposal、区域 `[1,3]` 预算、全 fine-probe setwise ranking 或冻结特征。
- 未来可以重启该方向的条件：必须使用语义不同的联合/setwise recoverability 表示或全 proposal 排序；不得仅调整现有阈值追认。
- 完整日志和结果路径：`results/E-004A/RESULT_CARD.md`；远端 `experiments/sparsegeo/runs/E-004A/run-01`。

### F-004：v0.8 Soft-expectation 稀疏递归共识与普通 Weighted LS

- 对应方法版本：`M001 v0.8`
- 对应实验编号：`E-005A`
- 日期：`2026-08-06`
- 原始假设：M256 top8、稀疏 local correlation、global tokens、四轮 affine feedback 和 dynamic K128，配合 candidate/dustbin 概率及 ordinary weighted LS，可替代 dense flow/GRU 并接近 GDROS 精度。
- 实现版本和审阅报告：source manifest `ec2596a30e65f30277989ffd6507188431d60964e4471cba8b6a282cb58865f4`；18 tests PASS，两轮独立审阅 `APPROVE/APPROVE`。
- 数据与 baseline 复核：group-isolated `1,607/404` train/dev；dense baseline parity 最大差 `0.0`；无 OS424/confirm、GT inference 或 hidden N2。
- 主要负面结果：main/review AEPE `4.6343/4.5593 px`、CMR@2 `7.92/11.14%`、PairLocalReady `96.53/96.29%`，未达到 AEPE `2.0 px`、CMR@2 `76.46%` 和 PairLocalReady `99%`。
- 失败集中在哪些样本或条件：main P50/P95/P99 AEPE `4.0648/9.7572/16.5752 px`；134/404 对 AEPE `>5 px`，18/404 对 `>10 px`。AEPE 与 final recoverable count 相关 `-0.562`，与 dustbin Brier 相关 `+0.610`。
- 已尝试的补救措施：两个冻结 seed 同方向；完整消融显示所有创新组件均有贡献，但 full model 仍失败。按冻结停止规则未调 T/D/G/K、loss、阈值或后端。
- Oracle 或上限实验：P0 top8+local oracle AEPE `0.1476 px`、PairLocalReady/valid `100%`，候选与局部坐标可达上限充分。
- 判断为方法失败而非实现失败的证据：P0/G0、18 tests、两轮审阅、双 seed、baseline parity、provenance 和复杂度审计均通过；结构契约满足。
- 失败边界：只否定当前 9-way candidate/dustbin objective、soft target expectation、hard dynamic K128 和 ordinary weighted LS 的组合；不否定 global-token recurrence、affine feedback、top8/local evidence 或所有 sparse recurrent consensus。
- 未来可以重启该方向的条件：先用冻结 checkpoint 做 oracle coordinate/candidate/weight 分解；新版本必须直接解决 active false-positive weight contamination，而不能只增加轮数或改 K。
- 后续诊断：GDROS matched-M256 2px GT/self inlier `72.40/96.75%`，E-005A active raw `14.70/47.41%`、weighted `22.20/73.33%`；只有 156/404 pairs 在 GT binary weights 下具备足够 predicted-coordinate 支持。首要失败为 GT coordinate/candidate recovery，次要失败为错误共识权重污染。
- 完整日志和结果路径：`results/E-005A/RESULT_CARD.md`；远端 `experiments/sparsegeo/runs/E-005A/run-01`。

### F-005：v0.9 E-006A 候选再生成覆盖门槛

- 对应方法版本：`M001 v0.9`
- 对应实验编号：`E-006A`
- 日期：`2026-08-06`
- 原始假设：global top8、region top8 和 B8 affine-local top4 的候选再生成空间，能暴露接近 GDROS 稠密内点率的 2px 正确 proposal，为 confidence-growing 训练提供高密度正样本池。
- 实现版本和审阅报告：E-006A source manifest `a26f9619ad395bdbc1d45b3ec09ae256d603512042681259efdc8aac4e486d2a`；自动测试 `12/12 PASS`，method/protocol reviewer 均 `APPROVE`。
- 数据与 baseline 复核：group-isolated `1,607/404` train/dev；formal P0 使用完整 404 dev；无 OS424/confirm、GT inference、hidden N2、dense state/GRU、fixed K128/Kmax 或 baseline fallback。
- 主要负面结果：P0 oracle AEPE `0.190768 px`、PairLocalReady/valid `100%` 通过，但 mean 2px-correct proposals 仅 `66.210/256`，低于冻结门槛 `140/256`。
- 失败集中在哪些样本或条件：correct proposals min/P10/P50/P90/max 为 `14/33/64/98/153`；只有 `0.743%` pair 达到 140，说明不是少数样本拖累，而是候选空间正样本密度整体不足。
- 已尝试的补救措施：旧 `p0-gate.json` 因使用 GT-fitted single basin 被 reviewer 判为无效；corrected `p0-gate-r2.json` 完整执行冻结 B8/512-token region path 并作为唯一正式结果。
- Oracle 或上限实验：正确点虽然不多，但分布足以 oracle 拟合极准 affine，AEPE P50/P95 为 `0.171/0.350 px`，condition P50/P90 为 `2.258/2.782`。
- 判断为方法失败而非实现失败的证据：12/12 自动测试、两路独立审阅、预算/泄漏/坐标/evaluator differential 均通过；失败项是冻结 P0 coverage gate。
- 失败边界：只否定 E-006A 的候选再生成覆盖目标和 `>=140/256` 高密度正样本假设；不否定稀疏观测、低维仿射反馈、多 basin、hard rejection 或从少量高纯度点拟合 affine 的路线。
- 未来可以重启该方向的条件：必须把任务改写为从约 30-100 个正确候选中学习高纯度 inlier selection，并重新冻结 P0/G1；不得仅调低 `140/256` 后追认 E-006A。
- 完整日志和结果路径：`results/E-006A/RESULT_CARD.md`；远端 `experiments/sparsegeo/runs/E-006A/run-01`。

### F-006：E-006A 候选空间的 26 维无 GT 特征可选择性

- 对应方法版本：`M001 v0.9 diagnostic`
- 对应实验编号：`E-006A-SELECT-DIAG`
- 日期：`2026-08-07`
- 原始假设：E-006A corrected candidates 虽然 2px 正样本极稀疏，但 26 维 no-GT candidate features
  和一个小型 pointwise selector 可能足以把正确候选排到前列，从而支持高纯 sparse affine fitting。
- 实现版本和审阅报告：19/19 tests PASS；feature/evaluator review 与两次增量复核均 `APPROVE`；
  2011 features+labels、6 checkpoints、404 dev evaluator 全完成。
- 数据与 baseline 复核：复用 group-isolated `1607/404` train/dev；E-006A corrected candidate
  generator parity PASS；old invalid P0、OS424 和 confirm 均未访问。
- 主要负面结果：main D2 dev AUROC/AP `0.9171/0.01386`，global topK-32 precision/recall
  `2.259/7.778%`，affine AEPE `63.516 px`，CMR@2 `0%`，valid `93.812%`。
- 失败集中在哪些样本或条件：dev candidate base rate 极低，2px positive 仅 `0.1343%`；
  in-bounds group 2px rate `0.1622%`，out-of-bounds group `0.00561%`。高 AUROC 在该 base rate
  下不能转化为高纯 topK。
- 已尝试的补救措施：D1 hand-crafted、D2 learned selector、两个 seed、feature ablations 和
  topK/threshold/region-balanced affine evaluation 均未产生可用精度。D2 相对 best D1 候选指标提升
  `0.0794`，但 affine AEPE 反而差 `1.883 px`，未达到 `PARTIALLY_SELECTABLE`。
- Oracle 或上限实验：D0 hard `<=2 px` inliers 在有效 pairs 上 selected precision `98.89%`，
  AEPE `0.653 px`；说明 solver 和高纯点拟合可行，失败在无 GT 高纯选择。
- 判断为方法失败而非实现失败的证据：自动测试、provenance、feature leakage audit、label semantics、
  evaluator review、split 隔离和两次修复复核均通过；负面结果由正式 dev 全量评估得到。
- 失败边界：只否定 E-006A corrected candidate space 上的 26 维 no-GT feature + <=100k pointwise
  selector 可选择性；不否定更强 patch evidence、descriptor 改造、改变 observation space、multi-basin
  robust consensus 或稀疏 affine feedback 主张。
- 未来可以重启该方向的条件：必须提供语义不同的观测证据，例如更强局部 patch/结构相似度、
  跨模态 descriptor 训练、区域到点联合匹配或其他能显著提高 topK precision 的机制；不得继续在
  同一 26 维特征上调 selector。
- 完整日志和结果路径：`results/E-006A-SELECT-DIAG/RESULT_CARD.md`；远端
  `experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01`。

### F-007：固定 E-006A 候选集上的第一版 Patch/Structure/Descriptor 观测证据

- 对应方法版本：`M001 v1.0`
- 对应实验编号：`E-007A`
- 日期：`2026-08-07`
- 原始假设：在不扩大 E-006A corrected candidate set 的前提下，raw/GDROS feature patch、
  跨模态结构通道和轻量 local descriptor adapter 可以显著提升候选可选择性，并改善 sparse affine。
- 实现版本和审阅报告：14/14 tests PASS；feature、descriptor/training、evaluator 三名定向 reviewer
  均 `APPROVE`；formal watchdog 全部结束。
- 数据与 baseline 复核：group-isolated `1607/404` train/dev；冻结 E-006A baseline parity PASS，
  topK-32 precision `2.2587%`、AEPE `63.5157 px`、CMR@2 `0%`；无 OS424/confirm、datum labels、
  candidate expansion、GDROS training、dense retrieval、RANSAC、IRLS 或 fallback。
- 主要负面结果：all-evidence 主 track topK-32 precision `1.7868%`、AP `0.00984`、AEPE
  `36.0761 px`、CMR@2 `0%`，判读 `NO_USEFUL_OBSERVATION_GAIN`。
- 失败集中在哪些样本或条件：all-evidence topK-16/32/64 precision 为 `2.3360/1.7868/1.4929%`，
  AEPE 为 `57.9356/36.0761/33.0043 px`；uncapped threshold 与 region-balanced diagnostics 仍
  CMR@2 `0%`。
- 已尝试的补救措施：raw patch、frozen feature patch、patch combined、handcrafted/learned
  structure、descriptor-only、descriptor+patch、descriptor+structure、all-evidence 和 review seed
  均未达到 weak/useful/strong gate。
- Oracle 或上限实验：E-006A/E-006A-SELECT-DIAG 已证明 hard 2px inlier oracle 可得到低 AEPE；
  E-007A 未改变候选空间，只测试更强 observation scoring。
- 判断为方法失败而非实现失败的证据：三路 reviewer APPROVE、自动测试 PASS、candidate parity、
  label materialization order、no-GT audit、train/dev split、resource trace 和 formal 404 dev evaluation
  均通过。
- 失败边界：只否定固定 E-006A candidate set 上第一版 patch/structure/descriptor verifier 排序；
  不否定改变 candidate generator、proposal expansion、更强跨模态 pretraining/backbone、dense
  descriptor learning 或不同 observation space。
- 未来可以重启该方向的条件：必须改变候选生成或底层跨模态表示，使正确候选的 base rate/topK
  precision 显著上升；不得继续在同一 candidate set 上堆类似 verifier。
- 完整日志和结果路径：`results/E-007A/RESULT_CARD.md`；远端
  `experiments/sparsegeo/runs/E-007A/run-01`。

### F-008：v1.1 Region-to-Point 跨模态候选生成器

- 对应方法版本：`M001 v1.1`
- 对应实验编号：`E-008A`
- 日期：`2026-08-07`
- 原始假设：把更强观测前移到候选生成阶段，通过多尺度 region matching、动态 source proposal
  generation 和 matched-region dense-lite descriptor/patch search，提高候选集中 2px 正确点覆盖，
  从而超过 E-006A corrected candidate generator。
- 实现版本和审阅报告：最终实现 source SHA
  `b8e4dd9f4fce7bf55f2f8ff5a875052e214248c2052ac0de5cb3fb6398ba04a5`；13 tests PASS；
  candidate/evaluator/protocol/result-interpretation reviewer 均 `APPROVE`。
- 数据与 baseline 复核：group-isolated `1607/404` train/dev；冻结旧候选生成 baseline identity SHA
  `eecff3522b8b9b26ffa196cdb559858ad9039291360bf050e791e1a1c13bd197`；candidate traces 在 GT
  label 接入前物化；未访问 OS424/confirm/datum labels，无 hidden 4096x4096 或 full-image blind search。
- 主要负面结果：P0 FAIL，P1 未运行。mean 2px-correct proposals 从 E-006A baseline `66.210`
  降到 `17.022`；P10/P50 从 `33/64` 降到 `7/15`；2px base rate 从 `0.1343%` 降到
  `0.09828%`；hard-2px oracle AEPE 为 `189.229 px`；PairLocalReady/valid 均为 `63.119%`。
- 失败集中在哪些样本或条件：动态生成器每对约产生 `1515` 个 source proposals 和 `17324` 个
  candidates，但正确覆盖整体下降。source ablation 中 region top4 的 2px base rate `0.14173%`
  略高于旧 aggregate baseline，而 descriptor top4、patch/structure top4 和 full union 均稀释最终纯度。
- 已尝试的补救措施：按冻结协议，P0 失败后停止，未启动 P1、E-008B、完整 sparse affine feedback
  或救援版本；不事后调整候选融合、ranking 或 dedup。
- Oracle 或上限实验：E-008A 的 hard-2px oracle AEPE `189.229 px`、GT-nearest-per-proposal oracle
  AEPE `37.848 px`，说明该最终候选空间本身不足以支持可靠 affine；相比之下，E-006A corrected
  candidate generator 的 oracle affine AEPE 为 `0.190768 px`。
- 判断为方法失败而非实现失败的证据：自动测试、四路 reviewer、baseline identity、GT isolation、
  trace materialization order、provenance 和 no-N2/no-blind-search gate 均通过；失败项是冻结 P0
  candidate coverage gate。
- 失败边界：只否定当前 v1.1 region-to-point v1 生成器及其最终候选融合；不否定所有区域匹配、
  学习型候选生成、跨模态 descriptor 或 sparse affine feedback。由于 P0 未记录 stage-wise GT recall，
  不能断言损失具体发生在 region selection、within-region retrieval 还是 final top4 ranking/dedup。
- 未来可以重启该方向的条件：必须先做 stage-wise GT recall 诊断或提出语义不同的新 candidate
  generator，并明确超过 `research/CANDIDATE_GENERATION_BASELINE.md` 的候选覆盖/可选择性门槛；
  不得把 E-008A v1 当作简单调参救援自动继续。
- 后续诊断：`E-008A-STAGE-DIAG` 判读 `MIXED_LOSS_DIAGNOSED`。S0/S1/S2/S4/S5 mean
  2px-correct proposals 为 `239.351/196.686/50.015/17.022/17.022`；top64 预筛是最大绝对损失，
  top4 ranking 条件损失也很大且与 top64 差距不足 `10 pp`；dedup 不丢 2px proposal。
- 完整日志和结果路径：`results/E-008A/RESULT_CARD.md`；远端
  `experiments/sparsegeo/runs/E-008A/run-01`。

### F-009：v1.2 raw/structure-only learned top64 retriever

- 对应方法版本：`M001 v1.2`
- 对应实验编号：`E-009A`
- 日期：`2026-08-07`
- 原始假设：固定 E-008A 的 source proposals 与 matched-region union，只替换 frozen cosine /
  region score top64 预筛；用 raw/structure-only 轻量跨模态 retriever 在 top64 前排序，提高正确候选覆盖。
- 实现版本和审阅报告：implementation source SHA
  `1eb43a36f88c69b661eaef2be3ba8655eef1f21ca78abf0167595395f32e8bf6`；reviewer R1
  `APPROVE_WITH_RESULT_FAIL`，未发现实现偏移。
- 数据与 baseline 复核：复用 group-isolated `1607/404` train/dev；frozen S2 parity PASS：
  mean `50.0149`、P10/P50/P90 `26/48/78`、base2px `0.05161%`。无 OS424/confirm/datum labels、
  hidden `4096x4096`、full-image blind search、dense flow/GRU 或 fallback。
- 主要负面结果：learned top64 mean/P10/P50/P90 为 `34.745/15/33/55.7`，低于 frozen S2；
  base2px `0.03585%` 低于 `0.05161%`；nearest P50/P90 `27.634/102.911 px` 差于
  `15.490/75.309 px`；G1 FAIL，G2 未晋级。
- 失败集中在哪些样本或条件：topR 曲线随 R 扩大提升覆盖，但纯度下降；top128 mean `55.394`
  和 P50 `53.5` 超过 frozen top64，但 base2px 仅 `0.02858%`，oracle AEPE `20.625 px`、
  PLR/valid `96.04%`，说明不是可直接追认的有效候选生成器。
- 已尝试的补救措施：按冻结协议 G1 失败即停止，未启动 E-009B、E-008B、rescue、完整 sparse
  affine feedback、OS424/confirm 或多 seed 追认。
- Oracle 或上限实验：E-008A-STAGE-DIAG 的 S1 region union mean correct `196.686`，说明 region
  union 中仍有大量可恢复空间；E-009A 失败在 raw/structure-only learned ranker 未能把正确点排进 top64。
- 判断为方法失败而非实现失败的证据：reviewer 确认 learned score 在 top64 前使用，且不读取 GDROS
  frozen feature；GT 仅用于训练标签和后置评估；资源/trace/provenance 均通过。
- 失败边界：只否定当前 `16k` 参数、raw/structure-only、sampled-negative CE 的 learned top64
  retriever；不否定更强跨模态 backbone/descriptor、full-union listwise loss、hard negative mining、
  或重新设计 topR 候选预算。
- 未来可以重启该方向的条件：必须提出语义不同的训练目标、负样本挖掘或底层表示，并预注册超过
  frozen S2 与 E-006A baseline 的 gate；不得仅把 `R=128` 作为 `R=64` 失败后的追认。
- 完整日志和结果路径：`results/E-009A/RESULT_CARD.md`；远端
  `experiments/sparsegeo/runs/E-009A/run-01`。

### F-010：v1.3 cached-evidence strong region-local retriever

- 对应方法版本：`M001 v1.3`
- 对应实验编号：`E-010A`
- 日期：`2026-08-07`
- 原始假设：固定 E-008A S0/S1，用 strong region-local retriever、full-union/listwise retrieval 和
  hard negative mining，在 matched-region union 内显著超过 frozen S2 top64。
- 实现版本和审阅报告：implementation source SHA
  `24a99d9d351e8fe00d55330bdd36010063381952864130a37ea358bb240d3719`；reviewer R1
  `APPROVE_WITH_RESULT_FAIL`，未发现实现偏移。
- 数据与 baseline 复核：复用 group-isolated `1607/404` train/dev；frozen S2 parity PASS。无
  OS424/confirm/datum labels、GDROS frozen feature input、frozen score teacher、hidden `4096x4096`、
  full-image blind search、dense flow/GRU 或 fallback。
- 主要负面结果：strong top64 mean/P10/P50/P90 为 `37.077/16/36/60`，低于 frozen S2
  `50.015/26/48/78`；base2px `0.03826%` 低于 `0.05161%`；nearest P50/P90
  `25.331/99.603 px` 差于 `15.490/75.309 px`；oracle AEPE/PLR 为 `48.541 px/90.59%`；
  G1 FAIL，G2 未晋级。
- 失败集中在哪些样本或条件：top128 mean `58.131`、P10/P50/P90 `29/55/89`，略强于 E-009A
  top128，但 base2px `0.02999%`、oracle AEPE `23.138 px`、PLR `95.54%`，说明扩大候选仍不能
  形成合格候选生成器。
- 已尝试的补救措施：按冻结协议 G1 失败即停止，未启动 E-010B、E-009B、E-008B、rescue、完整
  sparse affine feedback、OS424/confirm 或多 seed 追认。
- Oracle 或上限实验：E-008A-STAGE-DIAG 的 S1 region union mean correct `196.686`，说明 S1 内仍
  有大量可恢复空间；E-010A 没能利用 cached raw/structure/local evidence 把正确点排进 top64。
- 判断为方法失败而非实现失败的证据：reviewer 确认 score 在 top64 前使用，full-union Gaussian
  listwise loss 与 hard-negative auxiliary 已执行，GT 仅用于训练标签和后置评估，trace/provenance
  与资源检查通过。
- 失败边界：只否定当前使用 E-008A cached raw/structure/local tail 的 strong retriever；不否定从
  512 原图/SAR 重新提取多尺度 patch descriptor、更强跨模态预训练/backbone、或改变 S0/S1 候选
  生成的路线。
- 未来可以重启该方向的条件：必须真正改变 observation source，例如 raw-image patch descriptor
  backbone 或外部跨模态预训练；不得继续只在 cached tail 上更换 loss、调 hard negative 或追认 top128。
- 完整日志和结果路径：`results/E-010A/RESULT_CARD.md`；远端
  `experiments/sparsegeo/runs/E-010A/run-01`。

### F-011：v1.4 raw-image patch descriptor top64 retriever

- 对应方法版本：`M001 v1.4`
- 对应实验编号：`E-011A`
- 日期：`2026-08-10`
- 原始假设：从 official training archive 的原始 `512x512` optical/SAR 图像重新提取多尺度 patch
  descriptor，在固定 E-008A S0/S1 matched-region union 内检索，显著超过 frozen S2 top64。
- 实现版本和审阅报告：E-011A source SHA
  `dfd7fe1b13da5beb875aac878c439c1f7e8ad25b50faa136af53ccba88ccaca1`；baseline isolation
  audit `PASS`，targeted reviewer `APPROVE`。
- 数据与 baseline 复核：复用 group-isolated `1607/404` train/dev；GDROS baseline source、
  checkpoint、cache、run/results 只读隔离；`uses_raw_images=true`、
  `uses_e008a_cached_tail_main_input=false`；无 OS424/confirm/datum labels、hidden `4096x4096`、
  full-image blind search、dense flow/GRU 或 fallback。
- 主要负面结果：top64 mean/P10/P50/P90 为 `45.879/20/44/73.7`，低于 frozen S2
  `50.015/26/48/78`；base2px `0.04734%` 低于 `0.05161%`；nearest P50/P90
  `19.460/82.972 px` 差于 `15.490/75.309 px`；oracle AEPE/PLR `24.464 px/95.30%`；G1 FAIL，
  G2 未评估。
- 失败集中在哪些样本或条件：top128 mean/P10/P50/P90 为 `68.710/34/66/106`，超过 E-006A
  mean/P10/P50 数量线，nearest P50/P90 `13.129/70.895 px` 优于 frozen S2；但 oracle AEPE/PLR
  仍为 `14.249 px/97.28%`，说明正确候选进入较深排名但几何分布/质量不足以支撑仿射。
- 已尝试的补救措施：按冻结协议 G1 失败即停止，未启动 E-011B、rescue、完整 sparse affine
  feedback、OS424/confirm 或 top128 追认。
- Oracle 或上限实验：E-008A-STAGE-DIAG 的 S1 region union mean correct `196.686`；E-011A top128
  已显示 raw patch descriptor 能从 S1 中恢复更多正确点，但还不能形成合格 top64 或 oracle affine。
- 判断为方法失败而非实现失败的证据：reviewer 确认原图 patch 路径、GT 隔离、matched-region-only
  retrieval 和 baseline isolation；watchdog 正式 train+dev404 成功。
- 失败边界：只否定当前 raw-image patch descriptor 的 top64 ranking/budget配置；不否定 raw patch
  evidence 本身，因为 top128 有明确正信号；不否定 geometry-aware compression、分布约束选择、
  或更强预训练 descriptor。
- 未来可以重启该方向的条件：必须针对 top128 中正确点“数量足够但仿射不可用”的原因提出新诊断或
  几何感知选择机制；不得仅把 top128 作为 top64 失败后的追认。
- 完整日志和结果路径：`results/E-011A/RESULT_CARD.md`；远端
  `experiments/sparsegeo/runs/E-011A/run-01`。

### F-012：v1.5 deterministic no-training geometry-only top128 compression

- 对应方法版本：`M001 v1.5`
- 对应实验编号：`E-012A`
- 日期：`2026-08-10`
- 原始假设：E-011A top128 中已有足够且几何可压缩的正确候选，可通过无 GT 的确定性 geometry-aware
  compression、多个 affine basin、residual feedback、coverage/condition/duplicate 约束，得到动态高纯
  sparse active set。
- 实现版本和审阅报告：`results/E-012A/reviewer_conclusion.md` 为 `APPROVE_WITH_RESULT_FAIL`；
  G0 中 E-011A top128 parity、candidate-before-label、one-source-one-target、dynamic active size、
  no GT inference、no hidden N2/no RANSAC/131k 和 unit tests 均 PASS。
- 数据与 baseline 复核：复用 group-isolated dev404；E-011A top128 mean/P10/P50/P90 复现为
  `68.710/34/66/106`，top128 oracle AEPE 复现为 `14.249 px`；GDROS/E-011A read-only isolation PASS。
- 主要负面结果：G1 FAIL。active mean/P10/P50/P90 为 `327.723/218.6/294/497.7`，coverage pass
  `100%`；但 correct@2px mean/P10/P50/P90 仅 `2.861/0/0/7`，precision mean/P10/P50/P90 仅
  `0.673%/0/0/1.827%`。G2 未运行。
- 失败集中在哪些样本或条件：active 数量和空间覆盖并非瓶颈；失败集中在大量错误但几何自洽的
  basin 被选中，正确点虽存在于 top128 但未被无 GT 几何残差信号识别。
- 已尝试的补救措施：按冻结协议 G1 失败即停止，未启动 G2、E-012B、完整 sparse affine feedback、
  descriptor 重训、OS424/confirm 或 rescue。
- Oracle 或上限实验：E-011A-TOP128-GEOM-DIAG 显示 GT-guided compression-oracle-ready 为
  `401/404`；all-hard2px LS P50/P90/P95 AEPE `0.299/0.533/0.617 px`，GT-endpoint oracle P50/P90/P95
  约 `0.00004/0.00006/0.00007 px`。
- 判断为方法失败而非实现失败的证据：G0、自动测试、read-only isolation、no-GT/no-N2/RANSAC 负向测试、
  one-source-one-target 和 reviewer 均通过；负面结果由正式 dev404 G1 得到。
- 失败边界：只否定 deterministic no-training geometry-only top128 compression；不否定 E-011A top128
  evidence、learned/appearance-conditioned inlier scoring、多模态局部描述子重训、或将外观证据与几何
  basin 联合训练的 sparse affine feedback。
- 未来可以重启该方向的条件：必须引入能区分“几何自洽但跨模态错误”的新观测或监督，例如
  appearance-conditioned inlier likelihood、learned basin scoring、contrastive descriptor retraining、
  或用 GT/teacher 训练但 inference 无 GT 的候选纯化；不得仅调 residual threshold、basin 数、轮数或
  coverage/condition 权重作为 rescue。
- 完整日志和结果路径：`results/E-012A/RESULT_CARD.md`；远端
  `experiments/sparsegeo/runs/E-012A/run-01`。

### F-013：v1.6 轻量 geometry-supervised learned inlier-basin scorer

- 对应方法版本：`M001 v1.6`
- 对应实验编号：`E-013A`
- 日期：`2026-08-10`
- 原始假设：用 raw truth_flow 构造 endpoint、basin 和 affine 几何监督，训练轻量 learned
  inlier/basin scorer，可在推理无 GT 的条件下从 E-011A top128 candidates 中选出高纯动态 active set，
  避免 E-012A 的错误自洽 basin。
- 实现版本和审阅报告：`results/E-013A/reviewer_conclusion.md` 为 `APPROVE_WITH_RESULT_FAIL`；G0
  中 E-011A top128 parity、E-012A baseline parity、label construction audit、candidate-before-label、
  no GT inference、loss terms nonzero、gradient to scorer/basin heads、feedback_changed、
  one-source-one-target、dynamic active size、no hidden N2/no RANSAC/131k 均 PASS。
- 数据与 baseline 复核：复用 group-isolated `1607/404` train/dev；E-011A top128 parity PASS；
  E-012A deterministic baseline parity PASS；GDROS/E-011A read-only isolation PASS。
- 主要负面结果：G1 FAIL。active mean/P10/P50/P90 为 `0.658/0/0/2`，correct@2px 为
  `0.017/0/0/0`，precision 为 `1.246%/0/0/0`，coverage pass `0%`。G2 未运行。
- 失败集中在哪些样本或条件：训练后使用 train-calibrated threshold `0.9699785920`，dev active set
  几乎为空；从 E-012A 的“大规模错误自洽 basin”转为过度 abstention/generalization failure。
- 已尝试的补救措施：按冻结协议 G1 失败即停止，未启动 G2、E-013B、rescue、descriptor retraining、
  sparse affine feedback、OS424/confirm。
- Oracle 或上限实验：E-011A-TOP128-GEOM-DIAG 显示 top128 中 GT-guided compression-oracle-ready
  为 `401/404`，all-hard2px LS P50/P90/P95 AEPE `0.299/0.533/0.617 px`；E-012A 证明纯几何会误入
  错误自洽 basin。
- 后续诊断：E-013A-CALIB-DIAG 判读 `NO_USABLE_RANKING_SIGNAL`。final score train/dev AUROC
  `0.5036/0.5024`、AP `0.000477/0.000471`，threshold/topK/quota sweep 均远低于 G1；best dev
  threshold active `615.20`、correct `2.01`、precision `0.298%`，best dev global topK-256 active
  `235.91`、correct `0.98`、precision `0.395%`。pre-feedback endpoint score 有弱信号
  `AUROC/AP=0.7019/0.00256`，但 post-feedback/final score 近随机，basin-only 反向/无用。
- 判断为方法失败而非实现失败的证据：G0 全部关键协议、label/inference separation、loss/gradient、
  feedback、one-source-one-target、no-N2/RANSAC 和 reviewer 均通过；负面结果由正式 dev404 G1 得到。
- 失败边界：只否定当前 `10,630 params + 1 epoch + train-calibrated high threshold` 的轻量
  candidate/basin scorer 配置及其 final feedback/scoring path；不否定 learned geometry supervision、
  appearance-conditioned endpoint scoring、更强模型/训练、或 descriptor retraining。
- 未来可以重启该方向的条件：必须保护或强化 pre-feedback endpoint signal，并重新设计几何反馈/盆地
  分数的接入方式；不得仅降低阈值、延长训练、调 calibration 或把 over-abstention 追认为成功。
- 完整日志和结果路径：`results/E-013A/RESULT_CARD.md`；远端
  `experiments/sparsegeo/runs/E-013A/run-01`。

### F-014：v1.7 E-011A top128 上的手工/轻量跨模态信号排序

- 对应方法版本：`M001 v1.7 diagnostic`
- 对应实验编号：`E-014A`
- 日期：`2026-08-11`
- 原始假设：在 E-011A top128 candidates 中，raw/log-SAR patch、gradient/edge、RIFT/HOPC-style、
  census/rank/self-similarity、structure tensor/cornerness 等手工或轻量结构信号，可能比 E-013A
  final score 更好地排序 endpoint-correctness，为后续 purification 提供 observation。
- 实现版本和审阅报告：`results/E-014A/feature_reviewer_conclusion.md` 与
  `results/E-014A/evaluator_reviewer_conclusion.md` 均为 `APPROVE`；P0 sanity controls 全部 PASS。
- 数据与 baseline 复核：E-011A top128 parity PASS，candidate count `78,303,360`，correct@2px
  proposals mean/P10/P50/P90 `68.710/34/66/106`，oracle affine AEPE `14.249`。candidate-before-label
  join、optical source -> SAR target 方向、stride-8 token/pixel 转换、patch center/radius/padding、
  same-image identity、known-shift、gradient/RIFT/structure tensor toys、shuffled target、swapped direction
  和 random score controls 均 PASS。
- 主要负面结果：判读 `COMPLETED / NO_SIGNAL_FOUND`。best non-reference signal 为
  `census_rank_self_similarity/local_self_similarity`，dev AUROC/AP `0.531545/0.000443655`，best
  selection `region_quota=8` 的 correct mean `0.212871`、precision mean `0.000866667`。
- 失败集中在哪些样本或条件：所有测试信号在 E-011A top128 candidate-level endpoint ranking 上接近
  random 或明显弱于 E-013A pre-feedback endpoint reference；低 AP 与 topK/quota correct 表明无法转化为
  可用 candidate purification。
- 已尝试的补救措施：本实验按协议只做只读诊断，未训练、未 finetune、未接几何 feedback、未启动
  E-014B/rescue/descriptor retraining/sparse affine feedback。
- Oracle 或上限实验：E-011A-TOP128-GEOM-DIAG 显示 top128 中 GT-guided compression-oracle-ready
  为 `401/404`，说明正确候选存在；E-014A 说明这些手工/轻量信号无法无 GT 识别它们。
- 判断为方法失败而非实现失败的证据：P0 强 sanity、negative controls、candidate-before-label、
  两路独立 reviewer 均通过；负面排序结果不能归因为 patch 错位、方向反、label 泄漏或 evaluator 错误。
- 失败边界：只否定在现有 E-011A top128 candidates 上堆 raw/log-SAR/gradient/RIFT/census/structure
  这类手工或轻量 signal scorer；不否定跨模态预训练 descriptor、foundation/remote-sensing VFM 特征、
  区域级语义/结构重生成候选、重新训练 candidate generator 或改变 observation source。
- 未来可以重启该方向的条件：必须引入语义不同且更强的 observation source，例如预训练跨模态
  descriptor、foundation feature、区域级语义匹配后重新生成候选，或端到端重训候选生成器；不得仅在
  相同 top128 上继续堆类似 hand-crafted patch/structure score。
- 完整日志和结果路径：`results/E-014A/RESULT_CARD.md`；远端
  `experiments/sparsegeo/runs/E-014A/run-01`。

## 记录模板

### F-xxx：方法名称

- 对应方法版本：
- 对应实验编号：
- 日期：
- 原始假设：
- 实现版本和审阅报告：
- 数据与 baseline 复核：
- 主要负面结果：
- 失败集中在哪些样本或条件：
- 已尝试的补救措施：
- Oracle 或上限实验：
- 判断为方法失败而非实现失败的证据：
- 失败边界：
- 未来可以重启该方向的条件：
- 完整日志和结果路径：

## 禁止事项

- 不删除失败记录来美化研究过程。
- 不把数据协议不一致记录为方法失败。
- 不把显存不足、训练中断或未收敛直接记录为方法失败。
- 不因一个实现版本失败而概括整个方法族无效。
