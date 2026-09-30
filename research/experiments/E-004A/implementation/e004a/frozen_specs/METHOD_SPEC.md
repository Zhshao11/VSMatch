# 当前方法规范

方法编号：`M001`

暂定名称：`SparseGeo-GDROS`

版本：`0.7`

状态：`ADAPTIVE_ANCHOR_APPROVED / E-004A READY`

## 0. 版本变更

`v0.7` 根据 E-003A 的 G1 失败和用户批准，将研究焦点前移到 source anchor acquisition。
保留冻结 GDROS、stride-8 特征和 cosine top8，但不再直接在固定 K128 上训练联合
top8+dustbin selector。新方法先从按区域固定生成的 M256 proposal pool 中，以粗跨模态响应预测
recoverability；每个区域设置保底和封顶数量，动态分配 K128 primary，并允许最多 R64、同区域、
单轮 reserve replacement。E-004A 只判断 anchor acquisition 是否提高 recoverable 对应数量和
GT-oracle 仿射上限，不混入 RG-SAC、新 descriptor 或局部精修。

`v0.6` 根据 E-002B-ACC-DIAG 的完整 424 对机制诊断和用户批准，永久停止固定六键 B2
selector 及 131,072 hypothesis 枚举，但保留已证明上限充分的冻结 GDROS descriptor、K128
anchors 和 cosine top8 candidate set。新主线以共享小型网络学习候选权重，经过两轮 FP32
加权仿射和残差反馈得到变换，同时预测失败置信度。selector 复杂度限制为 `O(TNK)`，GDROS
feature extractor 完全冻结。E-003A 只使用官方 training archive 的隔离 train/dev；官方
424 testing 已参与前序决策，只能作为未来 secondary legacy benchmark。最终确认必须另行
冻结 E-003B 并绑定未参与项目决策的外部光学-SAR测试集。

`v0.5` 根据 E-002B 正式 G5-R 结果新增 `E-002B-R equivalent-backend rescue`。本版本不改变
B2-bridge 的任何候选、假设、排序、重指派或拟合语义，只将 v0.4 中由 Python 循环、大量
小张量分配和 kernel launch 主导的几何后端改为等价的向量化、融合或编译实现。必须先通过
reference/optimized 早期离散轨迹等价门槛，再通过新的资源资格审查和 424 对无 GT 全轨迹
等价；全部通过后才允许补跑
原 v0.4 被 G5-R 拦截的 424-pair accuracy。该后端救援是工程资格验证，不作为方法创新。

`v0.4` 根据 E-002A 正式结果选择 `B2 candidate-level geometric reranking`，但明确登记为
`B2-bridge` 救援探针，而不是把 E-002A 的 `G3=FAIL` 追认为通过。主路径冻结 K128、
post-Transformer 特征和无训练边界，以 cosine top8 一对多候选进行 source-aware 仿射假设
搜索、几何一致性选择、候选重指派和普通 LS。IRLS 延后为独立实验，避免掩盖候选重排是否
真正有效。正式 B2 accuracy 前必须先完成新的资源复核并解除
`EFFICIENCY_HOLD`；未解除时不得进入 E-002C。

`v0.3` 根据 E-001 正式结果锁定 E-002A：默认预算为 `K=128`，原生 `1/8` raw-dot、
hard argmax 和普通 LS 是主探针；其余分数、解码和分辨率方案均为预注册诊断。新增分层
精度、候选可恢复性、可见性和资源门槛，防止最简单解码器失败后错误否定整个方法。

`v0.2` 修正公开 GDROS 测试协议的 LSR 点池和评测 mask 描述。核心方法假设不变，但
E-001 的主对照必须从公开代码使用的固定点池内采样；全图覆盖实验单独标为 diagnostic。

## 1. 研究假设

OS 光学-SAR 配准的最终目标是六参数全局仿射。GDROS 先计算全对全相关，再通过多次 GRU
更新预测高自由度稠密流，最后将其投影到仿射空间，存在任务自由度与计算复杂度不匹配。

本方法研究：能否用少量全局分布且具有仿射可辨识性的对应点，直接估计粗仿射，再通过
几何引导的局部亚像素搜索保持精度，同时降低显存和推理耗时。

## 2. 目标结构

```text
跨模态特征
  -> 仿射可辨识的分布锚点
  -> K-to-all 全局搜索
  -> 置信度加权鲁棒粗仿射
  -> 粗仿射 warp
  -> 分布点局部亚像素精修
  -> 第二次鲁棒仿射
  -> 解析生成最终仿射 flow
```

核心要求是同时删除完整相关金字塔和迭代式稠密流更新。只压缩相关矩阵但保留 32 次 GRU
不能满足低延迟目标。

## 3. E-000：正式 baseline 校验

在任何方法结论之前，使用本地已下载的官方 GDROS `testing.zip` 校验 checkpoint 和评测
协议。记录文件大小、SHA256、解压测试、远端目标路径、实际样本数和完整指标。

本地文件：

`/Users/shihao/Downloads/os_dataset/testing.zip`

已观测大小：`1,973,635,350` bytes。

当前重新生成 OS 数据的结果只能用于诊断，详见 `REPRODUCTION_STATUS.md`。

E-000 已在完整官方 ZIP 的独立解压目录上正式完成：AEPE `1.461230444`，CMR@1/2/5 为
`33.726415/81.603774/99.528302%`，424 对完整评测通过，且与此前等价物化预检的最大逐对
差为 `0.0`。正式归档和结果绑定见 `research/results/E-000/formal-official-archive-run-01.md`。

## 4. E-001：稠密预测流的稀疏仿射充分性

目的：隔离验证“估计六参数仿射是否真的需要全部稠密 flow”。本实验不改变 GDROS 特征、
CorrBlock 或 GRU，只替换最终仿射拟合所使用的点集。

### 变量

- `K = 16, 32, 64, 128, 256`
- 主对照 `exact-public-pool`：只从公开 LSR 的固定 200,000 点池内进行一维等间隔、
  空间分区覆盖和固定随机采样
- 覆盖诊断 `full-image diagnostic`：从完整 `512 x 512` 图像进行边界网格或分区中心采样，
  用来测量覆盖区域变化，不能与主对照混合汇总
- `GT affine oracle`：在相同索引上采样 GT flow 并拟合仿射，只验证数值上限和协议正确性，
  不能作为方法结果
- 置信度选点不进入 E-001 第一版。后续追加时不得使用测试 GT；优先尝试最后两次 flow 的
  迭代稳定性、最终相关峰值差或其他可复现的无监督可靠性量，并单独报告定义
- 随机采样至少 5 个种子
- 仿射求解首先严格复用 baseline LS；WLS/IRLS 属于后续独立消融

### 公开协议事实

- `512 x 512` flow 按 row-major 展平。
- 公开 LSR 固定使用索引 `[10000:210000]`，即 200,000 点，并非全部像素或全部有效点。
- 公开最终 EPE 不使用 valid mask，在完整 `512 x 512` 输出上计算，黑边同样参与评测。
- 该固定切片具有不对称空间覆盖，因此“减少点数”和“改变覆盖区域”必须分别归因。

### 不变量

- 使用相同 checkpoint、输入、预处理和评测脚本。
- 稠密 flow 与抽样 flow 使用同一坐标方向和像素单位。
- 新拟合器使用完整 `[10000:210000]` 公开点池时，必须在规定数值容差内复现原始 LSR。
- baseline parity 的最终 EPE 必须与公开代码一样不使用 valid mask，并包含黑边。
- 方法内部若使用黑边或有效区域 mask，必须作为明确的方法组件单独消融；评测端仍保持
  公开无 mask 协议，不能把 mask 改动隐藏在 baseline 对照中。
- 不允许根据测试 GT 选择采样点或置信度阈值。

### 进入 E-002 的建议门槛

- `exact-public-pool` 中至少一种非 GT 的 `K=64` 或 `K=128` 策略，相对原公开 LSR 的
  AEPE 增量不超过 `0.05 px`。
- CMR@1 下降不超过 `1` 个百分点。
- 多随机种子或不同空间分区下结果稳定。

门槛用于决策而不是选择性报告。若 `exact-public-pool` 失败但 `full-image diagnostic`
通过，应判定覆盖位置而非点数主导，并重新设计锚点约束；若 GT oracle 也失败，应优先检查
拟合器、坐标协议和数据标签。未达到门槛时必须输出误差分组和覆盖/条件数分析。

E-001 已正式完成：14/14 个 K64/K128 候选通过。K128 stratified 的 AEPE 增量为
`+0.001100 px`，五个随机种子为 `+0.002345 +/- 0.002227 px`。这证明 128 个可靠分布点
足以保留仿射拟合精度，但没有证明冻结特征的 K-to-all 检索已经可行，也没有产生端到端
资源收益。

## 5. E-002：跳过稠密流的 K-to-all 仿射估计

目的：验证 GDROS checkpoint 的跨模态特征能否在不运行完整 CorrBlock 和 GRU 的情况下，
直接提供足够的稀疏对应点。

### E-002A 固定主路径

1. 先使用 checkpoint 已有的特征和分辨率，不立即更换 backbone，也不混入新特征训练。
2. 主 anchor 为 E-001 `public_stratified` 坐标到 stride-8 token 的确定性最近映射；
   `K=128` 默认，K64/K256 为 Pareto 对照。全图均匀 anchor 只作 coverage diagnostic。
3. optical 为 source，SAR 为 target，flow 为 `target-source`。原生 token `(xf,yf)` 对应
   像素 `(8xf,8yf)`。
4. 主分数严格复用 CorrBlock：`q^T k / sqrt(256)`；主解码为 hard argmax，主求解为
   归一化坐标上的 FP32 普通 LS。
5. 必须在选 source anchor 后才计算 `K x N`；target chunk 默认为 1024，禁止先构造
   `N x N` 再截取。
6. 主路径不使用 GT、valid mask、置信度、双向一致性或黑边过滤。最终 AEPE/CMR 保持
   公开全图无 valid-mask 评测。
7. 最终仅输出仿射矩阵；dense-contract 由矩阵解析生成 `512 x 512` flow，不能重新启用
   CorrBlock、context encoder 或 32 次 GRU。

### E-002A 预注册诊断

- cosine + hard argmax：独立检查 token norm hubness。
- raw-dot + `5 x 5` local soft-argmax：以 hard top1 为中心，`tau=1.0`。
- cosine + `5 x 5` local soft-argmax：`tau=0.1`。
- `K-anchor dual normalization`：
  `softmax_target(S/1.0) * softmax_anchor(S/1.0)` 后 hard argmax，只在 K128 运行；不得称为
  完整 dual-softmax。
- `1/16 decimation` 和 `1/16 average-pool`：只在 K128、raw-dot、hard argmax 下运行；二者
  都发生在 Transformer 之后，不能声称节省了 feature forward 计算。
- GT 只允许用于连续坐标 oracle、nearest-lattice oracle、可见性、rank/recall、top-k
  candidate oracle 和未来局部窗口捕获率分析，不得进入正式匹配或求解。

详细实施定义、门槛和决策树见 `research/experiments/E-002A.md`。

### E-002A 正式结论与分支决策

E-002A 状态固定为 `COMPLETED / EFFICIENCY_HOLD`：G0 通过，G1-G5 均失败。该结论不得
因后续实验修改。尤其是 cosine 的 `R@8,r1 / R@32,r1 / R@64,r2` 为
`26.3970 / 46.9063 / 79.4373%`，未达到 G3 的 `40 / 65 / 80%`，所以不能声称 G3 通过。

四个后续方向按冻结决策树解释如下：

- B1 不进入：top1 recall、置信度 AUROC 和 top32 precision 均不足。
- B3 不进入：in-bounds 诊断对 PairLocalReady 的提升为 `0 pp`。
- learned descriptor 暂不进入：top64 select-and-reject oracle 的 PairLocalReady 仍为 `100%`，
  冻结特征判负三条件没有同时成立。
- 选择 B2-bridge：top8 select-and-reject oracle 的 raw/cosine PairLocalReady 已达
  `99.2925 / 99.5283%`，top32 为 `100 / 100%`，说明候选集合具有上限；下一步只检验
  无 GT 的候选级仿射一致性是否能逼近该上限。

该选择是一次有止损线的救援实验，不是对 G3 门槛的事后放宽。完整公式、固定预算、资源
复核和停止条件见 `research/experiments/E-002B.md`。

### E-002B 正式结论与 v0.5 决策

E-002B/B2-bridge 的固定 v0.4 Python/PyTorch 后端已完成，终态为
`COMPLETED / EFFICIENCY_HOLD / G5-R FAIL`。其 p50/p95 为 `888.752/911.972 ms`，约为
同次 baseline 的 `11.67x`；峰值 allocated `276.455 MiB` 和无 hidden cross-image N2 检查
通过。主要瓶颈是 131,072 个假设的生成与 coarse consensus，而不是 K-to-all candidate
matching。由于 G5-R 是 accuracy 硬前置，官方 424-pair accuracy 未运行。

该资源结果只否定 v0.4 固定执行后端的资源资格。随后 E-002B-ACC-DIAG 已补充证明固定 B2
selector 精度同样失败：C8-main AEPE `10.4349 px`、PairLocalReady `75.71%`，而 cosine top8
GT oracle AEPE `1.4792 px`、PairLocalReady `99.53%`。因此固定 B2 selector 和全预算等价后端
均永久停止；该失败不外推到 candidate set、冻结 GDROS descriptor 或 SparseGeo 方法族。

v0.5 批准唯一下一步 `E-002B-R`：完整保留 K128 cosine top8、256 个三点组、全部
`256 x 8^3 = 131,072` hypothesis ID、BIND-001、32 sentinels、Q1024、R32、两轮 ordinary
LS 和 no-fit final validation，只重构等价执行后端。优化后端必须先与 v0.4 reference 在
synthetic、固定 smoke 和固定无 GT 官方输入上通过早期轨迹等价审计，再通过 p50/p95、
跨进程稳定性、显存和复杂度门槛；最后在全部 424 对、四个 variant 上通过无 GT 全轨迹
等价，才能运行 accuracy。具体绑定见 `research/experiments/E-002B-R.md`。

### 后续分段

- `E-002A`：现有特征、分区 anchor、K-to-all、普通 LS。
- `E-002B/B2-bridge`：已完成并停止。固定 selector 的资源与 accuracy 均失败。
- `E-002B-R`：已完成并停止。P0-early PASS，但 p50 `205.06 ms`、speedup `4.37x` 未通过 P1。
- `E-002B/B3`：主要问题来自图外或黑边时，加入 dustbin、可见性和越界点降权。
- `E-002C`：仿射反馈的局部亚像素精修。

无训练探针已经完成。当前批准分支为 M001 v0.6 / E-003A learned selector；详细结构、数据
隔离、梯度测试、消融与 gate 见 `research/experiments/E-003A.md`。不得加入局部精修、descriptor
微调、baseline fallback 或 hypothesis 枚举；禁止依据 OS 424 或未来 confirm 数据选择超参数。

## 5.1 E-003：轻量几何候选选择

E-003A 固定使用 K128 x cosine top8 稀疏候选。候选 MLP 只读取无 GT 的相似度、rank/margin、
坐标/位移、descriptor norm 和 target multiplicity；每个 anchor 另有 dustbin。主配置执行两轮
soft candidate weighting -> FP32 normalized weighted affine -> residual feedback，最终 confidence
head 预测 grid error，并以固定 5 px 阈值 abstain。GDROS feature extractor 全冻结且 no-grad，
selector 参数少于 100k，不允许任何 N2 correlation 或三点 hypothesis 维度。

E-003A 保持 official archive 的 `512x512` 图像，不 resize、crop、flip 或调用 flow augmentation；
stride-8 特征固定为 `64x64`。监督与指标的唯一权威为 raw `truth_flow/*.flo`，方向
`target-source`；`datum/*.npy` 不进入 candidate、affine、loss、confidence 或指标标签。anchor
标签在 `(8xf,8yf)` 整数像素直接读取 flow。16x16 grid loss/confidence 使用 bilinear
`grid_sample`、`align_corners=True`、zero padding；公开 AEPE/CMR 则解析生成完整 512x512 affine
flow 后与 raw truth_flow 逐像素计算 Euclidean endpoint error，不插值、不使用 mask、不裁黑边。

数据上，官方 `training.zip` 2,011 对按 decoded optical SHA 分组并固定哈希划分 80/20
train/dev。E-003A 只允许在 dev 上 early stopping 和报告 feasibility。OS 424 不进入本实验；
E-003A 通过后才可申请一次性 E-003B。E-003B 的 primary confirm 必须是未参与本项目方法决策
且在打开标签前完成 archive、split、源码、配置、checkpoint、阈值和 evaluator 冻结的外部
光学-SAR测试集。

E-003A 同时要求：dev AEPE `<=3 px`、PairLocalReady `>=90%`、grid mean `<=8 px`、固定 90%
coverage selective risk 至少降低 `20%`；selector-only p50/p95 `<=5/8 ms`，slim end-to-end
p50/p95 `<=15/20 ms`，相对同进程 GDROS p50 speedup `>=5x`，峰值 allocated `<=284 MiB`。
任一 gate 失败即停止，不得通过测试集调参或事后减少 K/top-k 救援。

## 5.2 E-004：覆盖约束的自适应 Anchor 预算

E-004A 固定使用 official `512x512` 输入和冻结 post-Transformer `64x64` 特征。source map 划分
为 `8x8` 个区域，每个区域以 `2x2` 子区域中心产生 4 个确定性 proposal，共 M256。SAR 特征以
`4x4` average pooling 得到 `16x16` 粗 target；M256 proposals 只对该粗 target 计算 cosine，
由轻量 pre-search head 预测 recoverability。

每个区域先选 1 个 primary，剩余 64 个名额按预测分数全局分配，但每区最多 3 个，最终严格
K128。未选 proposal 构成 reserve。对 primary 计算完整 `128x4096` cosine top8 后，post-search
verifier 使用 margin、entropy、collision 和粗/细一致性判断失败点；最多 R64 个失败点可由同区域
下一 reserve 替换，每个 primary 最多替换一次。推理不得使用 GT，最坏 fine-search 为
`192x4096`。

E-004A 的训练与评估继续使用 E-003A 已披露为 development 的 group-isolated train/dev；OS 424
和未来 external confirm 仍禁止访问。正式输出只评价 anchor recoverability、覆盖、condition、
replacement gain 和使用 GT-nearest top8 target 的 oracle affine；该 oracle 只隔离 anchor acquisition，
不能作为可部署匹配精度或端到端方法结果。

E-004A 先执行 proposal-pool oracle gate；无足够上限则不训练。通过后才训练 pre/post heads，并
执行固定资源 gate。不得加入 sparse graph、RG-SAC、hypothesis enumeration、descriptor 微调、
局部精修或 baseline fallback。详细绑定见 `research/experiments/E-004A.md`。

## 6. 计算与实现约束

以下 `400x400` 仅为历史 GDROS 论文训练配置的相关量估算，不是 E-003A 活动输入协议。
E-003A 的活动协议固定为 `512x512`、stride-8 `64x64`、target `N=4096`，禁止实现自行
resize/crop。历史 `400 x 400`、`1/8` 特征下 `N=2500`，GDROS level-0 相关为 `6.25M` 个分数；
四层相关金字塔约 `31.52 MiB/样本` FP32。推荐的 `1/16` 原型使用 `64 x 625` 全局匹配
和 `256 x 81` 局部匹配，共约 `60,736` 个分数。

该比例只描述相关项，不能直接作为端到端加速或整网显存下降结论。特征 Transformer 和
其他激活必须单独测量。

局部相关禁止直接生成完整 `C x K x window_area` 的大型 `unfold` 中间张量。使用分块、
逐偏移或经过验证的融合实现，并用峰值显存测量证明没有隐藏展开。

## 7. 精度与效率报告

除 GDROS 原始 AEPE 和 CMR 外，至少报告：

- P95/P99 和失败率；
- 参数量与理论匹配规模；
- batch-1 延迟、吞吐和峰值推理显存；
- 可训练时的峰值训练显存和训练时间；
- `K` 与精度、显存、延迟的 Pareto 曲线；
- 遮挡、散斑、重复纹理、黑边和大变换子集结果。

同一硬件比较必须固定输入、精度模式、warm-up、重复次数和 CUDA 同步方式。

## 8. 当前非创新型工程基线

以下优化应作为强工程对照，但不能单独作为论文贡献：

- 删除重复执行但未被使用的 context encoder 调用；
- 删除 forward 未使用的模块；
- 测试时只保留最终 flow，而不是全部迭代结果；
- 特征和相关使用 AMP，仿射求解保留 FP32；
- 使用 SDPA/FlashAttention 降低显式 attention 激活。

## 9. 方法创新边界

“稀疏点加加权最小二乘”不足以构成核心创新。暂定主线为：跨模态可靠性与仿射可辨识性
联合驱动的预算化对应点获取，加上几何不确定性停止条件和仿射反馈局部精修。

E-002B 的固定预算候选共识本身仍是可行性后端，不单独主张创新。StereoGlue 的一对多候选
与模型引导重匹配、GC-LAC 的局部仿射一致性图聚类均构成直接 prior。只有后续证明
“跨模态候选获取、仿射可辨识预算、几何反馈和停止条件”形成不可替代的整体，并在精度与
资源上同时优于强工程对照，才进入论文贡献判断。
