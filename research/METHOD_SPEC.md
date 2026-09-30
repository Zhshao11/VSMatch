# 当前方法规范

方法编号：`M001`

暂定名称：`SparseGeo-GDROS`

版本：`3.7-design`

状态：`E028B_NATIVE_MATCHER_PROPOSED / E-032A WAITING_USER_APPROVAL`

## 0. 版本变更

`v3.7 design` 根据 E-031A 诊断和用户决策停止 E028B-to-GDROS-backend adapter 路线。E-031A 显示
E-028B feature-only retrieval top32 `85.93%` 明显强于 GDROS native `23.77%`，但 GDROS backend
interface `[B,256,64,64]` 与 E-028B feature `[B,1024,32,32]` 不兼容，simple bridge AEPE `18.6692`、
CMR@2 `1.1656%`。下一步拟定 `E-032A-E028B-NATIVE-MATCHER-DIAG`：固定 E-028B 特征，设计 native
matcher，包括 mutual/dual-softmax/region-balanced matching、local refinement 和 affine consensus；
不训练新 descriptor，不复用 GDROS dense GRU/backend。详细绑定见
`research/experiments/E-032A-E028B-NATIVE-MATCHER-DIAG.md`。

`v3.6 diagnostic` 根据用户要求，将 E-031A full integration 暂缓，先执行
`E-031A-FEATURE-INTERFACE-DIAG`：不跑 120k，只在 official training.zip 固定 probe/small-overfit subset
上比较 GDROS native backend-interface feature 与 E-028B safe dual adapter feature 的分布差、后端兼容性
和小样本 bridge/backend overfit 能力。testing.zip 不参与选择或成功判读。详细绑定见
`research/experiments/E-031A-FEATURE-INTERFACE-DIAG.md`。

`v3.6 design` 根据 E-030A 结果固化后续 GDROS-style full training 参考 recipe：official training.zip
2011 训练、testing.zip 424 final only、batch `12`、OneCycle max LR `1.25e-4`、120k steps、train/test
iters `12/32`、public augmentor、public `flow_loss + mat_flow_loss`。下一步拟定
`E-031A-VALIDATED-RECIPE-E028B-INTEGRATION`：复制 E-028B safe dual adapter frontend 到实验副本，与
GDROS backend 在该 validated recipe 下联合训练；原 E-028B/DINO/GDROS baseline artifacts 只读。该实验
只验证系统级接入与 interface diagnostics，不启动轻量 matcher 或 sparse feedback。详细绑定见
`research/experiments/E-031A-VALIDATED-RECIPE-E028B-INTEGRATION.md`。

`v3.5 diagnostic` 追加 baseline protocol sweep。E-029A-GDROS-PAPER-PROTOCOL-RETRAIN 已按 official
training.zip 2011 -> testing.zip 424 完成：AEPE `1.7014 px`、CMR@1 `23.5849%`、CMR@2
`71.2264%`、CMR@5 `99.2925%`，未加载官方已训练 GDROS 权重；released checkpoint replay 为
AEPE `1.4612 px`、CMR@1 `33.726%`、CMR@2 `81.604%`、CMR@5 `99.528%`。数据/测试污染不明显，
但 CMR@1/2 仍低约 10pp。E-030A 因此只做 baseline 协议 sweep：公开仓库默认 batch2/LR1.25e-4、
batch12/high-LR 以及 checkpoint 曲线诊断；testing.zip 曲线只解释 training dynamics，不得用于 checkpoint
selection 或成功追认。详细绑定见 `research/experiments/E-030A-GDROS-PROTOCOL-SWEEP.md`。

`v3.4 diagnostic` 追加官方协议重训审计。E-029A-GDROS-RETRAIN-DIAG 判读
`E029A_NEGATIVE_NOT_TRUSTWORTHY`：official/frozen GDROS replay 很强，dense AEPE `1.460 px`，
但 `COPIED_GDROS_SAME_BUDGET_RETRAIN` dev AEPE `53.779 px`。用户进一步要求按官方训练协议重新
训练 GDROS，且不要使用官方已训练 GDROS 权重。`E-029A-GDROS-OFFICIAL-RETRAIN` 因此先审计官方
README/train scripts/config，绑定 optimizer、schedule、epoch/steps、augmentation、loss、checkpoint
selection 与所有初始化来源；primary track `OFFICIAL_RECIPE_GDROS_RETRAIN_FROM_SCRATCH` 禁止加载官方
已训练 GDROS checkpoint，只在实验副本中按 official recipe 训练，并用 formal dev404 最终评价。该实验
决定后续 E-028B 接入应使用完整官方 recipe 还是先解决 GDROS training reproduction。详细绑定见
`research/experiments/E-029A-GDROS-OFFICIAL-RETRAIN.md`。

`v3.4 diagnostic` 根据 E-029A 和用户质疑，暂停对 E-029A 负结果的结论外推，先审计 copied
GDROS-style backend/trainer/evaluator 是否能对原 GDROS baseline replay/retrain 出强结果。E-029A 判读
`NO_SYSTEM_INTEGRATION_GAIN`：best track AEPE `21.916 px`，没有超过 E-028B fair reference
`21.418 px`；虽然 P0/G0/reviewer 通过，但“强后端接入后比简单 matcher 还差”是红旗。
`E-029A-GDROS-RETRAIN-DIAG` 因此用 E-029A 同一 copied backend/training harness/evaluator，换回 GDROS
frontend/features，执行 no-train replay sanity 与 same-budget copied GDROS retrain。若 copied GDROS 能
接近 full baseline，E-029A 负结果才可信归因为 E-028B/backend mismatch；若 copied GDROS 也很弱，
则 E-029A 负结果不能用于方法结论。详细绑定见 `research/experiments/E-029A-GDROS-RETRAIN-DIAG.md`。

`v3.4 prototype` 根据 E-028B 正结果和用户批准，从 feature-only 前端评价进入系统级接入。E-028B 判读
`SAFE_DUAL_ADAPTER_STRONG_GAIN`：safe dual adapter dev AEPE `21.418 px`、recall@2 `0.669`、
correct@2 `0.520`，相对 E-027A 单 SAR adapter 分别提升 `18.4%/17.6%/18.6%`，且 optical drift
`4.38e-7 <= 0.02`。用户提出第一步应把 GDROS 的 feature extraction 替换为 E-028B，并将替换后的
frontend 与 GDROS 后半段一起训练，以适应我们的 feature distribution，再与 baseline 全流程比较。
E-029A 因此复制/隔离 GDROS-style backend 到实验目录，用 E-028B safe dual-adapter frontend 替换输入，
训练 bridge adapter、backend copy，并可端到端训练 E-028B frontend 副本；原始 GDROS baseline 与
E-028B formal checkpoint 只读。详细绑定见 `research/experiments/E-029A.md`。

`v3.3 prototype` 根据 E-028A 和用户批准，继续双 adapter 路线，但把 optical drift 作为硬约束做
bounded tuning sweep。E-028A 判读 `DUAL_ADAPTER_TRAINING_BLOCKED`：P0 PASS，G0 唯一失败项为
`optical_teacher_drift_bounded`，optical drift `0.178630 > 0.020000`，G1 未运行；同时 train sanity
显示潜力，AEPE `16.8356 px`、recall@2/correct@2 `1.0440/0.7188`。E-028B 因此不放松 drift gate，
而是在 `optical drift <= 0.02` 的安全边界内，通过 train_fit/train_calib 内部划分选择配置，再用
formal dev 单次评价，测量 dual adapter 能否安全超过 E-027A 单 SAR adapter。候选 sweep 覆盖 optical
adapter residual scale、teacher consistency weight、optical trainable subset、modality-invariant loss 与
local-structure consistency。dev 不得参与配置/阈值/early-stop 选择。详细绑定见
`research/experiments/E-028B.md`。

`v3.2 prototype` 根据 E-027A 正结果和用户批准，在强 DINOv2 迁移前端上加入双侧轻量适配与
modality-invariant auxiliary supervision。E-027A 判读 `TRANSFER_FRONTEND_STRONG_GAIN`：
`FROZEN_DINO_OPTICAL__SAR_ADAPTER` 在 dev primary fair matcher 下达到 AEPE `26.246 px`、
recall@2 `0.569`、correct@2 `0.438`，明显优于 frozen DINO、scratch opt-SAR 和 GDROS feature-only
AEPE，但 recall/correct 仍低于 GDROS feature-only。E-028A 因此保留 frozen DINOv2 teacher/reference
和 SAR adapter，同时给 optical branch 加 residual adapter/projection 或实验 checkpoint 副本上的 LoRA，
并用 optical teacher consistency 防止遗忘 DINO prior。新增 auxiliary supervision 包括 matched-token
contrastive、modality-invariant alignment、variance/covariance regularization、domain-adversarial /
modality-confusion、local-neighborhood structure consistency、cycle/mutual 和 soft affine/grid loss。
正式 dev 仍继承 E-025A primary fair matcher `topK256_nms2_quota4_mutual0`。详细绑定见
`research/experiments/E-028A.md`。

`v3.1 prototype` 根据 E-026A 和用户批准，将 opt-SAR feature frontend 从 scratch DINO-style network
改为 strong DINOv2 pretraining transfer。E-026A 判读 `NO_OPT_SAR_FEATURE_GAIN`：P0/G0/reviewer
均通过，loss 下降且梯度到达 optical stem、SAR adapter、ViT encoder 和 projection，但 dev primary
fair matcher 上 `OPT_SAR_DINOSTYLE_NET` AEPE `149.866 px`、recall@2 `0.0358`、correct@2
`0.0099`，明显弱于 `DINO_FOUNDATION_FAIR` 与 `GDROS_FEATURE_FAIR`。解释：只参考 DINOv2 结构、
从零训练小网络，不足以获得强跨模态表示。E-027A 因此必须真实使用 DINOv2 预训练权重：frozen
optical DINOv2 作为 teacher/reference，SAR log/structure/speckle-adaptive stem + adapter/projection
学习对齐 GT 对应的 optical DINO token；可选复制 DINOv2 到实验 checkpoint 后只训练 adapter/LoRA，
但原始 DINO/GDROS checkpoint 必须只读。正式 dev 仍继承 E-025A primary fair matcher
`topK256_nms2_quota4_mutual0`。详细绑定见 `research/experiments/E-027A.md`。

`v3.0 prototype` 根据 E-025A 和用户批准，将 feature frontend 从 frozen feature / 小型 descriptor
推进到我们自己的 DINOv2-inspired optical-SAR adaptive feature network。E-025A 判读
`FROZEN_PRIOR_FEATURE_GAIN_FOUND`：统一 fair matcher 下，`DINO_FOUNDATION_FAIR` 的 AEPE
`79.246 px` 优于 `GDROS_FEATURE_FAIR` 的 `96.430 px`，说明 DINO/foundation 全局先验有用；
但 DINO 的 recall@2 / correct@2 `0.198/0.163` 低于 GDROS feature-only 的 `0.729/0.762`，
小型 supervised / teacher descriptor 也未形成有效增益。因此 E-026A 不继续从候选池里后验挑点，
而是在网络最前端学习 opt-SAR 可匹配 descriptor：optical RGB stem + SAR log/structure adaptive stem，
ViT-style patch-token encoder，descriptor projection，并用 correspondence、hard-negative contrastive、
cycle/mutual consistency、SAR augmentation 和 soft affine/grid loss 训练。DINOv2 checkpoint 若使用，
只能作为只读 teacher、初始化副本或 reference，不能修改。正式 dev 比较仍继承 E-025A 的统一
`topK256_nms2_quota4_mutual0` fair matcher。详细绑定见 `research/experiments/E-026A.md`。

`v2.9 diagnostic` 根据 E-024A-BASEINIT-DIAG 和用户批准，将研究焦点前移到跨模态 feature frontend。
E-024A-BASEINIT-DIAG 判读 `BACKEND_ORACLE_ONLY`：baseline-init-only AEPE `1.474 px`，接入 E-024A
R1/R2/R3 后退化到 `3.626 px`，但 oracle local update 可到 `0.442 px`。这说明当前 local observation /
consensus 会污染好初始化，问题根源仍是 optical-SAR 特征与匹配观测。用户进一步指出，图像匹配本质是
特征建模与特征相似度匹配，多尺度/coarse-to-fine 主要用于效率和缩小搜索范围。E-025A 因此建立
feature-only fair benchmark：所有特征经过同一 matcher/NMS/affine evaluator，与 `GDROS_FEATURE_FAIR`
比较；完整 GDROS detector-free 系统只作 secondary reference。E-025A 同时训练 truth_flow-supervised
descriptor 与 GDROS-teacher/hybrid distilled descriptor，检验先验监督或教师蒸馏能否得到比 baseline
feature 更好的 optical-SAR feature。详细绑定见 `research/experiments/E-025A.md`。

`v2.8 prototype` 根据 E-023A-DIAG 和用户批准，将 E-023A 的同构递归改为角色分离的递归 coarse-to-fine。
E-023A-DIAG 判读 `COARSE_STATE_GENERATION_FAIL`：dev selected initial AEPE mean/P50 `50.546/41.432 px`，
oracle-best initial 仍为 `23.057/21.194 px`，round0 window coverage mean/P50 `0.498/0.435`，
但 oracle endpoint best-update AEPE mean/P50 可到 `6.206/5.705 px`。因此下一版不能继续加 T、调
threshold 或修 final LS，而必须先解决 coarse basin 与 window capture。E-024A 明确绑定 R0/R1/R2/R3
四轮角色：R0 产生 coarse affine/region basin，R1 用大窗口保证 GT endpoint 进入搜索范围，R2 做
geometry consensus 与 outlier rejection，R3 窄窗口精修。它与 LoFTR 类一次 coarse-to-fine 的区别是：
fine local observations 会反向更新 low-dimensional affine state，更新后的 state 再定义下一轮 local search。
详细绑定见 `research/experiments/E-024A.md`。

`v2.7 prototype` 根据 E-022A 和用户讨论，停止把主问题定义为 frozen DINO-K256 候选池里的
后验挑点。E-022A formal-r2 在 P0、train-small、资源恢复、4/4 shard 聚合与 reviewer 均通过后，
仍判读 `NO_TRAIN_FEATURE_SEPARABILITY`：best learned/fusion gate track precision@2 `1.3201%`、
correct@2 mean/P50 `1.805/1`、enrichment@2 `30.28x`、selected-set AEPE `34.001 px`。
这与 E-021A 的 set enrichment 失败一致，说明当前 DINO rank/local patch/raw/structure/neighborhood/
fusion 后验特征不足以从 hard negatives 中形成高纯、足量、可拟合 affine 的点集。用户指出固定挑点
问题本身不符合主流匹配范式；新主线改为 coarse-to-fine / state-conditioned matching：先产生低维
affine/region state，再由 state 限制局部搜索、产生稀疏/半稠密跨模态观测，并递归更新 affine state。
E-023A 是有效性优先原型，不以资源为第一版硬 gate；禁止 fixed DINO-K256 selector primary path、
dense flow state、dense GRU、RANSAC/三点/131k、hidden new full `4096x4096`、OS424/confirm/datum
和 baseline 写入。详细绑定见 `research/experiments/E-023A.md`。

`v2.6 diagnostic` 根据 E-021A 和用户批准，把问题进一步拆到候选区分特征层。E-021A 证明
probabilistic set selection 不是完全无信号：best track `expected_top256` 在 train-full 上达到
enrichment@2 `32.2557x`，但 selected precision@2 仅 `1.3979%`，correct@2 mean/P50 `1.919/1`，
AEPE `32.405 px`，PLR `22.34%`，G0 判读 `SET_ENRICHMENT_TRAIN_FAIL`，dev404 未运行。用户指出
应按输入到输出链路修复，当前最可疑的是候选区分阶段的特征不够强。E-022A 因此不直接救
E-021A selector，而是冻结 DINO-K256 candidate pool，比较 DINO rank/margin、raw/log-SAR patch、
gradient/census/structure、DINO local patch、learned patch verifier、neighborhood consistency、
region context、pair/source adaptive normalization 和 fusion tracks 对正确候选的可分性。主指标包括
per-source hit@1/4/8、selected precision/correct count、enrichment、coverage 和 selected-set affine。
详细绑定见 `research/experiments/E-022A.md`。

`v2.5 prototype` 根据 E-020B 和用户批准，将目标从逐候选硬分类改为 probabilistic set selection /
inlier enrichment。E-020B 在 train-full 上仍失败：endpoint@2 AP/AUROC `0.006255/0.916803`，
best top256 correct@2 mean/P50 `2.163/1`，precision `1.29%`，AEPE `35.686 px`。这说明在当前
DINO-K256 上用 hard candidate classification/ranking 学高纯前缀会被 hard negatives 淹没。用户指出最终
任务只是挑点：无需每个 candidate 拟合 GT，只要让正确点比错误点更高概率被选中，并让 selected set
支撑 affine 即可。E-021A 因此训练一个选择分布，优化 soft inlier enrichment、set-level affine/grid utility、
coverage/diversity 和 budget calibration；主指标改为 `P(inlier | selected)` 的富集倍率、correct count、
coverage 与 selected-set affine，而非 candidate AP。dev inference 仍禁止 GT，DINO/GDROS 与历史产物只读。
详细绑定见 `research/experiments/E-021A.md`。

`v2.4 diagnostic` 根据 E-020A 和用户批准，修正 G0 语义。E-020A 的 P0 和强网络通道通过，但旧 G0
要求 4-pair train-small 同时达到 AP `>=0.05` 和 selected-set affine AEPE `<=14.8 px`，导致实验在
train-full/dev404 前停止：held-in AP `0.039667`，top256 correct@2 mean/P50 `17/15`，AEPE
`19.957 px`。用户指出图像匹配跨模态 verifier 需要足够训练样本学习规律，4-pair 只能作为工程 sanity，
不能作为是否进入 train-full/dev404 的硬门槛。E-020B 因此保留 E-020A 的强网络和隔离边界，但将
1/4/8 pair overfit 降为 sanity，G0 主门改为 1607 train pairs 的 train-full sanity；同时允许把监督从
候选 BCE/简单 ranking 改为 source-choice listwise/sampled-softmax、hard-negative focal、supervised
contrastive/grouping、affine-aware set loss 与 coverage/diversity。dev404 仍只能在 G0 通过后 no-GT
运行，且不得用 dev 后验阈值追认。详细绑定见 `research/experiments/E-020B.md`。

`v2.3 diagnostic` 根据 E-019A 和用户批准，停止继续堆手工/浅层 no-GT proxy。E-019A 在 P0 与两路
reviewer 通过后判读 `NO_USABLE_PROXY_SIGNAL`：最好的 train-only linear probe endpoint@2 AP 仅
`0.003938`，region_quota128 correct@2 mean 仅 `0.649`、P50 `0`，selected-set PLR `9.41%`。这说明
DINO-K256 中正确候选的可分性不在简单 DINO rank/score、local/regional/global geometry consensus、
collision 或 coverage proxy 上。E-020A 因此改为 high-capacity candidate verifier learnability diagnostic：
冻结 DINO-K256 候选和 GDROS/DINO baseline，训练一个新的强候选验证网络，输入 raw optical/SAR patch、
DINO local feature、candidate token、within-source K256 competition、source-neighbor graph 和 global/
affine basin tokens；用 train truth_flow 构造 endpoint/source-choice/basin/affine labels，但 dev inference
禁止读 GT。E-020A 的目的不是直接宣布最终 sparse feedback 成功，而是区分：train 都不可过拟合、train
可过拟合但 dev 不泛化、candidate 可学习但 affine 压缩失败、或存在可用/强可学习信号。详细绑定见
`research/experiments/E-020A.md`。

`v2.2 diagnostic` 根据 E-018A 和用户批准，暂停 learned compression 连续救援，先用上帝视角分析
DINO-K256 中的高质量候选到底有什么可分特征。E-019A 不训练、不改模型、不启动 E-018B；它只在
候选 score/id 物化后用 raw truth_flow 构造 endpoint/inlier/oracle-set 标签，然后评估无 GT 可计算
features 对高质量点的 separability。重点检查是否存在局部到全局的几何共识特征：局部 displacement
一致性、region/neighbor support、target duplicate/collision、affine support proxy、coverage contribution、
DINO rank/score/margin 与多种组合。若几何共识 proxy 明显优于 appearance/rank，则下一步应围绕显式
inlier-set / correspondence grouping 设计；若所有无 GT proxy 都弱，则需要更强 observation source 或外部
matcher，而不是继续训练当前 DINO-K256 head。详细绑定见 `research/experiments/E-019A.md`。

E-019A 已完成并停止。P0、watchdog 与两路 reviewer 通过，主判读为 `NO_USABLE_PROXY_SIGNAL`。最佳
endpoint@2 AP：train-only linear probe `0.003938`，geometry_fusion `0.003748`，global_consensus
`0.002611`；region_quota128 最好 correct@2 mean `0.649`、precision `0.5067%`、P50 correct `0`。
因此停止把当前 proxy 当作可用 selector 的路线；但 E-016A 的 GT-guided DINO-K256 oracle 仍非常强，
不能据此否定候选池上限。

E-020A 已完成并停止。P0 PASS，强网络主通道存在并被调用，gradients 到达 required modules；
但 model/training reviewer 判 `CHANGES_REQUIRED / G0_STOP`，原因是旧 4-pair train-small overfit
阈值未过。按新方法解释，该结果不能说明输入不可分，只说明旧 G0/监督不足以完成可学习性验证。

E-020B 已完成并停止。P0 与两路 reviewer APPROVE，G0 判读 `TRAIN_FULL_SANITY_FAIL`，dev404 未运行。
train-full AP/AUROC `0.006255/0.916803`，best top256 AEPE `35.686 px`、correct@2 mean/P50 `2.163/1`、
precision `1.29%`。该结果停止继续修 hard verifier/classification/ranking loss 的路线。

E-021A 已完成并停止。P0 PASS，G0 判读 `SET_ENRICHMENT_TRAIN_FAIL`，dev404 未运行。best track
`expected_top256` 的 enrichment@2 `32.2557x` 说明 selected-set 富集信号存在；但 precision@2
`1.3979%`、correct@2 mean/P50 `1.919/1`、AEPE `32.405 px` 未达到可用门槛。该结果停止继续调
E-021A selection loss/threshold 的路线，下一步转向候选区分特征可行性诊断。

`v2.1 prototype` 根据 E-017A 和用户批准，停止修补 threshold、K512 或独立 candidate scorer。E-017A
证明当前 supervised compression head 有弱 AUROC 但 AP 极低，active set 被 outlier 淹没；K512 更差。
因此 E-018A 改为 `affine-state-first consensus`：冻结 DINOv2-L full4096 K256 candidates，让候选集合先
聚合/投票出少数低维 affine state tokens，再用 affine state 的残差和覆盖反向解释、筛选候选点。
训练时用 raw truth_flow 监督 affine tokens、basin assignment、inlier residual 和 final grid EPE；dev
推理不读取 GT。该版本把“候选打分→拟合仿射”改成“预测/维护仿射状态→解释候选→重拟合仿射”，更贴近
稀疏跨模态观测与低维仿射状态递归反馈的中心创新。E-018A 不微调 DINO/GDROS，不恢复 dense flow
state/dense GRU，不做 RANSAC/131k hypothesis enumeration，不启动轻量化蒸馏；full4096 仍为有效性
prototype，不宣称轻量化。详细绑定见 `research/experiments/E-018A.md`。

E-018A 已完成并停止。P0/G0 与两路 reviewer 通过，说明 affine token primary path、T=4 residual
feedback、梯度、toy overfit、strict load 和 no-GT dev inference 链路可信；但 G1 仍为
`OUTLIER_CONTAMINATED`。Direct affine token AEPE `29.806 px`、CMR@2 `1.75%`、PLR `25.50%`；
final active LS AEPE `29.580 px`、CMR@2 `1.96%`、PLR `28.96%`，active mean `292.1`，
precision@2 `0.0114`。该结果表明 affine-state-first token 没有学到正确 basin，失败不只是 final
selection；当前 DINO-K256 learned consensus 路线需要重新考虑监督/表示或更直接的外部几何先验，而非
继续修 threshold、K512 或当前 affine-token head。

`v2.0 prototype` 根据 E-016A 和用户批准，进入第一版有效性优先的 learned compression。E-016A 证明
DINOv2-L full4096 candidates 在 GT-guided selection 下已高度可配准：K256 all_hard:thr5 oracle
AEPE `0.294 px`、CMR@2 `99.169%`，region-balanced B128 oracle AEPE `0.364 px`、CMR@2
`99.252%`。同时 no-GT DINO score/region quota 完全不可用，best no-GT AEPE `54.991 px`、
CMR@2 `0.199%`，并且 K256 相对 K128 correct@2 mean 增加 `36.446`，说明 K128 偏浅。
E-017A 固定 DINOv2-L 为只读候选生成器，训练 geometry-aware compression/ranking 网络从 DINO topK
candidates 中输出动态、区域均衡、每 source 至多一个 target 的 sparse matches，再拟合 affine。
主线比较 K256 与 K512：K256 是主要继承 E-016A 的可压缩上限，K512 是 rank-tail headroom secondary
track。E-017A 不微调 DINO/GDROS，不恢复 dense flow state/dense GRU，不读取 OS424/confirm/datum；
由于 full4096 DINO 仍高成本，本实验只证明有效性，不宣称轻量化。详细绑定见
`research/experiments/E-017A.md`。

E-017A 已完成并停止。P0/G0 与两路 reviewer 通过，说明冻结候选、标签、梯度、toy overfit、strict load
和 no-GT dev inference 链路可信；但 G1 判读为 `OUTLIER_CONTAMINATED`。K256 dev AEPE `45.906 px`、
CMR@2 `0.3508%`、PairLocalReady `3.47%`，active mean `115.2` 中 correct@2 mean 仅 `1.047`；
K512 更差，AEPE `49.970 px`、CMR@2 `0.2097%`。ranking AUROC 达 `0.878941/0.906077`，但 AP
仅 `0.002845/0.002247`，没有转化为高纯 active set。该结果停止当前 v2.0 compression head，但不否定
DINO full4096 observation source 或 E-016A 的 GT-guided geometry-compressible 上限。

`v1.9 diagnostic` 根据 E-015A 结果和用户批准，承认 DINOv2-L full4096 retrieval 已经找到更强
candidate generation source：correct@2px proposals mean/P50 达到 `115.943/115`，明显超过 E-011A
top128 与 E-006A 数量线。但其 oracle affine AEPE 仍为 `12.903 px`，说明当前瓶颈转为候选的空间
分布、source/target 去重、rank tail、outlier 与几何一致性压缩。E-016A 不训练、不改变 baseline、
不启动完整 sparse affine feedback，只对 DINO full4096 topK candidates 做几何/分布诊断和 GT-guided
compression oracle，判断下一步应学习 candidate generation、geometry-aware compression 还是更强
observation source。高成本 full4096 仍标记为 diagnostic/resource-unqualified，不得宣称轻量化。
详细绑定见 `research/experiments/E-016A.md`。

`v1.8 diagnostic` 根据 E-014A 和用户批准，承认当前 E-011A top128 上的手工/轻量结构信号不足以
提供 endpoint ranking。E-015A 将研究退到 SparseGeo-GDROS 最前端的 observation/descriptor 阶段：
不计成本尝试强预训练 / foundation / 跨模态 matcher 特征，包括 DINOv2、OpenCLIP/RemoteCLIP、
SAM/SAM2/MAE encoder、MatchAnything、XoFTR、RoMa/DKM/LoFTR-style、PromptMID/SOMA official
code/checkpoint（若可用）。E-015A 可运行高成本 top128 reranking、S1 union retrieval 或 full 4096
target retrieval，但必须标为 diagnostic/resource-unqualified，不得宣称轻量化。完成后只判断是否存在
可用跨模态 observation source，不自动进入 E-015B 或完整 sparse affine feedback。详细绑定见
`research/experiments/E-015A.md`。

`v1.6` 根据 E-012A 和用户批准，停止 deterministic no-training geometry-only compression rescue。
E-013A 继续保留 E-011A raw-image top128 candidates 作为只读输入，但把几何从无训练裁判改为训练
监督和推理辅助特征：训练时用 raw truth_flow 构造 endpoint hard/soft labels、oracle affine/basin
labels 和 affine EPE loss；推理时不读取 GT，由 learned appearance-conditioned inlier/basin scorer
结合 residual feedback 输出动态 active sparse matches。每个 source 最终最多一个 target，active
点数不固定为 K64/K128。E-013A 不重训 E-011A descriptor，不训练或微调 GDROS
backbone/FeatureTransformer，不恢复 dense flow/GRU、RANSAC/131,072 hypotheses 或 hidden
`4096x4096` cross-image tensor。详细绑定见 `research/experiments/E-013A.md`。

`v1.5` 根据 E-011A-TOP128-GEOM-DIAG 和用户批准，停止把 top64 作为唯一候选生成出口。
E-012A 保留 E-011A raw-image top128 evidence，但将 `top128` 绑定为每个 source proposal 的
target candidate depth，而不是最终点数上限。新增 geometry-aware dynamic compression：从很多
source proposals 的 top128 candidates 中，维护多个 affine basin slots，执行 assignment、FP32
weighted affine solve、residual feedback 和 active/dustbin/pending 更新，输出动态 active sparse
matches。每个 source 最终最多一个 target，active 点数不固定为 64/128。E-012A 不训练或微调
GDROS backbone/FeatureTransformer/E-011A descriptor，不使用 GT 做 inference、阈值选择或 basin
初始化，不恢复三点 RANSAC、131,072 hypotheses、dense flow state、dense GRU 或 hidden
`4096x4096` cross-image tensor。详细绑定见 `research/experiments/E-012A.md`。

`v1.4` 根据 E-010A 和用户批准，停止继续在 E-008A cached raw/structure/local tail 上更换 loss 或
调 topR。E-011A 真正改变 observation source：从 official training archive 的原始 `512x512`
optical/SAR 图像重新提取多尺度 patch descriptor，在固定 E-008A S0 source proposals 与 S1
matched-region union 内做 full-union retrieval。GDROS baseline 必须完全只读：不得修改 GDROS
源码、checkpoint、baseline cache 或 baseline run 目录，不得训练/微调 GDROS backbone 或
FeatureTransformer。E-011A 的成功标准仍然是显著超过 frozen S2，并进一步判断是否超过 E-006A
candidate baseline；top128 只作诊断，不允许追认 top64 失败。

`v1.3` 根据 E-009A 和用户批准，明确 frozen S2 top64 只是低质量最低对照，不是学习目标。
E-010A 停止 E-009A 的 `16k params + raw/structure tail + sampled-negative CE` 小模型路线，建立
强跨模态 region-local retriever：在固定 E-008A S0/S1 的前提下，训练 modality-specific
raw/structure/patch encoder 与 stride-8 descriptor，在 matched-region union 内做 full-union/listwise
retrieval，并加入 hard negative mining。GDROS frozen features 不得进入 learned retriever，也不得
作为 teacher；只用于 frozen S2 baseline parity。E-010A 的 G1 要求显著超过 frozen S2，G2 再判断
是否超过 E-006A candidate baseline；top128 仅作诊断，不允许在 top64 失败后追认。

`v1.2` 根据 E-008A-STAGE-DIAG 和用户批准，停止 rescue E-008A v1 的 region-to-point 生成器，
将研究焦点收缩到最主要的可验证瓶颈：matched-region 内 frozen cosine / region score `top64`
预筛。新实验 E-009A 固定 E-008A 的 source proposals 和 matched-region union，只替换 top64
retriever：训练轻量双模态 stride-8 descriptor/retrieval module，使 learned local evidence 在 top64
之前参与检索。该 retriever 不读取 GDROS frozen features；frozen features 只用于旧 S2 baseline
parity。主对照是 E-008A-STAGE-DIAG 的旧 S2 top64；完整候选生成资格再与 E-006A
corrected candidate baseline 比较。E-009A 不启动完整 sparse affine feedback，不读取 OS424/confirm，
不使用 dense flow/GRU、hidden 4096x4096 或 full-image blind search。

`v0.9` 根据 E-005A-INLIER-DIAG 和用户批准，保留“稀疏跨模态观测和低维仿射状态递归反馈
替代 dense flow/dense GRU”的中心创新，但改变实现方式：从区域结构匹配出发，维护多个粗仿射
basin；每轮用当前 basin 引导未解决区域重新生成局部候选，让高置信点进入 active set，低置信点
保留为 unresolved，错误点进入 rejected；通过跨区域 leave-out 验证和 hard/robust inlier rejection
防止单一错误 basin 自我强化。active 点数不再固定为 128，也不设置人工上限；唯一自然边界是
冻结 proposal grid 的有限点数。详细绑定见 `research/experiments/E-006A.md`。

`v0.8` 根据 E-004A 的稳定 G1 失败和用户批准，停止把 anchor acquisition 作为独立于 target
candidate 和全局几何的分类问题。新主线固定 M256 覆盖 proposal，一次计算 `256x4096` cosine
top8；每个稀疏 candidate 维护隐状态与连续 target 坐标，以 16 个 global tokens 联合交换位移
模式，执行四轮 local correlation、候选/dustbin/reliability 更新、dynamic K128、FP32 weighted
affine solve 和 affine residual feedback。该结构借鉴 DPVO 的 learned update/geometry alternation、
IHN 的低维几何 unroll、CoTracker 的 virtual-token aggregation 和 SGMNet 的 seed-guided propagation，
但不得移植各方法的 dense correlation、SE(3)/BA、full Sinkhorn 或 target-all attention。最终 dense
flow 只能由六参数 affine 在 metric boundary 解析生成。详细绑定见 `research/experiments/E-005A.md`。

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

无训练探针已经完成。历史后续分支曾批准 M001 v0.6 / E-003A learned selector；详细结构、数据
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

E-004A 已完成并停止。P0 oracle 证明 M256/覆盖 K128 有足够上限，但三 seed 的 G1 均失败：主
配置 recoverable `41.80%`，区域分配相对 uniform 提升 `9.68 pp`，阈值 replacement 仅提升
`0.0522 pp`。因此只停止 v0.7 的独立 pre/post heads 和逐点 replacement；不否定 M256 proposal、
覆盖约束或在观察 target candidates 后进行集合级选择。

## 5.3 E-005：稀疏递归仿射共识

E-005A 固定使用 M256 coverage proposals，对全部 proposal 一次执行 exact cosine top8。每个
anchor 的八个 candidates 维护 96 维 recurrent hidden、连续 target coordinates、candidate logits
和 dustbin；16 个 persistent global tokens 以 cross-attention 汇聚 anchors，anchors 不做全连接
self-attention。每轮围绕 candidate 连续坐标采样 `5x5` quarter-token local correlation，并以共享
GRUCell 更新 candidate coordinate、logit、dustbin 和 reliability。

四轮中每轮都以 reliability 重新执行 coverage-constrained K128：每区至少 1、最多 3，inactive
anchor 后续可重新进入。active candidates 的条件概率产生 target expectation，dustbin/reliability
产生 weights；normalized FP32 ordinary weighted least squares 估计六参数 affine，并将下一轮的
`q-Hp` residual 反馈给 candidate 和 anchor。第 1 轮 geometry residual 显式为零，避免 identity
初始化抑制大位移候选。

网络不预测 dense flow。`512x512` flow 只在公开 metric 边界由最终 affine 解析生成；cross-image
global scores 固定为 `256x4096`，local samples 固定为 `4x256x8x25`。禁止 dense GRU、hidden
`4096x4096`、target Sinkhorn/self-attention、hypothesis enumeration、IRLS/BA 和 baseline fallback。

训练继续使用已冻结的 `1,607/404` group-isolated train/dev，raw truth_flow 为唯一标签，OS 424
和 external confirm 禁止访问。E-005A 必须同时通过 local/top8 oracle、创新契约/梯度审阅、dev
精度和端到端资源 gate；完成后不论成败均等待用户，不自动建立新版本。

E-005A 已完成并停止。P0/G0 和两轮审阅均通过，但 dev AEPE `4.6343 px`，CMR@2 `7.92%`，
明显低于 dense GDROS dev `1.4597 px/79.46%`。E-005A-INLIER-DIAG 进一步证明，E-005A
active K128 的 2px GT/self inlier 仅为 `14.70/47.41%`，加权后为 `22.20/73.33%`；权重主要强化
错误 basin 的内部自洽。K x rho 曲线显示 ordinary LS 只有 `rho=100%` 通过 dense 精度门槛，
因此只增加点数、T 或继续 soft expectation + ordinary weighted LS 不构成下一步。

## 5.4 E-006：区域引导的置信增长稀疏仿射反馈

E-006A 继续冻结 GDROS backbone/post-Transformer 特征、official `512x512` 输入、stride-8
`64x64` token 和 raw truth_flow 标签协议。source proposal pool 默认复用 M256：`8x8` source
regions，每区 4 个固定 proposal。M256 是第一版有限观察网格，不是 active 点数上限；所有达到
固定置信与几何验证条件的 proposal 都可进入最终拟合。

区域层先于点层执行。source/target 都划分为 `8x8` regions，每个 region descriptor 由区域均值、
四象限均值和坐标编码组成。每个 source region 只保留 top-L=8 target regions 作为结构候选，并
由轻量 region scorer 预测 `P(region center endpoint <= 8px)`。B=8 个 affine basin 由 region-level
候选的加权 affine solve 初始化；每个 basin 都有独立状态、置信和 active/unresolved/rejected 集合。

每轮递归执行：

1. 对 unresolved 和 active proposals 生成候选，候选来源为 global exact cosine top8、region-restricted
   top8、以及每个 basin 当前 `H_b(p)` 附近 `9x9` token window 的 affine-guided local candidates。
2. 候选保持离散，不做跨候选 soft expectation 作为最终坐标；训练可使用 soft loss，但最终候选
   必须通过 hard select 或 hard reject。local offset 以 pixel 为单位并 clip 到 `[-4,4] px`。
3. candidate confidence 监督目标为 `P(endpoint error <= 2px | appearance, region, basin residual,
   leave-region-out residual)`，不是与当前 H 自洽的概率。
4. 每个 source proposal 最多贡献一个最终 target vote；同一 source 的候选按 confidence、endpoint
   proxy、basin score 和确定性 tie-break 选择。
5. 对每个 basin 执行 region-balanced robust affine solve：先用通过阈值的候选拟合，再用
   leave-one-region-out affine residual 做 hard inlier rejection 并重拟合。每个 region 的总权重归一化，
   防止单一区域大量相关点支配全局仿射。
6. proposals 可在 `active / unresolved / rejected` 间迁移；停止条件是 active 集合和 affine 在连续
   两轮内稳定，或达到 T=4。最终有效条件为至少 12 个 effective inlier、至少 6 个 source regions、
   至少 3 个 quadrants、condition `<=20`。

active confidence threshold schedule 固定为 round 1/2/3/4 的 `0.90/0.85/0.80/0.75`；低于
`0.40` 且所有 basin leave-out residual `>8 px` 的 proposal 进入 rejected，其余保持 unresolved。

复杂度边界为 `O(T * M * (K_global + L_region_area + B * W_local) + T * B * M)`，其中
`M=256`、`K_global=8`、`L=8`、`W_local=81`、`B=8`、`T=4`。禁止构造 `4096x4096`
cross-image correlation、dense flow hidden state、dense GRU、RANSAC/三点 hypothesis 枚举、
descriptor 微调、baseline fallback 或跨样本缓存。

E-006A 的第一硬目标不是“把 K 调大”，而是把最终 GT endpoint inlier ratio 提到接近 GDROS：
dev active/inlier pool 的 2px GT inlier `>=65%`，self-GT inlier gap `<=20 pp`，同时公开仿射
AEPE 接近 dense GDROS。完整 gate、消融和停止条件见 `research/experiments/E-006A.md`。

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

“稀疏点加加权最小二乘”不足以构成核心创新。当前核心主张是：让稀疏/半稠密跨模态观测和低维
affine state 在多轮中相互反馈，以该联合优化过程替代 GDROS 的 dense flow state 和 dense GRU。
v2.9 暂不继续扩大几何后端，而是先验证跨模态特征前端。若 feature-only fair benchmark 找不到强于
`GDROS_FEATURE_FAIR` 的特征，则后续 sparse recurrent affine matcher 缺少可靠观测源；若找到强特征，
才能把该 feature 接回 R0/R1/R2/R3。完整 detector-free baseline 的系统性能不能被统一 NMS 代表，
但其 feature-only separability 可以作为 primary feature baseline。

DPVO 的 update/BA alternation、IHN 的低维 homography refinement、CoTracker 的 virtual tracks、
SGMNet 的 seeded propagation、StereoGlue 的一对多候选与 GC-LAC 的局部仿射一致性均构成 prior。
论文创新不能声称发明 recurrent geometry、global tokens 或 sparse matching；只能在证明该组合
对 optical-SAR 的 dense-GRU replacement 同时达到精度与资源 gate 后，主张“跨模态稀疏递归
仿射共识”这一任务化结构。
