# E-034C-E028B-COARSE-INLIER-SELECTION-SWEEP：No-Train Coarse Inlier Ratio Sweep

方法版本：`M001 v3.11 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

当前阶段主目标是提高 coarse matching 的正确点比例。E-033A 证明 GT-neighborhood 大多在 topK 内；
E-034B 证明 raw top1/IRLS 已优于 E-032A，但 CMR 仍远低于 GDROS。

E-034C 固定 E028B topK candidate set，系统 sweep 多种 no-train coarse selection / consensus 策略：

> 从已有 topK 候选中选出更高 inlier ratio、更均匀 coverage、可拟合 affine 的粗匹配集合。

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不训练、不 finetune descriptor；
- 不训练 matcher/decoder/head；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- 不使用 official testing.zip / confirm / OS424；
- topK candidate memory 与 selected matches 必须先落盘，再读取 truth-flow 做 metric join；
- 不使用 random RANSAC、random three-point、131k sampling；
- 允许 deterministic trimming、IRLS、ranked growth、neighborhood consistency、graph compatibility；
- 完成后停止，不自动启动后续实验或 rescue。

## 3. 借鉴路线

实现可借鉴但不必复刻：

- GMS：grid-based motion statistics，用局部网格运动一致性过滤 outlier；
- LPM：locality preserving matching，用邻域结构保持判断 correspondence；
- PROSAC：按匹配质量排序的渐进一致性，本实验只允许 deterministic ranked growth；
- NCNet/OANet：候选集合的邻域/上下文兼容性，本实验用 no-train graph compatibility proxy。

## 4. Tracks

### A. `BASELINES_FROM_E034B`

复现：

- raw_top1_all_source；
- top1_mutual；
- top1_trimmed50_irls。

作为硬对照。

### B. `RANKED_GEOMETRY_GROWTH`

按 raw score、margin、dual-softmax、mutual rank 排序，逐步扩大 candidate pool。

每步：

- deterministic weighted LS；
- IRLS/refit；
- residual threshold sweep；
- region coverage check。

报告 best selected coarse inlier ratio、selected count、coverage、AEPE/CMR。

### C. `NEIGHBORHOOD_CONSISTENCY`

GMS/LPM-style no-train filter：

- source grid neighborhood displacement consistency；
- region/quadrant local motion support；
- multi-scale cell support；
- affine residual consistency after local support filtering。

输出局部一致点集并拟合 affine。

### D. `TOPK_AFFINE_BASIN_COMPETITION`

使用 top4/top8 candidates 建立 deterministic affine basin：

- seeds 来自 high-score mutual/margin/region-balanced anchors；
- 不用 random 3-point/RANSAC；
- 每个 basin 对每个 source 在 topK 内重新选 residual+score 最优 candidate；
- basin score = inlier ratio + coverage + residual + raw score support；
- 输出 best basin 对应点集和 affine。

### E. `GRAPH_COMPATIBILITY_SCORING`

构建候选兼容图：

- nodes = source-token topK candidates；
- edge compatibility = displacement/relative geometry/affine residual compatibility；
- deterministic message passing / power iteration 得到 support score；
- 选择高 support、region-balanced 点集拟合 affine。

## 5. Gates

### P0：setup/parity

必须 PASS：

- official training/testing archive SHA 与 pair count verified；
- E028B checkpoint SHA、feature hook、feature shape recorded；
- E-032A / E-034A / E-034B references recorded；
- E-033A raw top1/topK capture parity reproduced；
- no GDROS backend adapter / dense GRU reuse；
- topK candidate memory and selected matches persisted before truth-flow metric join；
- baseline artifacts readonly；
- testing.zip / confirm / OS424 not used。

P0 fail：`E034C_SETUP_BLOCKED`。

### G0：execution

必须 PASS：

- E034B baseline replay complete；
- ranked growth, neighborhood consistency, affine basin, graph compatibility variants complete or typed failure；
- selected matches written before labels/flow are read；
- dev404 coarse inlier ratio / selected count / coverage / AEPE / CMR computed；
- reviewer approves no leakage, no hidden RANSAC, and result interpretation。

G0 fail：`E034C_EXECUTION_BLOCKED`。

### G1：diagnosis

判读：

- `COARSE_INLIER_SELECTION_GAIN`：best variant improves selected coarse inlier ratio and affine over E-034B best
  AEPE `9.969` / CMR@2 `6.136%`；
- `INLIER_RATIO_GAIN_AFFINE_LIMITED`：inlier ratio improves but affine/CMR does not, suggesting coverage/refinement/geometry issue；
- `NO_SELECTION_GAIN`：no variant improves over E-034B best；
- `E034C_RESOURCE_BLOCKED`：resource/time prevents completion。

## 6. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `selection_sweep_table.json`；
- `best_variant_metrics.json`；
- `coarse_inlier_coverage_report.json`；
- `variant_artifacts_manifest.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。

完成后只回传精简结果和核心路径，等待方法任务和用户决策。不得自动启动 E034D/E035、training、
descriptor finetune、full sparse feedback、OS424/confirm 或 rescue。
