# E-034A-E028B-RECURRENT-AFFINE-MATCHER：Trainable Decoder for E028B TopK Correlation

方法版本：`M001 v3.9 design`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-033A 证明 E028B feature correlation 中 GT-neighborhood 响应大多已经存在：

- rank≤32 visible `98.67%`；
- raw rank mean/P50/P90 `4.21 / 2 / 9`；
- top8 capture `89.05%`；
- matcher visible-miss `79.86%`；
- feature-not-visible `1.12%`。

因此 E-034A 固定 E028B feature，训练轻量 recurrent affine-conditioned matcher/decoder，目标是把 topK
correlation 中已有但未被当前规则 matcher 利用的响应解码成稳定 affine。

同时必须回答用户关心的分解问题：

> 当前失败是初始 coarse matching 已经错，还是 coarse token 正确但 local refinement 不够，还是全局 consensus
> 把好点用坏了？

## 2. 边界

- E-028B safe dual adapter checkpoint 固定只读；
- 不训练/finetune E028B descriptor；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- 允许训练 matcher/decoder/head；
- 只使用 official training.zip 的 train/dev split；official testing.zip、confirm、OS424 不使用；
- topK candidate memory 与 selected matches 必须先落盘，再读取 truth-flow 做 metric join；
- 不使用 hidden RANSAC / three-point / 131k sampling；
- 完成后停止，不自动启动 E-034B、descriptor training、full sparse feedback 或 rescue。

## 3. Tracks

### A. `TOPK_CORRELATION_MEMORY`

固定 E028B feature，对每个 source token 生成 target topK memory：

- K = 32 primary，K = 64 diagnostic；
- candidate fields：score、rank、margin、dual-softmax、mutual rank、source/target coords、displacement、
  local `3x3/5x5` correlation patch、source/target norm、region/quadrant id；
- 复现 E-033A GT-neighborhood visibility parity。

### B. `RECURRENT_AFFINE_CONDITIONED_DECODER`

训练一个轻量 matcher：

- per-source topK candidate encoder；
- candidate probability；
- source matchability；
- local sub-token offset；
- uncertainty；
- weighted affine state；
- affine residual feedback；
- 3-6 recurrent iterations。

至少包含 ablation：

- no affine feedback；
- single affine basin；
- multi-basin affine hypotheses；
- no local correlation patch；
- K32 vs K64 diagnostic if resource allows。

### C. `COARSE_VS_REFINEMENT_DIAG`

必须输出四个层级：

1. `coarse_token_selected`：只看 selected target token 是否落入 GT-neighborhood；
2. `coarse_only_affine`：用 token center endpoint 拟合 affine；
3. `local_refined_affine`：加入 predicted local offset / local soft-argmax；
4. `oracle_local_bound`：在 selected coarse token 条件下使用 GT local endpoint bound；
5. `gt_neighborhood_oracle_bound`：复现 E-033A 上帝视角 bound。

判读：

- selected coarse token 错：`COARSE_MATCHING_LIMITED`；
- coarse token 对但 local 后仍差：`LOCAL_REFINEMENT_LIMITED`；
- per-match 变好但 final affine 差：`CONSENSUS_LIMITED`。

### D. `AFFINE_CONSENSUS_OUTPUT`

用 learned probability、uncertainty、region coverage 做 weighted affine。

报告：

- AEPE；
- CMR@1/2/5；
- PLR / valid；
- selected count；
- correct@2 mean/P50；
- precision@2；
- coverage / region balance；
- visible-miss rate relative to E-033A。

## 4. Gates

### P0：setup/parity

必须 PASS：

- official training/testing archive SHA 与 pair count verified；
- E028B checkpoint SHA、feature hook、feature shape recorded；
- E-032A best consensus metrics reproduced or typed unavailable；
- E-033A GT-neighborhood visibility parity reproduced；
- no GDROS backend adapter / dense GRU reuse；
- topK candidate memory persisted before truth-flow metric join；
- baseline artifacts readonly；
- testing.zip / confirm / OS424 not used。

P0 fail：`E034A_SETUP_BLOCKED`。

### G0：training sanity

必须 PASS：

- candidate probability / matchability / local offset / affine consistency losses finite and nonzero where enabled；
- gradients reach decoder、affine feedback/residual heads、local offset head if enabled；
- train-small overfit sanity improves selected-set AEPE or GT-neighborhood probability mass；
- train-full selected-set utility improves over E-032A train reference or typed failure；
- checkpoint strict-load verified；
- reviewer approves candidate-before-label, leakage isolation, and training/evaluator interpretation。

G0 fail：`E034A_TRAINING_BLOCKED`。

### G1：dev404 system diagnosis

判读：

- `RECURRENT_MATCHER_GAIN_FOUND`：dev404 clearly improves over E-032A best consensus AEPE `15.3632` /
  CMR@2 `4.0633%` and reduces E-033A visible-miss；
- `COARSE_MATCHING_LIMITED`：GT topK visibility exists but selected coarse token remains wrong before refinement；
- `LOCAL_REFINEMENT_LIMITED`：selected coarse token near GT but local/pixel refinement prevents CMR@2；
- `CONSENSUS_LIMITED`：candidate probabilities improve but final affine remains unstable/outlier contaminated；
- `NO_MATCHER_GAIN`：no clear improvement over E-032A；
- `E034A_RESOURCE_BLOCKED`：resource/time prevents completion。

## 5. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `train_report.json`；
- `dev_metrics.json`；
- `coarse_vs_refinement_diag.json`；
- `ablation_summary.json`；
- `visible_miss_report.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。

完成后只回传精简 RESULT_CARD 与核心路径，等待方法任务和用户决策。不得自动启动 E-034B、descriptor
training、full sparse feedback、OS424/confirm 或 rescue。
