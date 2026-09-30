# 当前研究状态

更新时间：`2026-08-27`

## 当前身份

- 当前方法任务：`019fd11a-c6b4-72b2-afc8-658a9d728b7e`，ACTIVE。
- 实验工程任务：`019fc85d-350e-7941-ac69-89c626641fd9`，空闲 / 等待派工。
- 当前方法：`M001 v3.21 feature-extraction divergence / E-045A branching`。
- 当前状态：`E-037A COMPLETED / FEATURE_INTERACTION_GAIN_FOUND; gain is projection/norm, not cross-attention`。
- 近期主线已从 matcher-facing projection 转向 E-045A 特征提取发散；E-044A noenergy 作为当前最强 matcher 参考线冻结。

## 数据与协议边界

- 官方 training archive：`/Users/shihao/Downloads/os_dataset/training.zip`，SHA256
  `2887e7de8cd72290df688de0ce4071f42d7bfdec8507a24716ae2ea15c3f3cf5`，共 `2,011` 对。
- 官方 testing archive：`/Users/shihao/Downloads/os_dataset/testing.zip`，SHA256
  `e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218`，共 `424` 对。
- 后续训练参考 recipe：official training.zip 2011 训练、testing.zip 424 final only；batch `12`；
  OneCycle max LR `1.25e-4`；120k steps；train/test iters `12/32`；public augmentor；`flow_loss + mat_flow_loss`。
- 官方已训练 GDROS 权重、baseline 目录、DINOv2、E-028B formal checkpoint 只读；训练只能发生在实验副本。

## 中心创新

用稀疏/半稠密跨模态观测和低维仿射状态之间的递归反馈，替代 GDROS 的 dense flow state 与 dense GRU。
baseline 训练协议已校准。E-031A 已表明 E-028B 特征强，但 GDROS backend interface 不兼容；用户明确要求
不要继续设计 adapter，而是围绕 E-028B 特征原生设计后续 matcher。

## 最新证据

- E-028B：`COMPLETED / SAFE_DUAL_ADAPTER_STRONG_GAIN`。safe dual adapter AEPE `21.418`。
- E-029A：`COMPLETED / NO_SYSTEM_INTEGRATION_GAIN`，但后续诊断判定该负结果不可信。
- E-029A-GDROS-RETRAIN-DIAG：`E029A_NEGATIVE_NOT_TRUSTWORTHY`。short retrain AEPE `53.779`。
- E-030A Track B validated recipe：AEPE `1.4347`、CMR@1 `37.0283%`、CMR@2 `83.4906%`、
  CMR@5 `99.5283%`，final checkpoint 超过 released checkpoint replay 的 AEPE/CMR@1/2。
- E-030A Track A repo default：AEPE `1.5798`、CMR@2 `77.8302%`。
- E-028B：`COMPLETED / SAFE_DUAL_ADAPTER_STRONG_GAIN`。safe dual adapter feature-only AEPE `21.418`。
- E-031A：`COMPLETED / BRIDGE_REQUIRED_BUT_ADAPTABLE`。E-028B feature-only retrieval top32 `85.93%`
  vs GDROS native `23.77%`，但 direct replacement 因 `[B,1024,32,32]` vs `[B,256,64,64]` 失败；
  simple bridge AEPE `18.6692`、CMR@2 `1.1656%`，说明不应硬塞进 GDROS backend。
- E-032A：`COMPLETED / NATIVE_MATCHER_SIGNAL_FOUND`。best native consensus AEPE `15.3632`、
  CMR@2 `4.0633%`、CMR@5 `19.9378%`，相对 E028B feature-only AEPE `21.4176` 改善 `28.27%`。
  Reviewer APPROVE；无 GDROS backend adapter、无新 descriptor、无 testing.zip。
- E-033A：`COMPLETED / MATCHER_DECODER_BOTTLENECK`。GT-neighborhood rank≤32 visible `98.67%`，
  raw rank mean/P50/P90 `4.21/2/9`，topK capture raw top1/top4/top8/top16/top32
  `46.54/77.77/89.05/95.66/98.67%`；GT-neighborhood oracle affine AEPE `3.144`、CMR@2
  `29.93%`、CMR@5 `85.11%`；matcher visible-miss `79.86%`，feature-not-visible `1.12%`。
- E-034A：`COMPLETED / COARSE_MATCHING_LIMITED`。selected coarse token in GT-neighborhood `17.62%`，
  coarse-only AEPE `99.877`、CMR@2 `0.049%`，local-refined AEPE `97.067`、CMR@2 `0.051%`，
  selected-coarse oracle-local bound AEPE `99.154`；E-033A oracle parity AEPE `3.148`、CMR@2
  `29.84%`；visible-miss `66.99%`。说明 recurrent decoder 未保住 raw top1/topK correlation 信号。
- E-034B：`COMPLETED / RAW_TOP1_AFFINE_GAIN`。raw_top1_all_source coarse capture `46.111%`、
  AEPE `15.788`、CMR@2 `1.872%`；top1_mutual capture `63.653%`、AEPE `14.050`、CMR@2
  `3.306%`；top1_trimmed50_irls AEPE `9.969`、CMR@2 `6.136%`、CMR@5 `31.035%`，优于
  E-032A AEPE `15.363` / CMR@2 `4.063%`，但仍远低于 GDROS。
- E-034C：`COMPLETED / COARSE_INLIER_SELECTION_GAIN`。best inlier `graph_top4_r2_t32_top384_irls`
  inlier `78.960%`、AEPE `9.586`、CMR@2 `5.515%`、CMR@5 `27.836%`；best affine/CMR
  `rank_growth_dual_256_irls` inlier `60.491%`、AEPE `8.550`、CMR@2 `8.996%`、CMR@5
  `38.783%`。只读归因显示 coverage 非主因：quadrant `4/4`、region coverage 约 `49-52/64`；
  但 CMR@5 仍仅 `38.783%`，不足以靠 local refinement 追上 GDROS CMR@2。
- E-034D：`COMPLETED / MIXED_CMR5_BOTTLENECK`。failure pairs `265/404`；failure pair top32
  oracle CMR@5 `93.745%`、CMR@16 `99.946%`，说明大多可恢复；remaining outlier proxy ≥35%
  为 `217/265`。补算 CMR@16：E-034C best affine AEPE `8.550`、CMR@2 `8.996%`、CMR@5
  `38.783%`、CMR@16 `87.942%`；all-pair top32 oracle AEPE `1.728`、CMR@2 `72.191%`、
  CMR@5 `94.768%`、CMR@16 `99.964%`。因此当前需要把 16px basin 推向 5px/2px。
- E-035A：`COMPLETED / REFINER_LEARNING_LIMITED`。coarse replay AEPE `8.550`、CMR@2 `8.996%`、
  CMR@5 `38.783%`、CMR@16 `87.942%`；best refined stage1+stage2 AEPE `13.128`、CMR@2
  `8.254%`、CMR@5 `31.139%`、CMR@16 `73.877%`；direct head AEPE `13.175`、CMR@5
  `30.755%`；best window oracle CMR@5 `100.0%`。说明 local window 可达，但当前 learned refiner
  没解码出可用局部证据。
- E-035B：`COMPLETED / LOCAL_SIGNAL_WEAK`。coarse replay AEPE `9.629`、CMR@2 `7.037%`、
  CMR@5 `33.148%`、CMR@16 `85.019%`；best no-GT local signal `affine_predicted_r8/grad_edge_zncc`
  的 CMR@2/5 proxy `2.864%/14.027%`，top1/top4/top8 `5.314/19.059/35.721%`，GT rank
  P50 `12`；best signal top1-affine AEPE `12.943`、CMR@2 `4.244%`、CMR@5 `19.853%`。
  Window oracle CMR@5 `100.0%`，但传统 no-train local signals weak。
- E-035C：`COMPLETED / REFINER_OBSERVATION_STILL_WEAK`。coarse current AEPE `9.629`、CMR@2
  `7.037%`、CMR@5 `33.148%`、CMR@16 `85.019%`；reference 对照 AEPE `8.550`、CMR@5
  `38.783%`；window oracle CMR@5 `99.067%`。best trained `efficient_stage1_stage2_weighted`
  AEPE `16.110`、CMR@2 `4.787%`、CMR@5 `23.241%`、CMR@16 `66.034%`；LoFTR-style
  weighted CMR@5 `23.215%`，RoMa-style weighted CMR@5 `22.211%`。重要 caveat：watchdog
  分配 GPU0 但 torch CUDA unavailable，实际 CPU、1 epoch、8 tokens/pair，不能作为强训练否定。
- E-035D：`COMPLETED / STRONG_REFINER_STILL_WEAK`。CUDA gate PASS，device `cuda:0`，GPU
  `NVIDIA GeForce RTX 4090`；8 epochs、64 tokens/pair、1607 train pairs。coarse current
  AEPE `9.629`、CMR@2 `7.037%`、CMR@5 `33.148%`、CMR@16 `85.019%`；主 reference
  AEPE `8.550`、CMR@5 `38.783%`；window oracle CMR@5 `99.067%`。best strong
  `efficient_stage1_weighted` AEPE `15.978`、CMR@2 `5.130%`、CMR@5 `23.932%`、CMR@16
  `66.430%`；Strong RoMa CMR@5 `23.161%`；uncertainty cleanup only `+0.200pp`。
- E-035E：`COMPLETED / REFINEMENT_METHOD_SHOULD_CHANGE`。P0/G0 PASS；更多训练、容量与监督形式
  sweep 均没有超过 coarse reference。best scale `tokens64/epoch_16` AEPE `15.911`、CMR@5
  `24.302%`、CMR@16 `66.597%`；best supervision `direct_offset_huber` CMR@5 `19.479%`；
  window oracle CMR@5 `99.067%`。因此当前 isolated local refiner / paper-style refinement 路线停止。
- E-036A/B/C：三条替代 refinement 均未超过 E-034C coarse reference。有效 formal 为 `run-02`。
  - E-036A `GROUP_REFINER_FAIL`：best `no_message_passing` AEPE `12.513`、CMR@2 `5.676%`、
    CMR@5 `27.091%`、CMR@16 `75.436%`。
  - E-036B `FINE_FEATURE_REFINER_FAIL`：best `no_fine_feature` AEPE `12.970`、CMR@2 `5.143%`、
    CMR@5 `26.233%`、CMR@16 `74.486%`。
  - E-036C `ADAPTIVE_SPAN_FAIL`：best `uncertainty_span_no_train` AEPE `10.153`、CMR@2 `6.678%`、
    CMR@5 `31.691%`、CMR@16 `83.209%`。
  说明后处理式 refinement 不足，下一步应在 matching/correlation 之前做跨模态特征交互。
- E-037A：`COMPLETED / FEATURE_INTERACTION_GAIN_FOUND`。P0/G0/G1 PASS；训练 `60000`
  optimizer steps，CUDA `cuda:0`，E028B 只读，未用 GDROS backend/dense GRU/testing.zip/confirm/OS424。
  最佳为 `NO_INTERACTION_TRAINED_PROJECTION_lr125e4`：AEPE `7.247`、CMR@2 `10.720%`、
  CMR@5 `45.939%`、CMR@16 `92.337%`；E-034C baseline replay 为 AEPE `8.550`、CMR@2
  `8.996%`、CMR@5 `38.783%`、CMR@16 `87.942%`。关键判读：收益主要来自 matcher-facing
  projection/norm 适配；plain 2L/4L cross-attention 反而使 final AEPE/CMR 变差。

## 当前判断

停止 E-028B-to-GDROS-backend adapter 路线。E-028B 的价值在特征/检索质量，应设计原生 matcher：
利用 32x32 强特征做全局候选检索、局部 refinement、几何一致性聚合，而不是复用 GDROS dense GRU/backend。

## 唯一下一步

执行 E-038A：

- 在 E-037A 成功的 `NO_INTERACTION_TRAINED_PROJECTION` 上系统调参/增强；
- 目标是测 matcher-facing projection/norm 适配层的上限，而不是引入复杂 feature interaction；
- E-034C 后半段 coarse selection / rank-growth / IRLS 必须保持不变；
- 做 depth/width/norm/modality-specific/shared/whitening/rank-loss/hard-negative/LR/schedule ablation；
- 主要看是否把 AEPE 从 `7.247` 压到 `<=6.5` 或 CMR@5 从 `45.939%` 推到 `>=52%`；
- 不接 GDROS backend，不 finetune E028B 本体，不用 testing.zip，不自动启动 E039/rescue。
