# M001 v3.20 增量：E-038A Strong Matcher-Facing Projection

当前实验：`E-038A-E028B-STRONG-MATCHER-FACING-PROJECTION`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 背景

E-037A 已完成，判读 `FEATURE_INTERACTION_GAIN_FOUND`，但关键收益并不来自 plain cross-attention，而来自
`NO_INTERACTION_TRAINED_PROJECTION_lr125e4`：

- E-034C coarse reference：AEPE `8.550`、CMR@2 `8.996%`、CMR@5 `38.783%`、CMR@16 `87.942%`；
- E-037A best projection/norm：AEPE `7.247`、CMR@2 `10.720%`、CMR@5 `45.939%`、
  CMR@16 `92.337%`；
- E-037A plain 2L/4L cross-attention final AEPE/CMR 变差。

因此 E-038A 聚焦“matcher-facing projection/norm 适配层”的上限：系统调结构、归一化、训练目标与训练量，
看不引入复杂 interaction 时最多能把 E-034C coarse pipeline 拉到什么程度。

## 核心问题

> 在 E-034C `rank_growth_dual_256_irls` 后半段固定的前提下，增强 matcher-facing projection/norm
> 是否能继续提升 coarse ranking / inlier ratio / affine CMR？

## 协议边界

- E028B safe dual adapter checkpoint 固定只读；
- 不 finetune E028B backbone/adapter 本体；
- 不做 cross-attention / cross-image message passing；
- 只训练新增 matcher-facing projection / normalization / scoring-temperature head；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- E-034C 后半段 coarse selection、rank-growth、IRLS 必须保持不变；任何改动都不能计入主结果；
- 不使用 official testing.zip / confirm / OS424；
- projected features / selected matches 必须先落盘，再读取 truth-flow 做 metric join；
- 完成后停止，不自动启动 E039、rescue、descriptor finetune、cross-attention、full sparse feedback。

## 方法

### 输入输出

- 输入：E028B OPT/SAR feature maps，形状约 `[B, C, 32, 32]`；
- 输出：matcher-facing projected OPT/SAR feature maps，保持 token 网格与维度可被 E-034C correlation/selection 使用；
- 下游：复用 E-034C `rank_growth_dual_256_irls` coarse pipeline 计算最终 affine。

### 必做 tracks

1. `BASELINE_REPLAY`
   - 严格复现 E-034C reference；
   - 严格复现 E-037A best projection/norm；
   - 记录 raw/projected 前后的 top1/top4/top8/top16/top32 GT-neighborhood capture、rank P50/P90、
     entropy、top1-top2 margin、mutuality、inlier ratio、AEPE、CMR@2/5/16。

2. `PROJECTION_DEPTH_WIDTH_SWEEP`
   - linear、2-layer MLP、4-layer residual MLP；
   - widths `{256,512,1024}` 或等价；
   - shared projection vs modality-specific stem + shared projection；
   - 输出统一 L2 normalized descriptor。

3. `NORM_AND_WHITENING_SWEEP`
   - L2 norm、LayerNorm+L2、BatchNorm/GroupNorm variants；
   - learnable temperature；
   - optional train-set whitening / decorrelation head；
   - whitening/decorrelation statistics 只能来自 train split，禁止 dev/test leakage。

4. `LOSS_AND_NEGATIVE_SWEEP`
   - bidirectional coarse token CE；
   - soft GT-neighborhood target；
   - supervised contrastive / InfoNCE；
   - hard-negative mining：same-row/nearby/visually confusing negatives；
   - rank margin loss：提升 GT-neighborhood 相对 top wrong 的 margin；
   - entropy sharpening / confidence calibration。

5. `SCHEDULE_AND_LR_SWEEP`
   - 以 E-037A `lr=1.25e-4` 为中心；
   - 至少比较 `6.25e-5`、`1.25e-4`、`2.5e-4`；
   - final 至少评估 best train、best dev、EMA/final checkpoint。

## 训练要求：充分训练 hard gate

为了避免“训练不充分”的误判，G0 必须记录并满足：

- CUDA available true；允许多卡；
- train split 使用既有 `1607` train pairs，dev 使用既有 `404` dev pairs；official testing.zip 不用；
- train-small overfit：至少 `16` pairs，要求 coarse matching loss 明显下降，并且 top1/top4 capture 或 GT mass 明显上升；
- formal candidate train-full 至少满足以下之一：
  - 每个主候选配置 `>=60k` optimizer steps；
  - 或每个主候选配置 `>=40` full-train epochs；
  - 或 reviewer 认可的等价充分训练量；
- 主结果必须至少包含三种 LR：`6.25e-5`、`1.25e-4`、`2.5e-4`，除非 typed resource stop；
- 允许先用短 schedule 筛配置，但正式 claim 必须来自充分训练配置；
- 至少保存并 formal eval：best train、best dev、final/EMA checkpoint；
- 训练曲线必须包含 loss、train/dev topK capture、entropy/margin、final E-034C metric；
- 若资源不足，必须 typed stop 为 `E038A_RESOURCE_BLOCKED`，不能判方法失败。

## 监督目标

训练只允许用 train raw truth_flow 生成监督：

- bidirectional coarse token CE：source token 到 GT target token，target token 到 GT source token；
- soft neighborhood target：允许以 GT endpoint 周围 1-token/gaussian neighborhood 作为软标签；
- optional supervised contrastive / InfoNCE；
- optional hard-negative rank margin loss；
- optional mutual consistency / entropy margin loss；
- optional affine utility auxiliary loss：只能作为 train 辅助，dev/final 必须先输出 prediction 再 join GT metric。

必须报告每个 loss 是否非零、梯度是否进入 projection/norm/temperature heads。

## Gates

### P0：setup/replay

必须 PASS：

- E-034C reference artifacts located/replayed；
- E-037A success and projection-only interpretation recorded；
- E028B checkpoint SHA recorded and read-only；
- no E028B finetune；
- no cross-attention / cross-image interaction；
- no GDROS backend/dense GRU reuse；
- no testing.zip / confirm / OS424；
- output-before-GT-join verified。

### G0：training sufficiency

必须 PASS：

- CUDA available true；
- train-small overfit complete and positive；
- train-full training meets sufficient-training hard gate for formal candidates or typed resource stop；
- losses finite/nonzero；
- gradients reach projection/norm/temperature heads；
- strict checkpoint load verified；
- reviewer approves label construction, coordinate mapping, no leakage, training sufficiency, and baseline isolation。

### G1：dev404 projection effect

主对照 1：E-034C coarse reference AEPE `8.550`、CMR@2 `8.996%`、CMR@5 `38.783%`、
CMR@16 `87.942%`。

主对照 2：E-037A best projection/norm AEPE `7.247`、CMR@2 `10.720%`、CMR@5 `45.939%`、
CMR@16 `92.337%`。

判读：

- `PROJECTION_STRONG_GAIN_FOUND`：AEPE < `7.247` 且 CMR@5 > `45.939%`，CMR@16 不明显崩溃；
- `PROJECTION_MAJOR_GAIN_FOUND`：AEPE <= `6.5` 或 CMR@5 >= `52%`，且另一项不低于 E-037A；
- `RANKING_GAIN_FINAL_WEAK`：top1/top4 capture 或 margin/entropy 明显改善，但 final affine 不过 E-037A；
- `PROJECTION_TUNING_PLATEAU`：充分调参后不超过 E-037A；
- `IMPLEMENTATION_OR_LABEL_SUSPECT`；
- `E038A_RESOURCE_BLOCKED`。

## 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `training_curve.json`；
- `ranking_diagnostics.json`；
- `coarse_pipeline_metrics.json`；
- `projection_ablation_summary.json`；
- `per_pair.jsonl`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。
