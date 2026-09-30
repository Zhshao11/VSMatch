# E-006A-SELECT-DIAG：候选可选择性诊断

方法背景：`M001 v0.9`

状态：`USER_APPROVED_DIAGNOSTIC / FROZEN_FOR_ENGINEERING`

## 1. 目的

验证用户提出的怀疑：E-006A 的候选空间虽然有很强 oracle affine 上限，但已有 `30-100` 个正确候选
可能在无 GT 特征中区分度很低，导致“从候选里挑点”本身不可行。

本实验只诊断 selectability，不改变 E-006A 的 candidate generator，不扩大 proposal grid，不训练
GDROS descriptor/backbone/Transformer，不启动 E-006B。

## 2. 冻结输入

- 复用 E-006A corrected formal P0 的候选生成语义、source manifest、split、checkpoint 和 evaluator。
- 使用 official training archive 的 group-isolated train/dev：`1607/404`。
- raw truth_flow 只用于生成 train/dev 标签、oracle rows 和指标；禁止进入 inference feature。
- `datum`、OS424、external confirm 禁止访问。
- 旧无效 `p0-gate.json` 不得读取或解释；只允许使用 corrected `p0-gate-r2.json` 与重新生成的同语义候选 trace。

## 3. Candidate Unit

每条候选记录对应：

- source proposal id、source region id、source token/pixel coordinate；
- candidate target token/pixel coordinate；
- candidate source type：`global_top8`、`region_top8`、`affine_local`；
- rank、score、margin、entropy/local peakiness、target multiplicity、source/target region score；
- no-GT basin id、basin score、basin residual、leave-region-out residual proxy；
- in-bounds、geometry condition proxy、collision features。
- geometry condition proxy 是独立 feature 维度，不得混入 basin_score；当前实现绑定为第 26 维。

标签：

- primary label：raw-flow endpoint error `<=2 px`，只按离散 candidate 到 raw-flow endpoint 的距离判断。
- secondary endpoint labels：`<=1 px`、`<=5 px`，同样只看 endpoint 距离。
- secondary non-endpoint labels：same final oracle basin、in-bounds。
- endpoint labels 不与 in-bounds 做 conjunction；in-bounds 必须单独报告 base rate 和分组指标。

所有 method features 必须在读 raw flow 之前 materialize；测试需证明 feature dump 不含 GT-derived field。

## 4. 诊断模型

### D0 Oracle References

- GT candidate selector：每 proposal 选 endpoint error 最小候选，只报告上限。
- GT inlier selector：只保留 `<=2 px` candidates 并做 affine，只报告上限。

### D1 Hand-Crafted No-GT Selectors

按冻结特征测试以下排序或阈值：

- cosine/rank/margin；
- region score；
- basin residual；
- leave-region-out residual proxy；
- local correlation peakiness；
- collision/multiplicity penalty；
- simple fixed linear combination，权重在 train split 上网格选定，dev 只评估。

### D2 Learned No-GT Selector

- 小 MLP 或 logistic model，参数量 `<=100k`。
- 输入只允许第 3 节 no-GT candidate features。
- 训练目标：candidate primary label BCE + pairwise ranking loss；class imbalance 使用 train split 固定权重。
- 训练 seeds：`20260812`、`20260813`；主 seed 不得被复核 seed 替换。
- 必须报告 train/dev AUROC、AP、ECE、top-m precision/recall、pair-level valid rate。
- 必须做 feature ablation：no region、no basin residual、no local stats、score-only。

## 5. Affine Evaluation

对 D1/D2 的 dev 输出，按以下选择策略拟合 affine：

- `topK`: K in `{12,16,24,32,48,64}`，按 selector score 取前 K，要求至少 6 regions、3 quadrants。
- `threshold`: 固定 confidence threshold 取 active set，不设人工 Kmax；若超过 128 只报告，不裁剪。
- `region-balanced-top`: 每 region 最多先取 2 个，再按全局 score 补齐到 K；只作诊断。

求解：

- 主求解为 hard inlier selected ordinary FP32 affine。
- 诊断可附加 one-step trimmed LS：拟合后删去 residual top 20% 再 refit；不得称为主方法。
- 不使用 RANSAC、三点 hypothesis enumeration、IRLS、dense flow 或 baseline fallback。

## 6. 判读 Gate

本实验是诊断，不设“通过即可进入下一阶段”的成功 gate。按结果分三类解释：

- `SELECTABLE`：D2 dev AEPE `<=2.0 px` 且 CMR@2 距 dense GDROS 不超过 `3 pp`，同时 GT 2px
  selected-inlier precision `>=90%`、至少 `95%` pairs valid。
- `PARTIALLY_SELECTABLE`：D2 明显优于 D1 和 E-006A no-GT score，但 AEPE `>2.0 px` 或 CMR@2
  仍明显落后；需要新方法，不得直接训练 E-006A。
- `NOT_SELECTABLE_WITH_CURRENT_FEATURES`：D2 AUROC `<0.75` 或 top32 precision `<70%` 或 affine
  AEPE `>3.0 px`；说明当前候选特征区分度不足，下一步应转向更强 observation/descriptor/patch
  evidence，而不是继续几何 selector。

`PARTIALLY_SELECTABLE` 的“明显优于”唯一绑定如下：

- 比较 track 固定为 dev `full-main global topK-32`。
- D2 使用主 learned selector 的 `full-main global topK-32`。
- D1 使用所有 hand-crafted selectors 中 dev `full-main global topK-32` affine AEPE 最低者。
- E-006A no-GT score 以 `D1 cosine/topK-32` 作为同语义代理；除非方法任务另行冻结，不读取其他
  E-006A 旧产物作 no-GT score。
- D2 必须相对最佳 D1 同时满足：candidate AUROC 或 AP 至少一项提升 `>=0.03`，且 affine AEPE
  降低 `>=0.25 px`。否则不得判为 `PARTIALLY_SELECTABLE`。
- 若 D2 达到 `SELECTABLE`，优先判为 `SELECTABLE`，不再检查 partial 条件。

无论哪类结果，都只写 `RESULT_CARD.md` 后停止，等待方法任务和用户决策。

## 7. 自动验证和审阅

必须覆盖：

- E-006A candidate generator parity 和 corrected gate provenance；
- no GT/datum/OS424/confirm in inference features；
- train/dev split 隔离，feature normalization 只 fit train；
- labels 与 raw flow 的坐标方向、token/pixel单位；
- endpoint labels 不与 in-bounds conjunction，in-bounds 单独报告；
- geometry condition proxy 为独立第 26 维，不混入 basin_score；
- candidate feature dump 先于 label join；
- topK/threshold/region-balanced selection tie-break；
- full-image no-mask AEPE/CMR parity；
- MLP gradient、toy overfit、checkpoint load；
- reviewer 定向检查 feature leakage、label leakage、selection/evaluator 和 result interpretation。

## 8. 结果卡

`RESULT_CARD.md` 必须包含：

- E-006A frozen identity and candidate parity；
- candidate label base rate；
- D0/D1/D2 AUROC/AP/ECE/topK precision/recall；
- affine AEPE/CMR/PLR/valid；
- active count 与 selected precision 分布；
- feature ablation；
- 判读类别；
- 本诊断证明什么、不能证明什么；
- 全部 gate、review、logs、model 和 feature dump 路径。
