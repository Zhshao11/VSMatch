# E-045A-FEATURE-EXTRACTION-DIVERGENCE

方法版本：`M001 v3.21 draft`

状态：`USER_REQUESTED / READY_FOR_ENGINEERING`

## 1. 目标

从 E-044A 的最强参考线继续发散，但这次不再围绕 matcher 小修小补，而是从特征提取前端开始换范式：

> 先把 OPT/SAR 特征本身做强，再看它们是否能把候选质量、粗匹配纯度和最终 affine 指标抬上去。

本实验的核心不是“把 matcher 调顺”，而是验证：

1. 特征前端能否显著提高跨模态候选的可分性；
2. 哪种特征范式最适合 OPT-SAR；
3. 这些前端改动能否稳定超过 E-044A 的最佳 matcher-facing 参考线。

## 2. 当前参考线

E-044A 最强当前参考：

- `e044a_k64_l4_noenergy`
- AEPE `6.638`
- CMR@2 `11.951%`
- CMR@5 `49.571%`
- CMR@16 `93.999%`

这条线先冻结，作为 E045A 的比较基线，不改动、不污染。

## 3. 论文范式参考

这一步先借鉴“特征提取/预训练”而不是只看图像匹配论文：

- `CROMA`：mask reconstruction + cross-modal contrastive，适合做 OPT/SAR 联合表征。
- `DINO-MM`：把 SAR-optical 作为联合 self-supervision 输入，并用 modality dropout 逼模型学两模态共识。
- `DeCUR`：common / unique representation decoupling，适合把“模态共识”和“模态特有结构”拆开。
- `USat` / 类似 multi-sensor encoder：面向不同传感器做 patch embedding / positional encoding 适配。

## 4. 发散方向

### A. `SAR_STRUCTURE_STEM`

只增强 SAR 前端，让它更像“结构感知”的输入，而不是单纯的通道投影。

建议包含：

- log-intensity；
- gradient / edge；
- local variance / texture；
- census / speckle-robust structural cue；
- 轻量 residual stem 到 shared latent。

目标：提高 SAR 侧可区分性，尤其是把“高能量但不稳定”的点压下去。

### B. `COMMON_UNIQUE_SPLIT`

把特征拆成共识通道和模态特有通道：

- shared trunk 学跨模态共识；
- modality-specific residual 学 OPT / SAR 各自特有结构；
- common / unique 用解耦或冗余抑制约束；
- matcher 可先只看 common，再把 unique 作为辅助证据。

这条线对应 DeCUR 风格。

### C. `MASKED_CROSS_MODAL_PRETRAIN`

把一侧或局部 patch mask 掉，让另一模态或上下文去补：

- OPT masked → SAR / joint context 预测；
- SAR masked → OPT / joint context 预测；
- 位置对齐的 contrastive / reconstruction 组合；
- 适合测试“跨模态对应观测”是否真能学出来。

这条线对应 CROMA / multimodal masked modeling 风格。

### D. `TEACHER_CONSISTENCY`

保留强 optical teacher，把 SAR 往 teacher latent 靠：

- optical branch 继续稳住强预训练表征；
- SAR branch 学 teacher-consistent latent；
- 再加局部结构一致和模态不变约束；
- 优先避免两边一起漂。

这条线对应“强 teacher + SAR adaptation”路线。

## 5. 建议执行顺序

优先级从高到低：

1. `TEACHER_CONSISTENCY`：最接近现有 E028B/E027A 家族，改动最小，先确认“再强化 teacher 约束”是否还能继续抬高候选纯度。
2. `SAR_STRUCTURE_STEM`：把 SAR 侧结构输入做厚，看结构 cue 是否是真正增益来源。
3. `COMMON_UNIQUE_SPLIT`：检验“共识/私有”拆分是否比单一 latent 更利于跨模态区分。
4. `MASKED_CROSS_MODAL_PRETRAIN`：作为更激进的预训练/自监督分支，成本最高，放最后。

如果资源允许，前两条可先并行，后三条串行推进。

## 6. 统一比较协议

为避免前端发散把后端搅乱，E045A 统一保持 matcher 侧尽量固定：

- 先 materialize 前端特征；
- 再复用同一套固定 matcher / selection / affine evaluator；
- 不碰 GDROS baseline；
- 不回写 E-044A 参考结果；
- 所有 dev/formal 指标都必须先落盘特征，再 join truth-flow。

## 7. 评价指标

前端层面：

- GT-neighborhood capture @ top1/top4/top8/top16/top32；
- rank P50 / P90；
- feature AUROC / AP；
- modality gap / commonality；
- feature drift / collapse；
- optical-SAR agreement；
- SAR structure selectivity。

下游层面：

- candidate precision / recall；
- selected inlier ratio；
- AEPE；
- CMR@2 / CMR@5 / CMR@16。

## 8. Gate

### P0

必须 PASS：

- E-044A 冻结参考可复现；
- E-044A noenergy 作为比较基线记录；
- 不改 GDROS baseline；
- 不改 E-044A 参考结果；
- 前端特征先落盘，再做 GT join；
- 各分支参数与输入输出边界清楚。

### G0

必须 PASS：

- train-small 有正向信号；
- loss 非零且梯度能到前端模块；
- 至少一条前端分支在 train 上显示候选质量提升；
- reviewer 认可特征提取、label 对齐和隔离协议。

### G1

判读按 E-044A noenergy 基线：

- `FEATURE_FRONTEND_GAIN_FOUND`：前端发散后，AEPE / CMR@5 / CMR@16 至少一项超过基线，且候选纯度不崩；
- `COMMON_UNIQUE_PROMISING`：common/unique 或 teacher consistency 明显改善候选质量，但还没完全转化到 final affine；
- `MASKED_PRETRAIN_PROMISING`：masked cross-modal 预训练带来稳定特征收益；
- `FEATURE_FRONTEND_PLATEAU`：几条前端范式都不超过 E-044A；
- `IMPLEMENTATION_OR_LABEL_SUSPECT`。

## 9. 输出

- `RESULT_CARD.md`
- `feature_manifest.json`
- `feature_scorecard.json`
- `front_end_ablation_summary.json`
- `retrieval_metrics.json`
- `candidate_quality_metrics.json`
- `baseline_comparison.json`
- `per_pair.jsonl`
- reviewer conclusion
- `RUN_MANIFEST.json`

完成后停止，等待下一步方法决策；不要自动跳 matcher 侧新设计。
