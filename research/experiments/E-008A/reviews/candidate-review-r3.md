# E-008A Candidate Generator Review R3

结论：`APPROVE`

本轮仅定向复核 R2 F1：candidate region-scale trace、负分 membership 边界，以及 selector checkpoint 对 candidate trace/P0/P1 的 provenance 绑定。未修改实验代码，也未重审冻结且未受影响的方法模块。

## Finding Closure

### F1 CLOSED: region-scale membership

- `generator.py:67-73` 不再以 score 正负推断尺度命中。实现按每个 proposal 的 `proposal_region_ids` 索引对应尺度的 `region{grid}_indices`，再用 candidate 的 `target_region_ids` 对冻结 top-L matched target-region IDs 做显式集合 membership。
- 该关系直接表达“candidate target region 是否属于该 source region 的 top-L match”，与 region cosine score 的数值域无关；合法负分和 absent sentinel 不再混淆。
- `test_core.py:36-58` 的 fixture 将全部 `region{grid}_scores` 设为负值，同时提供明确的 matched target-region IDs；`test_core.py:98-119` 经过 hard union/dedup 后仍要求有效 candidate 具有非零 `region_scale_mask`。该回归路径覆盖了 R2 的负分反例。

### F1 CLOSED: checkpoint-bound candidate trace

- `evaluate.py:322-323` 从实际 selector checkpoint 文件计算 SHA256，再加载模型；不是信任 checkpoint 内部自报值。
- `evaluate.py:93-112` 将实际 `selector_checkpoint_sha256` 写入 candidate trace，并在已有 trace 复用时精确校验；checkpoint 被替换或 trace 缺字段会 fail closed。
- P0 在 `evaluate.py:231-239` 持久化同一 selector checkpoint SHA；P1 在 `evaluate.py:243-262` 要求 P0 SHA 与当前实际 checkpoint SHA 一致后才继续。
- P0/P1 均在 label 或 `read_flow` 访问之前完成 candidate generation 和 trace persistence，因此修复没有破坏 no-GT materialization 顺序。

## Verification

- 本地 stage `source_manifest()` SHA256：`415bde6e2572a441971f3e4f4847b8922a77645b98f82128beeafb8d4114c4bc`，与派工身份一致。
- 远端定向测试：`13 passed, 2 warnings in 3.61s`。
- 审阅范围内未发现新的方法语义、GT 泄漏、trace 完整性或缓存身份阻塞。

## Decision

R2 F1 已完整关闭。批准候选生成器通过本轮定向审阅，并按冻结 E-008A gate 顺序继续；本结论不预判 P0/P1 方法结果。
