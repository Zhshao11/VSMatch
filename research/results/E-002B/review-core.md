# E-002B Core Implementation Review

日期：`2026-08-04`

审阅者：独立子 agent `e002b_core_design_review`

结论：`APPROVE`

## 绑定与范围

- `METHOD_SPEC.md` SHA256：
  `53e83d376d5055d2f009b5943d74835f4830a18994461b34e8da6a1ac85aeb5f`
- `E-002B.md` SHA256：
  `19ec4d70b19a3d313323c3d6959e5059b8d30c224480e60ddea9567893bbaf49`
- 绑定：`BIND-001`
- K128 anchor SHA256：
  `2872a818932f138d79a0a0aebf3fec2bbadb1939e839788f6f103c1e1f16efb6`

## 已核验调用链

- optical 是 source、SAR 是 target，flow 为 `target-source`；stride-8 与 pixel affine 共轭
  沿用 E-002A 已审阅实现。
- C8-main 使用 K128 cosine top8；256 个 balanced triples 穷举 top8^3，coarse Q1024、
  full R32。
- BIND-001 排序键 4/5 对全部当前 scoring anchors 求和；collision source 分别贡献 `4`/`L`。
- 两轮 candidate reassignment + FP32 ordinary LS 后执行只验证的 final validation，没有第三次
  拟合。
- abstention 输出 identity 且计入 method failure；没有 baseline、hard-top1 或 GT fallback。
- GT 在非 GT 输出与 trace hash 固化后才读取；正式方法路径不依赖 truth flow。
- 未构造 cross-image `4096 x 4096` correlation，也未调用 context、CorrBlock、GRU、
  upsampler 或 cost-perceiver。

## 审阅修复闭环

- 修复 GT 提前加载、abstention failure accounting、C8 main-only 候选数和 gate provenance。
- 修复 hypothesis invalid reason 排他统计、collision/outlier 分离和 final validation 单测。
- anchor 归档改为直接调用 E-002A `anchor_manifest`；外层 key、全部字段、
  `normalized_design={point_count:128, rank:3, condition:2.30112624168396}` 与正式 K128 record
  逐字段一致。
- 最终远端 `28/28` 单测通过；核心 reviewer 确认无剩余代码阻断。

残余风险只属于正式 G5-R 和数据实证结果，不能解释为实现问题。
