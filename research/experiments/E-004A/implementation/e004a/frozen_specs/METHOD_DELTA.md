# M001 v0.7 增量：覆盖约束的自适应 Anchor 预算

当前实验：`E-004A`

状态：`APPROVED / READY`

## 方法决定

用户批准先解决 source anchor acquisition：每个空间区域保留最低点数，把剩余预算动态分配给
更易跨模态匹配的区域，并用同区域 reserve 替换低质量 primary。E-004A 只验证该模块，不加入
新的 candidate consensus。

## 相对 v0.6 的变化

- 固定 K128 anchor 改为 M256 proposal -> 动态 K128 primary。
- `64x64` source map 划分为 `8x8` 区域；每区 4 个固定 proposal、至少选 1、最多选 3。
- pre-search head 使用 source 局部特征和 `16x16` coarse SAR correlation 预测 recoverability。
- primary 完成 fine top8 后，由 post-search verifier 判断低质量点；最多 R64 个同区域单轮替换。
- 最大 fine search 固定为 `(128+64)x4096`，禁止无界重试。
- E-004A 的仿射精度使用 GT-nearest top8 oracle，只隔离 anchor acquisition，不能作为部署结果。

## 固定不变量

- official `512x512`、冻结 GDROS post-Transformer、stride-8 `64x64`。
- raw truth_flow 是唯一训练/诊断标签；datum 不作标签。
- optical source -> SAR target，flow 为 `target-source`。
- exact cosine top8 与 E-003A tie-break 一致。
- E-003A group-isolated train/dev；OS 424 与 external confirm 禁止访问。
- 不训练 descriptor/backbone/Transformer，不加入 RG-SAC、局部精修、hypothesis 或 baseline fallback。

## Gate 摘要

- P0 oracle：M256 在覆盖约束下必须显著提高可恢复 anchor 数，并把 oracle AEPE 降到 `<=2.5 px`，
  PairLocalReady 保持 `>=99%`；失败则不训练。
- G0：数据/坐标/缓存/梯度/预算/替换轨迹自动测试和两轮定向审阅全部 PASS。
- G1：最终动态 K128 recoverable fraction `>=50%`，median recoverable count `>=64`；replacement
  相对 no-replacement 提升至少 `3 pp`，且 oracle AEPE `<=2.5 px`、PairLocalReady `>=99%`。
- G2：anchor acquisition p50/p95 `<=6/10 ms`，slim end-to-end p50/p95 `<=18/25 ms`，
  warm allocated `<=284 MiB`，同进程 GDROS p50 speedup `>=5x`。
- 任一 gate 失败写 RESULT_CARD 后停止；不得自动建立 RG-SAC 或 E-004B。
