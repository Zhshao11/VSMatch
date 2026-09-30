# E-002B Active Method Decision

日期：`2026-08-04`

输入状态：`M001 v0.4 / BIND-001`，E-002B
`COMPLETED / EFFICIENCY_HOLD / G5-R FAIL`

决定：进入 `M001 v0.5 / E-002B-R equivalent-backend rescue`，暂不减少 hypothesis 预算，
不进入 E-002C、B1、B3、learned descriptor、WLS/IRLS 或局部精修。

冻结绑定：M001 v0.5 `METHOD_SPEC` SHA256
`eb383532c988569772ef34700067f932a14363b674f4baa0f67e5a4a1ef3f5ea`；E-002B-R SHA256
`5dc34e4efa1a9bde4e826828e0570af9131a01cb8687a3c95175042c42b28434`。

独立规范复核：`APPROVE`。确认 v0.4/BIND-001 语义与历史 SHA 未漂移，P0-full 覆盖
`424 x 4` 无 GT 全轨迹，G5-R2 汇总口径和原资源门槛无放宽。

## 结果解释

E-002B 的 424-pair accuracy 没有运行，因此没有证据否定 B2 candidate selector。已被正式
否定的是 v0.4 固定 Python/PyTorch 后端的资源资格：B2 p50/p95
`888.752/911.972 ms`，相对同次 baseline 慢约 `11.67x`。显存、score 元素数、无 hidden
cross-image N2、GT 隔离和 baseline parity 均通过。

分段证据指向执行后端而非 K-to-all matching：feature `5.678 ms`、candidate top8
`1.205 ms`，geometry reranker `886.345 ms`；其中 hypothesis generation、coarse consensus、
refit+final 分别约 `397.984/316.069/214.963 ms`。源码又包含 512 次 Python chunk 驱动、
重复小张量分配/排序和 32 个串行 refit/final-validation 循环，存在明确的等价工程优化空间。

## 为什么不先减预算

减少 triple 数、每点候选数、Q/R 或采用自适应早停会改变候选搜索覆盖率。由于当前精度
未知，把预算变化与后端重构放在同一轮会使 accuracy 成败无法归因。v0.5 因此完整保留
131,072 个 hypothesis ID 和 BIND-001，只允许执行等价优化。

从约 `889 ms` 到总 p50 `50 ms` 需要接近 18 倍加速，目标激进但不是没有依据：主要时间
落在 Python 循环和大量小 kernel launch，而不是一次不可避免的大矩阵运算。P1 先要求
geometry `<=80 ms` 且相对 reference `>=8x`；未达到即提前停止，避免无边界工程投入。

## 决策边界

- P0-early 必须在 synthetic/smoke/固定 10 对上证明 discrete trace 和最终选择等价，浮点量
  只允许固定小容差。
- P0-early 通过后仍须通过 P1 和正式 G5-R2，再在全部 424 对、四个 variant 上通过无 GT
  P0-full 全轨迹等价，才允许补跑 424-pair accuracy。
- 等价后端仍失败时，只能说明当前全预算 B2 在资源目标下不可行；随后是否减少预算必须
  另立方法版本，不得事后修改 E-002B-R。
- 当前不更新 `FAILED_METHODS.md`。

完整实施绑定：`research/experiments/E-002B-R.md`。
