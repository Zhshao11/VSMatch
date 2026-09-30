# M001 v0.5 方法增量

基线版本：`M001 v0.4 / E-002B / BIND-001`

当前实验：`E-002B-R`

状态：`FROZEN`

## 唯一变化

只把 v0.4 中由 Python 循环、小张量分配和大量 kernel launch 主导的几何后端，替换为语义
等价的向量化、融合或编译执行后端。

这不是新匹配方法，也不改变候选选择预算。当前实验验证的是：全预算 B2 是否能在保持执行
轨迹和输出语义的前提下达到资源门槛。

## 必须保持不变

- K128 public-stratified anchors，cosine top8。
- 256 triples 和全部 `256 x 8^3 = 131,072` hypothesis ID。
- BIND-001 collision、六级排序、32 sentinels、Q1024、R32。
- 两轮 ordinary LS 和第二轮后的 no-fit final validation。
- FP32、坐标方向、阈值、tie-break、abstention 和 identity dense contract。
- 四个预注册 variant；C8-main 为主结果。
- GT、valid mask、baseline flow 和测试标签隔离。

## 允许的实现增量

- 缓存与样本无关常量。
- 固定 block 向量化和确定性流式 top-Q。
- 等价的 stable sort/segment winner，避免 pairwise collision tensor。
- `torch.compile`、CUDA Graph、Triton 或 CUDA extension，但必须触发对应 reviewer 条件。
- 批量线性代数只有通过预注册数值容差才可使用；否则保留 reference 小段。

## 禁止变化

- 减少/抽样 hypothesis、数据依赖早停、自适应预算。
- 修改 T/L/Q/R、sentinel、阈值、排序或 tie-break。
- approximate top-k、混合精度、量化、跨样本缓存。
- WLS/IRLS、B1/B3、descriptor、训练、局部精修或 baseline fallback。

## 自动 gate

1. P0-early：synthetic、2-pair smoke、固定 10 对，逐轨迹 differential test。
2. P1：geometry p50 `<=80 ms`、相对 reference `>=8x`、warm allocated `<=284 MiB`。
3. 定向审阅：语义/checker；若使用编译或自定义 GPU 后端，再审 benchmark/cache/sync。
4. G5-R2：总 p50/p95 `<=50/63 ms`、5-process CV `<=5%`、cold/warm `<=284 MiB`。
5. P0-full：全部 424 对、四 variant、无 GT，`424 x 4` 全部轨迹 PASS。
6. 只有以上全部通过才运行预注册 accuracy。

## 当前实现修复范围

只修复 `assign_candidates_batched()` 的投影运算顺序，使 pair 47 的中间 assignment 离散轨迹
与 reference 一致。不得借此改容差、排序、预算或最终 gate。

## 停止条件

- differential test 失败：只允许本次实现错误修复；未通过则停止并返回用户。
- P1/G5-R2 失败：等价后端救援停止，不运行 accuracy。
- 资源通过但 accuracy 失败：B2 selector 停止并做失败审计。
- 全部通过：阶段成功，等待用户批准下一阶段，不自动进入 E-002C。
