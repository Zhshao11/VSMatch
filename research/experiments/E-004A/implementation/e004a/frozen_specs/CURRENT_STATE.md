# 当前研究状态

更新时间：`2026-08-05`

## 当前身份

- 当前方法：`M001 v0.7 / Coverage-Constrained Adaptive Anchor Budgeting`
- 当前实验：`E-004A`
- 方法任务：`019fca19-1b11-78b3-8106-3bf9cdaee3d9`，ACTIVE
- 实验任务：`019fc85d-350e-7941-ac69-89c626641fd9`，ASSIGNED
- 当前状态：`ADAPTIVE_ANCHOR_APPROVED / E-004A READY`

## 数据绑定

- 官方 training archive：`/Users/shihao/Downloads/os_dataset/training.zip`
- 大小：`5,727,075,338 bytes`
- SHA256：`2887e7de8cd72290df688de0ce4071f42d7bfdec8507a24716ae2ea15c3f3cf5`
- 内容：`2,011` 对，每对含 opt、sar_warped、truth_flow、datum。
- E-003A 活动输入固定为原始 `512x512`，stride-8 为 `64x64`；禁止 resize/crop/augment。
- raw truth_flow 是唯一监督/指标权威；datum 不作为训练或评测标签。
- 官方 424 testing：仅作为开发污染已披露的历史官方基准，不再用于训练、调参、早停或分支选择。

## 已完成诊断

- E-002B-R：P0-early PASS，optimized geometry p50 `205.06 ms`、speedup `4.37x`，P1 FAIL。
- E-002B-ACC-DIAG：官方 424 对完整、finite、GT 隔离和 evaluator parity PASS。
- C8-main AEPE `10.4349 px`、PairLocalReady `75.71%`、grid mean `17.233 px`，accuracy FAIL。
- cosine top8 GT oracle AEPE `1.4792 px`、PairLocalReady `99.53%`，候选集上限充足。
- 固定 B2 六键排序、131,072 hypotheses 和等价后端永久停止；不否定冻结特征和 top8 候选。
- E-003A 三个 main seed 的 best dev grid 为 `11.10/11.70/11.57 px`，稳定未达到 `8 px`。
- learned-main AEPE `10.6808 px`、PairLocalReady `67.08%`、coverage `0.74%`；G1 FAIL。
- corrected top8 oracle AEPE `2.8410 px`、PairLocalReady `100%`，候选容量仍充足。
- v0.6 的独立候选 MLP、九类 dustbin 竞争和单一仿射残差反馈停止；不外推到所有 learned selector。

## 唯一下一步

执行 E-004A：从 M256 区域 proposal pool 学习粗跨模态 recoverability，在每区保底 1、封顶 3
的约束下动态选择 K128，并允许最多 R64 个同区域单轮候补替换。先过 oracle headroom gate，
再训练和测资源；只隔离验证 anchor acquisition，不加入 RG-SAC 或新的 candidate selector。

## 停止边界

- 不训练或微调 GDROS descriptor/backbone/Transformer。
- 不枚举三点仿射假设，不恢复 131,072 hypotheses，不加入局部精修或 baseline fallback。
- 不读取 OS 424 GT 或未来 confirm GT 来选择 checkpoint、阈值、损失、结构或超参数。
- 不自动启动 RG-SAC、E-004B、descriptor、局部精修或救援实验。
