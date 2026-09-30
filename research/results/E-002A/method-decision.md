# E-002A 方法决策

日期：`2026-08-04`

方法任务：`019fca19-1b11-78b3-8106-3bf9cdaee3d9`

输入方法版本：`M001 v0.3`

输出方法版本：`M001 v0.4`

## 决策

E-002A 终态保持 `COMPLETED / EFFICIENCY_HOLD`，`G0 PASS`、`G1-G5 FAIL`。下一分支选择
`B2 candidate-level geometric reranking`，但只登记为 `B2-bridge` 救援探针，不追认
`G3 PASS`。

## 依据

- B1 不进入：raw/cosine top1 recall 仅 `5.9513 / 7.8920%`，最佳 AUROC 为
  `0.641329`，top32 precision 门槛没有 pair 通过。
- G3 仍失败：cosine `R@8,r1 / R@32,r1 / R@64,r2` 为
  `26.3970 / 46.9063 / 79.4373%`，未达到 `40 / 65 / 80%`。
- B2 有唯一正上限：top8 select-and-reject oracle 的 raw/cosine PairLocalReady 为
  `99.2925 / 99.5283%`，top32/top64 为 `100%`。
- B3 不进入：in-bounds PairLocalReady 提升为 `0 pp`。
- learned descriptor 暂不进入：top64 oracle PairLocalReady 为 `100%`，冻结特征判负
  三条件未同时成立。

top8 select-all 的 raw/cosine PairLocalReady 仅 `1.1792 / 2.3585%`。这说明“多保留候选”
本身不是答案；E-002B 必须解决无 GT 候选选择和拒绝。

## 方法范围

E-002B 冻结 K128 和 E-002A feature，主路径为 cosine top8、anchor-balanced source triples、
固定预算仿射假设、两级几何共识、两轮候选重指派和 ordinary LS。第一轮不加入 IRLS、
learned descriptor、可见性或局部精修，以隔离候选级几何的因果贡献。

完整规范：`research/experiments/E-002B.md`。

## 资源边界

E-002A 的旧 G5 不得追认。E-002B 必须先通过新预注册的 G5-R：五进程稳定性、绝对延迟、
峰值显存、实际张量形状、无 cross-image N2，以及把固定 layout 开销分离后的 K-scaling。
G5-R 未通过时不运行正式 E-002B accuracy，方法继续处于 `EFFICIENCY_HOLD`。

## 创新判断

固定 top-k 加仿射共识仍是强几何后端，不单独构成论文创新。StereoGlue 的一对多候选与
模型引导重匹配、GC-LAC 的局部仿射一致性已经覆盖相邻思想。本轮只验证 SparseGeo 后续
“预算化跨模态候选 + 仿射反馈 + 几何停止条件”是否有可行基础。

本地文献库已有上述 prior，因此本轮不新增 `LITERATURE_REQUESTS.md` 指令。

## 协议风险

E-002B 的分支和固定预算来自 E-002A 在同一 424-pair 测试集上的预注册诊断，虽未依据
E-002B 结果调参，仍属于 adaptive follow-up。其结果可用于机制可行性和同协议工程比较，
但论文最终泛化声明需要一个未参与分支选择的 confirmatory 数据集或预隔离 split。
