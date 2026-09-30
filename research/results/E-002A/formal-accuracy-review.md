# E-002A Formal Accuracy Independent Review

日期：`2026-08-04`

方法版本：`M001 v0.3`

结论：`APPROVE`。本结论只覆盖 accuracy、diagnostics 和 G0-G4；G5 由独立资源审计决定。

## 身份与完整性

- Watchdog：`20260804_102002_eb5f8e19`，GPU 3，`succeeded`
- Git：`ee6b6216fe780bc03bfb4759ae094daa0cfddc60`
- 正式源码快照聚合 SHA256：`05f428033300812330802c424fdafec99355754d48e94388ae32dc11c1c0d361`
- Checkpoint SHA256：`80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- 官方数据 manifest：`c129ae7147a7d9ad1c1a7ead35bb6da2d8c85ad8da73eef4723cf277e5ea7e96`
- 35/35 artifact、19/19 source snapshot、checkpoint、baseline、31 个 dependency 的运行前后
  SHA256/size 全部一致。

原始结构严格为：4,240 条非 GT variant 记录、848 条 retrieval-pair、108,544 条
retrieval-anchor、13,992 条 oracle 和 2,000 x 424 pair-bootstrap 索引；无重复、缺 pair、
意外非有限值或静默 solver failure。

## G0

- Source/target feature parity：`0 / 0`
- CorrBlock/K-row/chunked 最大差：`6.675720e-6 < 2e-5`，top1 mismatch `0`
- 6/6 anchor 设计 rank 3，最大条件数 `2.303243`
- Continuous-GT oracle：AEPE `2.0527853e-5 px`，P99 `3.6511464e-5 px`
- Nearest-inbounds oracle：AEPE `0.275215613 px`，pair P95 `0.653120843 px`，failure `0%`
- Finite：`424/424`；solver failure：`0`

因此 G0 独立重算为 `PASS`。

## G1-G4 与判负

10 个预注册非 GT 变体的 CMR@1/2/5 全为 `0%`，failure 全为 `100%`，
PairLocalReady 全为 `0%`。AEPE：

| 变体 | AEPE px |
| --- | ---: |
| cosine-hard K128 | `79.952541` |
| cosine local-soft K128 | `79.972727` |
| raw-hard full-grid K128 diagnostic | `93.211500` |
| raw-hard K64 / K128 / K256 | `92.023059 / 91.452119 / 90.659356` |
| K-anchor dual K128 | `87.115606` |
| raw local-soft K128 | `91.470398` |
| 1/16 average-pool / decimation | `72.578055 / 94.430083` |

Raw 的 R@1,r1 / R@8,r1 / R@32,r1 / R@64,r2 为
`5.9513 / 22.1111 / 42.5588 / 76.0058%`；cosine 为
`7.8920 / 26.3970 / 46.9063 / 79.4373%`。最佳 confidence AUROC 仅为 raw entropy
`0.641329`，任一信号满足 pair top32 precision >=60% 的图像比例均为 `0%`。

Top32 select-and-reject oracle 的 raw/cosine grid mean 为 `1.659629 / 1.601127 px`，
两者 PairLocalReady 均为 `100%`；top64 oracle 也均为 `100%`。但检索 recall 未同时达到
G3 的 `40/65/80%`，visibility 提升为 `0 pp`。

独立重建全部 13,992 个 oracle correspondence、FP32 ordinary LS、坐标共轭和 16x16
geometry 后，33 个 ID、availability、rank、point count 和 PairLocalReady 全部一致。

最终判定与 `gates.json` 逐字段一致：G1/G2/G3/G4 均 `FAIL`。完整判负条件未成立，因为
top64 oracle PairLocalReady 为 `100%`，故不能把冻结 checkpoint 特征或 SparseGeo 方法族
登记为失败。

## GT 隔离

4,240 条正式变体全部为 `uses_ground_truth=false`。Pipeline 在 GT 构造前完成全部正式
匹配与求解；GT 只进入 retrieval/oracle diagnostics。424 个 non-GT trace hash 全部有效，
未发现 GT、valid-mask 或测试标签进入正式方法路径。

正式路径：
`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002A/run-01/accuracy`
