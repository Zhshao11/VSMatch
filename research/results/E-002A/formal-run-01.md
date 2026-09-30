# E-002A Formal Run 01

日期：`2026-08-04`

方法规范：`M001 v0.3`

结论：`COMPLETED / EFFICIENCY_HOLD`

E-002A 的实现和坐标协议通过 G0，但所有非 GT 直接解码均未通过 G1-G4。资源路径取得
`9.139x` 中位延迟加速和 `12.368%` 峰值显存下降，但严格 G5 因五进程稳定性和 K-scaling
计时线性两项失败。因此实验在方法决策点停止，不启动 E-002B。

## 运行身份与完整性

- 远端 Git HEAD：`ee6b6216fe780bc03bfb4759ae094daa0cfddc60`
- E-002A 19 文件源码快照聚合 SHA256：
  `05f428033300812330802c424fdafec99355754d48e94388ae32dc11c1c0d361`
- Checkpoint SHA256：`80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- 官方数据 manifest：`c129ae7147a7d9ad1c1a7ead35bb6da2d8c85ad8da73eef4723cf277e5ea7e96`
- Accuracy watchdog：`20260804_102002_eb5f8e19`，GPU 3，`succeeded`，return code 0
- Benchmark watchdog：`20260804_102002_f4e2dea6`，GPU 4，`succeeded`，return code 0
- 正式 accuracy：424 对，4,240 条非 GT 变体、108,544 条 anchor retrieval、13,992 条
  oracle 记录
- 正式 benchmark：2 methods x 2 contracts x 5 fresh processes；每进程 20 warm-up、200
  measured
- 单元测试：`27/27` 通过；续跑复核日期 `2026-08-04`
- Accuracy manifest：`35/35`；benchmark manifest：`218/218`

Baseline 源码没有被修改或重置；E-002A 位于独立未跟踪目录。正式产物保存了源码快照、
直接依赖哈希、数据与 checkpoint 运行前后校验、原始 timing 和逐对指标。

## G0-G5

| Gate | 结果 | 依据 |
| --- | --- | --- |
| G0 | PASS | feature parity 0；score 最大差 `6.676e-6`；oracle、rank、condition 和 finite 全通过 |
| G1 | FAIL | 10 个非 GT 变体 CMR@1/2/5 与 PairLocalReady 均为 `0%` |
| G2 | FAIL | top1 recall、结构覆盖、AUROC 和 top32 precision 未同时过门槛 |
| G3 | FAIL | top32 oracle 可恢复，但实际 R@8/R@32/R@64 未同时达到 `40/65/80%` |
| G4 | FAIL | in-bounds PairLocalReady 相对全部 anchors 提升 `0 pp`，且 in-bounds 不满足 G2/G3 |
| G5 | FAIL | 五进程 p50 CV `13.713%`；K64/128/256 timing 线性 R2 `0.070739` |

最终状态必须为 `EFFICIENCY_HOLD`。完整冻结特征判负条件未成立：raw/cosine 虽都未过
G2，但两者 top64 oracle PairLocalReady 均为 `100%`。因此不得把冻结 checkpoint 特征或
SparseGeo 方法族写入 `FAILED_METHODS.md`。

## Accuracy 与检索

10 个非 GT 变体的 CMR@1/2/5 全为 `0%`、failure 全为 `100%`。主要 AEPE：

| 变体 | AEPE px | 95% pair-bootstrap CI |
| --- | ---: | ---: |
| raw-hard K64 | `92.023059` | - |
| raw-hard K128 | `91.452119` | `[89.7771, 93.1555]` |
| raw-hard K256 | `90.659356` | - |
| cosine-hard K128 | `79.952541` | `[78.1625, 81.7333]` |
| K-anchor dual K128 | `87.115606` | - |
| raw local-soft K128 | `91.470398` | - |
| cosine local-soft K128 | `79.972727` | - |
| 1/16 average-pool K128 | `72.578055` | `[70.9736, 74.2983]` |
| 1/16 decimation K128 | `94.430083` | - |
| full-grid K128 coverage diagnostic | `93.211500` | - |

Raw 的 R@1,r1 / R@8,r1 / R@32,r1 / R@64,r2 为
`5.9513 / 22.1111 / 42.5588 / 76.0058%`；cosine 为
`7.8920 / 26.3970 / 46.9063 / 79.4373%`。最佳 AUROC 是 raw entropy `0.641329`，任一
置信度信号满足 top32 precision >=60% 的图像比例均为 `0%`。

另一方面，top32 可恢复结构比例为 raw `99.7642%`、cosine `100%`，top32
select-and-reject oracle 的 grid mean 为 `1.659629 / 1.601127 px`，PairLocalReady 均为
`100%`。这说明候选集合包含几何可用对应，但当前 hard top1、固定 local soft 和普通 LS
无法读出它；它是方法任务下一步选择分支的证据，不授权本任务自行实施 B2。

## Oracle 与 GT 隔离

- Continuous-GT oracle：AEPE `2.0528e-5 px`，P99 `3.6511e-5 px`
- Nearest-inbounds oracle：AEPE `0.275216 px`，pair P95 `0.653121 px`，failure `0%`
- 全部 13,992 个 oracle 已从 retrieval trace 和 `.flo` 独立重建，ID、rank、point count、
  FP32 ordinary LS、坐标共轭和 PairLocalReady 一致
- 4,240 条正式变体全部为 `uses_ground_truth=false`；GT 只进入 retrieval/oracle diagnostics
- 正式公开评测使用完整 `512 x 512`，无 valid mask，包含黑边

## 资源结果

| 指标 | GDROS baseline | SparseGeo slim |
| --- | ---: | ---: |
| Dense-contract p50 | `78.951272 ms` | `8.638904 ms` |
| Dense-contract p95 | `97.157804 ms` | `9.157010 ms` |
| Cold/warm peak allocated | `315.472656 MiB` | `276.454590 MiB` |
| Registered parameters | `12,847,408` | `4,804,448` |
| Active parameters | `8,992,256` | `4,804,448` |

中位延迟加速为 `9.139x`，峰值显存降低 `12.368%`。K128 matcher 增量峰值为
`5.628418 MiB`。Slim 没有注册或执行 context、CorrBlock、GRU、upsampler 或
cost-perceiver，也没有构造 cross-image `[4096,4096]` correlation。

G5 的两个失败项必须保留：

1. Slim dense 五进程 p50 为
   `[8.628909, 8.659851, 11.433246, 8.638904, 8.549575] ms`，CV `13.712815%`。
   r2 与瞬态 DVFS/GPU 状态一致，但证据不足以追认通过；剔除 r2 后 CV `0.5597%` 只能
   作为诊断。
2. K64/128/256 matcher p50 为 `0.586000 / 0.589447 / 0.585670 ms`，斜率
   `-5.69085e-6 ms/K`、R2 `0.070739`。显存随 K 的 R2 为 `0.999999978`，score 数严格为
   `K x 4096`；计时平坦由固定 token copy 和固定 launch 开销主导，不是隐藏 N2，但仍不
   满足预注册 timing 门槛。

FeatureTransformer 仍有 `[8,1024,1024]` split-window attention。当前只能声称删除了
cross-image CorrBlock/N2 matcher，不能声称整网完全线性。

## Proxy 误差分组

分组均是可测 proxy，不是真实遮挡、重复纹理或语义标签。每个四分位 106 对：

- GT-in-bounds fraction 从最低到最高，cosine AEPE：
  `96.75 -> 83.41 -> 73.41 -> 66.25 px`
- Cosine mean entropy 从最低到最高，cosine AEPE：
  `71.62 -> 75.59 -> 83.41 -> 89.19 px`
- GT affine corner-Linf 从小到大，cosine AEPE：
  `73.11 -> 78.35 -> 80.21 -> 88.14 px`
- Collision proxy 没有稳定单调关系

## 审阅与产物路径

实现阶段已经过独立 accuracy、integration、reuse/resource 和 final-delta 审阅，高风险修复
复核后均为 `APPROVE`；正式运行后又分别完成 accuracy/G0-G4 与 resource/G5 的独立结果
审计。负面精度结果只有在这些审阅通过后才进入结论。

- 远端代码：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e002a`
- Accuracy：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002A/run-01/accuracy`
- Benchmark：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002A/run-01/benchmark`
- Accuracy log：`/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260804_102002_eb5f8e19.log`
- Benchmark log：`/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260804_102002_f4e2dea6.log`
- Accuracy watchdog report：`/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_102002_eb5f8e19.json`
- Benchmark watchdog report：`/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_102002_f4e2dea6.json`
- 正式离线审计 v2：`/mnt/sda/zhushihao/VS_Match/audits/E-002A/run-01-audit-v2`
- 本地完整只读副本：`/Users/shihao/Documents/图像匹配之二/tmp/e002a-formal-run01`
- 本地离线审计：`/Users/shihao/Documents/图像匹配之二/research/results/E-002A/offline-audit`
- Accuracy 独立审阅：`research/results/E-002A/formal-accuracy-review.md`
- Benchmark 独立审阅：`research/results/E-002A/formal-benchmark-review.md`
- 离线分析复现脚本：`tools/analyze_e002a_formal.py`

关键本地文件 SHA256：

- `accuracy/metrics.json`：`8301f83ae3e41840eba55843e3f0dd5c39fcf235d5d06ef730219e19c0ef884c`
- `accuracy/gates.json`：`718bcdea120f96a1a6f19ada59e91a34c2968e9cc9941b4cf8751b832e64b4db`
- `benchmark/aggregate.json`：`9d34c0a05a6109eaf3e228df85d3c56842830f5f27407200cf7e3e486c1037ec`
- `benchmark/g5_gate.json`：`267eeac270036907a87bb7e00d909b22412b66c94381a4dc183b8f1f5583b543`
- 离线 audit manifest：`e2506d4a773fb264d3852777c9044665a37d606506f7b9fcfe36eb56580f34fb`

正式复现命令逐字保存在 accuracy 与 benchmark 的 `command.txt`。本任务到此停止，等待
ACTIVE 方法任务依据该结果选择 B1、B2、B3 或 learned descriptor；不得自行启动 E-002B。
