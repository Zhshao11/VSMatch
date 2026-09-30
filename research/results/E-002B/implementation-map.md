# E-002B Implementation Map

日期：`2026-08-04`

方法版本：`M001 v0.4`

状态：`COMPLETED / EFFICIENCY_HOLD / G5-R FAIL`

## Frozen specification binding

- `research/METHOD_SPEC.md` SHA256：
  `53e83d376d5055d2f009b5943d74835f4830a18994461b34e8da6a1ac85aeb5f`
- `research/experiments/E-002B.md` SHA256：
  `19ec4d70b19a3d313323c3d6959e5059b8d30c224480e60ddea9567893bbaf49`
- 语义绑定：`BIND-001`；collision source 在键 4/5 分别贡献 `4`/`L`，两轮 LS 后执行
  只验证、不拟合的 final validation
- E-002A K128 anchor SHA256：
  `2872a818932f138d79a0a0aebf3fec2bbadb1939e839788f6f103c1e1f16efb6`

两份规范的冻结副本随 E-002B 源码部署，并进入 source snapshot。smoke、G5-R 和 accuracy
启动前都必须重新校验共享文件 SHA；若变化则停止并由 ACTIVE 方法任务确认新绑定。

## Specification to code/test mapping

| 规范 | 模块 | 测试与产物 |
| --- | --- | --- |
| 冻结 feature、K128 anchor、坐标与 dense render | 复用 `e002a.feature/anchors/geometry` | feature parity 0；anchor JSON/SHA 一致 |
| cosine/raw 稳定 C8/C16 top-k | `candidates.py` | chunked/unchunked index 全等、score 差 `<=2e-5`、tie-break |
| 256 balanced triples 与 32 sentinels | `triple_bank.py` | 104,781 合法组、覆盖/usage/condition/area、确定性 JSON/SHA |
| top8^3 流式 affine hypothesis | `hypotheses.py` | ID、过滤原因、chunk `<=256`、不物化 residual cube |
| 唯一 target 两级共识 | `consensus.py` | 六级字典序、hub 去重、Q1024/R32、tie-break |
| 两轮重指派、ordinary LS、abstention | `reranker.py` | residual/score/index tie、12点/3象限/rank/condition、identity fallback |
| 四个预注册 variant 与 GT 隔离 | `pipeline.py`, `evaluate_e002b.py` | 100+ synthetic、2-pair smoke、`uses_ground_truth=false`、GT 后置 |
| G5-R fresh-process 资源复核 | `benchmark_worker.py`, `benchmark_e002b.py` | raw wall/event、10 PID、live shapes、K64..1024、gate 单测 |
| provenance 与 watchdog 入口 | `provenance.py`, `run_*.sh` | 规范/代码/配置 SHA、manifest、命令、日志与绝对路径 |

## Stop boundary

实现、测试、smoke 和两轮独立实际 diff 审阅均已完成。正式 G5-R 未通过，因此保持
`EFFICIENCY_HOLD`，未创建 accuracy 正式结果，也未进入 E-002C。
