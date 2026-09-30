# E-002B Formal G5-R Result Audit

日期：`2026-08-04`

审阅者：独立子 agent `e002b_g5r_design_review`

结论：产物可信；`G5-R FAIL`；必须保持 `EFFICIENCY_HOLD`，不得运行 424 accuracy。

## Independent recomputation

- 10 个 unique PID，每个 200 primary rows，共 2,000；iteration、metadata、PID 和 exit code
  全部一致。
- B2 p50 `888.752 ms`、p95 `911.972 ms`、CV `5.7431%`，均未通过门槛；约比 baseline
  p50 `76.189 ms` 慢 `11.67x`。
- Baseline p95 `82.349 ms`；五进程官方 parity 全为 0。
- cold/warm peak `276.455 MiB`、K128 incremental `7.5317 MiB`、score count `K*4096`、
  memory R2 `0.99417`、timing positive slope、no N2/forbidden module、finite/FP32/no-GT 均通过。
- timing R2 独立重算为 `0.62108 < 0.90`，失败。

## Trace and artifact audit

- 每个 B2 worker 有 12 segments x 20 rows；名称、iteration、finite timing、throughput 全合法。
- 每 worker 2,107 条 live trace、529 个 hypothesis chunks、Hmax 256；assignment shapes 精确为
  `[128,2] / [128,8] / [128,128] / [128]`，bytes 有效，无 `4096 x 4096`。
- 完整 geometry reranker median p50 `886.345 ms`；hypothesis generation、coarse consensus、
  refit/final median 分别约 `397.984 / 316.069 / 214.963 ms`。
- Worker manifest 95 files、top manifest 135 files，逐文件 size/SHA 与 aggregate 重算均无问题。
- 当前源码、依赖、checkpoint、BIND-001 与 run_config/post-consistency 一致；benchmark input
  optical/SAR SHA 与 gate 一致。
- Watchdog `20260804_131250_4fbe4b65` return code 0 只表示程序正常完成；正式 gate
  `passed=false`。
- run-01 仅有 `synthetic/` 和 `g5r/`；无 accuracy 目录、metrics 或 evaluate 进程。

最终审阅建议：按预注册决策树返回 ACTIVE 方法任务，不运行 accuracy。
