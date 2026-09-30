# E-002B G5-R Implementation Review

日期：`2026-08-04`

审阅者：独立子 agent `e002b_g5r_design_review`

结论：`APPROVE`

## 已核验资源路径

- `B2Contract.candidates` 直接计算 stable top8；primary、segment 和 complexity audit 使用同一
  候选路径。
- 未插桩 `reranker_total` 覆盖 hypothesis generation、coarse/full consensus、两轮 refit
  和 final validation；后半段另记为 `refit_and_final_validation`。
- phase trace 明确标记为逐 phase CUDA synchronize 的 diagnostic，不能冒充未插桩延迟。
- actual matcher score elements 来自 live trace，K128 必须为 `128 x 4096`；候选增量显存、
  hypothesis chunk 和 reranker tensor shape 均从实际 tensor 记录。
- live-shape gate 要求 hypothesis `[H,3,3]` 且 `0 < H <= 256`，assignment tensor 分别为
  `[128,2]`、`[128,8]`、`[128,128]` 和 `[128]`，同时拒绝 `4096 x 4096`。
- segment gate 要求精确名称集合、逐段完整 iteration、finite wall/CUDA timing、明确 timing
  semantics 和正 throughput。
- 10 个 fresh PID、raw row/PID/replicate/protocol、worker artifact manifest、逐文件 SHA/size、
  path escape 和 post-consistency atomic gate 均有硬校验。

## 验证

- 远端 conda：`28/28` unittest PASS。
- 全部 E-002B Python：`py_compile` PASS。
- `run_synthetic.sh`、`run_g5r.sh`、`run_accuracy.sh`：`bash -n` PASS。
- 最新 10-PID smoke：所有 worker artifact validation 通过，candidate count `8`，score elements
  `524288`，无 hidden N2。短 smoke 资源结果不能替代正式 `20 warm-up / 200 measured`。

Reviewer 最终结论为可启动正式 G5-R；正式资源阈值仍必须由 watchdog run-01 决定。
