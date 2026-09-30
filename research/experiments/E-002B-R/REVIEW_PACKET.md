# E-002B-R Review Packet

更新时间：`2026-08-05`

状态：`COMPLETED / P0_EARLY_PASS / P1_FAIL`

## Reviewer 默认输入

- 当前状态：`research/CURRENT_STATE.md`
- 方法增量：本目录 `METHOD_DELTA.md`
- 创新契约：本目录 `INNOVATION_CONTRACT.yaml`
- 冻结实验单：`research/experiments/E-002B-R.md`
- 运行清单：本目录 `RUN_MANIFEST.json`
- 本轮精确 diff：由实验任务恢复后生成并写入本节，不读取完整任务对话

## 冻结 SHA

- M001 v0.5 METHOD_SPEC：`eb383532c988569772ef34700067f932a14363b674f4baa0f67e5a4a1ef3f5ea`
- E-002B-R：`5dc34e4efa1a9bde4e826828e0570af9131a01cb8687a3c95175042c42b28434`
- E-002B reference：`19ec4d70b19a3d313323c3d6959e5059b8d30c224480e60ddea9567893bbaf49`
- v0.4 implementation manifest：`3dd002c8f92c4ac58571ba04b92cbda6830843bce9662d93dd5d5ab67f35e58c`

## 当前改动文件

根目录：`tmp/e002br-work/experiments/sparsegeo/e002br/`

- `__init__.py`
- `binding_provenance.py`
- `fixture_producer.py`
- `geometry_benchmark_worker.py`
- `optimized_backend.py`
- `optimized_worker.py`
- `p0_official_shard.py`
- `p1_orchestrator.py`
- `profile_capture.py`
- `reference_worker.py`
- `select_block.py`
- `synthetic_parity.py`
- `test_optimized_backend.py`
- `test_trace_and_binding.py`
- `trace_checker.py`
- `trace_io.py`

## 规范到代码映射

- 冻结 reference 隔离：`binding_provenance.py`、`reference_worker.py`
- 131,072 hypotheses、coarse/full、BIND-001、LS/final validation：`optimized_backend.py`
- optimized trace：`optimized_worker.py`
- differential checker：`trace_checker.py`、`trace_io.py`
- synthetic/official P0：`synthetic_parity.py`、`p0_official_shard.py`
- block 锁定：`select_block.py`
- P1 资源计时：`geometry_benchmark_worker.py`、`p1_orchestrator.py`

## 高风险函数

- `optimized_backend.py::assign_candidates_batched`
- `optimized_backend.py` 的 hypothesis matrix/conjugation 顺序
- `optimized_backend.py` 的 stable collision winner 和六键排序
- `optimized_backend.py` 的 ordinary LS 与 no-fit final validation
- `trace_checker.py` 的离散精确比较和浮点容差
- `reference_worker.py` 的只读 snapshot/module namespace 隔离
- `geometry_benchmark_worker.py` 的 CUDA 同步、memory reset、compile/cache/RSS 统计

## 自动测试状态

- 远端单元/负向测试：`10 PASS`。
- synthetic block：`4096` 已锁定，不得重新选择。
- pair 0 C8-main：全轨迹 `PASS`；最大 assignment residual 差 `7.63e-6`。
- pair 47：`FAIL`；round-0 有 2 个 rank/target 差异，最终输出一致但不满足中间轨迹合同。
- P0-early：100 synthetic 与 44/44 official pair/variant 轨迹 PASS。
- P1：optimized p50 `205.055582 ms`、speedup `4.365844x`，资源 gate FAIL。
- 定向 reviewer、G5-R2、P0-full 和 accuracy 按 gate 顺序未运行。

## 当前唯一允许修复

把 `assign_candidates_batched()` 的批量投影改为 reference 单矩阵运算顺序。修复不得改变容差、
预算、排序、候选或最终 gate。修复后先跑相关单测和 pair 47，再补齐全部 P0-early。

## 未解决风险

- 逐矩阵投影可能降低 P1 加速，需要实际资源 gate 判断，不允许预先放宽。
- 恢复后必须核对已有 watchdog job 状态和失败产物，不能覆盖。
- optimized source/config/data input manifest 尚未在本地 `RUN_MANIFEST.json` 冻结。
- 正式 reviewer 尚未开始；只在 P0-early 自动 gate 通过后审阅本轮增量。

## Reviewer 输出要求

只输出 findings、结论和证据路径。已冻结 v0.4 reference 文件不重审；修复后只复核相关 diff。
