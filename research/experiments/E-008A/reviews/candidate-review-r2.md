# E-008A Candidate Generator Review R2

结论：`CHANGES_REQUIRED`

本轮仅定向复核 formal/smoke P0→P1 provenance、N2/full-image blind-search 派生 gate、shard 边界与受影响测试。未重审冻结且 SHA 未变的方法模块，也未运行 P0/P1 方法 gate。

## Finding Closure

### F1 [P1] P0→P1 provenance：部分修复，未关闭

- 已修复：smoke P0 写入 `SMOKE_PASS`，formal P0 只在所有 checks 通过时写入 `PASS`。formal P1 要求 P0 的 `status=PASS`、`run_mode=formal`、`pair_count=404`、split/source/checkpoint/baseline identity 一致且 `formal_gate_evaluated=true`；smoke P0 不能放行 formal P1。
- 未关闭 1：`evaluate_p1()` 使用 `all(p0.get("checks", {}).values())`，空字典会返回 true。定向反例中，具有正确 envelope 但 `checks={}` 的 formal P0 JSON 穿过了 provenance guard，随后才在 P1 数据聚合处以 `ValueError` 失败。因此伪造/缺字段 P0 仍可放行。需要求 P0 check key 集与预期精确一致且全为 true，或由 P0 `summary` 重算 checks。
- 未关闭 2：`load_checkpoint()` 只验证 checkpoint 内部 metadata，没有对 `args.checkpoint` 实际文件执行 `sha256_file(path) == CHECKPOINT_SHA256`。P0 JSON 写入的是常量 `CHECKPOINT_SHA256`，不是实际文件的计算结果；P1 也可使用元数据相同但权重不同的文件。需在 evaluator 入口 fail-closed 校验实际 checkpoint SHA。
- 受影响测试未覆盖：smoke P0 拒绝 formal P1、空/缺失/多余 P0 checks、旧 source SHA、错 checkpoint SHA、错 baseline identity 和实际 checkpoint 文件被篡改。

### F2 [P1] N2 / blind-search 派生 gate：代码已关闭，测试未关闭

- `summarize_p0()` 现在从 observation 的 `max_live_score_elements` 和 `max_target_union_tokens` 实际值派生 `hidden_4096_by_4096` 与 `full_image_blind_search`；`p0_checks()` 直接使用这两个派生值，不再依赖 hard-coded false 自证。
- 定向边界运行确认：将两个派生 resource 标记设为 true 时，`no_hidden_n2` 和 `no_blind_full_search` 均为 false。
- 但现有测试只断言正常 search trace 小于边界且 boolean 为 false，没有构造 `max_live_score_elements >= 4096^2` 或 `max_target_union_tokens >= 4096` 来断言 observation/P0 gate fail-closed。需将负边界加入自动回归测试。

### F3 [P1] shard 边界：已关闭

- `validate_shard()` 拒绝 `shard_count < 1`、`shard_index < 0` 和 `shard_index >= shard_count`。
- `build_observations.run()` 和 `build_labels.run()` 均在 modulo 分片前调用该验证。
- `test_shard_bounds_fail_closed` 覆盖 `(0,0)`、`(-1,5)` 和 `(5,5)` 三个非法边界，同时保留 `(0,1)` 正常路径。

## Test Evidence

远端命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/sda/zhushihao/VS_Match/.conda/gdros/bin/python \
  -m pytest -p no:cacheprovider -q \
  experiments/sparsegeo/e008a/test_core.py \
  experiments/sparsegeo/e008a/test_contract.py
```

结果：`12 passed, 2 warnings in 3.47s`。通过数不能关闭上述 provenance 反例和 resource 负边界的测试缺口。

## Reviewed Identity

- 远端源码根：`bl_tunnel:/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e008a`
- source manifest SHA256：`639f262b5e88cec382ff2ebd7dbc1ecebccb79c02a083a0103228bdb56a89d37`
- `evaluate.py`：`c63eae4b873546e79ce6a2450f5b588f6c03175184d67826ecf4c101d96ab81e`
- `build_observations.py`：`ce5b0927a8973b24c76d657a8c50cfb12e5cc0a3dea8c0c2105cf39608117e89`
- `build_labels.py`：`09b7e39c710302924ec93d77f18af23fe51496966a36bf56c286b9bc8cb8ceff`
- `spec_binding.py`：`8a1a201e268bbb0a5c8f909a757c3d36b9365ecb9bf11df6cec105276fdb50c5`
- `test_core.py`：`42d784bca390641086fbec383f9ce8a1d5f8396f9e37db82b93b4abbbe6ec4ae`
- `test_contract.py`：`21a93d74cd52a364a692ce6243f9132589e0393dc7170a3c0867ba94b7dec5e7`

## Decision

不批准进入后续 smoke/watchdog。先修复 formal P0 check schema/recompute、实际 checkpoint 文件 SHA 校验，并补齐 provenance 反例与 resource 负边界自动测试；随后只需对这些增量修复做下一轮定向复核。
