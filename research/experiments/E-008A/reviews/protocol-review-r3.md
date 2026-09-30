# E-008A Protocol Review R3

结论：`APPROVE`

本轮仅复核 r2 受影响路径：formal/smoke P0→P1 provenance、selector/GDROS checkpoint 身份、N2/full-image blind-search 边界及对应测试。未重审其他冻结模块。

## Closure

- P0 check schema：`P0_CHECK_KEYS` 固定 10 个完整 key；formal P1 同时要求 `checks` 是 dict、key 集精确相等且所有值为 true。空 `checks` 定向反例已 fail-closed。
- formal/smoke 边界：smoke 使用 `SMOKE_PASS`，formal P1 要求 `PASS` + `formal` + 404 pairs + split/source/baseline identity + `formal_gate_evaluated=true`；smoke 产物不能放行 formal P1。
- checkpoint 身份：observation 仍绑定冻结 GDROS checkpoint SHA；evaluator 对实际 selector checkpoint 文件计算 SHA，写入 P0 gate 和逐对 candidate trace，P1 要求与当前 selector 文件精确相等。selector SHA 不一致的定向反例已 fail-closed。
- N2/blind-search：`search_budget_flags()` 是唯一边界函数，由 `max_target_union_tokens` 和 `max_live_score_elements` 实际值派生标记；search trace、observation validation 和 P0 summary 使用同一语义。`4096` target tokens 和 `4096^2` live score elements 的精确边界均被判为禁止，边界前一值通过。
- shard：r2 已确认的 fail-closed 校验保持不变。

## Evidence

远端测试：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/sda/zhushihao/VS_Match/.conda/gdros/bin/python \
  -m pytest -p no:cacheprovider -q \
  experiments/sparsegeo/e008a/test_core.py \
  experiments/sparsegeo/e008a/test_contract.py
```

结果：`13 passed, 2 warnings in 3.41s`。warnings 为 `timm` deprecated import，与本轮协议路径无关。

额外定向反例：

- formal P0 `checks={}`：拒绝，`RuntimeError: E-008A P1 requires a PASS P0 gate`。
- P0 selector SHA 与当前 selector 文件 SHA 不一致：同样拒绝。

## Reviewed Identity

- 远程根：`bl_tunnel:/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e008a`
- source manifest SHA256：`415bde6e2572a441971f3e4f4847b8922a77645b98f82128beeafb8d4114c4bc`
- `evaluate.py`：`917251f37901f3b8818a60e7dc81a4f20d770e90ff2276d5c7b57340deb716fa`
- `build_observations.py`：`d71b3bc86bcdbd52666d7153ef8b3bf1b915941431ee5dc300789a56ea6d794a`
- `search.py`：`1482a226a88aa90bb73f2ff688bdeeafce949185b6caff5584ea9f7541172575`
- `test_contract.py`：`10875fac7d647bbd13c21285ffce2294e3c46b06041e4ff3c49a70a85302e980`
- `test_core.py`：`d6efaa7c47b0d0f83e4d68bcf1269c4289477a091a2fa7a0d134580478528ec0`

## Decision

批准进入后续 smoke/watchdog 流程。该 `APPROVE` 仅表示本轮协议和实现阻塞已关闭，不表示 P0/P1 方法 gate 或精度结果已通过。
