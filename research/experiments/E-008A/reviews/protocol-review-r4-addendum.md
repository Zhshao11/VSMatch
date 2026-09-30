# E-008A Protocol Review R4 Addendum

结论：`APPROVE` 保持有效，无新 blocker。

本轮仅复核 `build_labels.py` 与 `test_core.py` 的窄 SHA 增量：

- `endpoint_pixel_xy` 和 `endpoint_error_px` 现以 FP32 持久化；raw truth-flow 采样、candidate 坐标、in-bounds 以及 1/2/5 px 阈值语义未改变。
- `validate_label()` 显式要求两个 tensor 为 `torch.float32`，旧 FP16 label cache 会 fail-closed，不会被 evaluator 静默混用。后续运行应使用新 label cache 目录或重建旧 cache。
- 回归测试新增 endpoint/error FP32 dtype 断言，远程结果为 `13 passed, 2 warnings in 3.38s`；warnings 仅为无关的 `timm` deprecated import。

身份：

- source manifest SHA256：`b8e4dd9f4fce7bf55f2f8ff5a875052e214248c2052ac0de5cb3fb6398ba04a5`
- `build_labels.py`：`3156b78566bea2774122cffb1a44e4182dd905cf552a45ebcfafe6d390067745`
- `test_core.py`：`7656181009ac20e95d9ec1a26df54f87d77364bc70252990ff8b6ee2cb1602b6`

该增量关闭 evaluator label quantization finding，不改变 r3 对 smoke/watchdog 的协议批准，也不代表 P0/P1 方法 gate 已通过。
