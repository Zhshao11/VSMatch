# E-002B-ACC-DIAG Review Packet

状态：`COMPLETED / ZERO-DIFF REVIEW REUSED / ACCURACY FAIL`

## 默认读取

- `research/CURRENT_STATE.md`
- `research/METHOD_DELTA.md`
- `research/INNOVATION_CONTRACT.yaml`
- `research/experiments/E-002B-ACC-DIAG.md`
- 本目录 `RUN_MANIFEST.json`

## 冻结绑定

- diagnostic spec：`b4ab4b40ab1868e602ac5d59a0a7d336ea5be2ae22fdb51449bd81852506ae9d`
- method delta：`4cea7a97f2de562633c39c851129e31fd1ea3f1fcb37af55b83e40b120356315`
- innovation contract：`8835f2d5f95d3ec0ae67da2b44f8bdbff2dfac9908055b93383fa1f0c9b0cdda`
- E-002B spec：`19ec4d70b19a3d313323c3d6959e5059b8d30c224480e60ddea9567893bbaf49`
- reference implementation：`3dd002c8f92c4ac58571ba04b92cbda6830843bce9662d93dd5d5ab67f35e58c`

## 代码范围

计划代码 diff：`NONE`。只允许从正式只读 source snapshot 运行原 E-002B evaluator。

已有证据：

- E-002B core implementation review：`research/results/E-002B/review-core.md`，APPROVE。
- E-002B tests：28/28 PASS。
- E-002B-R P0-early 额外证明等价实现与 reference 在预注册集合轨迹一致；本诊断不调用
  optimized backend。

## 自动检查

- source/spec/config/checkpoint/anchor/data manifest SHA。
- 2-pair 四 variant non-GT smoke、finite、abstention、无 fallback。
- truth flow 只在方法输出后进入指标/oracle。
- evaluator 使用公开完整 512x512 无 mask 指标。
- 424 对和四 variant 完整性、逐文件 manifest 与 bootstrap 产物。

## 高风险点

- Python module path 必须来自只读 snapshot，不能加载 live `e002b` 或 `e002br`。
- evaluator 不得在方法 forward 前加载或传入 truth flow。
- resource gate 必须标记 `not_applicable_for_accuracy_diagnostic`，不能追认效率通过。
- diagnostics 不得根据结果晋升为主 variant。

## Reviewer 触发规则

source manifest 精确一致且无 evaluator/glue diff：不启动新 reviewer，复用已有 APPROVE 和自动
checker。若产生任何代码 diff，只将该 diff、上述高风险点和自动结果交一个最小上下文 reviewer；
review 非 APPROVE 时不得正式运行。
