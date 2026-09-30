# E-002B-ACC-DIAG Result Card

- 实验：`E-002B-ACC-DIAG`
- 方法：`M001 v0.5 / E-002B v0.4 / BIND-001`
- 最终状态：`COMPLETED / ACCURACY DIAGNOSTIC FAIL / EFFICIENCY_HOLD`
- 主结论：冻结全预算 B2 selector 未达到预注册 accuracy 门槛。

## Provenance

- diagnostic spec：`b4ab4b40ab1868e602ac5d59a0a7d336ea5be2ae22fdb51449bd81852506ae9d`
- E-002B reference implementation：`3dd002c8f92c4ac58571ba04b92cbda6830843bce9662d93dd5d5ab67f35e58c`
- checkpoint：`80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- anchor：`2872a818932f138d79a0a0aebf3fec2bbadb1939e839788f6f103c1e1f16efb6`
- 424 non-GT manifest：`7842997f758398f5879b22ff94071856e8aa554d99cbc99cc6954e390299cfe6`
- evaluator/glue diff：`NONE`；复用既有 `28/28 PASS + APPROVE`。
- 冻结 evaluator 以非-formal入口运行，因为其 `--formal-run` 硬要求 G5-R PASS；startup
  gate 在外部重验全部正式 SHA、424 完整性、公开 evaluator parity 和完整无 mask 协议。

## Gates

- Startup provenance/smoke：`PASS`；2 对×4 variant、finite、GT 隔离和 parity 均通过。
- 正式完整性：`PASS`；424 对×4 variant = 1,696 行，全部 finite、无 GT 方法输入。
- C8-main PairLocalReady：`75.71%`，门槛 `>=90%`，FAIL。
- C8-main non-abstention：`100%`，门槛 `>=90%`，PASS。
- C8-main grid mean：`17.233 px`，门槛 `<=8 px`，FAIL。
- 最终结构约束和 solver：全部 PASS，无静默 solver failure。
- Accuracy diagnostic：`FAIL`；资源 gate 为 `N/A`，EFFICIENCY_HOLD 不变。

## Main Result

- AEPE：`10.4349 px`；P50/P95/P99：`3.890/43.989/118.151 px`。
- CMR@1/2/5：`0.00% / 12.03% / 63.21%`。
- 103/424 对不满足 PairLocalReady；156/424 对误差 `>=5 px`。
- bootstrap 95% CI：AEPE `[8.539, 12.598]`；PairLocalReady
  `[71.70%, 79.48%]`。
- 结果呈明显重尾，且错误样本未被 abstention 识别。

## Attribution

- cosine top8 GT oracle：AEPE `1.4792 px`、PairLocalReady `99.53%`、grid mean
  `2.410 px`，说明候选集包含足够的正确匹配。
- C8-C16 diagnostic 略好于 C8-main：AEPE `8.2733 px`、PairLocalReady `79.72%`，
  仍未通过；扩大候选深度不能单独解决问题。
- consensus-only 与 raw-top8 更差。失败主要指向固定 B2 排序/几何选择与置信拒绝，而非
  top-k candidate set 本身。

## Evidence

- Startup gate：`research/results/E-002B-ACC-DIAG/gates/startup.json`
- Accuracy gate：`research/results/E-002B-ACC-DIAG/gates/accuracy.json`
- 远端 metrics：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B-ACC-DIAG/run-01/accuracy/metrics.json`
- 逐对结果：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B-ACC-DIAG/run-01/accuracy/per_pair.csv`
- diagnostics/oracle：同目录 `pair_diagnostics.jsonl`、`oracles_per_pair.csv`
- watchdog：`20260805_124412_c40ee093`

该结果是机制诊断，不是未见数据泛化证据；不解除资源 hold，不自动进入 E-002C、预算化方法、
descriptor 或新实验。
