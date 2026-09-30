# E-002B-ACC-DIAG：全预算 B2 Selector 精度诊断

方法：`M001 v0.5 / E-002B v0.4 / BIND-001`

状态：`READY / USER_APPROVED`

## 1. 目的

E-002B 与 E-002B-R 均因资源 gate 停止，尚未产生 selector accuracy。本实验经用户明确批准，
只诊断冻结全预算 B2 selector 的精度价值，不考核部署资源，不构成新方法或救援版本。

## 2. 冻结执行源

- 只使用 E-002B 正式只读 source snapshot，不使用 live `e002b` 或 `e002br`。
- Git：`ee6b6216fe780bc03bfb4759ae094daa0cfddc60`。
- implementation manifest：
  `3dd002c8f92c4ac58571ba04b92cbda6830843bce9662d93dd5d5ab67f35e58c`。
- E-002B spec SHA：`19ec4d70b19a3d313323c3d6959e5059b8d30c224480e60ddea9567893bbaf49`。
- checkpoint SHA：`80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`。
- source snapshot：
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B/run-01/g5r/source_snapshot`。

任何 source、spec、checkpoint、anchor、dataset 或 config SHA 不一致都必须停止。

## 3. 方法与数据不变量

- 完整复用 E-002B/BIND-001 的 K128 cosine top8、256 triples、131,072 hypotheses、
  32 sentinels、Q1024、R32、两轮 ordinary LS 和 no-fit final validation。
- 四个 variant 固定为 C8-main、C8-consensus-only、C8-C16-diagnostic、R8-diagnostic。
- 官方 OS 424 对、同一 preprocessing、完整 512x512 无 valid-mask 评测，黑边参与。
- 方法输出完成并落盘前不得加载 truth flow、valid mask、oracle 或测试指标。
- 禁止修改阈值、预算、排序、tie-break、variant 或根据结果选择配置。
- 禁止 baseline fallback、optimized backend、WLS/IRLS、descriptor、训练和局部精修。

## 4. 启动 gate

正式运行前只做自动检查：

1. 重验第 2 节全部 SHA、官方 424 input manifest 和 anchor SHA。
2. 复用 E-002B 已归档的 28/28 tests 与两轮 APPROVE；仅当 source manifest 变化才失效。
3. 运行 2-pair non-GT smoke，确认四 variant finite、uses_ground_truth=false、无 fallback。
4. evaluator 必须复用公开无 mask 指标，并通过 E-000 baseline artifact/parity checker。
5. 没有代码 diff 时不启动新 reviewer；若新增或修改 evaluator/glue，则只审该 diff 和 GT 隔离。

任一自动 gate 失败不得启动 424，也不得解释 selector。

## 5. 正式诊断

- watchdog 单独运行 424 对四 variant；保存逐对 CSV/JSON、trace manifest、bootstrap 和 oracle。
- C8-main 为唯一主结果；其余只作 attribution diagnostic，不得事后晋升。
- 资源耗时和显存可以记录，但不参与本实验 PASS/FAIL，也不能解除 EFFICIENCY_HOLD。

主精度门槛：

- PairLocalReady pairs `>=90%`。
- 非 abstention pairs `>=90%`，且 final validation 满足冻结结构约束。
- 16x16 grid mean 全数据均值 `<=8 px`。
- 424 对全部 finite，无静默 solver failure。

报告 AEPE、CMR@1/2/5、P50/P95/P99、abstention、grid/corner、inlier/condition、candidate rank、
collision、support margin、image-pair bootstrap 95% CI，以及三种 diagnostic 和 top-k GT oracle。

## 6. 结论与停止

- PASS：只说明 selector 机制有精度价值；资源仍 FAIL，等待用户决定是否批准预算化新方法。
- FAIL：停止当前固定 B2 selector，完成 selector/oracle gap 审计；不自动转 descriptor。
- 本结果因沿用已参与分支选择的 424 对，只能作为机制 diagnostic，不能单独作为最终泛化证据。
- 无论 PASS/FAIL，写 `RESULT_CARD.md` 后停止，不进入 E-002C 或任何新实验。

建议远端目录：

`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B-ACC-DIAG/run-01/`
