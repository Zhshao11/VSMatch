# E-009A Review Packet

## 版本

- Method: `M001 v1.2`
- Experiment: `E-009A`
- Spec: `research/experiments/E-009A.md`
- Contract: `research/INNOVATION_CONTRACT.yaml`

## 审阅目标

确认实现是否真正替换 E-008A 的 frozen cosine / region score top64 预筛，而不是在旧 top64 后做
verifier。

## 高风险点

- S0/S1 是否与 E-008A-STAGE-DIAG 一致。
- learned score 是否在 top64 之前参与排序。
- frozen GDROS feature/cosine/region score 是否没有进入 learned retriever 输入或主 top64 ranker。
- truth_flow 是否只用于训练标签和后置 eval，不进入 inference trace。
- topR 输出是否在 label join 前物化。
- 是否没有 hidden `4096x4096`、full-image blind search、dense flow/GRU 或 fallback。
- G1/G2 指标是否与冻结 baselines 可比。

## 必查证据

- `RUN_MANIFEST.json`
- frozen S2 parity test
- topR output trace
- gradient/toy-overfit test
- no-GT/no-N2 tests
- formal `topr_summary.json`

Reviewer 输出只需 findings、结论和证据路径。
