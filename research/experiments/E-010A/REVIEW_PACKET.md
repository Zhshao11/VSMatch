# E-010A Review Packet

## 版本

- Method: `M001 v1.3`
- Experiment: `E-010A`
- Spec: `research/experiments/E-010A.md`
- Contract: `research/INNOVATION_CONTRACT.yaml`

## 审阅目标

确认 E-010A 是语义不同的强跨模态 region-local retriever，而不是 E-009A 的小修补或 frozen S2 后
verifier。

## 高风险点

- S0/S1 是否与 E-008A-STAGE-DIAG parity。
- learned retriever 是否完全不读取 GDROS frozen features/frozen scores。
- 是否没有蒸馏或拟合 frozen S2 score。
- score 是否在 top64 前用于 full-union ranking。
- loss 是否覆盖 full union 或等价完整 mask，而不是退回 sampled CE 主目标。
- hard negative mining 是否真实执行并记录。
- truth_flow 是否只用于 train label 与 post-output eval。
- topR outputs 是否在 label join 前物化。
- 是否无 hidden `4096x4096`、full-image blind search、dense flow/GRU 或 fallback。
- G1 是否按“显著超过 frozen S2”判读，而不是略高追认。

## 必查证据

- `RUN_MANIFEST.json`
- S0/S1 parity 与 frozen S2 parity
- no-frozen-feature-input test
- loss/hard-negative trace
- gradient/toy-overfit test
- formal `topr_summary.json`

Reviewer 输出只需 findings、结论和证据路径。
