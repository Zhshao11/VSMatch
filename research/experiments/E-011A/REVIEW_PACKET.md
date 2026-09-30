# E-011A Review Packet

## 版本

- Method: `M001 v1.4`
- Experiment: `E-011A`
- Spec: `research/experiments/E-011A.md`
- Contract: `research/INNOVATION_CONTRACT.yaml`

## 审阅目标

确认 E-011A 真正从 512 原图/SAR 重新提取 patch descriptor，并且没有污染 GDROS baseline。

## 高风险点

- GDROS baseline 源码、checkpoint、cache、run/results 是否只读未改。
- E-011A outputs 是否全部写入独立 E-011A 路径。
- learned retriever 是否读取原始 optical/SAR patch，而不是 E-008A cached tail。
- GDROS frozen features/frozen scores 是否没有进入 learned input 或 teacher。
- S0/S1 是否与 E-008A-STAGE-DIAG parity。
- score 是否在 top64 前用于 S1 union ranking。
- loss 是否覆盖 full union 或 supervised contrastive equivalent。
- truth_flow 是否只用于 train label 与 post-output eval。
- topR outputs 是否在 label join 前物化。
- 是否无 hidden `4096x4096`、full-image blind search、dense flow/GRU 或 fallback。
- G1 是否按显著超过 frozen S2 判读，top128 是否未追认 top64 失败。

## 必查证据

- `RUN_MANIFEST.json`
- baseline isolation audit
- original image read / patch extraction trace
- no cached-tail main input test
- no frozen-feature-input test
- S0/S1 parity 与 frozen S2 parity
- loss/hard-negative trace
- gradient/toy-overfit test
- formal `topr_summary.json`

Reviewer 输出只需 findings、结论和证据路径。
