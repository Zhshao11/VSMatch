# E-033A-E028B-GT-NEIGHBORHOOD-CORRELATION-DIAG：E028B Correlation Response Bottleneck

方法版本：`M001 v3.8 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-032A 证明 E028B native matcher 相对 feature-only fair matcher 有收益，但 CMR@2 仍很低。当前需要回答：

> E028B feature correlation 在 GT endpoint / GT affine 邻域是否已经有正确响应？

判读逻辑：

- 如果 GT 附近响应存在、rank/mass 较高，但 E-032A matcher 没选中，则主瓶颈是 matcher / decoder / geometry feedback；
- 如果 GT 附近响应本身很弱，则主瓶颈是 feature；
- 如果 coarse token 响应存在但 pixel/local bound 差，则主瓶颈是 local resolution/refinement。

## 2. 边界

- 固定 E-028B safe dual adapter checkpoint，只读；
- 固定 E-032A formal artifacts，只读；
- 不训练、不 finetune、不改 descriptor；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend 作为 matcher；
- 不使用 official testing.zip / confirm / OS424；
- correlation / coarse candidates 必须先持久化，再读取 truth-flow 做后置 metric join；
- 本实验只做瓶颈诊断，不 claim 系统成功。

## 3. Tracks

### A. `FULL_CORRELATION_GT_NEIGHBORHOOD`

对 dev404 的 E028B source/target 32x32 tokens 计算 full target-token correlation。

对每个 source token，分别定义：

1. raw truth-flow endpoint 邻域；
2. fitted GT-affine endpoint 邻域。

报告：

- GT-neighborhood best rank；
- topK capture，K = 1/4/8/16/32/64/128/256；
- GT-neighborhood score percentile；
- softmax / dual-softmax mass inside GT neighborhood；
- global top1 与 GT-neighborhood best 的 margin / z-score；
- 按 pair、region/quadrant、source feature norm、texture bucket 汇总。

### B. `MATCHER_MISS_ATTRIBUTION`

复现或读取 E-032A best matcher / best consensus selected matches。

对每个 source token 分类：

- `VISIBLE_BUT_NOT_SELECTED`：GT-neighborhood rank/capture 好，但 E-032A 未选或选错；
- `SELECTED_WRONG_DESPITE_VISIBLE`：GT-neighborhood 响应好，但 selected target 错；
- `FEATURE_NOT_VISIBLE`：GT-neighborhood rank/mass 弱；
- `OUT_OF_BOUNDS_OR_AMBIGUOUS`：GT endpoint 越界或邻域定义不可判。

输出 pair-level 主导瓶颈比例。

### C. `ORACLE_RESPONSE_BOUND`

上帝视角只用于 bound，不参与 no-GT selection：

- GT-neighborhood best target token oracle；
- GT-neighborhood soft expectation oracle；
- region-balanced GT-neighborhood oracle。

报告 affine AEPE、CMR@1/2/5、PLR、valid、coverage。

## 4. Gates

### P0：setup

必须 PASS：

- training/testing archive SHA 与 pair count verified；
- E-028B checkpoint SHA、feature hook、feature shape recorded；
- E-032A formal RESULT_CARD / matcher artifact located or typed unavailable；
- no GDROS backend adapter / dense GRU reuse；
- baseline artifacts readonly；
- testing.zip / confirm / OS424 not used。

P0 fail：`E033A_SETUP_BLOCKED`。

### G0：execution

必须 PASS：

- E028B feature/correlation artifacts are persisted before truth-flow metric join；
- dev404 GT-neighborhood rank/mass/margin metrics complete；
- E-032A matcher miss attribution complete or typed unavailable；
- oracle response bound complete；
- reviewer approves leakage isolation, coordinate mapping, and result interpretation。

G0 fail：`E033A_EXECUTION_BLOCKED`。

### G1：diagnosis

判读：

- `MATCHER_DECODER_BOTTLENECK`：GT-neighborhood response is visible for enough sources, oracle bound is good, but E-032A misses it；
- `FEATURE_RESPONSE_BOTTLENECK`：GT-neighborhood response weak and oracle bound poor；
- `LOCAL_RESOLUTION_BOTTLENECK`：coarse response visible but local/pixel bound poor；
- `MIXED_BOTTLENECK`：multiple bottlenecks material；
- `E033A_RESOURCE_BLOCKED`：resource/time prevents completion。

## 5. 输出

- `RESULT_CARD.md`；
- `gt_neighborhood_summary.json`；
- `topk_capture_curve.json`；
- `matcher_miss_attribution.json`；
- `oracle_response_bound.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。

完成后停止，等待方法任务和用户决策。不得自动启动 E-033B、E-034A、descriptor training、full sparse feedback、OS424/confirm 或 rescue。
