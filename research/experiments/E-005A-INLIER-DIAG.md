# E-005A-INLIER-DIAG：点数、内点率与仿射精度二维诊断

方法身份：冻结 `M001 v0.8 / E-005A`，不建立新方法版本

状态：`USER_APPROVED / READY`

## 1. 问题与边界

回答两个问题：

1. GDROS dense predicted flow 中有多少点正确，以及有多少点支持其最终 affine？
2. 对 ordinary affine LS，点数 K 与内点比例 rho 如何共同影响 AEPE/CMR？

本诊断只读冻结 checkpoint/cache 和 raw truth_flow，不训练、不调参、不改变 E-005A G1/G2 状态，
不得作为 rescue、E-005B 或效率结果。OS 424 和 external confirm 禁止访问。

## 2. 数据与冻结身份

- 使用 E-005A 相同 `404` dev pairs、split SHA、official `512x512` 和无 mask/含黑边协议。
- dense：official GDROS checkpoint 和经 parity 通过的 dev evaluator。
- sparse：E-005A main seed `20260808` checkpoint、M256 proposals、final round coordinates/weights/active set。
- raw truth_flow 是唯一 GT；datum 不作标签。
- optical source -> SAR target，flow 为 `target-source`。

## 3. 统一点级定义

在 source pixel `p=(x,y)`：GT endpoint `g=p+truth_flow(p)`；模型 endpoint `q=p+pred_flow(p)` 或
E-005A continuous candidate endpoint。point error `e_gt=||q-g||_2`，单位 pixel。

对模型最终 affine `H_pred`，self-consensus residual 为 `e_self=||q-H_pred(p)||_2`。

阈值固定为 `tau in {1,2,5,8,16}` px，分别报告：

- GT inlier ratio：`mean[ e_gt <= tau ]`；
- self-consensus ratio：`mean[ e_self <= tau ]`；
- E-005A weighted GT inlier mass：`sum w_i I[e_gt<=tau]/sum w_i`；
- E-005A weighted self-consensus mass：同定义替换 `e_self`。

同时报告 pooled micro ratio、404-pair macro mean/median/P5/P95，禁止只报 pooled 数字。

## 4. GDROS 三个取点协议

分别计算，禁止混称：

1. `public-200k`：公开 LSR row-major indices `[10000:210000]`；
2. `full-512`：全部 512x512 pixels，仅作 coverage diagnostic，无 mask；
3. `matched-M256`：E-005A 同一 M256 source positions，用于与 sparse 方法公平比较。

GDROS 的 `H_pred` 必须复用公开 ordinary-LS 路径；不得按 GT 重新拟合。另以 GT affine endpoint
计算 correctness，但必须标为 GT correctness，不能称为模型自共识。

E-005A 的历史 `recoverable`（top8 距 GT <=2 stride-8 tokens，即 <=16 px 可达）单独列出，明确
禁止与上述最终 endpoint inlier ratio 等同。

## 5. 二维性能曲线

网格固定：

- `K in {16,32,64,128,192,256}`；
- controlled inlier ratio `rho in {0.50,0.60,0.70,0.80,0.90,0.95,1.00}`；
- 每 cell、每 pair 固定 32 个 repetitions；`rep_id=0..31`，随机种子为
  `sha256(pair_id|K|rho_string|rep_id)` 的前 64 bit unsigned integer。

使用 M256 固定 source pool。每次先以 source-only farthest-point sampling 得到 K 个空间分散 anchors；
首点为最小 proposal ID，距离 tie 取最小 ID。该步骤不得读取 GT 或模型误差。

对 K 个 anchors 构造 controlled observations：

- exactly `floor(rho*K+0.5)` 个 inliers：使用 raw GT endpoint 加上该 pair 的 GDROS M256 point residual，
  residual 只从 `e_gt<=2 px` 的点中有放回采样；
- 其余 outliers：使用该 pair E-005A M256 final endpoint residual，residual 只从 `e_gt>5 px` 的点中
  有放回采样；
- residual 以二维向量原样采样并加到当前 anchor GT endpoint，不旋转、不缩放；
- pair 缺少任一所需 residual pool 时该 cell 标记 ineligible，不跨 pair 借 residual。

主曲线使用 equal-weight normalized FP32 ordinary LS，直接隔离 K/rho。附加曲线使用 E-005A
对应 anchor weights，但不得替代主曲线。记录 condition、eligible pair fraction 和 sampling CI；
eligible `<90%` 的 cell 灰显并禁止作为主要结论。

输出：

- heatmap：AEPE、CMR@2、P95、condition；
- 每个 K 的 rho 曲线和每个 rho 的 K 曲线；
- 达到 dense GDROS dev `AEPE<=1.4597+0.35` 且 CMR@2 下降 `<=3 pp` 的最小 `(K,rho)` 区域；
- raw CSV/JSON，禁止只保存图片。

## 6. Oracle Decomposition

在 E-005A final active K128 上额外计算四条只读 oracle：

1. predicted coordinates + predicted weights；
2. GT coordinates + predicted weights；
3. predicted coordinates + GT binary weights (`e_gt<=2 px`)；
4. GT coordinates + GT binary weights。

保持 active set 不变，隔离 coordinate error 与 false-positive weight contamination。若 GT binary
有效点 `<12`、coverage `<3` quadrants 或 condition `>20`，显式 invalid，不 fallback。

## 7. 验证、审阅与停止

自动测试必须覆盖 public indices、flow direction、pixel/feature units、M256 parity、threshold boundary、
macro/micro、Farthest-Point tie、exact K/rho、sampling determinism、affine solver、GT isolation 和 CSV/plot
一致性。因新增指标与数据构造，使用两名最小上下文 reviewer：第一名审定义/坐标/采样，第二名审
evaluator/聚合/图表/无测试泄漏。

顺序：实现 -> tests -> reviewer -> 修复复核 -> watchdog 404-pair diagnostic -> RESULT_CARD。结果完成
后停止并回传当前方法任务；不得训练、启动 G2、修改 checkpoint 或自动建立新版本。
