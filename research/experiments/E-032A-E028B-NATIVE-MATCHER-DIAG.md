# E-032A-E028B-NATIVE-MATCHER-DIAG：Native Matcher for E028B Features

方法版本：`M001 v3.7 design`

状态：`PROPOSED / WAITING_USER_APPROVAL`

## 1. 目标

E-031A 表明 E-028B 特征很强，但不适合硬接 GDROS backend：

- E-028B feature-only retrieval top32 `85.93%`，明显强于 GDROS native `23.77%`；
- GDROS backend interface `[B,256,64,64]` 与 E-028B feature `[B,1024,32,32]` 不兼容；
- simple bridge 接入后 AEPE `18.6692`、CMR@2 `1.1656%`。

因此停止 adapter 路线，转向 E-028B 原生 matcher：直接围绕 E-028B 特征设计匹配、局部精修与几何一致性。

## 2. 边界

- 不设计 E028B-to-GDROS backend adapter；
- 不复用 GDROS dense GRU/backend 作为 primary matcher；
- 不训练新 descriptor；
- E-028B checkpoint、DINO checkpoint、GDROS baseline artifacts 只读；
- 优先做 no-train / light-head matcher 诊断；
- testing.zip / confirm 只有用户明确批准后才用于 final external evaluation。

## 3. Tracks

### A. `E028B_FEATURE_MATCHING_BASELINES`

固定 E-028B safe dual adapter feature，比较：

- mutual nearest neighbor；
- dual-softmax；
- ratio / margin filtering；
- region-balanced top-k；
- multi-quota source sampling。

输出 correspondence precision/recall、correct@1/2/5、selected count、spatial coverage、affine AEPE/CMR。

### B. `LOCAL_REFINEMENT_FROM_E028B`

对 32x32 coarse matches 做局部精修：

- nearest token endpoint；
- local window soft-argmax；
- small patch / feature correlation；
- optional raw-image local patch refinement。

目标是判断瓶颈是否在 32x32 coarse-to-pixel refinement。

### C. `AFFINE_CONSENSUS_SOLVER`

用 E-028B correspondences 拟合全图 affine：

- region-balanced weighted least squares；
- iterative residual reweighting；
- confidence from score/margin/mutuality/local-consistency；
- no hidden RANSAC/three-point/131k unless later explicitly approved。

目标是判断强 retrieval 信号能否转成稳定几何。

### D. `TRAINABLE_MATCHER_HEAD_OPTIONAL`

只有 A/B/C 显示 positive signal 时才启用：

- 输入 E-028B pair features、match score、local context、geometry residual；
- 预测 match probability、local offset、consensus weight；
- 训练目标：match probability + local offset + affine residual consistency；
- 不训练新 descriptor。

## 4. Gates

### P0

必须 PASS：

- E-028B checkpoint SHA / extraction hook recorded；
- no GDROS backend adapter in primary track；
- train/dev split and evaluator frozen；
- candidate/materialized correspondences written before truth-flow metric join；
- baseline artifacts readonly。

P0 fail：`E032A_SETUP_BLOCKED`。

### G0

必须 PASS：

- feature extraction and matcher tracks complete or typed failure recorded；
- no label/GT used before candidate materialization；
- evaluator/reviewer approve no-adapter and metric aggregation；
- resource report written。

G0 fail：`E032A_EXECUTION_BLOCKED`。

### G1

判读：

- `NATIVE_MATCHER_SIGNAL_FOUND`：native matcher clearly improves over E-028B feature-only fair baseline and gives usable affine estimates；
- `LOCAL_REFINEMENT_BOTTLENECK`：coarse matches strong but local/pixel refinement blocks affine accuracy；
- `CONSENSUS_BOTTLENECK`：matches exist but geometry aggregation fails；
- `NO_NATIVE_MATCHER_GAIN`：simple native matchers do not improve over current E-028B fair matcher；
- `E032A_RESOURCE_BLOCKED`：resource/time blocked。

## 5. 输出

只回传：

- `RESULT_CARD.md`
- matcher baseline table
- local refinement table
- affine consensus report
- optional trainable head report
- reviewer conclusions
- `RUN_MANIFEST.json`

完成后停止等待用户。
