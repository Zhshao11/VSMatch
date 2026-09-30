# 当前候选生成基准：E-006A Corrected Candidate Generator

状态：`FROZEN_BASELINE`

日期：`2026-08-07`

用途：后续所有“更好的候选生成器”必须优先超过本文件指标，不能只在 selector、verifier 或 affine
后端上做不可比改动。

## 1. 候选生成方案

当前候选集来自冻结 GDROS 特征上的稀疏检索与仿射局部再搜索。

- 输入：official training archive，原始 `512x512` optical/SAR。
- 特征：冻结 GDROS backbone + FeatureTransformer，stride-8 `64x64` token map。
- source proposals：M256，`8x8` source regions，每区 4 个固定 offsets。
- target candidate paths：
  - `global_top8`：每个 source proposal 对全部 `4096` target tokens 做 cosine top8。
  - `region_top8`：每个 source region 先取 top-L=8 target regions，再在对应区域内取 top8。
  - `affine_local`：B=8 no-GT affine basins，在 `H_b(p)` 附近 `9x9` token window 内取局部候选。
- 不使用：GT inference、datum labels、OS424/confirm、dense flow、dense GRU、hidden `4096x4096`、
  fixed K128/Kmax、proposal expansion、RANSAC/IRLS/baseline fallback。

正式身份：

- E-006A source manifest：`a26f9619ad395bdbc1d45b3ec09ae256d603512042681259efdc8aac4e486d2a`
- GDROS checkpoint：`80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- split manifest：`e8c990888123896af425923970215eb1a9ec8aca89eae0421998d2bdc13a99d3`
- formal dev：404 pairs，group-isolated dev split。

## 2. 候选覆盖上限指标

来自 `E-006A / p0-gate-r2`。

| 指标 | 当前值 |
|---|---:|
| Oracle affine AEPE | `0.190768 px` |
| PairLocalReady / valid | `100% / 100%` |
| Mean 2px-correct proposals | `66.210 / 256` |
| 2px-correct proposals min / P10 / P50 / P90 / max | `14 / 33 / 64 / 98 / 153` |
| Mean in-bounds proposals | `215.32 / 256` |
| In-bounds proposals 的 2px-correct rate | `30.16%` |
| 达到 `>=140` 个 2px-correct proposals 的 pair 比例 | `0.743%` |

解释：正确候选的空间分布足以 oracle 拟合极准 affine，但数量远不到高密度正样本池水平。

## 3. 候选可选择性指标

来自 `E-006A-SELECT-DIAG`，使用当前候选集和 26 维 no-GT 特征。

| 指标 | 当前值 |
|---|---:|
| Candidate base rate `<=1 / <=2 / <=5 px` | `0.0338% / 0.1343% / 0.7577%` |
| In-bounds candidate 占比 | `82.155%` |
| In-bounds / out-of-bounds 2px base rate | `0.1622% / 0.00561%` |
| D1 cosine AUROC / AP / P@32 / AEPE | `0.7482 / 0.00476 / 1.230% / 119.853 px` |
| Best D1 AUROC / AP / P@32 / AEPE | `0.8377 / 0.00805 / 1.740% / 61.633 px` |
| Main D2 AUROC / AP / P@32 / recall@32 | `0.9171 / 0.01386 / 2.259% / 7.778%` |
| Main D2 affine AEPE / CMR@2 / CMR@5 | `63.516 px / 0% / 0.990%` |
| D0 hard 2px inlier oracle precision / valid-pair AEPE | `98.89% / 0.653 px` |

解释：当前候选在无 GT 特征下不可高纯选择；高 AUROC 在 `0.1343%` 正样本率下不能转化为可用 topK。

## 4. 第一版更强观测证据结果

来自 `E-007A`，仍固定同一 E-006A candidate set。

| Track | P@32 | AP | AEPE | CMR@2 |
|---|---:|---:|---:|---:|
| E-006A D2 baseline parity | `2.2587%` | `0.01386` | `63.5157 px` | `0%` |
| raw patch only | `0.2321%` | `0.00166` | `138.0443 px` | `0%` |
| frozen feature patch only | `1.2067%` | `0.00586` | `90.5367 px` | `0%` |
| patch combined | `1.2531%` | `0.00601` | `87.7608 px` | `0%` |
| handcrafted structure | `0.1238%` | `0.00129` | `180.9078 px` | `0%` |
| learned structure | `0.3094%` | `0.00159` | `140.5526 px` | `0%` |
| descriptor only | `0.8663%` | `0.00240` | `127.8814 px` | `0%` |
| descriptor + patch | `1.3072%` | `0.00495` | `99.9055 px` | `0%` |
| descriptor + structure | `0.8509%` | `0.00239` | `128.2480 px` | `0%` |
| all evidence | `1.7868%` | `0.00984` | `36.0761 px` | `0%` |

解释：在固定 E-006A candidate set 上堆第一版 patch/structure/descriptor verifier 不能提高候选点正确性。

## 5. 后续候选生成器必须超过的指标

后续若设计新的 candidate generator，至少要同时报告并超过以下冻结基准。

### 候选覆盖门槛

- Mean 2px-correct proposals：必须 `>66.210/256`；建议目标 `>=100/256`。
- P50 2px-correct proposals：必须 `>64`；建议目标 `>=96`。
- P10 2px-correct proposals：必须 `>33`；建议目标 `>=64`。
- 2px candidate base rate：必须 `>0.1343%`；建议目标 `>=0.5%`。
- Oracle affine AEPE 必须保持 `<=0.5 px`，不能用更差上限换更多候选。
- PairLocalReady/valid 必须保持 `>=99%`。

### 可选择性门槛

- P@32：必须 `>2.259%`；建议最低有用目标 `>=10%`，强目标 `>=20%`。
- AP：必须 `>0.01386`；建议目标 `>=0.05`。
- Selector affine AEPE：必须 `<36.0761 px` 才算超过 E-007A best affine；建议目标 `<=5 px`。
- CMR@2：必须 `>0%`；建议接近 dense GDROS dev 时再进入正式方法阶段。

### 资源边界

- 不得回到 hidden `4096x4096` cross-image tensor 或 dense GRU。
- 必须报告 candidate generation p50/p95、selector/verifier p50/p95、peak memory。
- 如果使用更强 descriptor/backbone，必须单独报告训练成本和推理成本，不能只报候选质量。

## 6. 判定原则

- 只提高 oracle AEPE 不够；E-006A oracle 已经很强。
- 只降低 affine AEPE 但 P@32/CMR@2 不升不够；E-007A all-evidence 已出现这种情况。
- 真正有用的候选生成器必须让正确候选“更多”且“更可分”。
- 新方法若不改变 candidate generator 或底层 observation space，不应再进入同类 selector/verifier 实验。
