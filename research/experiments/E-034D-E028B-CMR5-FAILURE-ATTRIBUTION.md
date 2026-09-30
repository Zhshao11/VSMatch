# E-034D-E028B-CMR5-FAILURE-ATTRIBUTION：Why Only 38.8% CMR@5?

方法版本：`M001 v3.12 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-034C best affine/CMR variant `rank_growth_dual_256_irls` 达到 AEPE `8.550`、CMR@2 `8.996%`、
CMR@5 `38.783%`。用户指出：即使后续 local refinement 把当前所有 CMR@5 pair 都推进到 CMR@2，
CMR@2 上限也只有 `38.783%`，远低于 GDROS baseline CMR@2 `81.604%/83.491%`。

因此 E-034D 不设计新选择器，只做诊断：

> 为什么 E-034C best affine/CMR 只有 38.8% pair 到 CMR@5？要拉高 CMR@5 应该优先改什么？

## 2. 边界

- 只读 E-034C formal artifacts；
- 可重算 diagnostic metrics，但不得改变 E-034C selection result；
- 不训练、不 finetune descriptor、不训练 matcher/head；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- 不使用 official testing.zip / confirm / OS424；
- 不启动新 selection sweep、rescue 或后续实验。

## 3. Tracks

### A. `PAIR_LEVEL_SUCCESS_FAILURE_SPLIT`

以 E-034C best affine/CMR variant 为主分析对象，按 pair 是否 CMR@5 success 分组。

比较：

- inlier ratio；
- selected count；
- correct@2 / correct@5；
- region/quadrant coverage；
- AEPE / residual P50/P90；
- PLR / valid。

### B. `BASIN_AND_OUTLIER_ATTRIBUTION`

对 CMR@5 failure pair 判断：

- selected basin 是否整体偏离 GT；
- remaining outlier proxy 是否显著高；
- IRLS trim 后是否仍有大 residual；
- high-score outliers 是否主导 weighted affine。

### C. `ORACLE_BOUND_ON_FAILURE_PAIRS`

在 CMR@5 failure pair 上复查 E-033A GT-neighborhood oracle：

- GT-neighborhood best coarse oracle；
- region-balanced oracle；
- topK oracle if available。

若 oracle 可到 CMR@5，说明 selection/consensus 仍未吃干净；若 oracle 也失败，说明 coarse feature/token bound
不足。

### D. `COVERAGE_CONDITIONING_ATTRIBUTION`

比较 success/failure 的：

- region coverage；
- quadrant coverage；
- spatial covariance / condition proxy；
- source point spread；
- affine parameter stability proxy。

判断是否“点多但几何条件差”。

### E. `TOKEN_QUANTIZATION_LOCAL_BOUND`

估计：

- coarse token quantization residual；
- selected coarse candidate 的 GT local offset bound；
- 当前失败 pair 中有多少已经在 5px 附近，多少远大于 5px。

判断 local refinement 是否是当前 CMR@5 的主杠杆，还是仍需先改 basin/selection/consensus。

## 4. Gates

### P0：read-only setup

必须 PASS：

- E-034C RESULT_CARD / selection_sweep_table / best_variant_metrics / coverage report located；
- E-033A oracle bound reference recorded；
- no E-034C selection result mutation；
- no GDROS backend adapter / dense GRU reuse；
- baseline artifacts readonly；
- testing.zip / confirm / OS424 not used。

P0 fail：`E034D_SETUP_BLOCKED`。

### G0：diagnostic execution

必须 PASS：

- pair-level success/failure split complete；
- basin/outlier attribution complete or typed unavailable；
- oracle-bound-on-failure complete or typed unavailable；
- coverage/conditioning attribution complete or typed unavailable；
- token/local-bound attribution complete or typed unavailable；
- reviewer approves read-only use of artifacts and interpretation。

G0 fail：`E034D_EXECUTION_BLOCKED`。

### G1：bottleneck diagnosis

判读：

- `SELECTION_CONSENSUS_STILL_LIMITED`：failure pairs have usable oracle bound but selected basin/outlier handling fails；
- `COARSE_BOUND_LIMITED`：failure pairs remain poor even under GT-neighborhood coarse oracle；
- `COVERAGE_CONDITIONING_LIMITED`：failure pairs mainly suffer poor spatial support/conditioning；
- `LOCAL_REFINEMENT_NOT_YET_PRIMARY`：many failures are >5px and cannot be solved by 5→2px refinement alone；
- `MIXED_CMR5_BOTTLENECK`：multiple material causes；
- `E034D_RESOURCE_BLOCKED`。

## 5. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `cmr5_success_failure_table.json`；
- `failure_attribution_counts.json`；
- `oracle_failure_bound.json`；
- `coverage_conditioning_report.json`；
- `local_bound_report.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。

完成后只回传精简结果和核心路径，等待方法任务和用户决策。不得自动启动 E035、training、descriptor
finetune、full sparse feedback、OS424/confirm 或 rescue。
