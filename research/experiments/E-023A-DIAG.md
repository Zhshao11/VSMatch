# E-023A-DIAG：Stage Decomposition for Sparse Affine-State Recurrent Matcher

方法版本：`M001 v2.7 diagnostic`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-023A 已完成：P0 PASS、G0 PASS、G1 `NO_STATE_FEEDBACK_SIGNAL`，train final AEPE `5.502 px`，
dev404 final AEPE `50.578 px`、CMR@2 `0.302%`。E-023A-DIAG 不提出新方法，只诊断 dev failure
发生在哪个阶段。

核心问题：

```text
coarse state
-> basin selection
-> state-conditioned window coverage
-> local matcher ranking
-> recurrent affine update
```

到底是哪一步导致 dev 上没有反馈信号。

## 2. 输入

只读：

- E-023A formal result artifacts：
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-023A/run-01/formal`
- E-023A checkpoint/source/RUN_MANIFEST；
- 官方 train/dev split；
- raw `truth_flow`，仅用于 diagnostic labels、oracle 和 metrics；
- raw 512x512 optical/SAR。

禁止：

- 不训练新模型；
- 不修改 E-023A checkpoint、formal result、baseline 或历史产物；
- 不启动 E-023B/rescue；
- 不读取 OS424/confirm/datum 做诊断判读；
- 不使用 dev oracle 选择新 threshold 后追认 E-023A。

## 3. 必做诊断

### D0：artifact/replay sanity

- 复核 E-023A P0/G0/G1/reviewer/RUN_MANIFEST；
- strict-load E-023A checkpoint；
- 小批 train/dev replay，确认 no-GT inference finite，round/state metrics 可复现；
- 若 replay 失败，判 `DIAGNOSTIC_IMPLEMENTATION_BLOCKED`。

### D1：coarse state generation 与 basin selection

对 train1607/dev404 分别报告：

- selected initial basin AEPE/CMR；
- oracle-best initial basin AEPE/CMR；
- selected-vs-oracle basin gap；
- final selected AEPE 与 round-wise AEPE。

判读：

- oracle-best basin 也差：`COARSE_STATE_GENERATION_FAIL`；
- oracle-best basin 明显好但 selected basin 差：`BASIN_SELECTION_FAIL`；
- train selected 好、dev selected 差：`TRAIN_MEMORIZATION_OR_STATE_INIT_GENERALIZATION_FAIL`。

### D2：state-conditioned window coverage

对每轮 `t=0..T` 报告：

- GT endpoint 落入 local window 的 anchor 比例；
- train/dev mean/P10/P50/P90；
- per-basin 与 selected-basin coverage；
- window size sweep：`5x5 / 9x9 / 17x17 / 33x33 / 65x65` token-equivalent，若实现资源受限可分块。

判读：

- dev window-hit 低，扩大 window 明显提高 oracle upper bound：`WINDOW_COVERAGE_LIMITED`；
- window-hit train 高/dev 低：`TRAIN_MEMORIZATION_OR_STATE_INIT_GENERALIZATION_FAIL`。

### D3：local matcher ranking

只在 GT endpoint 落入窗口的 anchors 上评估：

- local matcher top1/top3/top5 hit；
- GT local rank / local AP；
- predicted local target 与 oracle-nearest local target 的 endpoint error；
- 使用 oracle-nearest local target 拟合 affine 的 AEPE/CMR。

判读：

- window-hit 高但 local topK/rank 差：`LOCAL_MATCHER_RANKING_FAIL`。

### D4：state update oracle

保持 E-023A no-GT initial state，但替换局部观测为 oracle：

- oracle local residual update；
- oracle window target update；
- oracle confidence/inlier mask update；
- 对比 predicted update final AEPE/CMR/PLR。

判读：

- oracle update 好、predicted update 差：`STATE_UPDATE_CONTAMINATION_FAIL` 或 local observation/weight bottleneck；
- oracle update 仍差：`NO_RECOVERABLE_STATE_MATCH_SIGNAL` 或 source/window geometry 不可恢复。

### D5：train/dev gap and bucket

输出 train/dev 对齐 bucket：

- initial selected basin good/bad；
- oracle basin good/bad；
- window coverage good/bad；
- local ranking good/bad；
- oracle update recoverable/unrecoverable；
- final no-GT success/failure。

必须给出 primary failure bucket 占比。

## 4. 判读优先级

若多项同时成立，按最早瓶颈优先：

1. `DIAGNOSTIC_IMPLEMENTATION_BLOCKED`
2. `COARSE_STATE_GENERATION_FAIL`
3. `BASIN_SELECTION_FAIL`
4. `WINDOW_COVERAGE_LIMITED`
5. `LOCAL_MATCHER_RANKING_FAIL`
6. `STATE_UPDATE_CONTAMINATION_FAIL`
7. `TRAIN_MEMORIZATION_OR_STATE_INIT_GENERALIZATION_FAIL`
8. `NO_RECOVERABLE_STATE_MATCH_SIGNAL`
9. `MIXED_STAGE_FAILURE`

必须同时报告 secondary buckets，不能只给一个口号。

## 5. 审阅

需要一名定向 reviewer，默认只读：

- 本实验单；
- E-023A RESULT_CARD/G0/G1；
- diagnostic summary JSON；
- 新增诊断 diff；
- oracle 使用位置。

重点核查：

- truth_flow 只用于 diagnostic/oracle/metric；
- dev oracle 没有被用于训练、阈值选择或追认；
- window-hit/local rank/oracle update 定义正确；
- train/dev aggregation 无缺失/重复；
- E-023A formal 产物只读。

## 6. 输出

只回传：

- `RESULT_CARD.md`
- `diagnostic_summary.json`
- `stage_decomposition.json`
- `window_sweep.json`
- `local_ranking.json`
- `oracle_update.json`
- `train_dev_gap.json`
- reviewer conclusion
- `RUN_MANIFEST.json`
- watchdog log/report

完成后停止，等待方法决策。
