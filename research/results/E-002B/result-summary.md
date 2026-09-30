# E-002B/B2-bridge Result Summary

日期：`2026-08-04`

方法：`M001 v0.4 / BIND-001`

终态：`COMPLETED / EFFICIENCY_HOLD / G5-R FAIL`

## Decision

正式 G5-R 未通过，按照预注册停止条件没有启动官方 424-pair accuracy，也没有进入 E-002C。
因此本轮只否定当前固定 B2-bridge 几何后端的资源资格，不能据此否定其 accuracy、冻结特征
candidate set 或整个 SparseGeo 方法族。

E-002A 终态不变：`G0 PASS; G1-G5 FAIL`，不得追认 G3 通过。

## Provenance

- METHOD SHA256：
  `53e83d376d5055d2f009b5943d74835f4830a18994461b34e8da6a1ac85aeb5f`
- E-002B SHA256：
  `19ec4d70b19a3d313323c3d6959e5059b8d30c224480e60ddea9567893bbaf49`
- BIND-001；anchor SHA256：
  `2872a818932f138d79a0a0aebf3fec2bbadb1939e839788f6f103c1e1f16efb6`
- Git HEAD：`ee6b6216fe780bc03bfb4759ae094daa0cfddc60`
- E-002B implementation manifest SHA256：
  `3dd002c8f92c4ac58571ba04b92cbda6830843bce9662d93dd5d5ab67f35e58c`
- Checkpoint SHA256：
  `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- Dataset：官方 424-pair archive，
  `/mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official_e000_archive`
- Formal source snapshot：
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B/run-01/g5r/source_snapshot`

## Implementation verification

- 两轮独立代码审阅：`APPROVE / APPROVE`，见本目录 `review-core.md` 和 `review-g5r.md`。
- 单测：`28/28 PASS`；Python compile 与三个 shell 入口语法检查通过。
- 2-pair smoke：四个 variant finite，C8-main 两对 PairLocalReady，GT 前后 non-GT trace hash
  一致；仅作 smoke。
- Synthetic watchdog `20260804_130950_5a402155`：100/100 PASS，failure 0，最大 grid mean
  `6.396593e-6 cell`，无 abstention/solver failure。

## Formal G5-R

Watchdog job：`20260804_131250_4fbe4b65`，GPU 3/4/5/6，进程本身 `succeeded`；方法 gate
为 `passed=false`。协议为 RTX 4090、batch 1、FP32、5 个 fresh process/method、每进程
20 warm-up + 200 measured。

| 指标 | B2 | Baseline | 门槛 | 判定 |
| --- | ---: | ---: | ---: | --- |
| p50 wall | 888.752 ms | 76.189 ms | <=50 ms | FAIL |
| p95 wall | 911.972 ms | 82.349 ms | <=63 ms | FAIL |
| 五进程 p50 CV | 5.743% | 2.298% | <=5% | FAIL |
| cold peak allocated | 276.455 MiB | 315.473 MiB | <=284 MiB | PASS |
| warm peak allocated | 276.455 MiB | 315.473 MiB | <=284 MiB | PASS |
| K128 incremental peak | 7.532 MiB | N/A | <=8 MiB | PASS |
| memory linear R2 | 0.994170 | N/A | >=0.99 | PASS |
| timing slope | +2.238e-5 ms/K | N/A | >0 | PASS |
| timing linear R2 | 0.621083 | N/A | >=0.90 | FAIL |

其余 gate 项全部通过：5+5 fresh PID、score elements 精确线性、K128 `128 x 4096`、无 hidden
cross-image N2/禁用模块、worker artifacts、finite output、RTX4090/FP32/inference/no-truth。

B2 五进程 p50 为 `911.629 / 888.752 / 832.313 / 945.386 / 828.703 ms`；p95 为
`936.069 / 911.972 / 879.104 / 966.272 / 880.128 ms`。baseline 五进程 parity 最大差为 0。

K64/128/256/512/1024 的 score elements 分别为
`262144/524288/1048576/2097152/4194304`；增量峰值为
`2.786/7.532/14.063/22.227/44.446 MiB`。

未插桩分段的五进程 median p50：feature `5.678 ms`、candidate top8 `1.205 ms`、完整
geometry reranker `886.345 ms`、refit+final-validation `214.963 ms`、dense render `0.292 ms`。
逐 phase 同步 diagnostic 的 median p50：hypothesis generation `397.984 ms`、coarse
consensus `316.069 ms`、full consensus `9.111 ms`、candidate reassignment `68.002 ms`、
ordinary LS `112.153 ms`、final validation `34.343 ms`、final rescore `0.991 ms`。

每个 B2 worker 都记录 131,072 attempted hypotheses，其中 130,612 accepted、176 duplicate
target、284 collinear、0 nonfinite，max chunk 256；每次完整 probe 为 64 fit calls 和 32 次
final validation。分段数据包含同步开销，只用于定位瓶颈，不与未插桩总延迟相加。

## Artifacts

- G5-R root：
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B/run-01/g5r`
- Gate：`.../g5r/g5r_gate.json`，SHA256
  `fb297007f151edd6a1fd4c66d814f9ff96480379fed8cf0353c740565e7534ef`
- Raw primary timing：`.../g5r/workers/*/timing_total.jsonl`，共 2,000 rows。
- Raw segment timing：`.../g5r/workers/b2__r*/timing_segments.jsonl`，共 1,200 rows。
- Raw K scaling：`.../g5r/workers/b2__r*/k_scaling_raw.jsonl`，共 250 rows。
- Worker manifest：`.../g5r/worker_artifact_manifest.json`，逐文件 SHA/size 重验无问题。
- Watchdog log：
  `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260804_131250_4fbe4b65.log`
- Watchdog report：
  `/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_131250_4fbe4b65.json`

## Interpretation

B2 将 allocated peak 降低约 `39.02 MiB`，但当前固定的 Python/PyTorch 穷举几何后端使
端到端延迟约为 baseline 的 `11.7x`，明显不符合“低推理耗时”目标。最主要成本在 131k
hypothesis generation 和 32-sentinel coarse scoring，而不是 K-to-all candidate matching。

由于 G5-R 是 accuracy 的硬前置，正式逐对 AEPE/CMR、candidate/abstention、capture@32 和
结构诊断均未产生；不能用缺失的 accuracy 结果判定 B2 候选选择机制失败。本轮不写入
`FAILED_METHODS.md`，等待 ACTIVE 方法任务决定是否重构几何搜索预算/实现，或继续保持 hold。
