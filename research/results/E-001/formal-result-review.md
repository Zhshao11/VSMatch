# E-001 Formal Result Independent Review

日期：`2026-08-04`

结论：`APPROVE`，无阻塞项。M001 v0.2 的 E-001 正式门槛成立；建议 E-002A 默认从
`K=128` 起步，但本结果不能证明 K-to-all 已经可行。

## 审阅绑定

- Accuracy watchdog：`20260804_040321_a1ab77de`，GPU 2，`succeeded`
- Benchmark watchdog：`20260804_040325_fa3cd863`，GPU 3，`succeeded`
- Evaluator：`a16fe7586053ea8fb6ea33a9607fdff4b75eee215b264af13d4ebcfbca8a226e`
- Benchmark：`3dbcb88f2a443018fdc7c7312b81051f6d522173140c5900cebd8dc9ba3f0473`
- Tests：`39e00d837376cd74e701e549cc1715d9751985ce9ec01a2e46168dcf91d9b3d0`
- Checkpoint：`80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- Metrics：`f6e0c0afab6f6665a9fb0ecf883f88c216a130c7e3643133b46fa8b63ab86217`
- Per-pair CSV：`fe4220fe97b25a57c84dadd5a8117336016a855fe79d02791fbf1106bc6a3e28`
- Benchmark JSON：`77c0df510699da2a9058c8b4050a848e2f3a2ae7c0d3cba27c1934050c7faab2`

## 独立复核内容

两名独立 agent 与主实验任务分别用结构化 JSON/CSV 解析器重算，而不是只读取程序写入的
pass 字段。结论一致：

- 正式 archive、provenance、published/validated root、loader 000000 至 000423 和运行后
  manifest 全部绑定并再次重哈希通过。
- `per_pair.csv` 为 `38,584 = 424 x 91` 条唯一记录；91 个结果每项恰好 424 对，无重复、
  缺失、路径逃逸或非有限值。
- 45 个采样设计全部 rank 3；结果构成为 1 reference、35 exact-public predicted、10
  full-image diagnostic、45 GT oracle。
- CSV 重算与 JSON 汇总最大差为 `1.42e-14`；CSV public/vector delta 字段重算差为 `0`。
- Baseline AEPE `1.461230482`，CMR@1/2/5
  `33.726415/81.603774/99.528302%`，P95/P99 `3.247386/4.597962 px`，failure `2/424`。
- 14 个正式 decision ID 与从结果反向推导的 K64/K128、exact-public、非 GT 集合完全一致，
  独立重算 `14/14 PASS`。
- K64 随机 delta AEPE 为 `0.005654 +/- 0.004379 px`；K128 为
  `0.002345 +/- 0.002227 px`，K128 更稳定。
- K128 coverage 逐对抵消复算为 exact `1/0`、full-grid `130/138`、full-center `99/107`
  对越过 `+/-0.05 px`。
- 45 个 oracle 的 AEPE 为 `1.5981e-5` 至 `3.5416e-5 px`，failure 为 0，且未进入正式门槛。
- 所有结果满足 `failure = 100 - CMR@5`，最大浮点残差低于 `1e-14`。
- Benchmark 12 个 case 均为 FP32、batch 1、warm-up 5、repeat 30，并在计时前后 CUDA
  synchronize；throughput 和显存字段算术一致。

## 解释边界

Generalized 200k、K64、K128 的端到端 peak allocated 都是 `316.298 MiB`，postprocess
incremental peak 都是 `7.061 MiB`。约 1 至 2 ms 的端到端均值差不能作为稳定加速证据，
因为完整 CorrBlock 和 32 次 GRU 未改变，且 case 在同一进程顺序执行。

剩余非阻塞限制：benchmark 没有持久化 30 次原始 duration；所有 design 在测量前共享驻留
状态。后续部署级 benchmark 应保存 raw timing，并用隔离进程复核小于毫秒级的差异。

## 产物路径

- Accuracy：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-001/formal-official-archive/run-01/accuracy`
- Benchmark：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-001/formal-official-archive/run-01/benchmark`
- Accuracy watchdog report：`/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_040321_a1ab77de.json`
- Benchmark watchdog report：`/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_040325_fa3cd863.json`
