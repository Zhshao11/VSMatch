# E-002B-R Result Card

- 实验/方法：`E-002B-R / M001 v0.5`
- 最终状态：`COMPLETED / EFFICIENCY_HOLD / P1 FAIL`
- 停止原因：P1 optimized geometry latency 与 reference speedup 均未达冻结门槛。

## Provenance

- METHOD_SPEC SHA256：`eb383532c988569772ef34700067f932a14363b674f4baa0f67e5a4a1ef3f5ea`
- E-002B-R SHA256：`5dc34e4efa1a9bde4e826828e0570af9131a01cb8687a3c95175042c42b28434`
- optimized source manifest：`d7030507e8fb4a43b7e1da786d2a17c73b058aa349d7205ce0feb06d982990d1`
- optimized backend：`15e7e2747ed28578f83e5217f39780697767b93d85479d83a3b13ac8f2723ccd`
- P1 config：`54db6658d46d3b2d62ef540d93147ce5054efe618fc596cb8b492693a5afb80e`
- non-GT input manifest：`7842997f758398f5879b22ff94071856e8aa554d99cbc99cc6954e390299cfe6`
- checkpoint：`80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- reference implementation：`3dd002c8f92c4ac58571ba04b92cbda6830843bce9662d93dd5d5ab67f35e58c`

## Gates

- Unit/negative tests：`10/10 PASS`。
- P0-early：`PASS`；100/100 synthetic、11 unique official pairs、44/44 variant traces；
  全部 `truth_flow_loaded=false`。
- P1 latency：`FAIL`；optimized p50 `205.055582 ms`（门槛 `<=80 ms`）。
- P1 speedup：`FAIL`；reference p50 `895.240630 ms`，加速 `4.365844x`
  （门槛 `>=8x`）。
- P1 memory：`PASS`；warm allocated `54,758,400 bytes`，约 `52.22 MiB`
  （门槛 `<=284 MiB`）。
- 定向 review：`NOT RUN`；按冻结顺序仅在 P1 PASS 后启动。
- G5-R2 / P0-full / accuracy：`NOT RUN`；P1 FAIL 后硬停止。
- Baseline parity：`NOT REACHED`。

## Evidence

- P0 gate：`research/results/E-002B-R/gates/p0-early.json`
- P1 gate：`research/results/E-002B-R/gates/p1.json`
- 远端 run：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B-R/run-01`
- P1 raw reference timing：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B-R/run-01/p1/reference/timing_raw.jsonl`
- P1 raw optimized timing：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B-R/run-01/p1/optimized/timing_raw.jsonl`
- watchdog job：`20260805_113440_4029a120`
- watchdog report：`/mnt/sda/zhushihao/experiment_watchdog/reports/20260805_113440_4029a120.json`

## Interpretation

本轮唯一许可的 assignment 运算顺序修复成功关闭 pair 47 的中间轨迹差异，P0-early 已证明
预注册早期集合上的语义等价。P1 表明当前全预算、精确 FP32 等价后端仍不具备资源资格。
由于 accuracy 未获准运行，本结果不能判断 B2 selector 精度，也不否定候选选择思路本身。
不写 `FAILED_METHODS.md`，不自动创建新版本或进入 E-002C。
