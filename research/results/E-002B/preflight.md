# E-002B Preflight

日期：`2026-08-04`

状态：`COMPLETED / G5-R FAIL`

## Tests

- 远端环境：`/mnt/sda/zhushihao/VS_Match/.conda/gdros`
- 命令：
  `python -m unittest experiments.sparsegeo.e002b.test_core experiments.sparsegeo.e002b.test_benchmark -v`
- 结果：`28/28 PASS`，耗时约 `18.8 s`。
- `py_compile`：全部 E-002B Python 通过。
- `bash -n`：`run_synthetic.sh`、`run_g5r.sh`、`run_accuracy.sh` 全部通过。
- 2-pair evaluator smoke：
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B/smoke-2-v1`

## Synthetic gate

- Watchdog job：`20260804_130950_5a402155`，GPU 3，`succeeded`。
- 结果：100/100 PASS，failure 0，最大 grid mean `6.396592652890831e-06 cell`，
  平均 `926.715648 ms/case`。
- Result SHA256：
  `4a42a56332cbcd97502fd4e77caed1708352a95a3bb7c09a397b6a32c5609508`
- Result：
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B/run-01/synthetic/synthetic_100.json`
- Watchdog log：
  `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260804_130950_5a402155.log`
- Watchdog report：
  `/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_130950_5a402155.json`

## Resource smoke

- 单 worker：
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B/benchmark-worker-smoke-b2-v3`
- 10-PID orchestrator：
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B/benchmark-orchestrator-smoke-v2`
- Smoke candidate count 8，K128 score elements 524,288，4 个 target chunks，worker artifact
  validation 全通过，无 hidden N2。
- Smoke B2 p50 约 `952.17 ms`，峰值 `276.45 MiB`。短 smoke 未满足正式 timing/CV/K-scaling
  协议，只用于验证执行和 gate 链，不能替代 G5-R。

## Formal job

- G5-R watchdog：`20260804_131250_4fbe4b65`
- G5-R output：
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002B/run-01/g5r`
- G5-R watchdog 进程成功，gate `passed=false`；accuracy 未启动。
