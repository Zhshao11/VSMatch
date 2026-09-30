# E-002A Offline Analysis

日期：`2026-08-04`

输入是正式运行的本地只读副本：
`/Users/shihao/Documents/图像匹配之二/tmp/e002a-formal-run01`。

在远端已有 NumPy 的 GDROS conda 环境运行；输出目录必须是新的空路径：

```bash
/mnt/sda/zhushihao/VS_Match/.conda/gdros/bin/python \
  /mnt/sda/zhushihao/VS_Match/audits/E-002A/analyze_e002a_formal.py \
  --accuracy-dir /mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002A/run-01/accuracy \
  --benchmark-dir /mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-002A/run-01/benchmark \
  --dataset-root /mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official_e000_archive \
  --output-dir /mnt/sda/zhushihao/VS_Match/audits/E-002A/run-01-audit-rerun
```

该分析不重新运行模型，也不改写正式产物。它完成：accuracy 35/35 manifest 复验、benchmark
218/218 manifest 与 raw timing 重算、G0-G5 独立重算、2,000 次 image-pair bootstrap CI，
以及 424 对的 proxy 四分位误差分组。

结果：accuracy audit `PASS`、benchmark artifact/raw audit `PASS`、G5 `FAIL`，与两份独立
正式审阅一致。`error_groups.csv` 中的 visibility、entropy、margin、collision 和
large-transform 都只能解释为 measurable proxy，不能冒充遮挡、重复纹理或语义标签。

远端 v2 审计位于 `/mnt/sda/zhushihao/VS_Match/audits/E-002A/run-01-audit-v2`。早期 v1
位于 `run-01-audit-v1-superseded`，已因分析脚本的 NPZ 重复解压、affine shape 和 sample-CV
口径问题被替代，禁止引用。
