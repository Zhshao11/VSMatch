# E-000 Formal Official Archive Run 01

日期：`2026-08-04`

方法规范：`M001 v0.2`

结论：`COMPLETED / PASS`

## 正式数据绑定

- 官方归档：`/mnt/sda/zhushihao/VS_Match/datasets/official_archives/gdros/testing.zip`
- 归档大小：`1,973,635,350` bytes
- 归档 SHA256：`e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218`
- 发布目录：`/mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official_e000_archive`
- 发布 provenance SHA256：`bff6f565e9a8a12bb2675d54b97fccaeb3a88d39b49ae56a1571c57d8489aad9`
- 2120-file manifest：`c129ae7147a7d9ad1c1a7ead35bb6da2d8c85ad8da73eef4723cf277e5ea7e96`
- 样本：424 对，`testing`，`512 x 512`，全图无 valid-mask，包含黑边

官方归档和旧等价物化目录的 1272 个评测输入在 GDROS loader 语义下全部一致：PIL `L`
图像平铺为三通道，PIL `RGB` 使用前三通道，FLO 比较原始字节。原始文件字节并非全部
一致：optical `0/424`、SAR `424/424`、truth flow `424/424`；这属于 PNG 编码差异，不能
表述为 1272 个文件 raw bytes 全部相同。

## 正式指标

| 指标 | 结果 |
| --- | ---: |
| AEPE | `1.461230444` |
| CMR@1 | `33.726415%` (`143/424`) |
| CMR@2 | `81.603774%` (`346/424`) |
| CMR@5 | `99.528302%` (`422/424`) |
| P95 / P99 | `3.247386 / 4.597962 px` |
| Failure (`>=5 px`) | `0.471698%` (`2/424`) |
| Pair-error standard deviation | `0.862292` |
| Legacy/public-code variance | `0.743547` |
| Batch-1 evaluation peak allocated | `315.8 MiB` |

论文表中的约 `0.75` 对应公开代码记录的 pair-error variance，而不是数学上的标准差。正式
结果与论文 AEPE/CMR 量级一致，并与已审等价预检逐对完全相同：最大绝对 pair delta
为 `0.0`。

## 审阅与执行

- Validator：`da6d59041a926eba98bb59bdbc85aa1302c7744bc23bc6b4a58680df2a6ba609`
- Extractor：`e9c17149999e6f5bfcc1a36f457284f9afe8398f543ef7fc718662dc775c7f3e`
- Validator tests：`11c6a629e79463a6c5bc1d579088b24b3712912351d2882da29db3986702ef5c`
- Checkpoint：`80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- 独立提取/语义终审：`APPROVE`；远端 validator `9/9 PASS`
- Extraction watchdog：`20260804_035335_8b16c356`，CPU job，`succeeded`
- Smoke watchdog：`20260804_035723_b11d54c3`，GPU 2，`succeeded`，单对 EPE `1.248353`
- Formal watchdog：`20260804_040000_19368569`，GPU 2，`succeeded`

## 绝对产物路径

- 正式结果：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-000/run-01/formal-official-archive`
- Metrics：上述目录下 `metrics.json`，SHA256
  `933d638a6d5a4bff52861ba432bf834f6b04a2ef6e29f54c93eb1a2b1bb23953`
- Formal checker：上述目录下 `validation.json`，`passed=true`
- Watchdog log：`/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260804_040000_19368569.log`
- Watchdog report：`/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_040000_19368569.json`
- 预发布记录：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-000/run-01/prepublish-validation-v3/E000_PROVENANCE_PREPUBLISH.json`

E-000 门槛已关闭，可以使用该发布目录运行 E-001 formal。
