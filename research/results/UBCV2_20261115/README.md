# UBCv2 跨数据集泛化：原始结果（2026-11-15）

| 文件 | 内容 |
|---|---|
| `ubcv2_official_20261115.json` | **主结果**：GDROS released 与 E075A-M4@80K 在 1625 对 UBCv2 上的官方 `test.py` 口径全量指标 + 布局校验 + 耗时 |
| `ubcv2_sanity8.json` | 8 对管线 sanity（GDROS + M4），**数字不可用作结论**，前 8 对恰好都是易配准样本 |
| `ubcv2_sanity8_negflow.json` | 把 8 对 `truth_flow` 整体取反后的 GDROS 对照，AEPE `21.84 → 81.69`，用于排除 GT 方向写反 |
| `manifest.json` | 生成协议元数据：1625 对、seed 901、变换范围、SAR 8-bit LUT（`lo_db=-28.3079`, `hi_db=10.5459`）、未预筛 |

## 主结果（官方口径，`iters=32`、`LSR NPOINT=200000`）

| 模型 | AEPE | CMR@1 | CMR@2 | CMR@5 | RMSE_var | >5px 对 | 耗时 |
|---|---|---|---|---|---|---|---|
| GDROS released | 22.0091 | 0.06 | 0.74 | 11.26 | 679.0462 | 1442 | 131.5 s |
| E075A-M4@80K | 3.8941 | 46.46 | 62.95 | 74.52 | 38.7571 | 414 | 1286.4 s |

两个 checkpoint 都只在 **OS 训练集**上训练过 ⇒ 这是**跨数据集泛化**，
不是论文 Table III 的 in-domain 结果（官方 released 权重没有 UBCv2 版）。
本实验还是**单 seed、单测试集、未预筛**（论文是预筛 1447 对、三个随机
变换测试集 mean±std，预筛规则与 seed 未公开），绝对数值不可与论文并列。

复现命令、SAR 位深陷阱、flow 方向约定的反推过程与完整边界见
[`docs/UBCV2_GENERALIZATION.md`](../../docs/UBCV2_GENERALIZATION.md)；
脚本说明见 [`research/eval/README.md`](../eval/README.md)。
生成的数据集本体（4.5 GB）留在 zzvr
`/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/generated/seed901/`，不入库。
