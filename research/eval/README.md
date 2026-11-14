# 评测与消融脚本

本目录存放 OS 数据集评测脚本。它们是历史实验在部署工作区运行的副本，
**路径常量硬编码为 zzvr 布局**，换环境时需修改各文件顶部的
`ROOT` / `TMP` / `OFFICIAL` / `TEST_ROOT` / `M4_CKPT`，详见
[`docs/REPRODUCE.md`](../../docs/REPRODUCE.md)。

| 脚本 | 用途 | 口径 |
|---|---|---|
| `eval_os_unified.py` | 早期统一对比脚本（前 404 对，baseline + M4） | 逐像素向量 EPE |
| `eval_os_full_metrics.py` | 全量 424 对全指标，附前 404 对历史协议 | 配对级 + 逐像素向量 EPE |
| `eval_os_official_protocol.py` | 按 GDROS 官方 `test.py` 语义报数，可与论文 Table II 并列 | 逐分量 `\|Δx\|,\|Δy\|` |
| `diag_gdros_epe.py` | 同一批预测流只换误差定义，验证 π/2 比值 | 两种口径对照 |
| `diag_m4_top25.py` | Top-25 失败配对的 B / C / oracle / LSR 前后分解 | 官方配对级 EPE |
| `ablate_v2_off.py` | **零训练消融**：旁路 V2（DINO 辅助）那一路，同一权重下跑两路 / 单路 | 与 `eval_os_official_protocol.py` 一致（官方） |

## `ablate_v2_off.py` 的实现与状态

E073-F3 的相关体融合是单一标量：

```python
a = torch.sigmoid(self.corr_gate_logit)          # 训练后 a ≈ 0.5
corr = (1 - a) * Corr1 + a * Corr2               # Corr1 = V1(GDROS), Corr2 = V2(DINO)
```

消融把 `corr_gate_logit` 设为 `-80`（`a ≈ 0`），F2 分支把 `fuse_gate` 换成
恒定大负输出使 `gs ≈ 0`，因此只剩 GDROS 兼容那一路的 CorrBlock。
M4 的 B 分支（F2/F3）与 C 分支（F2/F3/student）共 **5 个**
`E073FusionMatcher` 实例一起切换；不改层维度、不重新训练。

```bash
# 官方 test.py 口径：GDROS 官方权重 + M4 两路 + M4 单路，三者可直接并列
CUDA_VISIBLE_DEVICES=2 python ablate_v2_off.py \
  --configs both,v2off --max-pairs 424 \
  --out ablate_v2_off_result_official.json

# 仅验证消融生效（1 对、iters=1，自动跳过 baseline，指标无效）
CUDA_VISIBLE_DEVICES=2 python ablate_v2_off.py \
  --configs both,v2off --max-pairs 1 --iters 1 --probe
```

脚本复用 `eval_os_official_protocol.py` 的 `Collector` 与 `run_one("baseline")`：
`diff = post - gt[None]` 后按 `sum(diff**2, dim=0).sqrt()` 得逐分量误差，
配对级 `EPE(k)` 聚合出 `AEPE / CMR@τ / AEPE@τ / RMSE_official_var`。
`--no-baseline` 可跳过 GDROS 参照行（此时 sanity check 会判 FAIL）。

输出还包括：

- `SANITY PASS/FAIL`：把 GDROS 与 `both` 对照 2026-11-13 官方口径 424 对历史值
  （AEPE 1.4618 / 1.4738），不通过则 `v2off` 结论不可用；
- `pair_EPE` 原始数列，用于 `both` / `v2off` 的逐对差值与灾难失败归因；
- `AEPE_vector_EPE_reference`，仅作旧口径交叉验证，**不参与并列比较**。

**状态**：GPU 驱动已修复，CPU 结构自检与 GPU probe 均通过 ——
权重加载 `missing=[] unexpected=[]`、命中 5 个
`E073FusionMatcher`（`variants=['F2','F3','F2','F3','F3']`）、训练后
`corr_gate_logit ∈ [0, 0.007]` 即 `a ≈ 0.5`；probe 首对
`max|Δpost|=56.04 / mean|Δpost|=14.73`，证明旁路 V2 确实改变了输出。
`--device cpu` 会把 `e074.lsr` 换成恒等映射（`LSR` 内部硬编码 `.cuda()`），
仅用于链路自检，该模式下指标无效。

**2026-11-14 正式结果**（官方口径 424 对，`SANITY PASS`，10/10 项复现历史值）：

| 配置 | AEPE↓ | CMR@1↑ | CMR@2↑ | CMR@5↑ |
|---|---|---|---|---|
| GDROS 官方权重 | 1.4618 | 33.73 | 81.60 | 99.53 |
| M4 两路 `both` | 1.4738 | 72.41 | 84.91 | 94.10 |
| M4 单路 `v2off` | 7.7457 | 0.94 | 4.25 | 24.76 |

逐对 `ΔEPE` mean `+6.2719` / median `+6.0892`，仅 `15/424` 变好。
完整结论与「零训练消融不等于重训单路」的边界见
[`docs/ABLATION_V2.md`](../../docs/ABLATION_V2.md)，原始数据见
[`research/results/OS_20261114/`](../../research/results/OS_20261114/)。

## 未入库的探针脚本

10–11 月还有大量一次性诊断脚本（`m1_*` / `m2_*` / `m3_*` / `g*` / `cov_*` /
`tail_finetune.py` 等）用于探查尺度、等变性、鲁棒拟合与尾项续训，
它们是被否定的探索路径，不属于当前方法，因此不入库。其**结论**已提炼进
[`docs/METHOD_DESIGN.md`](../../docs/METHOD_DESIGN.md) §5。
