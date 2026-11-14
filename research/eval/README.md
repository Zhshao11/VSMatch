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
| `ablate_v2_off.py` | **零训练消融**：旁路 V2（DINO 辅助）那一路，同一权重下跑两路 / 单路 | 与 `eval_os_full_metrics.py` 一致 |

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
# 两配置在同一进程、同一权重、同一口径下评测
CUDA_VISIBLE_DEVICES=2 python ablate_v2_off.py \
  --configs both,v2off --max-pairs 404 --out ablate_v2_off_result.json
```

**状态**：脚本已完成语法检查与 CPU 侧结构自检（权重加载
`missing=[] unexpected=[]`、命中 5 个实例、训练后 `corr_gate_logit` 全在
0–0.007）；**尚未产出指标**，因为运行时 GPU 驱动与用户态库版本不匹配导致
CUDA 不可用。`--device cpu --probe` 可用于链路自检（该模式把 `e074.lsr`
替换为恒等映射，指标无效）。

## 未入库的探针脚本

10–11 月还有大量一次性诊断脚本（`m1_*` / `m2_*` / `m3_*` / `g*` / `cov_*` /
`tail_finetune.py` 等）用于探查尺度、等变性、鲁棒拟合与尾项续训，
它们是被否定的探索路径，不属于当前方法，因此不入库。其**结论**已提炼进
[`docs/METHOD_DESIGN.md`](../../docs/METHOD_DESIGN.md) §5。
