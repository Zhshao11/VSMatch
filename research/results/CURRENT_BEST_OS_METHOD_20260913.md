# 当前最佳 OS 方法快照（2026-09-13）

> 2026-11-15 口径纠正：下文历史表中的 M4 使用 404 对标准向量 EPE，GDROS released replay 使用 424 对公开分量误差，不能直接相减或作为同口径优势证据。正确的配对重汇总见 [OS 评测复核](OS_GDROS_PROTOCOL_AUDIT_20261115.md)。M4 继续作为保护主模型；其“最高 CMR@1”描述仅适用于本快照的比较范围，后续 E082-S3 曾达到 58.1683%，但其他指标退化，未晋级。

> 随后 E092A 已完成 M4/GDROS 全部 424 对官方口径重评估，见 [最新同口径结果](OS_OFFICIAL424_E092A_SUMMARY_20261115.md)。GDROS32 / M4-32 的公开平均误差为 1.461486 / 1.474430，CMR@1 为 33.7264% / 72.4057%，CMR@2 为 81.6038% / 84.9057%，CMR@5 为 99.5283% / 94.1038%。该新表替代下文跨口径的 GDROS 对比；原始历史结果保留。

## 结论

截至 2026-09-13，当前 OS 数据集主线中应固定保留的最佳方法是：

**E075A-M4 Region Gate，checkpoint `M4_step80000.pt`，推理输出 `iter24_post_lsr`。**

正式状态为 `PRESERVE_RECOVER_SUCCESS`。该方法是当前 LuckyR1 的主要对照。LuckyR1 仍在实验阶段，尚未产生可以替换 E075A-M4 的有效 model-facing 性能结果。

这里的“最佳”指当前选定的 OS 主方法：M4 同时保持最高的 CMR@1 和当前 M4 系列中较好的 CMR@5。E075A-M2 在 AEPE、CMR@2 上略有优势，作为 Pareto 次级对照保留，不能与 M4 混写成同一个结果。

## 正式指标

评估对象：OS 404 pairs；正式统计口径为先计算每个 pair 的 dense mean EPE，再计算 pair-level CMR。`CMR@t` 表示 pair AEPE `< t`。

| 方法 / 输出 | AEPE ↓ | CMR@1 ↑ | CMR@2 ↑ | CMR@5 ↑ | CMR@10 ↑ | CMR@16 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **E075A-M4 step80K / iter24 post-LSR** | **2.3613** | **56.6832%** | **78.9604%** | **88.8614%** | **94.8020%** | **97.0297%** |
| E075A-M2（Pareto 次级对照） | 2.3477 | 55.9406% | 79.2079% | 88.1188% | 未在本快照核定 | 未在本快照核定 |
| GDROS released replay | 1.4615 | 33.7300% | 81.6000% | 99.5300% | 未核定 | 未核定 |

E075A-M4 相对 GDROS released replay：

- CMR@1：`+22.9532` 个百分点，当前方法的主要优势。
- CMR@2：`-2.6396` 个百分点。
- CMR@5：`-10.6686` 个百分点，当前最主要的性能缺口。
- AEPE：高 `0.8998 px`，说明长尾误差仍明显更大。

## 方法结构

E075A-M4 不是单一 backbone，也不是只替换一层特征。完整路径如下：

```text
OPT / SAR
   ├── E074A-B branch
   │     ├── E073-F2 matcher
   │     ├── E073-F3 matcher
   │     └── B 内部空间 gate 融合 F2/F3 flow
   ├── E074A-C branch
   │     └── F3 初始化的 student，经 teacher-average distillation 训练
   │
   └── E075A-M4 region gate
         ├── 读取 B flow、C flow
         ├── 读取 B-C、|B-C|
         ├── 读取 LSR(B)、LSR(C)
         ├── 读取 |LSR(B)-B|、|LSR(C)-C|
         ├── 16-channel gate feature
         ├── Conv 16→32→16→1 + sigmoid
         ├── 4×4 average pooling 后上采样为区域 gate
         ├── fused = (1-g) * B + g * C，其中 g 是 C 分支权重
         └── iter24 输出再经过 LSR
```

M4 训练时冻结 B/C 分支，只训练 region gate。正式训练运行至 120K，但正式选择的最佳 checkpoint 是 80K；训练使用 12 次 recurrent iteration，评估比较 iter12/24/32，最佳正式输出为 `iter24_post_lsr`。

## 可复现入口

### M4 checkpoint

```text
/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/
E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/
run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt
```

SHA256：

```text
55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b
```

### B branch checkpoint

```text
/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/
E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-b-stage180k/checkpoints/B_step180000.pt
```

SHA256：

```text
f40d80f2dca62e74ec86c1823963282a490d47f336f2cfce4a28fc211ea33218
```

### C branch checkpoint

```text
/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/
E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-c-stage180k/checkpoints/C_step180000.pt
```

SHA256：

```text
c3c67e6d65aca0dd55cf7af2c2a9eedccadadc0fec5ad11fe6e493e08ecc85b9
```

### 实现源码

```text
/mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated/.codex_tmp/
e075a_preserve_high_precision_recover_c2c5.py
```

已核定源码 SHA256：

```text
1b74c437306e236538e24d110c609d349c56bd7f5d26435c6cc166ae86ec95c3
```

### 正式结果卡

```text
/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/
E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/
run-m4-regiongate120k-r2/formal/RESULT_CARD.md
```

## Checkpoint 曲线中的重要区别

| Checkpoint / 输出 | AEPE ↓ | CMR@1 ↑ | CMR@2 ↑ | CMR@5 ↑ |
|---|---:|---:|---:|---:|
| M4 step40K / iter24 post-LSR | 2.3730 | 55.4455% | 78.4653% | 88.6139% |
| **M4 step80K / iter24 post-LSR** | **2.3613** | **56.6832%** | **78.9604%** | **88.8614%** |
| M4 step120K / iter24 post-LSR | 2.3708 | 56.9307% | 79.4554% | 88.3663% |

120K 的 CMR@1/2 略高，但 AEPE 和 CMR@5 比 80K 更差。因此正式卡选择 80K，而不是简单选择训练最久的 checkpoint。

## 已确认的诊断边界

E075B/C/D 系列对 E075A-M4 的诊断得到以下稳定结论：

- 2011 个模型训练来源 pair 上，M4 的 pair-level CMR@5 为 100%，但这些样本不是 model-unseen。
- 404-pair 历史测试视图中有 45 个 `AEPE >= 5 px` 失败。
- 其中 `41/45 = 91.11%` 在 B/C 两个分支的精确 iter24 post-LSR 输出中都失败。
- 只有 `4/45 = 8.89%` 至少有一个分支已经 `<5 px` 而 M4 仍失败，因此单独修改 M4 gate 无法解释主要长尾。
- 没有发现 iter12 已 `<5 px`、iter24 又失败的普遍漂移，也没有发现 pre-LSR 成功而 post-LSR 失败的系统性 LSR regression。
- GT 通常处于 recurrent CorrBlock query 支持范围内；简单扩大窗口缺少针对性。
- 失败样本的初始 raw correlation 和正确 basin token 覆盖更弱，但“特征/相关性表达不足”与“recurrent 更新未充分利用证据”尚未完全因果分离。

因此 LuckyR1 当前尝试改进的是 B/C 共有的“相关性证据组织与 recurrent 更新”，而不是替换这份 E075A-M4 最佳记录。

## 当前未晋级项目

以下内容不得写入“当前最佳方法”指标：

- `LuckyR1-A/B/C`：尚无有效 model-facing 正式结果。
- LuckyR1 初始 primitive G0：只证明局部张量、unmatched normalization 和玩具梯度可运行，不是完整网络结果。
- LuckyR1 初次 F2/F3 120K：已判定为 `NON_EQUIVALENT_PILOTS`，不是有效 lineage。
- LuckyR1 repaired F2/F3：属于上游重训中间结果，尚未完成 B/C、M4 和 LuckyR1 方法评估。
- E075B/C/D：诊断实验，不是新方法性能结果。
- GT oracle、历史测试失败筛选和 token-level evidence 指标：只能用于诊断，不能作为 no-GT 方法最终结果。

## 更新规则

只有满足以下条件的新实验才能替换本文件中的 E075A-M4：

1. 完成真实 model-facing 接入，而不是 primitive 或离线 proxy。
2. 有严格 checkpoint、源码、数据和推理路径记录。
3. 按 pair-level 正式口径报告 AEPE、CMR@1/2/5/10/16。
4. 至少在预声明验证协议中超过 E075A-M4，并说明是否 model-unseen。
5. CMR@5 或 AEPE 改善时，不能用明显损害 CMR@1/2 作为代价。
6. 失败、OOM、外部终止和非等价训练必须单独留卡，不能覆盖成功记录。

在这些条件满足前，**E075A-M4 step80K / iter24 post-LSR 始终是当前 OS 主方法快照。**
