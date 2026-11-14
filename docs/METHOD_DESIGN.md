# 当前方法设计：E075A-M4

> 本文描述**当前已实现并评测**的方法（截止 2026-11-14），不是候选方向。
> 历史探索、被否定的假设与阶段性结果卡放在 `research/` 下，不作为当前设计。
> 指标见 [RESULTS_OS.md](RESULTS_OS.md)，评测口径见 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)。

## 0. 一句话

在 GDROS 稠密匹配框架上，保持官方前端分布不变，用**两路相关体证据融合 +
双学生分工 + 区域级门控**把紧阈值精度（CMR@1/2）大幅提升；代价是松阈值召回
（CMR@5 及以上）略低，误差分布呈「尖峰 + 重尾」。

## 1. 任务与数据

| 项 | 值 |
|---|---|
| 任务 | 光学（opt）→ SAR 全局仿射配准，输出 512×512 稠密流，再由 LSR 投影为 6 参数仿射 |
| 数据集 | OS（`gdros_official_e003a`），训练 2011 对 / 测试 424 对，512×512 |
| 标签 | 6 参数全局仿射生成的 `truth_flow` |
| 基线 | GDROS released `GDROS-os512.pth`：4 层全对相关金字塔 + GRU 迭代 32 次 + LSR |
| 评测 | `iters=32`、LSR `Npoint=200000`、valid mask 实测全 1；黑边参与统计 |

## 2. 分层结构

| 层 | 模块 | 具体做法 | 训练状态 |
|---|---|---|---|
| L1 | **V1 前端**（GDROS 兼容） | 保留官方 `FeatureEncoder`（stride-8、256 通道、raw dot-product）+ `FeatureTransformer`（1 层 swin）；只加 optical / SAR 各自的残差模态适配器（3→24→3 conv，末层零初始化） | E072 起联合训练 |
| L1' | **V2 前端**（DINO 辅助） | 在 V1 特征上以 **FiLM + 残差注入**一个冻结的 DINO 系特征（E046，只读），补全局语义 | E046 冻结，注入层训练 |
| L2 | **相关体级门控融合**（E073-F3） | V1、V2 各自算 CorrBlock，查表结果按可学习标量门控融合：`corr = (1−a)·Corr₁ + a·Corr₂`，其中 `a = sigmoid(corr_gate_logit)`；再送 GRU 更新 | 端到端联合训练 |
| L3 | **双学生**（E074-B / E074-C） | B：冻结 F2/F3 两路流，用逐像素 gate（输入含各自 LSR 残差）融合，目标偏平衡；C：从 F3 热启动，朝 F2/F3 均值蒸馏 + GT 监督，目标偏紧阈值精度 | B/C 各自独立训练 |
| L4 | **区域门控融合**（E075A-M4） | 冻结 B、C，16 通道证据 → 两层 3×3 conv → sigmoid → **4×4 平均池化 + 双线性上采样**得到空间平滑门控图，`out = (1−g)·B + g·C` | 只训门控，**约 9.3k 参数** |

前向数据流（单对、单次迭代）：

```
a, b ──┬─ V1 前端 ─→ F₁ ─→ CorrBlock₁ ─┐
       └─ V2 前端 ─→ F₂ ─→ CorrBlock₂ ─┴─(1−a)· + a·─→ GRU ×32 ─→ flow ─→ LSR ─→ affine
                                                                  │
                        B 分支 flow 序列 ─┐                        └─ 监督：Lseq + Lgeo
                        C 分支 flow 序列 ─┴─(1−g_reg)·B + g_reg·C ─→ 最终流 ─→ LSR ─→ 指标
```

### 2.1 L4 门控的输入与标签

门控输入是 16 通道显式证据：B / C 两支的预测流、二者之差、差的绝对值与范数、
各自 LSR 投影、以及 `|LSR(p) − p|`（局部流与全局几何的自洽度）。
标签不需要人工标注：

```
t = sigmoid( (e_B − e_C) / 0.75 )
```

其中 `e_B`、`e_C` 是两支各自经 LSR 后的逐像素 EPE —— 即「在这个像素上 C 是否真的
比 B 好」，用的正是最终评测所处的仿射域。

### 2.2 损失与超参

```
L = Lseq + Lgeo
  + 0.35 · BCE(g, t)          # 门控监督
  + 0.20 · CVaR₁₀%(EPE)       # 压尾部
  + 0.03 · H(g)               # 熵，防塌缩
  + 0.05 · (mean(g) − mean(t))²   # 均值平衡
```

| 超参 | 值 |
|---|---|
| 训练 | `train_iters=12`，`steps=120000`，AdamW `lr=2e-5`，均匀采样单对 batch |
| 保护 checkpoint | **step 80000**（`M4_step80000.pt`） |
| M4 超参 | `gate_temp=0.75`、`gate_weight=0.35`、`tail_weight=0.20`、`tail_frac=0.10`、`entropy_weight=0.03`、`balance_weight=0.05` |
| 评测 | `iters=32`，LSR `Npoint=200000` |

E074 阶段另用 1/2/5/10 px 分桶加权，把优化目标从单点 AEPE 改为多阈值曲线。

## 3. 设计依据（每一条都有实测支撑）

**① 为什么不动 backbone，而用零初始化适配残差。**
跨模态迁移的常见做法是直接把特征换成 DINO / CLIP 描述子，但官方匹配器的
context 编码器、CorrBlock 与 GRU 都是在「stride-8、256 通道、raw dot-product
激活」这一分布上训练的；换成 1024 维 L2 归一化描述子会把它们推入 OOD ——
早期直接桥接实验（E-031A）实测 AEPE 18.67、CMR@2 1.17%，基本不可用。
因此先对齐分布，再谈融合。

**② 为什么在相关体层融合，而不是特征拼接或结果集成。**
V1 强在局部纹理与几何细节，V2 强在全局语义，两者维度与归一化方式不同，无法直接拼接。
相关体是二者唯一的公共度量空间：融合发生在「同一个 anchor 对同一批 target 的相似度」上，
权重具有明确的几何含义。

**③ 为什么拆成双学生 + 区域门控。**
单一模型同时优化 CMR@1 与 CMR@5 会互相拉扯。先训练两个目标不同的学生
（B 偏平衡、C 偏紧阈值），再用极轻门控逐区域裁决；两个设计细节是关键：
(a) 门控输入包含各支与自身 LSR 投影的偏差，把「局部流与全局几何是否自洽」作为显式
可靠性证据；(b) 逐像素门控先算再 4×4 池化 + 上采样平滑，避免高频抖动造成碎片化切换。
B/C 全部冻结，不破坏已收敛的解。

**④ 门控标签直接对齐多阈值 CMR。** 见 §2.1；标签定义与评测指标同源，
不需要额外的可靠性标注。

## 4. 已确证的边界（读结果前必须知道）

1. **评测口径不可混用。** GDROS 官方 `test.py` 的 `sum(dim=0).sqrt()` 算的是逐分量
   绝对误差 `|Δx|,|Δy|`；本项目全部实验用 `linalg.vector_norm(dim=1)` 真向量 EPE，
   两者 AEPE 相差约 π/2。内部消融在同一口径下有效，但与论文 Table II 并列时必须
   统一到官方口径。官方栏位 "RMSE" 实际是 `np.var(MAE_list)`（方差）。
   详见 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)。
2. **重尾继承自双学生，门控救不了。** M4 ≈ `0.5·(B+C)`；逐像素 oracle(B,C) 的
   CMR@5 仍只有 91.22%，低于 baseline 93.51%。门控是软加权，既没制造也没消除重尾。
3. **失效发生在跨模态稠密对应阶段。** Top-25 灾难配对的 `ΔLSR` 平均仅 +0.0287
   （23/25 对 `|ΔLSR|<0.5`），拟合出的尺度/旋转几乎正确，错的是**平移**
   （`t_err=16.64 px`，增益 `α≈0.80`）；LSR 前 EPE 已 >5 px。
   GDROS 官方权重在同一 25 对上平均仅 1.2595，说明是本文前端特有的失败。
4. **高度集中在 1.10 尺度档。** Top-25 中 23/25 对 `gt_scale≈1.10`；
   该档 123 对里有 23 对 EPE>5 px，其余三个尺度档合计只有 2 对。
   机制是平移估计向内收缩、迭代在第 2 步后停在伪不动点，而 GDROS 能继续收敛。
5. **门控置信度未校准。** 门控最自信（g→0/1）的区间，恰恰是 oracle 误差最高的像素
   （U 形），即置信度与难度正相关，无法为尾部提供保护。

以上结论的一手证据集中在
[`research/results/OS_20261113/`](../research/results/OS_20261113/)。

## 5. 已被实验否定的方向（避免重复投入）

- **鲁棒仿射拟合（IRLS / trimmed-LS）不是重尾主因**：trim70 一致最优但仅
  `CMR@5 +0.25pp`、`CMR@1 +1.3pp`；46–48 个灾难配对在鲁棒拟合下几乎不动
  （C 13.31 → 13.17）。错误是「配对本身匹配错」，不是「离群污染了最小二乘」。
- **像素级尾权重续训不可转移**：训练批 `mat5≈100%`，训练集没有 >5 px / >10 px
  样本，尾损失只作用在约 2.5 px 残差上；4000 步续训还让 CMR@1 崩塌。
- **门控硬路由假设不成立**：`PairFusion.forward` 是 `(1-g)·B + g·C` 软融合，
  不是 argmax；且存在约 4.55% 像素 B/C 同时 >10 px。
- **数据侧缺覆盖不是主要解释**：训练与测试的几何分布几乎一致，难格本来就在训练集里，
  但训练同格 0% 失败、测试 23–33% 失败 —— 是泛化问题而非覆盖问题。

## 6. 实现索引

| 内容 | 文件 |
|---|---|
| V1 / V2 前端 | [`e072a_layerwise_frontend_alignment.py`](../research/remote_stage/os/e072a_layerwise_frontend_alignment.py) |
| F2 / F3 融合、`corr_gate_logit` | [`e073a_v1_v2_fusion_longtrain.py`](../research/remote_stage/os/e073a_v1_v2_fusion_longtrain.py)（`E073FusionMatcher`） |
| B / C 双学生、`lsr` | [`e074a_f2_f3_twostage_fusion.py`](../research/remote_stage/os/e074a_f2_f3_twostage_fusion.py)（`E074Model`） |
| 区域门控、损失 | [`e075a_preserve_high_precision_recover_c2c5.py`](../research/remote_stage/os/e075a_preserve_high_precision_recover_c2c5.py)（`PairFusion`） |
| 评测脚本 | [`research/eval/`](../research/eval/README.md) |

> **实现层注记（2026-11-15）**：E074 variant C 里的 f2 / f3 teacher 只服务
> E074 训练期的 `distill` 损失；E075A 的 C 分支在训练和推理中都只取 student
> 输出。当前实现默认跳过这两条冻结链（`need_teacher_avg=False` /
> `PairFusion(c_skip_teachers=True)`），424 对官方口径指标**逐位不变**。
> 这是工程剪枝，**不改变上表的方法结构**，详见
> [GRU_ITER_SWEEP.md](GRU_ITER_SWEEP.md) §6.1。

## 7. 下一步方向

重尾的主病灶在 E072/E073 的跨模态稠密对应层，因此优先：

1. **难度递增的难例合成**：针对 1.10 尺度档造出平移失效样本，让训练批出现
   >5 px / >10 px 案例（`mat5` 从 100% 降到 ~85–90% 作为验证信号）。
2. **从 GDROS released 蒸馏并加正则**：在训练期恢复难配对匹配能力，
   而不是在推理期做基线兜底。
3. **门控置信度校准与弃权**：当 B/C 分歧大或自洽度低时回退到软平均。
4. **上游分支消融**：两路相关体减为单路，量化 V2（DINO 辅助）那一路的边际贡献。

前三项目前是方向，尚未纳入保护版本。第 4 项**只完成了前置的诊断性旁路**：
2026-11-14 在官方口径 424 对上把 V2 贡献置零（不重训，`SANITY PASS`），
AEPE `1.4738 → 7.7457`、`96.5%` 配对变差。这说明冻结管线与两路融合分布强耦合、
推理时不能直接拆，**但不能量化 V2 的边际贡献**（退化里混有下游 OOD）。
第 4 项本身**仍未完成**，需要固定单路后重训或至少适配性续训。实现见
[`research/eval/ablate_v2_off.py`](../research/eval/ablate_v2_off.py)，诊断结论、
边界与后续方案见 [ABLATION_V2.md](ABLATION_V2.md)。
