<div align="center">

# VSMatch

**面向光学–SAR 跨模态图像匹配的双路相关体融合与区域门控方法**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![Dataset](https://img.shields.io/badge/Dataset-OS%20(optical–SAR)-2E8B57)](#)
[![Method](https://img.shields.io/badge/Current%20method-E075A--M4%4080K-8A2BE2)](#)
[![Evaluation](https://img.shields.io/badge/Evaluated-424%20pairs-1F6FEB)](#)
[![Status](https://img.shields.io/badge/Status-research%20snapshot-FF8C00)](#)

</div>

> 本仓库记录方法实现、评测协议与结果。克隆本仓库后，从
> **[当前方法设计](docs/METHOD_DESIGN.md)** 开始读；需要数字看
> **[当前结果与已知短板](docs/RESULTS_OS.md)**，需要复现看
> **[评测口径](docs/EVALUATION_PROTOCOL.md)** 与 **[复现指南](docs/REPRODUCE.md)**；
> 跨数据集泛化结果看 **[UBCv2 泛化评测](docs/UBCV2_GENERALIZATION.md)**。
> 数据集、权重与预测张量不入库。

<details>
<summary><b>目录</b></summary>

- [最近更新](#最近更新)
- [核心特性](#核心特性)
- [方法架构](#方法架构)
- [当前结果](#当前结果)
- [已知短板](#已知短板)
- [仓库结构](#仓库结构)
- [复现边界](#复现边界)
- [路线图](#路线图)

</details>

---

## 最近更新

| 日期 | 内容 |
|---|---|
| **2026-11-15（UBCv2 诊断）** | 查清「M4 `3.8941` 优于论文 GDROS in-domain `4.49`」这个对比**不成立**：用官方阈值分档反解否定「论文筛掉病态对」的解释（要压到 4.49 需保留 7965 对 > 现有 1625）；逐对诊断显示 GDROS 在 UBCv2 上的误差与旋转幅度 / warp 损失面积**全不相关**且预测流近似恒等变换；**零流基线 `29.7910`** 下 GDROS 只比「什么都不预测」好 `26.1%`，M4 好 `86.9%`。另定量了「旋转基准点未公开」带来的 **1.83×** 不可比幅度。**起步在 UBCv2 上按官方 recipe 训 GDROS**（120K 步，约 7.7 h）。见 [UBCV2_DIAGNOSIS.md](docs/UBCV2_DIAGNOSIS.md) |
| **2026-11-15（UBCv2）** | 新增 **UBCv2 跨数据集泛化评测**：官方 RGB/SAR 测试集 1625 对，按论文变换参数构造 GDROS 布局、修正 SAR float32 位深陷阱、用 OS 官方 `datum` 反推并反证 flow 方向；同一份 OS 权重下 M4 AEPE `3.8941` vs GDROS `22.0091`（CMR@5 `74.52` vs `11.26`）。**跨数据集泛化，不与论文 Table III 并列**，见 [UBCV2_GENERALIZATION.md](docs/UBCV2_GENERALIZATION.md) |
| **2026-11-15** | 落地 C 分支 `teacher_preds` 死计算剪枝（GRU 160→96 次，424 对指标逐位不变，推理 1.339× / 训练前向 1.455×，见 [GRU_ITER_SWEEP.md](docs/GRU_ITER_SWEEP.md) §6.1）；在剪枝后 M4 上补齐**对称单路旁路**（双 GPU、官方口径 424 对），并澄清 V2 不是独立 backbone，见 [ABLATION_BACKBONE_SYMMETRIC.md](docs/ABLATION_BACKBONE_SYMMETRIC.md)；再补 **2000 步适配性短训**（方案 A），证明零训练旁路 95–96% 的 AEPE 落差来自下游 OOD，见 [ABLATION_SHORT_FINETUNE.md](docs/ABLATION_SHORT_FINETUNE.md) |
| **2026-11-14** | 补齐当前方法设计、统一评测口径与全量 424 对结果文档；同步评测与消融脚本；完成 V2 分支的**诊断性零训练旁路**（不构成模块消融结论，见 [ABLATION_V2.md](docs/ABLATION_V2.md)）；补跑 GRU 迭代次数官方口径 sweep，确认 `iters=32` 过冲、`24` 更优 |
| **2026-11-13** | 核实 GDROS 官方复现无误：官方 `test.py` 计算的是**逐分量绝对误差**，本项目用**向量 EPE**，两者 AEPE 相差约 π/2；补跑官方口径下 M4 结果；完成 Top-25 失败配对归因 |
| **2026-11-02 ~ 10-31** | 定位 CMR 阈值交叉现象：重尾继承自 E074 B/C 双分支，门控为软加权且 oracle 也无法补救；鲁棒仿射拟合被实验否定为主因 |
| **2026-09-30** | 冻结 E072 → E073 → E074 → E075A 实现链与阶段性结果卡 |

> ⚠️ `research/CURRENT_METHOD_STATE.md` 及 `research/results/` 下的 2026-09 结果卡是**历史快照**，
> 其指标口径与最新结论以 `docs/` 下的文档为准。

---

## 核心特性

1. **分层对齐的跨模态前端，而不是替换 backbone。** 保留 GDROS 官方
   `FeatureEncoder`（stride-8 / 256 通道 / raw dot-product）与 `FeatureTransformer` 的分布，
   只叠加零初始化的模态适配残差，以及一层冻结 DINO 特征的 FiLM 注入。直接把特征换成
   1024 维描述子会把 context / CorrBlock / GRU 推入 OOD（早期实验 AEPE 18.67、CMR@2 1.17%）。
2. **在相关体层面融合两路证据。** V1（GDROS 兼容，强局部纹理）与 V2（DINO 辅助，强全局语义）
   各自构建 CorrBlock，按可学习标量门控融合
   `corr = (1-a)·Corr₁ + a·Corr₂`，再送 GRU 迭代。相关体是二者唯一的公共度量空间。
3. **双学生 + 区域级门控，把紧阈值精度与松阈值召回拆开。** E074-B（平衡）/ E074-C（紧精度）
   两个学生冻结后，由仅 **9.3k 参数**的区域门控在 4×4 区域尺度上裁决，
   `out = (1-g)·B + g·C`，避免逐像素高频抖动。
4. **门控标签由评测指标本身定义。** 标签 `sigmoid((e_B − e_C)/τ)` 来自两支各自 LSR 后
   的逐像素 EPE，直接对齐多阈值 CMR；损失为 `Lseq + Lgeo + 0.35·BCE + 0.20·CVaR₁₀%
   + 0.03·熵 + 0.05·均值平衡`。
5. **评测口径全量核查并公开。** 官方 `test.py`、论文 Table II、项目内部向量 EPE 三套数字
   的关系已逐位对齐，见 [评测口径](docs/EVALUATION_PROTOCOL.md)。

> 定位说明：CorrBlock、GRU 迭代、LSR、DINOv2 特征均为已有组件（GDROS / DINOv2）。
> 本工作的贡献是**融合结构与门控监督方式**，不主张发明 recurrent geometry 或稠密匹配本身。

---

## 方法架构

```mermaid
flowchart LR
    A[光学图 a] --> V1["L1 · V1 前端<br/>GDROS 兼容 + 模态适配残差"]
    B[SAR 图 b] --> V1
    A --> V2["L1' · V2 前端<br/>FiLM 注入冻结 DINO 特征"]
    B --> V2
    V1 --> C1["CorrBlock₁<br/>4 层相关金字塔"]
    V2 --> C2["CorrBlock₂<br/>4 层相关金字塔"]
    C1 --> F["L2 · 相关体级门控<br/>corr = (1−a)·C₁ + a·C₂"]
    C2 --> F
    F --> G["GRU × 32 → 稠密流"]
    G --> LSR["LSR 仿射投影<br/>Npoint=200000"]
    LSR --> E074B["L3 · E074-B 学生"]
    LSR --> E074C["L3 · E074-C 学生"]
    E074B --> M4["L4 · 区域门控 M4<br/>9.3k 参数 · B/C 冻结"]
    E074C --> M4
    M4 --> OUT[最终流 → 评测]
```

链路对应实现：[E072 前端](research/remote_stage/os/e072a_layerwise_frontend_alignment.py) →
[E073 相关体融合](research/remote_stage/os/e073a_v1_v2_fusion_longtrain.py) →
[E074 双学生](research/remote_stage/os/e074a_f2_f3_twostage_fusion.py) →
[E075A-M4 区域门控](research/remote_stage/os/e075a_preserve_high_precision_recover_c2c5.py)。

---

## 当前结果

保护版本 **E075A-M4@80K**（`M4_step80000.pt`），OS 数据集全量 **424 对**，
`iters=32`、LSR 200k。两套口径**必须分开读**，不能直接与论文表格混排：

| 口径 | 方法 | AEPE↓ | CMR@1↑ | CMR@2↑ | CMR@5↑ |
|---|---|---|---|---|---|
| **官方 `test.py`**（逐分量误差，可与论文并列） | GDROS 官方权重 | **1.4618** | 33.73 | 81.60 | **99.53** |
| | **E075A-M4@80K** | 1.4738 | **72.41** | **84.91** | 94.10 |
| **项目内部向量 EPE**（配对级，内部消融用） | GDROS 官方权重 | **2.3216** | 9.43 | 52.12 | **94.34** |
| | **E075A-M4@80K** | 2.3669 | **52.59** | **79.72** | 88.92 |

**结论**：官方口径下 M4 的 CMR@1 是论文 GDROS（33.88%）的两倍以上，
三档 `AEPE@τ` 低 36–47%，全部 AEPE 基本持平（1.4738 vs 1.4618）；
代价是 CMR@5 落后 5.42 pp。完整指标、逐像素口径、截断式 `AEPE@τ`
与三种 RMSE 定义见 [当前结果](docs/RESULTS_OS.md)。

**诊断性旁路（非模块消融，不进结果主表）**：2026-11-14 在冻结权重上直接把
V2 那一路的贡献置零（不重训），AEPE `1.4738 → 7.7457`。这测的是
**下游对输入分布偏移的敏感性**，不是 V2 的边际贡献 —— 大幅退化里混合了
「少了一路信息」和「下游 OOD」两个无法分离的原因，因此不能写成
「去掉 DINO 模型就废了」。真正的模块消融需要重训单路，见
[V2 旁路诊断](docs/ABLATION_V2.md) §4.1。

> `iters=32` 是与论文 / GDROS 并列的前提。零训练 sweep 显示 `24` 已更准
> （AEPE 1.4449、68% 配对更好）、`16` 可省一半评测成本，但**改 iters 就换口径**：
> 内部筛选可用，最终报数仍须 32。见 [GRU 迭代次数 sweep](docs/GRU_ITER_SWEEP.md)。

---

## 已知短板

- **尖峰 + 重尾**：多数配对 <1 px，但约 6% 的灾难配对有 10–20 px 误差，
  把官方栏位 `RMSE`（实为 `np.var(MAE_list)` 方差）从 0.7463 顶到 8.3350。
- **失败集中在 1.10 尺度档**：按官方口径排序的 Top-25 中 23/25 落在该档；
  主因是**跨模态稠密对应阶段的平移估计失效**（`α≈0.80`），
  不是 LSR 几何重建（23/25 对 `|ΔLSR|<0.5`），也不是区域门控（两支 oracle 仍 >5 px）。
  GDROS 官方权重在同一 25 对上平均仅 1.2595，说明不是数据本身难。
- **门控置信度未校准**：最自信处恰是 oracle 也最难的像素（U 形）。

细节与证据见 [Top-25 归因](research/results/OS_20261113/M4_Top25误差归因_20261113.md)
和 [CMR 现象分析](research/results/OS_20261113/CMR_现象分析.md)。

---

## 仓库结构

| 路径 | 内容 |
|---|---|
| [`docs/METHOD_DESIGN.md`](docs/METHOD_DESIGN.md) | **当前方法设计**（分层结构、损失、超参、设计依据） |
| [`docs/RESULTS_OS.md`](docs/RESULTS_OS.md) | 当前指标、误差结构、口径对照 |
| [`docs/ABLATION_V2.md`](docs/ABLATION_V2.md) | V2（DINO 辅助）分支零训练旁路：定义、结果、边界（诊断性，非模块消融） |
| [`docs/ABLATION_BACKBONE_SYMMETRIC.md`](docs/ABLATION_BACKBONE_SYMMETRIC.md) | 剪枝后 M4 的**对称单路旁路**（双 GPU）：两方向结果、V2 结构边界、推论限制 |
| [`docs/ABLATION_SHORT_FINETUNE.md`](docs/ABLATION_SHORT_FINETUNE.md) | **适配性短训（方案 A）**：受控 2000 步续训，零训练旁路 OOD 的定量证据 |
| [`docs/GRU_ITER_SWEEP.md`](docs/GRU_ITER_SWEEP.md) | GRU 迭代次数 sweep：代码事实、历史证据、省训练时间的边界 |
| [`docs/UBCV2_GENERALIZATION.md`](docs/UBCV2_GENERALIZATION.md) | **UBCv2 跨数据集泛化**：数据构造、SAR 位深、flow 方向反推、全量结果与不可并列边界 |
| [`docs/UBCV2_DIAGNOSIS.md`](docs/UBCV2_DIAGNOSIS.md) | **UBCv2 结果可比性诊断**：筛选假设的反解否定、逐对诊断（几何不敏感 + 预测恒等）、零流基线 `29.7910`、旋转基准点 1.83× 自由度 |
| [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) | CMR / AEPE / RMSE 定义与 π/2 口径更正 |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | 环境、路径约定、复现命令与限制 |
| `research/remote_stage/os/` | E072–E075A 训练与模型实现（冻结版本） |
| [`research/eval/`](research/eval/README.md) | 评测脚本与零训练消融脚本 |
| `research/results/OS_20261113/` | 2026-11-13 一手评测报告与原始指标 JSON |
| `research/results/OS_20261114/` | 2026-11-14 分支消融与 GRU sweep 原始结果 JSON（含逐对 EPE） |
| `research/results/OS_20261115/` | 2026-11-15 剪枝验证、计时与对称单路旁路原始结果 JSON |
| `research/experiments/`、`research/results/` | 早期方法搜索的规格、结果卡与审计记录 |
| `research/CURRENT_METHOD_STATE.md` | 2026-09 历史方法状态快照 |

---

## 复现边界

仓库不附带 GDROS 源码、OS 数据集和 checkpoint。实验脚本从部署工作区导入上游 GDROS 代码，
通过 `SPARSEGEO_ROOT` 指定根目录；历史 zzvr 布局仍是默认值，以保留冻结实验的原始行为。
`research/eval/` 下脚本的路径常量需要按实际工作区调整，见该目录 README。

本地可做的静态检查：

```bash
python -m pip install -r requirements-research.txt
python -m compileall -q tools research/remote_stage research/eval e081a_whu_our_native_train.py
python -m unittest discover -s tools -p 'test_*.py'
python scripts/preflight.py --scope all
```

---

## 路线图

- [x] E072 分层前端对齐 → E073 相关体级融合 → E074 双学生 → E075A 区域门控
- [x] OS 424 对全量评测，两套口径分别与 GDROS released / 论文 Table II 对齐
- [x] 重尾归因到跨模态对应层，排除 LSR 与门控主因
- [ ] 难度递增的难例合成，覆盖 1.10 尺度档的平移失效
- [ ] 从 GDROS released 蒸馏并加正则，恢复难配对匹配能力
- [ ] 门控置信度校准与弃权机制
- [x] 上游前端**对称**单路旁路（剪枝后 M4、双 GPU、官方口径 424 对；两个方向都大幅掉点，**仍只是敏感性诊断，不等于模块消融**）
- [x] 适配性续训（固定单路、冻结前端、2000 步、双 GPU）：AEPE `7.7457→1.7127` / `41.3298→3.3824`，落差恢复 96.2% / 95.2%，**零训练旁路不能量化模块贡献**已被定量证实
- [ ] 同预算（80K 步）严格重训单路 E073 → E074 → M4，才能量化 V2 的边际贡献；或把短训预算扩到与保护版本可比并解冻更多下游
- [x] GRU 迭代次数官方口径 sweep：确认 `iters=32` 过冲、`24` 更准、`16` 省一半
- [x] 核清 M4 调用链（剪枝前）：`iters=32` 前向 = 5 条 GRU 链 / 160 次 update + 129 次 LSR
- [x] 落地 C 分支 `teacher_preds` 死计算剪枝：GRU 160→96 次；424 对官方口径 AEPE / CMR / RMSE **逐项 Δ=0**（`SANITY PASS`），全量 422→317 s
- [x] 同进程 A/B 计时：推理 **1.339×**、训练前向 **1.455×**；剪枝同时覆盖 `PairFusion.forward` 与 M4 手工训练循环
- [x] **UBCv2 跨数据集泛化**（1625 对、官方口径、单 seed）：M4 `3.8941` vs GDROS `22.0091`；证实 M4 迁移退化 2.64× 而 GDROS 15.06×，**属泛化对比，非 Table III in-domain 复现**
- [ ] **UBCv2 in-domain 训练**：GDROS 官方 recipe（120K 步，约 7.7 h，2026-11-15 启动中）；随后需补 M4 在 UBCv2 整链重训（≈16.5 h）才有严格同条件的四格表
- [ ] 实测剪枝后的完整训练 step（前向 1.455× 不等于总时长 1.455×）
- [ ] 诊断两处逐迭代 LSR（`B.gate_input` / `PairFusion.gate_features`）的缓存冗余 —— 129 次 LSR 全保留，现为最大可压缩项
- [ ] 固定 `train_iters=8` 短训 2K 步，验证训练步数是否真能压缩（零训练 sweep 不能回答）
