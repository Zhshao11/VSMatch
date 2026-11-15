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
| `sweep_gru_iters.py` | **零训练 GRU 迭代次数 sweep**：一次 32-iteration 前向拿到 `[4,6,8,12,16,24,32]` 全曲线 | 与 `eval_os_official_protocol.py` 一致（官方） |
| `verify_c_teacher_pruning.py` | C 分支 teacher 剪枝正确性：同进程 A/B bit-exact + 424 对官方口径 sanity check | 与 `eval_os_official_protocol.py` 一致（官方） |
| `probe_c_prune_timing.py` | 同进程逐对计时：推理 `iters=32` / 训练前向 `iters=12` 的剪枝前后加速 | 不报指标，仅计时 |
| `ablate_backbone_single.py` | 剪枝后 M4 的**对称单路旁路**：一个进程一个配置（`both` / `v1only` / `v2only`），便于两张 GPU 并行 | 与 `eval_os_official_protocol.py` 一致（官方） |
| `merge_backbone_ablation.py` | 合并两份单路结果做逐对 Δ 统计，纯标准库、无需 GPU | 官方 |
| `short_finetune_single_branch.py` | **适配性短训（方案 A）**：锁死单路门控、冻结前端、2000 步续训下游，再同进程官方口径评测 | 与 `eval_os_official_protocol.py` 一致（官方） |
| `merge_short_finetune.py` | 合并两份短训结果，对比零训练旁路与 `both`，纯标准库 | 官方 |
| `build_ubcv2_testset.py` | 从 UBCv2 官方 RGB/SAR 精配准影像构造 GDROS 布局测试集（仿射 + `.flo`） | 论文 UBCv2 变换参数 |
| `verify_ubcv2_testset.py` | 构造结果自检：flow = `M@[x,y,1]−[x,y]`、图像级 warp 一致性、数据集可加载 | 只读验证，不报指标 |
| `eval_ubcv2_official_protocol.py` | **换数据集**：把官方口径评测指向 UBCv2（只覆盖 `TEST_ROOT`，复用 `Collector` / `run_one`） | 与 `eval_os_official_protocol.py` 一致（官方） |
| `summarize_ubcv2.py` | 把 UBCv2 结果 JSON 转成 markdown 表 + 与 OS 基准的泛化落差，纯标准库 | 官方 |
| `diag_ubcv2_pairs.py` | UBCv2 **逐对诊断**：官方 EPE、只算有效重叠区的 EPE、warp 损失面积、预测流的最小二乘仿射 vs 真值 `M` | 官方口径 + 附加诊断字段 |
| `build_ubcv2_split.py` | 为 UBCv2 **train/val** 构造 GDROS 布局 split，复用测试集 SAR LUT 保证辐射一致 | 与 `build_ubcv2_testset.py` 同一套变换 |
| `prepare_ubcv2_train_data.sh` | 解压 `train_images.zip` → 生成 `training/`(4008) 与 `validation/`(1681) | 幂等 |
| `train_gdros_ubcv2.sh` | 用**官方 recipe**（120K 步 / `iters=12` / bs 2 / 400² / OneCycleLR）在 UBCv2 上训 GDROS | 官方 `configs/default.py`，未改上游代码 |

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

**2026-11-14 诊断性旁路结果**（官方口径 424 对，`SANITY PASS`，10/10 项复现历史值；
**非模块消融，不能作为 V2 贡献度引用**）：

| 配置 | AEPE↓ | CMR@1↑ | CMR@2↑ | CMR@5↑ |
|---|---|---|---|---|
| GDROS 官方权重 | 1.4618 | 33.73 | 81.60 | 99.53 |
| M4 两路 `both` | 1.4738 | 72.41 | 84.91 | 94.10 |
| M4 单路 `v2off` | 7.7457 | 0.94 | 4.25 | 24.76 |

逐对 `ΔEPE` mean `+6.2719` / median `+6.0892`，仅 `15/424` 变好。
完整结论、能/不能成立的推论，以及「有效消融需重训单路」的方案与成本见
[`docs/ABLATION_V2.md`](../../docs/ABLATION_V2.md) §4；原始数据见
[`research/results/OS_20261114/`](../../research/results/OS_20261114/)。

## `sweep_gru_iters.py`：零训练 GRU 迭代次数 sweep

`iters` 不是结构层，而是**同一份 GRU 权重的循环展开次数**
（`e073a...:248`）。训练用 12、正式评测用 32。脚本只做一次 `iters=32` 前向，
对 `preds[k-1]` 各补一次 LSR + 官方 `Collector`，一次拿到
`[4,6,8,12,16,24,32]` 全曲线，不必逐 `k` 重跑模型。

```bash
CUDA_VISIBLE_DEVICES=2 python sweep_gru_iters.py --max-pairs 424 --gpu 0 \
  --out sweep_gru_iters_official.json
```

`k=32` 与历史官方值做 sanity check（2026-11-14 运行 `SANITY PASS`，451 s）。
结果：`24` 的 AEPE `1.4449` 反而优于 `32` 的 `1.4738`（68.2% 配对更好），
`16` 的 AEPE `1.4611` 且 CMR@2 持平；`12` 以下明显退化。
这只测**推理截断**，不能推断「训练 `train_iters` 砍到 5/6 重训后也行」；
完整边界与省训练时间的验证路径见
[`docs/GRU_ITER_SWEEP.md`](../../docs/GRU_ITER_SWEEP.md)。

## C 分支 teacher 剪枝：验证与计时

`E074Model(variant="C")` 的 `teacher_preds`（f2 + f3 两条冻结 GRU 链）只写
`last_teacher_avg`，而该属性只被 E074 自己的 `distill` 损失读取。E075A 的
`PairFusion.forward`、M4 手工训练循环、评测链都不读 ⇒ 纯死计算。
2026-11-15 已落地开关：`E074Model.forward(need_teacher_avg=True)`、
`PairFusion(c_skip_teachers=True)`；M4 训练循环里手工调用的 `model.c(...)`
也显式传 `need_teacher_avg=False`（**漏掉这处就只省推理、不省训练**）。
`c_skip_teachers=False` 可逐位恢复原始链。

```bash
CUDA_VISIBLE_DEVICES=5 python verify_c_teacher_pruning.py \
  --max-pairs 424 --ab-pairs 8 --gpu 0 --out verify_c_teacher_pruning.json
CUDA_VISIBLE_DEVICES=5 python probe_c_prune_timing.py \
  --pairs 30 --warmup 3 --gpu 0 --out probe_c_prune_timing.json
```

2026-11-15 运行结果（`SANITY PASS`）：8 对 × 32 个迭代点与 post-LSR 全部
`max|Δ|=0`；424 对 `AEPE=1.473827834446231`、CMR@1/2/5 `72.41/84.91/94.10`、
`RMSE_var=8.3350` 与 11-13 历史值 **Δ=0**；全量 `317.2 s`（剪枝前空闲 GPU
`422.4 s`）。同进程 30 对计时：推理 `0.9638 → 0.7197 s/对`（**1.339×**）、
训练前向 `0.6006 → 0.4128 s/对`（**1.455×**）。

⚠️ 计时只测前向，**不能**把 `0.741 s/step` / `16.5 h` 直接除以 1.455；
完整 step 还含 `LSR_list(fused)`、loss、反向、数据加载与 dev eval。
11-14 单次 probe 报的 `1.796×` 偏高，以 30 对均值为准。
细节见 [`docs/GRU_ITER_SWEEP.md`](../../docs/GRU_ITER_SWEEP.md) §6.1，
原始数据见 [`research/results/OS_20261115/`](../../research/results/OS_20261115/)。

## 对称单路旁路（2026-11-15，剪枝后 M4，双 GPU）

`ablate_backbone_single.py` 把 11-14 只有单方向的 `v2off` 补成对称的两个方向。
只改 E073 的两处融合门控，5 个 `E073FusionMatcher` 一起切换，不重训：

| 配置 | F3 `corr_gate_logit` | F2 `fuse_gate` |
|---|---|---|
| `v1only` | `-80`（`a≈0`，只留 GDROS 那路） | 常数 `-80`（`gs≈0`） |
| `v2only` | `+80`（`a≈1`，推向含 DINO 那路） | 常数 `+80`（`gs≈1`） |

`c_skip_teachers=True` 显式启用 2026-11-15 的剪枝，结果 JSON 里记录该标志；
`v1only` 与 11-14 历史 `v2off` 5 项指标 **Δ=0**，再次验证剪枝逐位等价。
每个进程先做 1 对 probe（`max|Δpost|` 分别为 46.95 / 143.67）自证门控生效。

```bash
CUDA_VISIBLE_DEVICES=5 python ablate_backbone_single.py \
  --config v1only --max-pairs 424 --gpu 0 --out ablate_v1only_pruned.json
CUDA_VISIBLE_DEVICES=1 python ablate_backbone_single.py \
  --config v2only --max-pairs 424 --gpu 0 --out ablate_v2only_pruned.json
python merge_backbone_ablation.py
```

结果（官方口径 424 对）：`both` AEPE `1.4738`，`v1only` `7.7457`，
`v2only` `41.3298`（CMR@5 仅 0.24%）。两个方向都大幅掉点，属
**敏感性诊断，不是模块消融**；尤其 `v2only` 里的 V2 不是独立 backbone
（DINO 描述子注入到 V1 特征上），只是把门控推向 V2 侧。完整结构边界、
逐对统计与推论限制见
[`docs/ABLATION_BACKBONE_SYMMETRIC.md`](../../docs/ABLATION_BACKBONE_SYMMETRIC.md)。

## 适配性短训（2026-11-15，方案 A）

`short_finetune_single_branch.py` 把零训练旁路升级成受控实验：两个配置
**只有门控锁在哪一侧不同**，其余（初始化、训练数据、seed、2000 步预算、
`lr=2e-5`、`train_iters=12`、4,401,475 个可训练参数、损失项）完全相同。
冻结所有 E072 前端（含冻结 DINO 注入）与门控参数；只解冻真正有梯度的下游：
L4 `gate`、B 的 `GateNet`/`scalar`、C 的 `student.context/update/fuse_proj`。

> **结构事实**：`E074Model(B).teacher_preds` 整体在 `torch.no_grad()` 下，
> B 的 f2/f3 递推链**拿不到梯度、无法续训**；C.student 是唯一可训练递推链。

```bash
CUDA_VISIBLE_DEVICES=5 python short_finetune_single_branch.py \
  --config v1only --steps 2000 --gpu 0 --out short_ft_v1only.json
CUDA_VISIBLE_DEVICES=7 python short_finetune_single_branch.py \
  --config v2only --steps 2000 --gpu 0 --out short_ft_v2only.json
python merge_short_finetune.py
```

结果（官方口径 424 对）：`v1only` `7.7457→1.7127`、`v2only`
`41.3298→3.3824`，分别恢复零训练 AEPE 落差的 **96.2% / 95.2%**，
两者的 >10px 灾难失败都降到 17 对（`both` 为 12 对）。
这定量证明零训练旁路的大幅掉点主要是下游 OOD；但**残余差距
（+0.24 / +1.91 AEPE）仍不能当作任一路的边际贡献**（预算只有 2000 步、
解冻范围受限、`v2only` 未稳定收敛）。完整边界见
[`docs/ABLATION_SHORT_FINETUNE.md`](../../docs/ABLATION_SHORT_FINETUNE.md)。

## UBCv2 跨数据集泛化评测（2026-11-15）

### 数据与 SAR 位深

官方 `UBCv2_test_images/multi-modal.../test_images.zip` 解压后是 **1625 对**
精配准 512×512 RGB/SAR。⚠️ 官方 SAR TIFF 是 **float32 后向散射幅度**
（约 0.05–13，稀疏离群到 130），直接 `cv2.imwrite` 写 PNG 会被静默截断成
`0..13`（几乎全黑）。`build_ubcv2_testset.py` 用**固定全局对数幅度 LUT**
（`20·log10(x+1e-3)`，从 300 张抽样得 p2 `-28.3079 dB` / p98 `10.5459 dB`）
统一映射到 uint8 再复制 3 通道；LUT 写入 manifest，所有配对一致、可复现。

### 构造口径

变换按论文 UBCv2：平移 `[-15,15]` px（1 px 步长）、缩放 `[0.9,1.1]`（0.05）、
旋转 `[-10°,10°]`（1°）；保持 512×512（不做训练的 400 center-crop，因为官方
`test.py` 在 512 上评测）。`seed=901`，**未做论文预筛**（规则未公开，
论文 1447 对 vs 本实验 1625 对）。

`truth_flow` 方向用 OS 官方 `datum/*.npy[2]` 反推确认：官方 flow 精确等于
`M@[x,y,1] − [x,y]`（3 对最小二乘残差 ~1e-6），M 是**绕原点**的
旋转+缩放+平移（不是 `getRotationMatrix2D` 的中心化形式）。生成脚本用同一形式，
`warped_sar = cv2.warpAffine(sar, M, ...)` 默认 inverse mapping，与 flow 自洽。

### 验证（`VERIFY PASS`）

`verify_ubcv2_testset.py`：flow vs 解析式 `max|Δ|=4e-6`、反拟合 M `max|Δ|=0`；
从原始 SAR TIFF 用同一 LUT 重算 warp `warp_max_delta=0`；`opt_sar_test`
可加载 1625 对，shape `(3,512,512)` / `(2,512,512)`。
**方向反证**：把 flow 取反后跑 GDROS 8 对，AEPE `21.84 → 81.69`（更差），
排除 GT 方向写反。

### 复现

```bash
cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
PY=/home/zzvr/miniconda3/envs/compared/bin/python
RAW=/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/raw/raw_extracted
OUT=/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/generated

$PY .codex_tmp/build_ubcv2_testset.py --raw-root $RAW --output-root $OUT \
  --seeds 901 --workers 8
$PY .codex_tmp/verify_ubcv2_testset.py --root $OUT/seed901 \
  --raw-root $RAW --check-pairs 3
CUDA_VISIBLE_DEVICES=5 $PY .codex_tmp/eval_ubcv2_official_protocol.py \
  --test-root $OUT/seed901 --which both --gpu 0 --max-pairs 1625 \
  --out .codex_tmp/ubcv2_official_20261115.json
```

`eval_ubcv2_official_protocol.py` 只覆盖 `P.TEST_ROOT`，其余（模型、权重、
`Collector`、`run_one`）逐字复用 OS 官方口径；表格打印做了 `AEPE=None`
容错（UBCv2 上 CMR@1 可能为 0）。

**边界（先读）**：`M4_step80000.pt` 与 `GDROS-os512.pth` 都只在 OS 上训练过，
直接跑 UBCv2 是**跨数据集泛化**，不能与论文 Table III 的 GDROS 行并列
（后者是 UBCv2 in-domain 训练权重，官方 released 权重里没有 UBCv2 版）。
论文还要求预筛后 1447 对、三个随机变换测试集报 mean±std；本实验是
**单 seed、单测试集、未预筛**，预筛规则与随机 seed 均未公开 ⇒ 即便重训
也只能做「同口径条件下的公平对比」，不能逐位复现 Table III。

## UBCv2 结果可比性诊断 + in-domain 训练（2026-11-15）

用户质疑「M4 跨数据集 `AEPE 3.8941` 竟然优于论文 GDROS UBCv2 in-domain `4.49`」。
三步拆解，结论见 [`docs/UBCV2_DIAGNOSIS.md`](../../docs/UBCV2_DIAGNOSIS.md)：

1. **筛选假设被反解否定**。论文三个 split 的保留率一致（87.7/85.5/89.0%），但用官方
   `@5px` 分档反解，把总 AEPE 压到 4.49 需要保留 7965 对 > 现有 1625，结构上不可能。
2. **逐对诊断**（`diag_ubcv2_pairs.py`）：`AEPE_overlap 21.84 ≈ AEPE_full 22.01`，
   误差与旋转幅度、缩放偏差、warp 损失面积的相关性全 ≈ 0，预测流仿射 `pred_scale`
   中位 0.9958、`pred_rot` 均值 0.08° ⇒ GDROS 在 UBCv2 上**基本输出恒等变换**。
3. **零流基线 29.7910**：GDROS 22.01 只比「什么都不预测」好 26.1%，M4 好 86.9%。

顺带确定了一个**新的不可比边界**：论文没写旋转/缩放基准点，绕原点（与官方 OS
`datum/*.npy[2]` 一致）零流基线 29.79，绕图像中心 16.30 —— **1.83×**。所以任何
绝对 AEPE/CMR 都不能与 Table III/V 并列。

为了把「域差异」和「几何难度」拆开，在 UBCv2 上按**官方 recipe** 训 GDROS：

```bash
# ⑫ 解压 + 生成 train(4008)/val(1681)，复用测试集 SAR LUT
bash .codex_tmp/prepare_ubcv2_train_data.sh

# 官方 recipe：120K 步 / iters=12 / bs=2 / 400² / OneCycleLR / RAFT-Sintel 初始化
NAME=gdros-ubcv2-512 GPU=5 bash .codex_tmp/train_gdros_ubcv2.sh
```

两个上游硬编码用软链满足、**不改官方代码**：`train.py` 的 `restore_ckpt` 只当布尔开关
（实际 `torch.load('./checkpoints/raft-sintel.pth')`，`cfg.restore_ckpt` 从未被加载）；
数据根写死在 `opt_sar_train(root='./datasets/os_dataset')`。`LSR_list` 是无参数最小二乘，
故 `LSRnet.pth` 与训练无关。实测 `0.214 s/step`，120K 步约 7 h。

**仍未闭合**：GDROS in-domain 也只是「UBCv2 训 vs UBCv2 测」，而 M4 是「OS 训 vs
UBCv2 测」。严格同条件表还需在 UBCv2 重训 M4 整链（≈16.5 h）。

## 未入库的探针脚本

10–11 月还有大量一次性诊断脚本（`m1_*` / `m2_*` / `m3_*` / `g*` / `cov_*` /
`tail_finetune.py` 等）用于探查尺度、等变性、鲁棒拟合与尾项续训，
它们是被否定的探索路径，不属于当前方法，因此不入库。其**结论**已提炼进
[`docs/METHOD_DESIGN.md`](../../docs/METHOD_DESIGN.md) §5。
