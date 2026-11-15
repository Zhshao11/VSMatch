# 复现指南

## 1. 仓库里有什么、没有什么

**有**：E072–E075A 的模型与训练实现、评测脚本、口径定义、结果报告、原始指标 JSON。

**没有**（按设计不入库）：

- GDROS 上游源码（`external_repos/E068A_GDROS_official`）
- OS 数据集（`datasets/extracted/gdros_official_e003a_testing` 等）
- 任何 checkpoint（`.pt` / `.pth`，含 `M4_step80000.pt`、`GDROS-os512.pth`）
- 预测张量、运行日志、生成的图表与密集诊断数据

因此**克隆仓库不能直接跑推理**，需要先准备部署工作区（下节）。但方法可以完整读懂：
从 [METHOD_DESIGN.md](METHOD_DESIGN.md) 开始，指标见 [RESULTS_OS.md](RESULTS_OS.md)。

## 2. 部署工作区约定

历史实验在 zzvr 上完成，脚本里的路径常量沿用该布局：

```
/mnt/sda/zhushihao/VS_Match/                        # ROOT / SPARSEGEO_ROOT
├── image_matching_part2_migrated/.codex_tmp/      # TMP：E072–E075A 模块与评测脚本
├── external_repos/E068A_GDROS_official/           # OFFICIAL：上游 GDROS 代码
├── datasets/extracted/gdros_official_e003a_testing/
├── GDROS/official_models_reference/GDROS-os512.pth
└── GDROS/experiments/sparsegeo/runs/E-075A-.../checkpoints/M4_step80000.pt
```

换到自己的工作区时，需要改各脚本顶部的 `ROOT` / `TMP` / `OFFICIAL` / `TEST_ROOT` /
`M4_CKPT` 常量（`research/eval/` 下脚本集中列在
[README.md](../research/eval/README.md)）。训练脚本通过环境变量 `SPARSEGEO_ROOT`
指定根目录，默认值仍是 zzvr 布局，以保留冻结实验的原始行为。

## 3. 环境

| 项 | 值 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.11.0+cu128（历史运行）；CUDA GPU |
| 其它 | `timm`、`numpy`；见 `requirements-research.txt` |

`timm` 兼容 shim：vendored GDROS `encoder.py` 导入 `timm.layers`（新版 API），
旧版 `timm` 只有 `timm.models.layers`。评测脚本顶部已做兼容处理：

```python
import timm
if not hasattr(timm, "layers"):
    import timm.models.layers as _l
    sys.modules.setdefault("timm.layers", _l)
```

## 4. 复现命令

以下命令都在 `image_matching_part2_migrated/` 目录下执行 —— 部分路径解析依赖 cwd。

```bash
PY=/home/zzvr/miniconda3/envs/compared/bin/python

# ① 向量 EPE 全指标（424 对；含配对级与逐像素两套口径）
CUDA_VISIBLE_DEVICES=2 $PY .codex_tmp/eval_os_full_metrics.py \
  --which both --max-pairs 424 --gpu 0 \
  --out .codex_tmp/os_full_metrics_20261113.json

# ② 官方 test.py 口径（论文 Table II 风格）
CUDA_VISIBLE_DEVICES=2 $PY .codex_tmp/eval_os_official_protocol.py \
  --which both --max-pairs 424 --gpu 0 \
  --out .codex_tmp/os_official_protocol_20261113.json

# ③ Top-25 失败配对归因
CUDA_VISIBLE_DEVICES=2 $PY .codex_tmp/diag_m4_top25.py \
  --max-pairs 424 --gpu 0 --top 25 \
  --out .codex_tmp/m4_top25_diag_20261113.json

# ④ 零训练消融：旁路 V2（DINO 辅助）那一路，官方 test.py 口径
#    同一次运行附带 GDROS 官方权重参照行 + 2026-11-13 历史值 sanity check
CUDA_VISIBLE_DEVICES=2 $PY .codex_tmp/ablate_v2_off.py \
  --configs both,v2off --max-pairs 424 --gpu 0 \
  --out .codex_tmp/ablate_v2_off_result_official.json

# ④-a 只验证链路/门控是否生效（1 对、iters=1，指标无效）
CUDA_VISIBLE_DEVICES=2 $PY .codex_tmp/ablate_v2_off.py \
  --configs both,v2off --max-pairs 1 --iters 1 --probe

# ⑤ 零训练 GRU 迭代次数 sweep：一次 32-iteration 前向，拿全 [4,6,8,12,16,24,32] 曲线
CUDA_VISIBLE_DEVICES=2 $PY .codex_tmp/sweep_gru_iters.py \
  --max-pairs 424 --gpu 0 \
  --out .codex_tmp/sweep_gru_iters_official.json

# ⑥ C 分支 teacher 剪枝验证：同进程 A/B bit-exact + 424 对官方口径
CUDA_VISIBLE_DEVICES=5 $PY .codex_tmp/verify_c_teacher_pruning.py \
  --max-pairs 424 --ab-pairs 8 --gpu 0 \
  --out .codex_tmp/verify_c_teacher_pruning.json

# ⑦ 同进程逐对计时：推理 iters=32 / 训练前向 iters=12
CUDA_VISIBLE_DEVICES=5 $PY .codex_tmp/probe_c_prune_timing.py \
  --pairs 30 --warmup 3 --gpu 0 \
  --out .codex_tmp/probe_c_prune_timing.json

# ⑧/⑨ 对称单路旁路（剪枝后 M4）：两张不同 GPU 并行，一个进程一个配置
CUDA_VISIBLE_DEVICES=5 $PY .codex_tmp/ablate_backbone_single.py \
  --config v1only --max-pairs 424 --gpu 0 \
  --out .codex_tmp/ablate_v1only_pruned.json
CUDA_VISIBLE_DEVICES=1 $PY .codex_tmp/ablate_backbone_single.py \
  --config v2only --max-pairs 424 --gpu 0 \
  --out .codex_tmp/ablate_v2only_pruned.json

# 本地合并两份结果（纯标准库，不需要 GPU）
python research/eval/merge_backbone_ablation.py

# ⑩ 适配性短训（方案 A）：两张 GPU 并行，2000 步 + 同进程官方口径评测
CUDA_VISIBLE_DEVICES=5 $PY .codex_tmp/short_finetune_single_branch.py \
  --config v1only --steps 2000 --gpu 0 \
  --save-delta .codex_tmp/short_ft_v1only_delta.pt \
  --out .codex_tmp/short_ft_v1only.json
CUDA_VISIBLE_DEVICES=7 $PY .codex_tmp/short_finetune_single_branch.py \
  --config v2only --steps 2000 --gpu 0 \
  --save-delta .codex_tmp/short_ft_v2only_delta.pt \
  --out .codex_tmp/short_ft_v2only.json
python research/eval/merge_short_finetune.py

# ⑪ UBCv2 跨数据集泛化：构造测试集 → 验证 → 官方口径评测（1625 对）
RAW=/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/raw/raw_extracted
OUT=/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/generated
$PY .codex_tmp/build_ubcv2_testset.py --raw-root $RAW --output-root $OUT \
  --seeds 901 --workers 8
$PY .codex_tmp/verify_ubcv2_testset.py --root $OUT/seed901 \
  --raw-root $RAW --check-pairs 3
CUDA_VISIBLE_DEVICES=5 $PY .codex_tmp/eval_ubcv2_official_protocol.py \
  --test-root $OUT/seed901 --which both --gpu 0 --max-pairs 1625 \
  --out .codex_tmp/ubcv2_official_20261115.json
python research/eval/summarize_ubcv2.py \
  research/results/UBCV2_20261115/ubcv2_official_20261115.json

# ⑫ UBCv2 逐对诊断（GDROS only，1625 对，约 180 s）
CUDA_VISIBLE_DEVICES=3 $PY .codex_tmp/diag_ubcv2_pairs.py \
  --which baseline --gpu 0 --max-pairs 1625 \
  --out .codex_tmp/diag_ubcv2_pairs_20261115.json

# ⑬ UBCv2 in-domain 训练 GDROS：解压 → 生成 train/val → 官方 recipe
bash .codex_tmp/prepare_ubcv2_train_data.sh
NAME=gdros-ubcv2-512 GPU=5 bash .codex_tmp/train_gdros_ubcv2.sh
```

`CUDA_VISIBLE_DEVICES=2` 把物理 GPU 映射为进程内 `cuda:0`，因此脚本 `--gpu 0`。
耗时参考（424 对）：GDROS 59–87 s；M4 **剪枝前** 422–787 s（空闲 GPU 约 422 s），
**2026-11-15 剪枝后 317 s**；Top-25 诊断约 742 s（剪枝前）。
消融脚本共跑 GDROS / both / v2off 三路，约 25–35 min；⑤ 一次前向 + 候选点各补
一次 LSR，实测 **451 s**；checkpoint 加载本身约 30–170 s（受页缓存与共享机 I/O 影响）。
M4 checkpoint 约 6.4 GB，权重加载应报 `missing=[] unexpected=[]`。

> **口径**：④ 与 ② 完全相同，都是 GDROS 官方 `test.py` 语义（逐分量 `|Δx|,|Δy|`
> + 配对级聚合 + 424 对），三者结果可直接并列。① 是项目内部向量 EPE 口径，
> **不能**与 ②/④ 混用；差异见 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)。
> 消融脚本会打印 `SANITY PASS/FAIL`：只有 GDROS 与 both 复现 2026-11-13 官方数值，
> `v2off` 的结论才成立。⑤ 的 `k=32` 行同样做 sanity check。
>
> ⚠️ **⑤ 扫的是推理 `iters`**：只有 `k=32` 与 ②/④ 同口径、可与论文 Table II 并列；
> 其他 `k` 是内部筛选用的敏感性曲线。改 `iters` 会改变与 GDROS baseline 的可比性，
> 详见 [GRU_ITER_SWEEP.md](GRU_ITER_SWEEP.md) §5。
>
> ⚠️ **⑥/⑦ 是工程剪枝验证，不是新口径**：⑥ 的 `SANITY PASS` 要求剪枝后 424 对
> 指标与 11-13 历史值**逐项 Δ=0**；⑦ 只计前向时间，不能当完整训练 step 的加速比。
> 详见 [GRU_ITER_SWEEP.md](GRU_ITER_SWEEP.md) §6.1。
>
> ⚠️ **⑧/⑨ 是诊断性旁路，不是模块消融**：`v2only` 里的 V2 不是独立 backbone
> （冻结 DINO 描述子只是注入到 V1 特征上），它表示「把两处门控推向 V2 侧」。
> 两个方向的大幅掉点**不能**读作任一路的边际贡献；结构边界与推论限制见
> [ABLATION_BACKBONE_SYMMETRIC.md](ABLATION_BACKBONE_SYMMETRIC.md) §1.1 / §4。
>
> ⚠️ **⑩ 是受控短训，不是同预算严格重训消融**：两个配置只变门控锁在哪一侧，
> 预算 / 数据 / seed / 可训练参数完全相同；但只有 2000 步（保护版本 80K 步），
> 且 B 的 `teacher_preds` 在 `no_grad` 下、只有 C.student 一条递推链可训练。
> 短训后的残余差距**不能**当作任一路的边际贡献；边界见
> [ABLATION_SHORT_FINETUNE.md](ABLATION_SHORT_FINETUNE.md) §4。
>
> ⚠️ **⑪ 是跨数据集泛化，不是 Table III 复现**：两个 checkpoint 都只在 OS 上
> 训练过，UBCv2 上只是同权重推理；且本实验单 seed、未预筛 1625 对，
> 论文是预筛 1447 对、三个随机变换测试集 mean±std。绝对数值不可与论文并列；
> 边界与 SAR 位深处理见 [UBCV2_GENERALIZATION.md](UBCV2_GENERALIZATION.md)。
>
> ⚠️ **⑫ 只改「怎么读 ⑪」，不改 ⑪ 的数**：⑫ 逐对记录官方 EPE、只算 warp 后有效
> 重叠区的 EPE、以及预测流的最小二乘仿射。它证明 GDROS 在 UBCv2 上的误差与旋转
> 幅度 / 缩放偏差 / warp 损失面积**都不相关**，且预测流近似恒等变换；配合由
> `datum/*.json` 解析算出的**零流基线 29.7910**，可知 GDROS `22.0091` 只比
> 「什么都不预测」好 26.1%。再叠加「旋转基准点未公开」带来的 **1.83×** 自由度
> （绕原点 29.79 vs 绕中心 16.30），结论是**绝对数一律不可跨实现比较**。
> 详见 [UBCV2_DIAGNOSIS.md](UBCV2_DIAGNOSIS.md)。
>
> ⚠️ **⑬ 给出的是 GDROS「UBCv2 训 vs UBCv2 测」**，而 M4 是「OS 训 vs UBCv2 测」，
> 两者仍不同条件；严格同条件表还需在 UBCv2 重训 M4 整链（≈16.5 h）。官方 recipe
> 的 120K 步实测 `0.214 s/step`（约 7 h），每 20K 步存档。上游两处硬编码
> （`./checkpoints/raft-sintel.pth`、`./datasets/os_dataset`）由脚本用软链满足，
> **不修改官方仓库代码**。

## 5. 已知限制与坑

- **2026-11-15 起 M4 默认剪掉 C 分支 `teacher_preds`**（`PairFusion(c_skip_teachers=True)`）：
  指标逐位不变，GRU 从 5 条链 / 160 次 update 降到 3 条 / 96 次，129 次 LSR 不变。
  复现剪枝前的耗时 / 调用链要传 `c_skip_teachers=False`；E074 variant C 的蒸馏训练
  仍走 `need_teacher_avg=True` 默认路径，不受影响。
- **UBCv2 的 SAR TIFF 是 float32 幅度图**（约 0.05–13，离群到 130），不是 8-bit
  影像。直接 `cv2.imwrite` 到 PNG 会被静默截断成 `0..13`（几乎全黑），评测无意义。
  `build_ubcv2_testset.py` 用固定全局对数幅度 LUT（p2/p98 dB）统一映射到 uint8，
  LUT 参数写在 `protocol_metadata/manifest.json`；换数据集/重跑务必用同一 LUT。
- **必须从 `image_matching_part2_migrated/` 启动**：E038B 相关路径解析依赖 cwd。
- `LSR` 内部有硬编码的 `.cuda()`（`LSmodel.py:138`），因此**纯 CPU 不能跑完整推理链**；
  `ablate_v2_off.py --device cpu` 会自动把 `e074.lsr` 换成恒等映射，仅用于链路自检，
  该模式下指标无效。
- 非 0 物理 GPU 上运行 GDROS 的 `DataParallel` 时，需要
  `DataParallel(..., device_ids=[gpu])` + `torch.cuda.set_device(device)`，
  否则 LSR 内部 grid 落在 `cuda:0` 会导致设备不一致。
- 官方 `test.py` 的 `--Npoint` 使用 `nargs='+'`，解析为 list，`10000 + Npoint` 会
  `TypeError`；复现官方数字时需把该值转成 int（唯一改动）。**直接不传该参数最省事**，
  默认值就是 200000。
- 上游 `train.py` / `test.py` 依赖 `tensorboard`（`from torch.utils.tensorboard import
  SummaryWriter`），最小环境里要单独 `pip install tensorboard`，否则 import 阶段就崩。
- **官方没有发布 UBCv2 权重**（README 只列 WHU-OPT-SAR 与 OS；GitHub Releases 为空；
  数据下载也不给 UBCv2 筛后 split）⇒ 论文 Table III 的 UBCv2 行没有可用的公开权重
  直接验证，只能自己按代码 recipe 训。
- **论文正文与发布代码的超参不一致**：正文写 `lr=1.2e-5 / bs=12`，
  `configs/default.py` 是 `canonical_lr=12.5e-5 / batch_size=2`，且
  `core/optimizer/__init__.py` 不按 batch size 缩放 LR。照代码跑 = 6× 台阶更少样本、
  10.4× 更高 LR。复现时**照代码**，并在对比表里注明这一差异。
- 共享机器上 checkpoint 加载受磁盘 I/O 与页缓存影响很大，冷盘且高负载时启动可能
  显著变慢；判活应看日志是否打印 `missing=/unexpected=`，而不是看进程是否存在。
- 训练脚本默认 `steps=120000`，保护版本取 **step 80000**；不要误用 `120000`
  的 checkpoint 与本文指标对照。

## 6. 静态检查（不需要数据集和 GPU）

```bash
python -m pip install -r requirements-research.txt
python -m compileall -q tools research/remote_stage research/eval e081a_whu_our_native_train.py
python -m unittest discover -s tools -p 'test_*.py'
python scripts/preflight.py --scope all
```

`research/experiments/` 下的测试另外需要上游 GDROS 源码与各自 run manifest 描述的 fixture。
