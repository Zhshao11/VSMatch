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

# ④ 零训练消融：旁路 V2（DINO 辅助）那一路，同一权重下对比两路 / 单路
CUDA_VISIBLE_DEVICES=2 $PY .codex_tmp/ablate_v2_off.py \
  --configs both,v2off --max-pairs 404 \
  --out .codex_tmp/ablate_v2_off_result.json
```

`CUDA_VISIBLE_DEVICES=2` 把物理 GPU 映射为进程内 `cuda:0`，因此脚本 `--gpu 0`。
耗时参考（424 对）：GDROS 70–87 s，M4 554–787 s；Top-25 诊断约 742 s。
M4 checkpoint 约 6.4 GB，权重加载应报 `missing=[] unexpected=[]`。

## 5. 已知限制与坑

- **必须从 `image_matching_part2_migrated/` 启动**：E038B 相关路径解析依赖 cwd。
- `LSR` 内部有硬编码的 `.cuda()`（`LSmodel.py:138`），因此**纯 CPU 不能跑完整推理链**；
  `ablate_v2_off.py --device cpu` 会自动把 `e074.lsr` 换成恒等映射，仅用于链路自检，
  该模式下指标无效。
- 非 0 物理 GPU 上运行 GDROS 的 `DataParallel` 时，需要
  `DataParallel(..., device_ids=[gpu])` + `torch.cuda.set_device(device)`，
  否则 LSR 内部 grid 落在 `cuda:0` 会导致设备不一致。
- 官方 `test.py` 的 `--Npoint` 使用 `nargs='+'`，解析为 list，`10000 + Npoint` 会
  `TypeError`；复现官方数字时需把该值转成 int（唯一改动）。
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
