# OS 数据集复现结果（baseline + 最好方法 E075A-M4）

> ⚠️ **2026-11-13 更正（详见 `GDROS复现口径核查_20261113.md`）**：下文 §2 关于
> 「2.298 与 1.461 是真实差距，不是评测脚本 bug / 极可能是论文数字而非本机实测」的判断
> **作废**。官方 `test.py`（仅修 `--Npoint` 的 list bug）用同一个 `GDROS-os512.pth`、
> 同一份 `gdros_official_e003a_testing`、424 对，实测 `AEPE 1.461846 / CMR@1 33.73 /
> CMR@2 81.60 / CMR@5 99.53`，与 `REFS` 及论文 Table II（`33.88±0.73%`）一致。
> 差距 100% 来自误差度量：官方 `test.py` 的 `sum(dim=0).sqrt()` 算的是**逐分量绝对误差**，
> 本仓库脚本用**向量 EPE**（比值 1.588 ≈ π/2）。权重、数据集、LSR、iters 均无问题。
> §2 以下内容保留作历史记录，结论不再成立。

> 机器：zzvr（8× RTX 5090）｜conda env：`compared`（torch 2.11.0+cu128，支持 sm_120）
> 数据：`/mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official_e003a_testing`（OS 测试集，424 对，512×512）
> 评测脚本：`image_matching_part2_migrated/.codex_tmp/eval_os_unified.py`（统一 per-pixel 口径）
> 评测对：前 404 对（与 E075A `dev_eval_pairs=404` 一致）

## 1. 结论速览（per-pixel 口径，与 E075A `eval_model` 一致）

| 方法 | AEPE | CMR@1 | CMR@2 | CMR@5 | CMR@10 | CMR@16 |
|---|---|---|---|---|---|---|
| **GDROS released（`GDROS-os512.pth`，本次复现）** | 2.298 | 18.58 | 53.21 | 93.51 | 99.44 | 99.99 |
| E074-B（仓库 REFS 记录值） | 2.505 | 44.80 | 72.77 | 89.36 | 95.30 | 97.52 |
| E074-C（仓库 REFS 记录值） | 2.570 | 48.76 | 76.98 | 87.87 | 93.56 | 96.78 |
| **E075A-M4 step80K（本次复现）** | **2.402** | **54.94** | **77.29** | **89.22** | **94.32** | **97.25** |

**关键判断（同一评测协议下，baseline 与 M4 完全可比）：**
- **E075A-M4 相对仓库里两个冻结分支（E074-B/C）全面占优**：相对 E074-C，CMR@1 +6.2pp、CMR@2 +0.3pp、CMR@5 +1.35pp、AEPE −0.17；相对 E074-B，CMR@1 +10.1pp、CMR@2 +4.5pp、AEPE −0.10（仅 CMR@5/10/16 松阈值低 <1pp）。→ 区域门控（region gate）确实在保 CMR@1 的同时压低了 AEPE，是有效的改进方向。
- **相对 GDROS released baseline**：M4 把 CMR@1 从 18.58 拉到 54.94（+36.4pp）、CMR@2 从 53.21 到 77.29（+24.1pp），代价是 AEPE 略升（2.298→2.402）以及 CMR@5/10/16 这类“饱和级”松阈值回落（精度-召回权衡）。这说明 M4 的核心收益在**紧阈值精度**，符合 E075A 实验目标（“保 CMR@1 同时回收 CMR@2/CMR@5、压低 AEPE”）。

## 2. ⚠️ 关于 baseline 口径差异（重要）

仓库 `REFS["gdros_released"]` 记录的是 **AEPE 1.461 / CMR@1 33.73 / CMR@2 81.60 / CMR@5 99.53**，但用本机 `GDROS-os512.pth` 在 `gdros_official_e003a_testing` 上实跑得到的是 **AEPE 2.298 / CMR@1 18.58 / CMR@2 53.21 / CMR@5 93.51**（per-pixel）。

**这是真实差距，不是评测脚本 bug**（已交叉验证）：
- 我的 `eval_os_unified.py` 与官方 `test.py` 的 LSR 后处理、`iters=32`、`Npoint=200000` 完全一致（官方 `test.py` 本身有 bug：`--Npoint` 用 `nargs='+'` 变成 list，`10000 + Npoint` 直接 `TypeError`；我修成整数 200000 才是其本意）。
- 两个 checkpoint 都跑在同一套脚本、同一数据集、同一 per-pixel 指标上，所以 **M4 vs baseline 的相对结论不受该差距影响**。
- 1.461 极可能是 **GDROS 论文公布的 OS 数字**，而非本机 `GDROS-os512.pth` 在该提取测试集上的实测值（released checkpoint 在该测试集上表现弱于论文公布值）。若你要对外展示“baseline 1.461”，需要确认是否换用论文 checkpoint 或另一个测试集版本；否则应以本次复现的 2.298 为准。

## 3. 复现时的环境修复（已在 zzvr 上落地）

为了让代码在本机跑通，做了以下非侵入式修复（没有改业务代码逻辑，只补依赖/路径）：
1. **装 `timm` 进 `compared` 环境**：`/home/zzvr/miniconda3/envs/compared/bin/pip install timm`（装的是 1.0.30，仅新增 `certifi/safetensors`，**未动 torch/CUDA**）。原因：GDROS 前端 `encoder.py` 用了 `from timm.layers import ...`（timm≥0.6 API），原机只有 `xoftr` 里的 timm 0.4.12（只有 `timm.models.layers`，且无 sm_120 支持）。
2. **`timm.layers` shim**：在 `eval_os_unified.py` 顶部注入 `sys.modules['timm.layers'] = timm.models.layers`，兼容老 API，不改外部仓库文件。
3. **E038B 路径**：`_load_e038b_module()` 按 `Path.cwd()` 拼候选路径，必须**从 `image_matching_part2_migrated/` 目录启动脚本**（不能 cd 进 `.codex_tmp/`），否则找不到 `.codex_tmp/e038b_projection_sweep/run_projection_sweep.py`。这是构建 E074→E073→E072→E046→E044A 教师栈的必需文件。
4. **设备对齐**：`LSR`/E075A `eval_model` 的 LSR 输出在 CPU，ground-truth 也在 CPU，预测流在 CUDA——已在脚本中统一到 CPU 做 EPE 比较（与 `eval_model` 原逻辑一致）。
5. **DataParallel 加载**：`GDROS-os512.pth` 是 `DataParallel` 保存的（`module.` 前缀），按官方 `test.py` 方式用 `nn.DataParallel` 包一层再 `load_state_dict`，避免 missing-keys。

## 4. 复现命令（可重跑）

```bash
# baseline（GDROS released）
cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
CUDA_VISIBLE_DEVICES=6 /home/zzvr/miniconda3/envs/compared/bin/python \
  .codex_tmp/eval_os_unified.py --which baseline --max-pairs 404 --gpu 0

# E075A-M4 step80K（冻结 E074 B/C 分支 + 区域门控，加载 M4_step80000.pt）
cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
CUDA_VISIBLE_DEVICES=6 /home/zzvr/miniconda3/envs/compared/bin/python \
  .codex_tmp/eval_os_unified.py --which m4 --max-pairs 404 --gpu 0
```

## 5. 原始日志（证据）

- `baseline_run.log`、`m4_run.log`（已落到本机 `VSmatch/` 目录）
- E075A-M4 加载干净：`missing=[] unexpected=[]`（M4_step80000.pt 与 PairFusion 架构完全对齐）
