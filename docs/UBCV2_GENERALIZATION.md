# UBCv2 跨数据集泛化评测（2026-11-15）

> **定位（先读）**：这是**跨数据集泛化**测试，不是论文 Table III 的复现。
> `E075A-M4@80K` 与 `GDROS-os512.pth` 都只在 **OS 数据集**上训练过，这里
> 直接拿同一份权重在 UBCv2 上推理，比较的是**两个方法迁移到未见数据集的
> 相对表现**，两者用完全相同的权重口径、数据、评测协议，因此横向对比公平。
> 但结果**不能与论文 Table III 的 GDROS 行并列**（那是 UBCv2 in-domain
> 训练权重；官方 released 权重里没有 UBCv2 版本）。

## 1. 数据集与预处理

UBCv2（Urban Building Classification V2，2023 IEEE GRSS DFC）是超高分辨率
（0.5 m）光学/SAR 建筑数据集。GDROS 论文把它作为第三个评测集，并指出它
模态差异更大、噪声与云遮挡更严重。本次使用官方
`multi-modal_fine-grained_building_roof_instance_segmentation/test_images.zip`
解压后的 **1625 对**精配准 512×512 RGB/SAR 影像。

⚠️ **关键预处理**：官方 SAR TIFF 是 **float32 后向散射幅度**（约 0.05–13，
稀疏离群到 130），而 GDROS/M4 都在 8-bit SAR 上训练。直接 `cv2.imwrite`
写 PNG 会被静默 fallback 到 `CV_8U`，把数值截断成 `0..13`（几乎全黑）。
因此用**固定全局对数幅度 LUT** 统一转换：

```
db = 20·log10(x + 1e-3)
lo, hi = percentile(db, [2, 98])   # 从 300 张抽样 tile 估计，seed=901
u8     = clip((db - lo)/(hi - lo)·255, 0, 255)
sar    = repeat(u8, 3 channels)    # 实测 lo=-28.3079 dB, hi=10.5459 dB
```

LUT 写进 `protocol_metadata/manifest.json`，所有配对共用同一套辐射映射，
不引入逐图自适应。论文的 SAR 8-bit 转换方式未公开，这是本实验与论文
不可逐位对齐的原因之一。

## 2. 测试集构造

`research/eval/build_ubcv2_testset.py` 输出 GDROS 官方布局
（`testing/opt|sar_warped|truth_flow|datum`），变换参数按论文 UBCv2：

| 项 | 值 |
|---|---|
| 平移 | `[-15, 15]` px，1 px 步长（整数采样） |
| 缩放 | `[0.9, 1.1]`，0.05 步长 |
| 旋转 | `[-10°, 10°]`，1° 步长 |
| 裁剪 | 无，保持 512×512（训练用的 400 center-crop 不适用，`test.py` 在 512 评测） |
| 边界 | `cv2.BORDER_CONSTANT 0` |
| seed | `901`（单测试集） |
| 预筛 | **无**，保留全部 1625 对（论文预筛后 1447 对，规则未公开） |

### 2.1 flow 方向约定（已用官方数据反推）

读 OS 官方 `datum/000000.npy`，其第 3 个元素存的就是仿射矩阵
`M = [[s·cos, −s·sin, tx], [s·sin, s·cos, ty]]`（**绕原点**旋转+缩放+平移，
不是 `cv2.getRotationMatrix2D` 的中心化形式）。对 3 对数据做最小二乘：
官方 `truth_flow` 精确等于 `M·[x,y,1] − [x,y]`，残差约 `1e-6`。
生成脚本用同一形式，`warped_sar = cv2.warpAffine(sar, M, ...)`（默认
inverse mapping，即 `warped_sar(M·p) = sar(p)`），与 flow 完全自洽。

## 3. 正确性验证

`research/eval/verify_ubcv2_testset.py` 在 1625 对全量集上跑前 3 对，
`VERIFY PASS`：

- flow vs 解析式 `max|Δ| = 4e-6`（float32 舍入）；从 flow 反拟合 M `max|Δ| = 0`；
- 从原始 SAR TIFF 用同一 LUT 重算 warp，与生成的 `sar_warped` 逐像素
  `warp_max_delta = 0`；
- `core.datasets.opt_sar_test` 可加载 1625 对，图像 `(3,512,512)`、
  GT `(2,512,512)`、`valid_frac=1.0`。

**方向反证**：把 8 对样本的 `truth_flow` 整体取反后重跑 GDROS，
AEPE 从 `21.8368` 恶化到 `81.6948`（CMR@5 归零）。方向若写反，取反后
应变好；实测显著变差 ⇒ GT 方向正确。原始 sanity 结果见
`research/results/UBCV2_20261115/ubcv2_sanity8*.json`。

## 4. 评测配置

| 项 | 值 |
|---|---|
| 模型 | 剪枝后 `E075A-M4@80K`（`M4_step80000.pt`，`c_skip_teachers=True`）与 GDROS released `GDROS-os512.pth` |
| 协议 | GDROS 官方 `test.py`：逐分量 `\|Δx\|,\|Δy\|` → 配对级 `EPE(k)` → `AEPE = mean(MAE_list)` |
| 官方栏位 `RMSE` | 实为 `np.var(MAE_list)`，记作 `RMSE_var` |
| `iters` / LSR | `32` / `NPOINT=200000`（固定连续切片，确定性） |
| GPU | zzvr 物理 GPU 5（空闲），`CUDA_VISIBLE_DEVICES=5 --gpu 0` |

## 5. 结果

**全部 1625 对、单 seed `901`、官方 `test.py` 口径**，`SANITY` 与方向反证见 §3。

| 模型 | AEPE↓ | CMR@1↑ | CMR@2↑ | CMR@3↑ | CMR@4↑ | CMR@5↑ | RMSE_var | >5px 对 | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| GDROS released `GDROS-os512.pth` | 22.0091 | 0.06 | 0.74 | 2.83 | 6.83 | 11.26 | 679.0462 | 1442/1625 | 131.5 s |
| **E075A-M4@80K（剪枝版）** | **3.8941** | **46.46** | **62.95** | **68.68** | **71.51** | **74.52** | 38.7571 | 414/1625 | 1286.4 s |

同口径的 OS 基准值（424 对）是 GDROS `1.4618` / M4 `1.4738`。换到 UBCv2 后：

- GDROS 退化到 **15.06×**，且 CMR@1 从 33.73% 掉到 0.06%、CMR@5 从
  99.53% 掉到 11.26% —— 不是精度下降，是**近乎整体失效**；
- M4 退化到 **2.64×**，仍保有 46.46% 的 <1px 正确率和 74.52% 的 <5px 正确率；
- 两者在同一份 UBCv2 数据上 **AEPE 相差 5.65×**，CMR@5 相差 **63.3 pp**。
  由于权重口径、数据、评测协议完全相同，这个差距是方法层面的泛化差异，
  不是评测口径造成的。

**误差结构**：M4 的 `AEPE@5 = 1.0968` 但整体 `AEPE = 3.8941`，说明
74.5% 的配对已经很准（平均约 1.1 px），剩下的 **414 对 >5px**（25.5%）
贡献了几乎全部误差，训练日志里可见单对 50 px 级的灾难失败。这与 OS 上
「尾部来自配对级灾难失败」的既有结论一致，只是 UBCv2 上尾部占比从约
10% 升到约 25%。GDROS 的 `AEPE@5 = 3.6080` 且只有 11.26% 的配对能进
<5px，属于整体失配。

交叉验证：内部向量 EPE 参考值 `35.0028 / 6.1157`，与官方逐分量 AEPE
的比值约 π/2，与 [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) 的
口径关系一致，说明本次评测没有把两套误差定义混用。

> ⚠️ 8 对 sanity 曾给出 M4 `1.4621`、GDROS `21.8368`，前 8 对恰好都是
> 易配准样本；全量 M4 是 `3.8941`。**不要引用 8 对数字**，它只用于
> 管线自检与方向反证。

## 6. 能成立 / 不能成立的推论

- ✅ 可以说：在**同一份 OS 训练权重、同一套官方口径、同一批 UBCv2 变换
  样本**下，两个方法的跨数据集泛化表现差距。这是两个方法之间的公平对照。
- ✅ 可以说：UBCv2 的 SAR 位深/辐射分布与 OS 不同，需要显式 8-bit 映射；
  这一步没做会让任何方法的评测失去意义（已写入生成脚本）。
- ❌ **不能**与论文 Table III 的 GDROS 数字（CMR@2 14.89 / @3 37.08 /
  @5 72.50）并列或说「超过/不如 SOTA」：那是 in-domain 训练 + 论文自造
  测试集 + 三个随机变换集 mean±std，本实验是单 seed、未预筛、泛化推理。
- ❌ 不能把本结果当作「UBCv2 上的绝对性能」：单 seed 的随机仿射抽样、
  未公开的预筛规则、以及我们自定义的 SAR 8-bit LUT 都会改变绝对数值。
- ❌ 不能反推 M4 在 UBCv2 上「已解决超高分辨率跨模态配准」；要回答这个
  需要在 UBCv2 train split 上重训并与同样重训的 GDROS 对比。

## 7. 复现

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

脚本索引与更多实现细节见 [`research/eval/README.md`](../research/eval/README.md)，
原始结果见 [`research/results/UBCV2_20261115/`](../research/results/UBCV2_20261115/)。
