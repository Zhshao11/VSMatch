# 评测口径与 GDROS 复现核查

> 本文是**报数规范**：定义了 CMR / AEPE / RMSE 在本项目与 GDROS 官方代码中的
> 真实含义，并给出 2026-11-13 的口径更正。任何对外数字都应能追溯到本文。

## 1. 结论先行

**GDROS 官方权重在 OS 数据集上的复现没有问题。** 权重、数据集、模型配置、
LSR、`iters=32` 全部正确；此前测到 CMR@1 只有 9.43% 是**误差度量定义不同**，
不是 checkpoint / 测试集版本 / 实现 bug。论文 Table II 与官方代码可以逐位复现。

官方 `test.py` 原样跑（只修 `--Npoint` 的 `nargs='+'` 类型 bug，其余一字不动），
同一权重、同一份数据、424 对：

```
Validation (testing) EPE: 1.461846, 1px: 0.469459, 3px: 0.886792, 5px: 0.972654
test (testing) EPE: 1.461846, RMSE: 0.746347
T=1 CMR: 33.73%, MAE: 0.7420, RMSE: 0.0273
T=2 CMR: 81.60%, MAE: 1.1249, RMSE: 0.1553
T=5 CMR: 99.53%, MAE: 1.4431, RMSE: 0.6749
```

| 来源 | AEPE | CMR@1 | CMR@2 | CMR@5 |
|---|---|---|---|---|
| 仓库 `REFS["gdros_released"]` | 1.461486 | 33.73 | 81.60 | 99.53 |
| 官方 `test.py` 本次实测 | **1.461846** | **33.73** | **81.60** | **99.53** |
| GDROS 论文 Table II（OS，mean±std） | 1.48±0.02 | 33.88±0.73 | 80.19±1.77 | 99.45±0.11 |

三组数字逐项吻合（实测值都落在论文 ±std 内）。

## 2. 根因：`test.py` 算的不是端点误差

```python
# GDROS 官方 test.py
epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
```

实测形状：`flow_pre (1,2,512,512)`、`flow_gt (2,512,512)`、`diff (1,2,512,512)`。
广播后在 **dim=0（batch 维）** 求和开方 → `[2,512,512]`，即 `|Δx|`、`|Δy|` 两个**分量**。
真正的端点误差应是 `sum(dim=1).sqrt()` 或 `linalg.vector_norm(diff, dim=1)` ——
本项目 `eval_os_unified.py` / `eval_os_full_metrics.py` / E075A `eval_model` 用的正是后者。

同一批预测流、只换误差定义：

| 误差定义 | AEPE | CMR@1（配对） | CMR@2 | CMR@5 |
|---|---|---|---|---|
| A 官方：逐分量 `\|Δx\|,\|Δy\|` | **1.461846** | **33.726** | 81.604 | 99.528 |
| B 向量 EPE `sqrt(Δx²+Δy²)` | 2.321585 | 9.434 | 52.123 | 94.340 |

`AEPE_B / AEPE_A = 1.5881`，与各向同性误差下的理论比 `π/2 = 1.5708` 一致。
阈值施加在更小的分量误差上，CMR@1 自然从 9.43 抬到 33.73。

## 3. 指标定义（三种容易混的写法）

| 记号 | 定义 | 备注 |
|---|---|---|
| `EPE(k)` | 第 k 对图像在官方误差图上的**平均值** | 官方 `MAE_list`，配对级聚合 |
| `CMR@τ` | `100 × \|{k : EPE(k) < τ}\| / 总对数` | **配对级**，不是逐像素 |
| `AEPE@τ` | 通过阈值配对的 `EPE(k)` 均值 | inlier-mean，失败配对被筛掉 |
| 全局 `AEPE` | 全部 424 对的 `EPE(k)` 均值 | 直接被灾难配对抬高 |
| 逐像素 `CMR@k` | 有效像素中 `e < k` 的占比 | 项目历史协议 |
| 官方 "RMSE" | `np.var(MAE_list)` | **是方差，未开根号**，名实不符 |
| 真 RMSE | `sqrt(mean(e²))` | 本报告同时给出 |

另一种常见写法是截断式 `mean(min(e, τ))`：把失败样本按 τ 计入惩罚，
同时反映精度与召回。它与 inlier-mean 的 `AEPE@τ` **会给出相反结论**
（见 [RESULTS_OS.md](RESULTS_OS.md) §2），因此必须写明用的是哪种。

官方 / 论文里另外两处名实不符：

1. 论文公式 (15) 写的是带 `sqrt` 的标准差，但表格 OS `RMSE 0.75±0.02`
   对应的是方差 `0.7463`（开根后 0.864）—— 数字跟代码走。
2. 论文正文称 EPE 是 "averaged pixel-wise Euclidean distance l2"，但表格数字
   与官方代码（逐分量误差）一致。要引用论文数值，必须按官方代码口径。

## 4. 已排除的疑点

| 疑点 | 核实结果 |
|---|---|
| 权重不对 | 官方 `test.py` 用同一个 `GDROS-os512.pth` 复现论文数字 |
| 数据集版本不对 | `testing.zip` 与提取目录结构一致：424 对，各字段各 424；`truth_flow/scene/` 另有副本，`glob('*.flo')` 只取根目录 424 个并与 `opt/*.png` 按序对齐 |
| 模型配置不对 | `configs/default.py` 默认 `featureformerwithlbl`、`image_size=[512,512]`，与官方一致 |
| `LSRnet.pth` 没加载 | 官方 `test.py` 也不加载；`LSR()` 走无权最小二乘 `LSR_()`，只用 `[10000:10000+Npoint]` 采样点 |
| LSR / iters / mask | `Npoint=200000`、`iters=32`、`test_mode=True` 一致；valid mask 实测全 1 |

## 5. 推理链三段（数字从哪来）

1. **数据读取**：官方 `core/datasets.py` 的 `FlowDataset_test.__getitem__` 返回
   `opt / sar_warped / truth_flow / valid`，512×512。
2. **GRU → 全分辨率**：`E073FusionMatcher.recurrent_from_corr` 在 1/8 分辨率迭代，
   `upsample_flow` 以 8 倍双线性 + 学习到的 mask 回到 512×512。
3. **B/C 双分支 + 区域门控**：`PairFusion.forward` 输出 `(1−g)·B + g·C` 的流序列。
4. **LSR 仿射重建**：`e074.lsr` → `LSR(1, 512, 512, flow, 200000)`，
   与官方 `test.py` 完全一致（无权最小二乘，取连续中段 20 万点）。
5. **误差计算**：官方口径 `sum(dim=0).sqrt()`，项目口径 `vector_norm(dim=1)`。
   代码摘录见 [`评测代码三段_20261113.md`](../research/results/OS_20261113/评测代码三段_20261113.md)。

## 6. 历史结论更正（此前判断是错的）

- 「这是真实差距，不是评测脚本 bug」「1.461 极可能是论文公布值而非本机实测」→ **错**。
  1.461 就是本机 checkpoint 在同一数据集、官方口径下的实测值。
- 「baseline 数字存在口径差，是 checkpoint 与测试集版本的差异」→ **错**，同上。
- 早期「官方配对级口径」的表实际是**配对级聚合 + 向量 EPE** 的混合口径，
  不能与论文 / `REFS` 数字对照。

**正确表述**：同一脚本、同一口径下 baseline 与自研方法可比；与论文表格对比时
必须统一到官方 `test.py` 口径，或明确声明用了向量 EPE 口径。
此前所有「baseline 弱于论文」的说法作废。

E072 → E075A 整条链路都用 `vector_norm`，因此内部消融（B/C/M4 之间）**不受影响**；
但**不能**把 M4 的向量 EPE 数字与论文 Table II 的 GDROS 行直接并列。
按论文口径重算的 M4 结果见 [RESULTS_OS.md](RESULTS_OS.md) §1.1。

**新增实验统一用官方口径**：2026-11-14 起的两路/单路相关体消融
（[`ablate_v2_off.py`](../research/eval/ablate_v2_off.py)）直接复用
`eval_os_official_protocol.py` 的 `Collector`，GDROS 官方权重、M4 两路、M4 单路
三者在同一份官方口径下报数（424 对），不再产出向量 EPE 主表；向量 EPE 只作为
`AEPE_vector_EPE_reference` 交叉验证字段保留。这样消融结论可以直接与 §1.1 的
baseline 行并列，避免再次出现混合口径。

## 7. 报数检查清单

- [ ] 写明误差度量：逐分量 `|Δx|,|Δy|` 还是向量 `sqrt(Δx²+Δy²)`
- [ ] 写明聚合单位：配对级还是逐像素
- [ ] `RMSE` 写明是 `np.var(MAE_list)` / `sqrt(var)` / `sqrt(mean(e²))` 中的哪一个
- [ ] `AEPE@τ` 写明是 inlier-mean 还是截断式 `mean(min(e,τ))`
- [ ] 与论文 Table II 并列时，只使用官方口径的数字
- [ ] 同时报告 CMR@5 与灾难配对占比，如实标注松阈值短板
