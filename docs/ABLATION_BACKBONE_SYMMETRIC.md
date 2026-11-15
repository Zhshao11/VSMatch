# 对称单路旁路：只留 GDROS 前端 / 只留含 DINO 的前端

> **定位（先读）**：这是**冻结权重、零训练的敏感性诊断**，不是合格的模块消融。
> 两个配置都是在同一个已收敛的 M4 上把融合门控硬推向某一侧，测的是
> 「这个管线能不能被直接拆成单路」。**不能**用来量化 V1 / V2 任一路的边际贡献，
> 也**不宜进论文消融主表**。理由见 §4，边界沿用
> [`ABLATION_V2.md`](ABLATION_V2.md) §4.2。
>
> **2026-11-15 后续（重要）**：已补做
> [适配性短训（方案 A）](ABLATION_SHORT_FINETUNE.md)。2000 步下游适配把两个
> 配置的 AEPE 落差分别拉回 **96.2% / 95.2%**（`7.7457→1.7127`、
> `41.3298→3.3824`），>10px 灾难失败都降到 **17 对**（`both` 为 12 对）。
> 这把「大幅掉点主要是下游 OOD」从定性判断变成了定量证据，本页的零训练
> 数字**更不能**被当作任一路的边际贡献。

2026-11-15 在 **剪枝后的 E075A-M4@80K**（`c_skip_teachers=True`，与剪枝前
指标逐位相同）上，把 11-14 只有单方向的 `v2off` 补成对称的两个方向，
分别放到 zzvr 的两张 GPU 上并行跑；都是 **GDROS 官方 `test.py` 口径**：
OS 测试集全量 424 对、`iters=32`、`LSR NPOINT=200000`。

## 1. 实验定义

E073 用两处门控混合两路描述子：

```python
# F3（相关体级）
a = sigmoid(corr_gate_logit)
corr = (1 - a) * Corr1 + a * Corr2

# F2（特征级）
gs = sigmoid(fuse_gate(cat([f1s, f2s])))
fs = (1 - gs) * f1s + gs * fuse_proj(cat([f1s, f2s]))
```

训练后 `corr_gate_logit ∈ [0, 0.007]`，即 `a ≈ 0.5`；`fuse_gate` 是学出来的
空间门控。M4 的 B/C 分支共 **5 个** `E073FusionMatcher`
（`F2,F3,F2,F3,F3`），两个配置都把 5 个一起切换，不改层维度、不重训：

| 配置 | F3 `corr_gate_logit` | F2 `fuse_gate` | 含义 |
|---|---|---|---|
| `both` | 原值（≈0，`a≈0.5`） | 原 module | 原始两路 |
| `v1only` | `-80`（`a≈0`） | 常数 `-80`（`gs≈0`） | 只保留 GDROS 兼容那一路 |
| `v2only` | `+80`（`a≈1`） | 常数 `+80`（`gs≈1`） | 把门控全部推向含 DINO 的那一路 |

1 对 probe 证明开关确实生效：`v1only` 相对 `both` `max|Δpost|=46.95`，
`v2only` `max|Δpost|=143.67`。两个进程都 `SANITY PASS`。
门控在三个配置间的切换过程可用自包含动画
[`ablation_gate_animation.html`](ablation_gate_animation.html) 逐步查看
（浏览器直接打开，无需依赖或网络）。

### 1.1 结构边界：「只留 DINO」不是物理上可实现的

E072 的 V2 **不是独立 backbone**。`E072Matcher._features` 里：

```python
fs, ft = self.front_v1(img1n, img2n, cnet)      # GDROS 兼容前端，V1/V2 都会跑
if self.variant == "V2":
    fs, ft = self.e046(raw1, raw2, fs, ft)      # 冻结 DINO 描述子做 FiLM/residual 注入
```

所以 `f2` **本身就是在 V1 特征上调制出来的**，已经含 V1 信息；冻结的
DINO 描述子只提供调制信号，没有自己的稠密特征图。此外 F2 里 `gs=1` 得到的是
`fuse_proj(cat([f1s, f2s]))`，**仍显式包含 `f1s`**。

因此 `v2only` 的准确含义是「**把两处融合门控全部推向 V2 侧**」，
不是「删除 GDROS backbone」。同理，`v1only` 的 V2 前端前向仍会执行，
只是输出被乘到 ≈0，数学上等价于只用 `Corr1` / `f1s`，这一侧才是严格的单路。

## 2. 结果（官方口径 424 对，可与 GDROS 并列）

| 配置 | AEPE↓ | CMR@1↑ | CMR@2↑ | CMR@5↑ | RMSE↓（官方栏位/方差） | 耗时 |
|---|---|---|---|---|---|---|
| GDROS 官方权重 `GDROS-os512.pth` | 1.4618 | 33.73 | 81.60 | 99.53 | 0.7463 | 59–87 s（历史） |
| **M4 两路 `both`** | **1.4738** | **72.41** | **84.91** | **94.10** | **8.3350** | 317.2 s |
| `v1only`（只留 GDROS 那路） | 7.7457 | 0.94 | 4.25 | 24.76 | 16.0190 | 350.2 s |
| `v2only`（门控推向含 DINO 那路） | 41.3298 | 0.00 | 0.00 | 0.24 | 287.7100 | 950.7 s ⚠️ |

`both` / GDROS 行是 11-13 官方口径历史值；`v1only` 本次结果与 11-14 的
`v2off` **5 项指标全部 `Δ=0`**（AEPE `7.745715116406876`），
再次确认 2026-11-15 的 C 分支剪枝与原模型逐位等价。

⚠️ `v2only` 运行在 GPU 1，期间该卡被另一任务占满（97% 利用率），
每对约 2.2 s 而非空闲时的 0.72 s；**耗时不可与 `v1only` 的 350.2 s 直接比较**，
指标不受影响。

## 3. 逐对 ΔEPE（配对级，正数 = 前者更差）

| 对比 | mean Δ | median Δ | 前者变好 | Δ>+5px | 灾难失败 >10px（前者/后者） |
|---|---|---|---|---|---|
| `v1only − both` | +6.2719 | +6.0892 | 15/424（3.5%） | 261 对 | 104 / 12 |
| `v2only − both` | +39.8560 | +40.0243 | 2/424（0.5%） | 419 对 | 416 / 12 |
| `v1only − v2only` | −33.5841 | −33.8624 | 420/424（99.1%） | 2 对 | 104 / 416 |

`v2only` 几乎全军覆没：424 对里 **416 对**的配对 EPE 超过 10 px，
CMR@5 只剩 0.24%。`v1only` 虽也严重退化，但仍有 24.76% 的配对落在 5 px 内。
两者相差 5.3 倍，且逐对比较里 `v1only` 在 420/424 对上更好。

## 4. 能成立 / 不能成立的推论

**能成立（诊断层面）**：

1. 这个已收敛的 M4 管线对**训练后的门控分布强耦合**。两处门控的收敛值
   （`a≈0.5`、学到的 `fuse_gate`）是解的一部分，硬推到任一侧都会让
   B/C 分支与 L4 门控面对训练时从未见过的输入分布。
2. 两个方向都大幅掉点，且 `v2only` 更极端 —— **对称性反而加强了
   「推理时不能直接拆」的判断**，比只有单方向的 11-14 结果更不容易被
   「恰好那一侧弱」解释掉。
3. `v2only` 更差符合结构预期（**推测，未验证**）：F2 的 `gs=1` 让下游
   第一次单独见到 `fuse_proj` 的输出，`Corr2` 的统计量与 GRU 的收敛区域
   也不匹配；而 V2 特征本身是 V1 特征被 DINO 调制的结果，并非独立表示。

**不能成立**：

- ❌ 不能说「DINO 那一路不重要 / V1 更重要」。这是**不对称推断**：
  零训练旁路只有「几乎不掉点」才对冗余有较强解释力；「大幅掉点」
  几乎不携带信息，只能说明「不能直接拆」。
- ❌ 不能把 `7.7457` 或 `41.3298` 当作某一路的边际贡献。退化里混有
  「少一路信息」与「下游 OOD」两个不可分离的原因；
  `v1only 7.7457` 比 GDROS `1.4618` 差 5.3 倍本身就是下游失配的证据 ——
  一个真正只依赖 GDROS 前端的模型没有理由比 GDROS 自己差这么多。
- ❌ `v2only` 尤其不能被叫做「纯 DINO 单路」，见 §1.1。
- ❌ 不宜进论文消融主表。

**要量化边际贡献**，仍必须控制训练预算 / 数据 / 下游适配，只变「有无该模块」：
A. 固定单路、冻结前端、续训 B/C + L4 gate 2K–8K 步（≲2000 步稳定，
4000 步 CMR@1 崩塌）；B. 同预算严格重训单路（约 27 h）。
详见 [`ABLATION_V2.md`](ABLATION_V2.md) §4.1。

## 5. 复现

```bash
cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
PY=/home/zzvr/miniconda3/envs/compared/bin/python

# 两张不同 GPU 并行；一个进程一个配置
CUDA_VISIBLE_DEVICES=5 $PY .codex_tmp/ablate_backbone_single.py \
  --config v1only --max-pairs 424 --gpu 0 \
  --out .codex_tmp/ablate_v1only_pruned.json
CUDA_VISIBLE_DEVICES=1 $PY .codex_tmp/ablate_backbone_single.py \
  --config v2only --max-pairs 424 --gpu 0 \
  --out .codex_tmp/ablate_v2only_pruned.json

# 本地合并（纯标准库，无需 GPU）
python research/eval/merge_backbone_ablation.py
```

原始数据：[`research/results/OS_20261115/`](../research/results/OS_20261115/)
（`ablate_v1only_pruned.json`、`ablate_v2only_pruned.json`、
`backbone_ablation_merged.json`），均含 424 项逐对 EPE。
脚本说明见 [`research/eval/README.md`](../research/eval/README.md)。
