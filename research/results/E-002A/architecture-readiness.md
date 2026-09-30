# E-002A 架构就绪性审计

日期：`2026-08-04`

方法版本：`M001 v0.2`

结论：`WAITING_FOR_METHOD_DECISION`

本报告只记录远端代码、checkpoint 和数据协议的只读核查。未修改 GDROS 代码，未启动
E-002A 实验。

## 1. 实际 checkpoint 路径

- 仓库：`bl_tunnel:/mnt/sda/zhushihao/VS_Match/GDROS`
- 分支和 commit：`codex/reproduction@ee6b6216fe780bc03bfb4759ae094daa0cfddc60`
- checkpoint 配置：`featureformerwithlbl`，1 层 FeatureTransformer，
  `attn_splits=2`
- checkpoint 严格加载：318 个 tensor，无 missing/unexpected key

官方 `512 x 512` 输入的实际调用链为：

```text
optical/SAR [B,3,512,512]
  -> 2 * (image / 255) - 1
  -> BasicEncoder [2B,256,64,64]
  -> 1x1 channel converter [2B,256,64,64]
  -> position encoding + FeatureTransformer
  -> optical/SAR post-Transformer features [B,256,64,64]
  -> CorrBlock N x N
  -> 32 GRU iterations
  -> convex upsample [B,2,512,512]
  -> public LSR
```

E-002A 应直接复用 `model.featureNet` 的两个 post-Transformer 输出，并跳过 context、
CorrBlock 和 GRU。`FeatureEncoder.forward` 不使用传入的 context；实测传 context 与
`None` 时两个输出的最大绝对差均为 `0.0`。

参数审计：全模型 12.847M，`featureNet` 对象 8.660M，实际 feature forward 路径
4.804M；其中 3.855M 参数的 `cost_perceiver_encoder` 被 checkpoint 加载但未执行。删除
未执行模块只能作为工程优化对照，不能计作 SparseGeo 方法贡献。

## 2. 无歧义的 1/8 主探针

原 CorrBlock 的分数为：

```text
S(i,j) = qi^T kj / sqrt(256)
```

因此当前唯一由 checkpoint 与 baseline 调用点完全确定的 E-002A 探针是：

```text
source = optical_feature.flatten(2).transpose(1, 2)  # [B,4096,256]
target = sar_feature.flatten(2).transpose(1, 2)      # [B,4096,256]
anchors = source[:, anchor_indices]                  # [B,K,256]
scores = anchors @ target.transpose(1, 2) / 16       # [B,K,4096]
target_index = argmax(scores, dim=-1)
```

必须在矩阵乘法前选取 source anchors，禁止先构造 `4096 x 4096` 再截取 K 行。默认
`K=128`，`K=64/256` 为 Pareto 对照；target chunk 建议为 1024。流式 argmax 只在新值
严格大于旧值时更新，以保持 `torch.argmax` 的首次命中语义。

| K | 空间分区 | 分数数目 | FP32 分数存储 | 点积量 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 8 x 8 | 262,144 | 1 MiB | 0.067 GMAC |
| 128 | 8 x 16 | 524,288 | 2 MiB | 0.134 GMAC |
| 256 | 16 x 16 | 1,048,576 | 4 MiB | 0.268 GMAC |

主探针不使用 GT、valid mask、置信度或双向一致性，也不在 source/target 端过滤黑边；
最终仍按公开的完整 `512 x 512` 无 mask 协议评测。

## 3. 1/16 不是 checkpoint 原生输出

checkpoint 没有原生 `1/16` 特征。以下方案不是等价实现：

1. post-Transformer decimation：`f[:, :, ::2, ::2]`，保留已有 token，像素坐标为
   `(16x, 16y)`；
2. post-Transformer `avg_pool2d(2, 2)`，混合相邻描述子，像素坐标为
   `(16x+4, 16y+4)`；
3. Transformer 前降采样，会改变位置编码和 window attention，不应进入 E-002A。

建议把原生 `1/8` 固定为正式主路径，只把 decimation 和 average-pool 作为两个明确命名
的 K128 resolution diagnostics。

## 4. 坐标、方向与解析 flow

方向为 optical source 到 SAR target，flow 定义为 `target - source`。BasicEncoder 总 stride
为 8；其 padding 与 GDROS 的 flow/upsample 约定对应 token `(xf,yf)` 的像素坐标
`(8xf,8yf)`，不是 `(511xf/63,511yf/63)`，也不是 `(8xf+4,8yf+4)`。

令：

```text
N(h,w) = [[2/(w-1), 0,       -1],
          [0,       2/(h-1), -1],
          [0,       0,        1]]
```

先在归一化 lattice 坐标中用 FP32 普通 LS 求 `q_norm = H_norm p_norm`，再转换：

```text
H_lattice = inverse(N) @ H_norm @ N
H_pixel = P @ H_lattice @ inverse(P)
flow(x,y) = (H_pixel @ [x,y,1])[:2] - [x,y]
```

其中原生 1/8 使用 `P=diag(8,8,1)`；1/16 decimation 使用 `P=diag(16,16,1)`；
1/16 average-pool 使用 `P=[[16,0,4],[0,16,4],[0,0,1]]`。

## 5. v0.2 尚未确定的定义

- raw-dot 是 baseline 原生分数；cosine 会改变 token norm 信息，必须明确主次关系。
- soft-argmax 尚未固定 raw/cosine 分数及温度 `tau`。
- K-to-all 下可实现的 dual-softmax 只能在 K 个 anchors 上做 source 方向归一化，应明确
  命名为 `anchor-restricted dual-softmax`。
- 若 dual-softmax 在全部 4096 个 canonical source token 上归一化，就重新引入
  `N x N` 计算，违反 E-002A 复杂度边界。
- 尚缺 E-002A 进入 E-002B 的精度门槛和最低资源收益门槛。

这些选择会改变实验含义，不能由实验工程任务自行决定。完整 E-002A 实现前需要
`M001 v0.3` 或等价的版本化实验单。

## 6. 建议的 E-002A 最小矩阵

1. 单元测试：feature hook parity、K 行 score parity、chunked argmax parity、已知仿射
   oracle、方向、stride-8 conjugation 和无 GT/mask 调用路径。
2. Smoke：2 个官方 pair，`1/8 + raw-dot + hard argmax + K128 + ordinary LS`。
3. 正式主探针：424 pair，`1/8 + raw-dot + hard argmax`，K64/K128/K256。
4. K128 相似度诊断：cosine-hard、soft-argmax、anchor-restricted dual-softmax；后两者仅在
   分数类型与温度预注册后运行。
5. K128 分辨率诊断：1/16 decimation 和 1/16 average-pool。
6. 资源：独立进程，warm-up 5、repeat 30、CUDA 同步并保存全部原始 timing；记录特征、
   matching、solver 和 end-to-end 的增量/峰值显存与耗时。

该阶段无训练成本。K128 原生匹配理论量仅 0.134 GMAC，但保留的 backbone/Transformer
预计成为主要耗时，端到端收益必须实测，不能由相关矩阵缩减比例外推。

## 7. 推进条件

方法讨论任务需要在 `M001 v0.3` 中至少固化：

1. `1/8` 主路径和两个 `1/16` diagnostic 的身份；
2. raw-dot/cosine 主次；
3. soft-argmax 与 dual-softmax 的分数类型和温度；
4. dual-softmax 的 K-anchor 归一化集合；
5. token 坐标、矩阵共轭、flow 方向公式；
6. E-002A 到 E-002B 的精度与资源门槛。

收到该版本后，E-002A 才从 `DRAFT` 转为 `READY` 并进入实现、独立审阅、smoke、baseline
回归和 watchdog 正式评测。
