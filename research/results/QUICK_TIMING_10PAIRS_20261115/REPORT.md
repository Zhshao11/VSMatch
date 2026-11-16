# M4 与 GDROS：10 对图像推理耗时实测

日期：2026-11-15（Asia/Shanghai）。本次没有训练、修改模型实现或更新权重。

**结果：原始 M4 平均耗时为 GDROS 的 7.93 倍。**

| 方法 | 平均毫秒/对 | 中位数毫秒/对 | 10对计时总秒数 | 对/秒 | 每对LSR次数 |
|---|---:|---:|---:|---:|---:|
| GDROS 官方 OS 权重 | 121.195 | 118.759 | 1.212 | 8.251 | 1 |
| OS E075A-M4 step80K 原始实现 | 960.787 | 952.100 | 9.608 | 1.041 | 129 |

条件：zzvr，同一张 RTX 5090（物理 GPU 6，UUID GPU-c32175d3-7e65-83aa-a3f6-fb85fe6a6c17）；512×512、batch=1、FP32、关闭 TF32/AMP、32 轮 GRU；每个模型独立进程，先预热3对，再依次测量相同10对，每对1次。CPU计算线程数为1。

样本：OS 官方424对的排序索引 0, 47, 94, 141, 188, 235, 282, 329, 376, 423，通过等间隔规则预先选取，未使用GT挑样本。

计时范围：输入已加载到GPU，CUDA同步后使用墙钟计时，包含模型forward、所有内部LSR、设备传输、门控及一次末端官方LSR；不含检查点加载、图片解码、输入传输、输出验证、结果存盘和nvidia-smi采样。

元数据更正：实测使用的远端E075默认 c_skip_teachers=True，已跳过C分支两个teacher；先前将其描述为保留teacher不准确。E074/E075源码哈希与测量JSON一致，耗时和LSR计数不受此文字更正影响。实测保留全部中间轮次门控，尚未应用仅最终轮融合。原始JSON中的implementation描述也有同一文字错误，应以此更正为准。计数包装器实际确认每对M4为129次LSR、GDROS为1次；不是仅根据源码推算。

**限制：这是共享GPU上的10对快速实测。** 所有卡均已有外部任务，本次未独占GPU；两次测量均记录到同卡外部进程 945911。nvidia-smi快照不能证明计时区间完全没有竞争。样本量小、每对只测1次，不能作为独占环境正式部署基准。

本次仅比较整个原始实现的耗时，不能将7.93倍的差异全部归因于LSR；多个匹配器、DINO前端、门控及同步也包含在内；C teacher未执行。本次没有测定移除某模块后的加速或精度变化。

逐对记录：

| Pair ID | GDROS毫秒 | M4毫秒 |
|---|---:|---:|
| 000000 | 117.218 | 953.547 |
| 000047 | 119.171 | 945.768 |
| 000094 | 118.347 | 1003.479 |
| 000141 | 121.311 | 954.030 |
| 000188 | 116.397 | 988.960 |
| 000235 | 143.987 | 941.758 |
| 000282 | 120.665 | 990.906 |
| 000329 | 116.747 | 943.434 |
| 000376 | 117.933 | 950.652 |
| 000423 | 120.177 | 935.339 |

逐对CSV和SUMMARY.json位于本报告同目录；完整原始JSON位于 outputs/quick_timing_10pairs_20261115 的 gdros.json、m4.json，包含权重SHA256、环境、逐对耗时、实测LSR次数和外部进程快照。

运行入口：research/remote_stage/os/quick_timing_10pairs_20261115.py，依赖已有E092A加载器及其数据、权重；必须在远端 image_matching_part2_migrated 项目根目录运行（历史E044A加载器按工作目录定位实现）。

```bash
cd "$SPARSEGEO_ROOT/image_matching_part2_migrated"
for model in gdros m4; do
  CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$BENCH_PYTHON" -u .codex_tmp/quick_timing_10pairs_20261115.py \
    --model "$model" --output-dir "$BENCH_OUTPUT"
done
```

SPARSEGEO_ROOT、BENCH_PYTHON、BENCH_OUTPUT分别设为项目根路径、带PyTorch 2.8/CUDA12.8依赖的Python和新的结果目录；GPU编号应按实际环境选择。脚本需与e092a_official_eval.py一同放在远端.codex_tmp目录。

远端原始记录目录：GDROS/experiments/sparsegeo/runs/QUICK-TIMING-10PAIRS-20261115。
