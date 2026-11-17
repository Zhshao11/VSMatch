# 独立精简 C/student

2026-11-17同步。方法名 `M4-streamlined-C-student`，入口为 [M4StreamlinedCStudent](../research/remote_stage/os/m4_streamlined_c_student.py)。这是从历史 C 学生权重提取的单匹配器推理版，不包含后续自适应编码、独立矩阵修正器或完整 M4 的 B/C 区域门控。

保留两套独立 CNN/Transformer 特征前端、冻结 DINO/描述子与 V2 FiLM 调制、两个四层相关金字塔、全局标量查询融合、单套32轮 GRU、末轮学习上采样及一次官方 LSR。完整说明见[逐层架构](M4_STREAMLINED_C_STUDENT_ARCHITECTURE.md)和[历史训练/蒸馏来源](C_STUDENT_DISTILLATION_AND_TRAINING.md)。

## 已验证结果

| 数据范围 | 平均component误差/px ↓ | CMR@1 ↑ | CMR@2 ↑ | CMR@5 ↑ |
|---|---:|---:|---:|---:|
| OS424 | 1.592903 | 69.5755% | 85.1415% | 92.9245% |
| WHU700，冻结OS权重 | 2.742063 | 63.5714% | 80.8571% | 89.1429% |
| UBCv2原始1625对×3组，冻结OS权重 | 4.117085 | 46.7487% | 61.7026% | 72.5949% |

OS424全部输出与历史C缓存逐元素一致，参数319,303,309，其中DINO为304,367,616。RTX5090 GPU3独立平均143.476ms/对，包含网络和唯一末端LSR，不含读取图像/H2D/保存；这不是与其他模型同轮配对的速度比较。

UBCv2为既有独立seeds901/902/903的均值，不是作者筛选1447对的精确复现。指标为逐对全图x/y绝对分量均值，CMR使用严格小于阈值，不与标准向量EPE混用。详见[E099 OS报告](../research/results/E099A_STREAMLINED_C_STUDENT_OS424_20261116/REPORT.md)和[E100跨域报告](../research/results/E100A_STREAMLINED_C_WHU_UBCV2_20261116/REPORT.md)。

## 权重和依赖

Git仅存代码、协议、结果与哈希，不存模型权重、数据集或稠密预测。精简权重保留在已有项目服务器zzvr：

```text
/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-099A-STREAMLINED-C-STUDENT-OS424-20261116/M4-streamlined-C-student-state.pt
Bytes: 1277449423
SHA256: c2d4f79b45ce190df48f52f588e40a7654fa6cce0a32da1bc0c32020b35636ad
```

原始受保护M4权重SHA256为 `55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b`，提取其 `c.student.*`。完整机器清单见[模型版本清单](../research/models/M4-streamlined-C-student.json)。

运行仍需已配置的官方GDROS、DINO及迁移模块；当前先构造历史C再提取有效模块，尚无不依赖历史构造器的冷启动工厂。`M4StreamlinedCStudent.forward` 是inference-only，不可直接用于训练。代码依赖的 `m4_clean_deploy.py` 与E072–E075/E092–E093脚本均在本仓库，外部资产布局见[复现指南](../research/REPRODUCIBILITY.md)。

## 集成推理

将 `research/remote_stage/os/` 中依赖脚本放入既有迁移工程 `.codex_tmp/` 并加入Python搜索路径，从迁移工程根目录执行。按既有布局配置 `SPARSEGEO_ROOT`，用项目环境Python：

```python
import torch
import e093a_bc_ablation_profile as branches
import e092a_official_eval as official
from m4_streamlined_c_student import M4StreamlinedCStudent

branches.configure_backends()  # FP32，TF32关闭
payload, state = branches.checkpoint_state()
student, load_report = branches.build_c_student(torch.device("cpu"), state)
model = M4StreamlinedCStudent(student).eval()
del student, state, payload
model = model.to("cuda:0")
_, _, LSR = official.official_imports()

# source=光学，target=SAR；float32 RGB 0..255，[1,3,512,512]，同一GPU。
raw = model(source, target, iters=32)
final = LSR(1, 512, 512, raw, 200000)
```

`raw` 是GPU上的目标减源位移；官方 `LSR` 返回同形状 `[1,2,512,512]` 的CPU仿射位移场。上述路径严格加载历史权重，无需运行完整M4或构造B/教师。

## 评测与离线核验

OS入口：[e099a_streamlined_c_student_eval.py](../research/remote_stage/os/e099a_streamlined_c_student_eval.py)，具体依赖与新运行目录要求见[E099协议](../research/experiments/E-099A-STREAMLINED-C-STUDENT-OS424.md)。它额外使用E098已验证缓存来证明精简等价性。

无需GPU/权重即可复算OS紧凑结果：

```bash
python3 research/remote_stage/os/verify_e099a_results.py \
  research/results/E099A_STREAMLINED_C_STUDENT_OS424_20261116
```

输出PASS，424条精度/计时和全部424对缓存等价性检查通过。加 `--verify-predictions` 需访问zzvr原预测与权重，用于重新计算密集文件哈希。

跨域入口：[e100a_streamlined_c_crossdataset.py](../research/remote_stage/os/e100a_streamlined_c_crossdataset.py)，搭配本仓库 `research/remote_stage/ubcv2/e092cr_ubcv2_eval.py`，按[E100协议](../research/experiments/E-100A-STREAMLINED-C-WHU-UBCV2.md)执行predict、join及[独立核验](../research/remote_stage/os/verify_e100a_results.py)。已有5575份预测核验PASS；跨域完整重算仍需服务器上的数据、模型与预测。

此版本作为独立精简C基线归档；完整M4-streamlined的默认配置保持不变，后续试验结果不冒充本版本性能。
