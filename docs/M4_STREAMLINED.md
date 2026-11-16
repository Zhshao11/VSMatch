# M4-streamlined：精简推理版存档

命名日期：2026-11-16。**M4-streamlined** 是 E095A 已验证的 M4 精简推理实现，原代码名为 `CleanM4`。这是原受保护 OS M4 的等价工程精简，没有重新训练。默认保留 **3 个匹配器 × 32 轮 GRU、5 个四层相关金字塔、4 次内部 LSR + 1 次最终 LSR**。

本名称不指 E097A 的仅最终一次或最后两次 LSR 消融，也不指尚未验证部署耗时的 C-only 候选。

## 入口与实现

- 统一入口：[m4_streamlined.py](../research/remote_stage/os/m4_streamlined.py)，类 `M4Streamlined`。
- 冻结核心：[m4_clean_deploy.py](../research/remote_stage/os/m4_clean_deploy.py)，类 `CleanM4`。新类直接继承其构造和前向，不改权重键或算式。历史脚本继续使用旧名以保留源码哈希和证据。
- 机器可读版本清单：[M4-streamlined.json](../research/models/M4-streamlined.json)。
- 当前工作入口：[CURRENT_METHOD_STATE](../research/CURRENT_METHOD_STATE.md)；复现条件：[REPRODUCIBILITY](../research/REPRODUCIBILITY.md)。

删除内容包括闲置教师模块、未消费的嵌套更新/上下文、中间 mask 与高分辨率上采样、无用诊断计算；冻结且状态完全一致的 DINO 和描述子适配器按图像对共享。有效匹配器、FiLM、门控及迭代位移更新保留。仅推理使用此路径，训练仍用历史结构。

## 已验证结果

OS 作者公开测试组全部 424 对、512×512、FP32、batch=1、32 轮、官方最终 LSR：

| 指标 | M4-streamlined |
|---|---:|
| 公共分量平均误差 / px | 1.4744179696 |
| CMR@1 / @2 / @5 | 72.40566038% / 84.90566038% / 94.10377358% |
| 每对分量均值 ≥5 px | 25 / 424 |
| 向量 EPE / px（不同口径） | 2.3679955441 |
| 注册参数数目（含冻结权重） | 347,975,753 |

E095A 的全部 424 对原始流和最终流与五次 LSR 对照逐元素完全一致；该对照由 E094A 与原 129 次路径建立等价链。E096A 再次全量推理复核。

E095A 同轮配对耗时为精简前五次路径 621.689 ms → M4-streamlined 340.450 ms（1.826×）。E096A 同轮官方 GDROS / M4-streamlined 为 221.343 / 571.037 ms（2.58×）。两次都使用共享 RTX 5090，不同 GPU/负载的绝对时间不能拼接为总加速比。

该版本改善效率与存储，不解决原来的误差重尾。后续诊断显示主要成本是三套迭代匹配器、DINO 和特征前端；单 C 分支已具有 69.5755% 的 CMR@1，但不能据此归因到某个独立模块。

| 证据 | 内容 |
|---|---|
| [E094A](../research/results/E094A_M4_FINAL_ONLY_LSR_424_20261116/REPORT.md) | 129→5 次拟合及中间门控清理、等价性 |
| [E095A](../research/results/E095A_M4_CLEAN_DEPLOY_424_20261116/REPORT.md) | 核心精简、全量逐元素等价、注册参数与配对耗时 |
| [E096A](../research/results/E096A_GDROS_CLEAN_M4_OS424_20261116/REPORT.md) | 官方 GDROS 与精简 M4 全量新推理 |
| [E097A](../research/results/E097A_M4_LSR_SLOT_ABLATION_OS424_20261116/REPORT.md) | 五次/仅最终一次/最后两次 LSR 的固定权重消融 |
| [E098A](../research/results/E098A_M4_COST_CONTRIBUTION_20261116/REPORT.md) | 组件耗时、历史分支缓存重新审计、精度归因边界 |

公共分量误差为 `mean_pixels((abs(dx_error)+abs(dy_error))/2)`；CMR@t 为每对该均值严格小于 t 的比例。所有 424 对参与，无困难样例剔除。历史向量 EPE、UBCv2 不同构造及不同数值设置的结果须分开读。

## 权重与依赖边界

原始受保护权重为 `os_e075a_m4_80k`，路径和完整 SHA256 见 [CHECKPOINTS.json](../research/CHECKPOINTS.json)。精简状态 `clean_deploy_state.pt` 保留在 zzvr：

```text
$SPARSEGEO_ROOT/GDROS/experiments/sparsegeo/runs/E-095A-M4-CLEAN-DEPLOY-424-20261116-r2/clean_deploy_state.pt
SHA256: b68832855a3770dbcb0eed5d7be421d0379ae94b4774e161334b10079a3519a5
Bytes: 1392216504
```

Git 不包含数据、权重或密集预测。克隆后可直接阅读方法、脚本和紧凑证据；实际推理仍需项目存储中的权重、上游 GDROS/DINO 和迁移模块。**当前还需构造历史模型后提取精简模块，尚无独立冷启动工厂。** 精简 state_dict 可严格加载到新旧两个类，键名不变。

## 推理集成

先按 [复现指南](../research/REPRODUCIBILITY.md) 准备 `$SPARSEGEO_ROOT` 布局，将所需冻结 OS 脚本放到迁移工程的 `.codex_tmp/`，并从迁移工程根目录运行。依赖见根目录 `requirements-research.txt` 和上游项目环境。

```python
import torch
import e092a_official_eval as base
from m4_streamlined import M4Streamlined

# base.load_m4 依赖已准备好的历史模块和初始化权重；CPU 上严格加载。
legacy, _ = base.load_m4(torch.device("cpu"))
model = M4Streamlined(legacy).eval()
del legacy
device = torch.device("cuda:0")
torch.cuda.set_device(device)
model = model.to(device)
_, _, LSR = base.official_imports()

def official_lsr(flow):
    return LSR(1, 512, 512, flow, 200000)

with torch.inference_mode():
    raw = model(source, target, iters=32, lsr_fn=official_lsr)
    final = official_lsr(raw)  # 第五次；不能省略
```

`source` 是光学、`target` 是 SAR；输入均为 float32 RGB 0..255，形状 `[1,3,512,512]`，位于同一 GPU。原始输出 `[1,2,512,512]` 是目标减源的位移；官方最终 LSR 返回同形状 CPU 张量。需要完整配对复核时使用 [E095A 入口与运行条件](../research/experiments/E-095A-M4-CLEAN-DEPLOY-424.md)，它还要求 E094A 历史预测归档。

无需 GPU、数据或权重即可重算已归档统计：

```bash
python3 research/remote_stage/os/verify_e095a_results.py research/results/E095A_M4_CLEAN_DEPLOY_424_20261116
python3 research/remote_stage/os/verify_e096a_results.py research/results/E096A_GDROS_CLEAN_M4_OS424_20261116
python3 research/remote_stage/os/verify_e097a_results.py research/results/E097A_M4_LSR_SLOT_ABLATION_OS424_20261116
python3 research/remote_stage/os/verify_e098a_results.py research/results/E098A_M4_COST_CONTRIBUTION_20261116
```

## 存档范围

本次同步 E092–E098 的方法相关脚本、协议、指标/哈希清单及报告，同时保留已在 GitHub 的旁路消融和 UBCv2 工作。不同对话的 UBCv2 构造/数值设置各有结果目录，不合并成一个“官方复现数字”。生成的 `outputs/`、调度器运行状态、数据、模型与密集输出排除在 Git 外。

本次存档的验证记录见 [archive-validation.json](../research/models/M4-streamlined.archive-validation.json)：E094–E098 离线校验通过；Python 语法检查通过；使用 zzvr 的已配置 Python 在临时目录运行 15 项 CPU 工具测试通过，并核实新类继承原构造与前向实现。此命名步骤没有重跑 GPU 推理。独立克隆若尚未安装依赖/外部资产，preflight 会报告 INCOMPLETE。
