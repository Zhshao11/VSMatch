# GDROS 三数据集论文与公开代码协议核对

2026-11-15，方法对话直接读取论文HTML、Crossref记录、Semantic Scholar记录、官方README及zzvr已审test.py。本文记录文献事实，不是新评测结果。

## 数据与训练身份

| 数据集 | 分辨率 | 论文 train/val/test 对数 | 公开README对应训练权重 |
|---|---:|---|---|
| WHU-OPT-SAR | 5 m | 5600 / 700 / 700 | GDROS-whu-opt-sar512.pth |
| OS | 1 m | 2011 / 238 / 424 | GDROS-os512.pth |
| UBCv2 | 0.5 m | 3517 / 1437 / 1447，作者从7170对筛除后划分 | README未列出 |

论文为三个数据集分别列出训练、验证、测试划分；README明确两个公开模型分别在WHU、OS上训练。当前没有直接核验UBCv2训练checkpoint，不以其他数据集的权重泛化结果冒充论文UBCv2训练结果。

用户本轮指定M4全程使用OS step80K，因此WHU是M4跨数据集测试对比GDROS目标数据集权重，UBCv2在缺少对应官方权重时为两个OS模型的迁移测试。此实验设计与论文各数据集训练后评测的设定不同，报告须列出训练来源。

## 测试生成与重复次数

- 统一图像512×512。WHU/OS：平移[-30,30] px、步长1；缩放[0.8,1.2]、步长0.05；旋转[-20,20]°、步长1。
- UBCv2单独采用：平移[-15,15] px、步长1；缩放[0.9,1.1]、步长0.05；旋转[-10,10]°、步长1。
- 论文IV-B明确使用三组不同seed的随机仿射测试集，最终mean±std跨三组计算。
- 作者公开一个现成测试组时可以回放该组，但不能声称重现三组均值/标准差。不能用逐图误差标准差替代跨组三次标准差。
- UBCv2原始数据服务于建筑检测/分类，原始测试图像不自动等于GDROS作者筛选并生成变换与GT的注册测试集；作者1447对ID与GT、变换生成细节需要单独核验。

## 论文表格参考值

以下全部是论文原文GDROS行，仅作协议/趋势参考，**不是本轮重新运行的结果**。

| 数据集 | CMR@1 | CMR@2 | CMR@3 | CMR@5 | All AEPE | All RMSE（原文标签） |
|---|---:|---:|---:|---:|---:|---:|
| WHU，Table I | 72.05±1.06% | 96.86±0.65% | 未列 | 99.57±0.11% | 0.90±0.04 | 0.62±0.15 |
| OS，Table II | 33.88±0.73% | 80.19±1.77% | 未列 | 99.45±0.11% | 1.48±0.02 | 0.75±0.02 |
| UBCv2，Table III | 未列 | 14.89±0.38% | 37.08±0.04% | 72.50±0.42% | 4.49±0.00 | 11.97±0.31 |

论文还列出每个阈值成功子集的AEPE，UBCv2表的主要阈值是2/3/5而非1/2/5。新评测保留全部1..5指标，避免遗漏对应列。

## 公开代码与论文公式差异

论文定义每像素二维向量L2误差后求均值。实际公开test.py中prediction维度为[1,2,H,W]、GT为[2,H,W]，`sum(dim=0).sqrt()`在batch=1时计算每分量绝对误差，再对通道/像素求均值。

- 公开代码口径：mean((|dx|+|dy|)/2)。
- 标准向量EPE：mean(sqrt(dx²+dy²))。
- 两套结果分表；无法用常数比例从一个转换成另一个。先真实张量parity，再报告本轮结果。
- 公开`Registration_index_calculation`以逐对误差严格`<T`判成功，并对成功子集计算均值、方差。
- 公开所谓`RMSE`调用`np.var`，是方差。报告注明代码原标签，另列真实标准差；不把该方差当以px为单位的RMSE。
- 官方主流程32轮推理、post-LSR、Npoint=200000；全图无mask。每个数据集仍需核对实际loader和数据格式。

## 原始来源

1. [论文全文，arXiv 2511.00598v1](https://arxiv.org/html/2511.00598v1)，2025-11-01提交，IV-A、IV-B、IV-C.3，Tables I–III。
2. [出版DOI](https://doi.org/10.1109/TGRS.2025.3627132)。Crossref返回题名一致；Semantic Scholar返回相同DOI与arXiv ID。
3. [GDROS官方仓库](https://github.com/Zi-Xuan-Sun/GDROS)，Data Preparation、Models、test.py。
4. zzvr已审官方源码commit：ee6b6216fe780bc03bfb4759ae094daa0cfddc60；本次在线main README关于数据与模型的声明与其一致。

本轮执行合同：[E092B/C](../experiments/E-092BC-WHU-UBCV2-FROZEN-OS-M4-OFFICIAL-EVAL.md)。本次OS已完成公开单组：[E092A总结](OS_OFFICIAL424_E092A_SUMMARY_20261115.md)。
