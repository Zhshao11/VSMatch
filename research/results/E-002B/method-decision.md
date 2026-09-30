# E-002B Return To Method Task

日期：`2026-08-04`

实验任务结论：`G5-R FAIL -> EFFICIENCY_HOLD`

按照 M001 v0.4 的预注册决策树，官方 424-pair accuracy 未启动，E-002C 未启动。当前实验
任务停止在方法决策点，不自行增加 B1/B3/learned descriptor、WLS/IRLS、局部精修或 baseline
fallback。

G5-R 失败不是隐藏 N2 或显存失败：K128 score count、K-scaling memory、峰值显存和禁用模块
检查均通过。失败由固定 B2 geometry search 的绝对延迟、跨进程 CV 和 timing-linear R2
造成；hypothesis generation 与 coarse consensus 是主要瓶颈。

方法任务下一步需要决定：保持 hold，或另立版本改变 hypothesis/coarse search 的算法预算与
实现。任何新预算、剪枝、CUDA/vectorized backend 或精度-资源折中都必须形成新规范，不能
在本轮 424 accuracy 之前事后改动。E-002B 当前固定选择器的 accuracy 尚未被测试。
