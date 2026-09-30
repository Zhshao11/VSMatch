# GDROS 轻量化研究记录

本目录是方法讨论任务、实验工程任务和后续文献更新任务之间的共享事实源。
聊天记录用于推导和交流，下面这些文件用于长期追踪，发生冲突时以最新版本文件和
明确标注的实验结果为准。

## 文件入口

- `WORKFLOW_PROTOCOL.md`：任务分工、代码审阅、看门狗运行和结果交付协议。
- `METHOD_SPEC.md`：当前方法假设、实现约束和验收门槛。
- `EXPERIMENT_INDEX.md`：全部实验的状态、结果路径和下一步动作。
- `FAILED_METHODS.md`：经过实现复核和补救后仍失败的方法。
- `LITERATURE_REQUESTS.md`：需要文献库更新任务补充的检索指令。
- `TASKS.md`：方法讨论任务与实验工程任务的 ID、职责和同步规则。
- `results/`：从远端同步回来的实验摘要、指标 JSON 和审阅报告。
- `REPRODUCIBILITY.md`：当前 E081A-E091A 方法链的数据、环境、指标和运行说明。
- `CURRENT_METHOD_STATE.md`：当前 OS/WHU 最优方法、最新实验和下一研究问题的唯一入口。
- `CHECKPOINTS.json` / `CHECKPOINTS.md`：不入库权重的路径、作用与 SHA256。
- `remote_stage/`：在 zzvr 执行的冻结方法实现与 watchdog DAG 源码。

## 目录职责

- 本地研究目录：`/Users/shihao/Documents/图像匹配之二/research`
- 远端代码根目录：`bl_tunnel:/mnt/sda/zhushihao/VS_Match/GDROS`
- 远端实验产物建议目录：
  `bl_tunnel:/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo`

不要把聊天中的临时猜测直接登记为实验结论。所有结论必须关联实验编号、代码版本、
配置、日志和结果路径。

当前最新 held-out 结论是 E091A `STOP_HELDOUT_GATE_FAILED`。它没有访问 WHU 700-test，
因此不能被描述为最终测试集上的方法提升。仓库入口和提交边界见根目录 `README.md`。
