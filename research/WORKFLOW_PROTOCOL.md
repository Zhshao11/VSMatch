# 图像匹配研究低 Token 工作流

版本：`2.0`

生效日期：`2026-08-05`

## 1. 目标与优先级

目标是在降低 Codex、子 agent 和跨任务沟通 Token 的同时，保留方法实现完整性、独立审阅、
可复现性和失败审计。降低 Token 不得放宽预注册 gate 或掩盖实现偏移。

冲突优先级：用户当前明确指令 > 已冻结实验规范及其 SHA > 创新契约 > 本工作流 > 一般流程建议。

当前已经冻结并执行过的实验规范不得因采用本工作流而改变 SHA 或语义。

## 2. 单一信息源

每轮默认只读取：

1. `research/CURRENT_STATE.md`
2. `research/METHOD_DELTA.md`
3. 已完成实验的 `RESULT_CARD.md`；实验未完成则读取当前实验单

除非出现红旗，不默认重读历史对话、历史方法规范、全部报告、完整日志/CSV/JSON、源码快照，
或 SHA 未变化且已 APPROVE 的模块。

文件职责：

- `CURRENT_STATE.md`：当前方法、实验状态、唯一下一步，最多 100 行。
- `METHOD_SPEC.md`：完整方法语义，冻结后不修改。
- `METHOD_DELTA.md`：本轮相对上一版的变化，最多 150 行。
- `INNOVATION_CONTRACT.yaml`：创新点与机器检查合同。
- `RESULT_CARD.md`：关键结果、gate、结论和产物路径，最多 100 行。
- `REVIEW_PACKET.md`：reviewer 的精简输入，最多 150 行。
- `RUN_MANIFEST.json`：规范、源码、数据、checkpoint 和配置 SHA。
- `EXPERIMENT_INDEX.md`：只记录状态和结果路径。
- `TASKS.md`：只记录任务归属。
- `DECISION_LOG.md`：追加式方法决策记录。

完整证据只保留路径和 SHA 引用，不在多个文件复制。规范自身 SHA 不写回规范文件；统一记录
在 `RUN_MANIFEST.json` 或外部索引。

## 3. 方法任务

只负责方法假设、创新判断、增量规范、创新契约、验收/停止条件、最终结果解释和分支停止。

不得重复审阅实验任务已 APPROVE 的完整实现，不默认读取源码和原始产物，不在实验结束后
自动设计或派发新版本，不经用户批准进入新分支。

一次派工必须包含：版本/实验号、`METHOD_DELTA.md`、`INNOVATION_CONTRACT.yaml`、关键输入输出
与坐标尺度、禁止模块、自动验收测试、资源/精度 gate 和停止条件。只有会改变方法语义的歧义
才返回方法任务绑定。

## 4. 创新契约

每个创新点的 YAML 至少包含：

```yaml
claim: 要解决的问题
must_implement: []
must_not_implement: []
implementation_hooks: []
observable_evidence: []
acceptance_tests: []
ablations: []
stop_conditions: []
```

训练型方法额外验证梯度、loss 反传、toy overfit、数据隔离和 checkpoint 参数。

等价后端额外验证只读冻结 reference、离散轨迹一致、预注册浮点容差，并禁止减预算、近似计算
和跨样本结果缓存。

## 5. 自动验证优先

固定顺序：

```text
静态检查 -> 单元/边界测试 -> synthetic/oracle -> reference differential
-> 定向人工审阅 -> watchdog 正式实验
```

机器检查至少覆盖 shape、坐标/单位、hypothesis/candidate 数量、排序/tie-break、GT/mask/标签隔离、
禁用模块/fallback、baseline parity、finite/abstention/失败原因、显存/同步计时和 provenance/SHA。

自动 gate 失败先修复，不得用人工解释绕过。

## 6. 低 Token 审阅

默认只审本轮增量。已 APPROVE 且 SHA 不变的文件不重审。

reviewer 默认只读取 `REVIEW_PACKET.md`、本轮 diff、高风险函数和自动测试结果，不读取完整对话
或全部历史。子 agent 使用最小上下文或无历史上下文。

第一 reviewer 检查方法语义、创新契约、坐标/GT/梯度/指标以及 checker 覆盖。

第二 reviewer 仅在冻结实验单要求、新训练/梯度、自定义 CUDA/Triton、新指标/数据协议、第一
reviewer 发现高风险，或 benchmark 可能受缓存/异步/错误汇总影响时启动。

修复后只复核相关 diff。输出只保留 findings、结论和证据路径。

`REVIEW_PACKET.md` 包含版本、规范 SHA、改动文件、规范到代码映射、高风险函数、自动测试、
未解决风险、精确 diff 和产物路径。

## 7. 实验任务

一次执行：

```text
规范映射 -> 实现 -> 自动测试 -> 定向审阅 -> 修复复核 -> watchdog -> RESULT_CARD
```

一般实现决策不返回方法任务。长实验交 watchdog，不持续轮询；只在异常退出、长期无进展、
资源冲突、gate 完成、产物缺失或 provenance 不一致时介入。

完成后只发送结果卡，不在任务消息粘贴完整日志。

## 8. 结果卡与红旗

`RESULT_CARD.md` 固定包含实验/版本、最终状态、规范/源码/数据/checkpoint SHA、baseline parity、
自动测试、review、精度/资源、各 gate、失败原因、创新契约状态、绝对路径、能/不能证明什么，
以及不自动执行的下一步建议。

方法任务默认只读结果卡、gate JSON 和 review 结论。仅以下红旗触发完整审计：baseline parity
失败、review 非 APPROVE、自动测试与正式结果矛盾、provenance/SHA 不一致、指标异常、使用
未批准模块、证据无法解释，或负面结果将否定核心创新点。

## 9. 停止规则

每个方法分支最多：1 次主实验、1 次实现错误修复、1 次预注册救援。达到上限必须返回用户，
不得自动创建连续 R2/R3/R4。

```text
gate 失败 -> 当前实现或分支停止
资源通过但精度失败 -> 当前选择器/方法停止并完成失败审计
精度和资源均通过 -> 阶段成功，等待用户批准下一阶段
```

没有用户明确批准，不得进入新方法版本、descriptor、训练模块、新数据集、局部精修或新救援。

## 10. Token 规则

- 不重复读取完整历史对话，不在多文件复制相同方法说明。
- 不默认输出完整日志和大 JSON；使用 `rg`、精确行段和结构化摘要。
- shell 输出限制到完成判断所需范围。
- 子 agent 不继承无关历史，不让多个 agent 重复审同一文件。
- 方法任务不重复实验任务已完成的实现审阅。
- 证据一致时直接接受；仅红旗触发深读和额外 reviewer。
