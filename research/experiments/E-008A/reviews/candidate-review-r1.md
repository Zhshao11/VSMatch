# E-008A Candidate Generator Review R1

结论：`CHANGES_REQUIRED`

审阅范围：`M001 v1.1 / E-008A` 的 multiscale region matching、dynamic source proposals、matched-region dense-lite search、hard candidate union/dedup/source trace、no-GT/no-hidden-N2、cache identity，以及训练目标和梯度。

冻结身份：本轮最终读取的 `E-008A.md`、`REVIEW_PACKET.md`、`INNOVATION_CONTRACT.yaml` SHA 与 `spec_binding.py` 一致。本地阶段源码与 `bl_tunnel:/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e008a` 对应文件 SHA 一致。

## Findings

### F1 [P1] 冻结的候选级 trace 没有保存完整

- 规范要求每条 generated candidate 保存 source proposal id、region scale、target region id、candidate source type、score、rank、target coordinate 和 no-GT trace（`E-008A.md` 第 60-66 行）。
- `generator.py:13-19` 的单路输出只有 pool slot、candidate id、score、valid 和 source bit；`generator.py:38-55` 的 full union 也只有这些字段。`target_pixel_xy` 在 `generator.py:56-59` 补充，但 proposal id、source-specific rank、region scale、target region id 和候选级 no-GT trace仍缺失。
- `build_observations.py:120-145` 只保存 pool 与全局 region-match 张量；learned candidate 在之后才生成。`evaluate.py:86-114` 和 `evaluate.py:189-239` 仅保存逐 pair 聚合指标，不持久化可审计的逐候选 trace。全局 region tensors 和 source-type aggregate 不能替代规范要求的逐候选记录。

影响：无法验证某个 final candidate 究竟由哪个尺度/target region/source rank 进入 union，也无法对 candidate source type 与 region scale 做可靠的逐候选复核或结果分组。当前结果不能满足冻结的 trace 证据要求。

要求：在不改变 candidate IDs、排序或预算的前提下，为每个 track 的每条有效候选持久化 proposal id、各来源 rank、source bit、贡献尺度/target-region id（多尺度命中可用 mask/list）、score、hard target token/pixel coordinate及 no-GT provenance，并增加 dedup 后 trace parity 测试。

### F2 [P1] `nearby_basin` hard negative 没有实现冻结语义

- `train.py:43-50` 将 `nearby_basin` 定义为 GT endpoint error 位于 `(2, 32] px` 的错误 candidate。
- 该条件没有读取 coarse hypothesis、matched-region 邻接或任何 basin 标识；它是“GT 附近错误点”，不是规范第 73-74 行要求的“邻近 basin 错误 target”。

影响：hard-negative 分布和训练目标发生实质变化。虽然 raw flow 在 train-only mining 中允许使用，但不能用 endpoint 距离静默替代已冻结的 basin/geometry category。

要求：从 no-GT coarse region/hypothesis trace 定义并实现 neighboring-basin negative；若方法任务本意确为 `(2,32] px` endpoint band，需先正式绑定规范，不能由实现自行解释。

### F3 [P1] 自动 gate 没有执行规范要求的 toy overfit

- `test_core.py:126-138` 仅执行一次 forward/backward、检查全部参数具有有限非零梯度，并验证 state-dict save/load parity。
- 测试没有 optimizer step、没有重复拟合 toy labels，也没有断言 loss 明显下降或小样本可过拟合。因此测试名称虽覆盖 gradient/checkpoint，却没有覆盖 `E-008A.md:75` 和 contract acceptance test 要求的 toy overfit。

影响：目前不能排除创新模块虽有梯度但训练目标/优化路径无法学习的实现错误，正式负面结果不应在该 gate 缺失时用于否定方法。

要求：为 descriptor、patch/structure 和 full 三条可训练分支加入确定性 toy overfit，验证各分支 loss 明显下降，并保留现有全模块 gradient 与 checkpoint exact-load 检查。

## Passed Checks

- Multi-scale region matching 精确覆盖 `4x4/8x8/16x16`，top-L 为 `4/8/8`，exact tie 按 target-region id 稳定。
- Region descriptor 包含 frozen GDROS feature summary 与 raw/structure auxiliary mean/std；optional region adapter 未启用，符合 optional 边界。
- Dynamic proposals 由 source-only structure、local feature distinctiveness、region confidence 构成；`4x4` 每区至少 2、`8x8` 每区至少 1，threshold selection 无 semantic cap。
- Matched-region search 的 target IDs 严格来自三尺度 matched-region union。smoke 最大 union `1584`，最大 live score tensor `21,504`，没有 full-image blind `4096` search 或 `4096x4096` tensor。
- Dense-lite pool 为 hard discrete token IDs；full union 按 target token 去重并保留 source bit，最终坐标为 `token_xy * 8`，没有 soft expectation。
- Observation materialization 不读取 truth flow、datum、OS424 或 confirm；label join 在独立入口中发生且 content/time identity 绑定 observation。
- GDROS frontend 在 `torch.inference_mode()` 下显式 frozen no-grad；cache content hash 覆盖动态 proposal、region matches、pool IDs/scores/evidence 与输入/源码身份。
- Train split mining、BCE + ranking loss、全创新模块有限非零梯度和 checkpoint exact load 已实现；缺口仅为 F2 的 negative category 语义与 F3 的 toy-overfit gate。
- 远端自动测试当前为 `10/10 PASS`，但不能覆盖上述三个 finding。

## Decision

在 F1-F3 修复并完成定向复核前，不批准正式训练或将 P0/P1 负面结果用于方法结论。
