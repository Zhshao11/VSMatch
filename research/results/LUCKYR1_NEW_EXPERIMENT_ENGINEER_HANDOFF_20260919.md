# LuckyR1 新实验工程交接

更新时间：2026-09-19 20:09（Asia/Shanghai）<br>
模式：普通实验工程对话；禁止创建或依赖 Codex Goal / heartbeat。

## 1. 当前结论

LuckyR1 的 current-model exploratory A/B/C 已全部完成成本门，没有任何 arm 可以替换受保护的 E075A-M4。

clean lineage 已完成 split-safe F2/F3、clean B/C 基座选择、M4 raw control、clean-v3 G0 和 Lucky-A。Lucky-B/C 虽然已有有效 checkpoint 与 201-dev 指标，但遭遇外部 GPU 冲突和多次 SIGTERM，orchestrator 的 8 次 attempt 已耗尽，导致 `lucky_reduce` 与 clean official-404 finalizer 尚未执行。

当前不应从零重训。下一步应先冻结 Lucky-B step2000 与 Lucky-C step10000 的有效结果，按成本门判断是否停止，然后补 `lucky_reduce`。只有 clean arm 相对 clean M4 raw control 出现可信信号，才允许进入 untouched official 404 finalizer。

## 2. 远端基础设施

- SSH：`ssh zzvr`
- 专用 watchdog：`/mnt/sda/zhushihao/experiment_watchdog_luckyr1`
- 父实验：`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN`
- 远端代码：`/mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated/.codex_tmp`
- watchdog daemon PID：`2174727`，健康。
- orchestrator PID：`2324644`，健康；2026-09-19 20:09 heartbeat 正常推进。
- watchdog 当前计数：running=0、queued=0。
- GPU 策略：始终 `candidate_gpus=null`，由 watchdog 选择真正无 compute PID 的独占卡；进程内只使用逻辑 `cuda:0`。

重要：基础设施健康不等于 DAG 在推进。当前 `lucky_b`/`lucky_c` attempts 已耗尽，依赖链停在 `lucky_reduce` 前。

### 已修复的 orchestrator 故障

旧 orchestrator PID `2631514` 因写入不存在的 `collisions/` 目录触发未捕获 `FileNotFoundError` 而死亡。当前版本启动时创建 `collisions/` 和 `attempt_cards/`，并为退休任务补终态 attempt 卡。

本地对应文件：

- `research/remote_stage/luckyr1/luckyr1_orchestrator.py`
- `research/remote_stage/luckyr1/lucky_plan.json`

## 3. 不可违反的实验边界

- 不覆盖 E075A-M4；其 SHA256 必须保持：`55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b`。
- current-model exploratory 404 不是 clean model-unseen 结果。
- clean 1810/201 结果只能与 clean M4 raw control 比较，不能与历史 E075A-M4 混比。
- 指标为 pair-level：先算每个 pair 的 dense mean EPE，再统计 AEPE 和 CMR@1/2/5/10/16。
- prediction 必须先完整物化，再单独做 GT join；不得在 inference 阶段使用隐藏 GT。
- 成功、失败、OOM、SIGTERM、kill、超时和协议阻塞都必须保留 JSON/Markdown RESULT_CARD。
- 不删除旧 attempt，不复写正式 checkpoint/结果。
- WHU 仍未启动，在 OS clean 收口前继续禁止启动。

## 4. 方法实现现状

### B：geometry-aligned intra-modal relation correlation

- 作用路径：真实 `CorrBlock -> relation evidence -> BasicUpdateBlock`。
- raw correlation 作为加性基底保留。
- 官方 CorrBlock 的 81 通道候选顺序、坐标轴和金字塔尺度已核对。
- target offset 使用当前映射的局部 Jacobian 对齐。

### C：relation + sparse multimode geometry belief

v1 C 被否决：它把 81 个候选压成单一全局期望，再平滑 expected flow，违背“保留分离模式与候选间几何兼容”的设计。

v2/v3 改为：

- 保留多个绝对 target modes；
- 保留 unmatched mass；
- 对 cardinal source neighbors 做候选到候选的 Jacobian 几何消息；
- 跨轮状态使用绝对 target 坐标并 detach；
- raw correlation 仍为加性基底。

主要文件：

- `research/remote_stage/luckyr1/luckyr1_models_v3.py`
- `research/remote_stage/luckyr1/luckyr1_pipeline_v3.py`
- `research/remote_stage/luckyr1/luckyr1_historical_v3_g0.py`
- `research/remote_stage/luckyr1/luckyr1_clean_v3_g0.py`

### v2 零梯度根因与 v3 修复

历史 E074-B 的 `teacher_preds()` 原本把 F2/F3 matcher 包在 `torch.no_grad()` 中。这对原冻结 teacher 合理，但在注入 trainable LuckyR1 evidence 后会断开 4 个 relation 参数。

v3 只移除了 evidence 所在 matcher 调用的 `no_grad`；所有 base 参数仍为 `requires_grad=False`。

- historical-M4 v3 G0 job：`20260915_093624_331c5869`，成功。
- B：6/6 trainable tensors present、finite、nonzero。
- C：21/21 trainable tensors present、finite、nonzero。
- 峰值 reserved memory：B 14820 MB，C 14828 MB。
- clean full-M4 v3 G0 job：`20260916_085312_87f76f3c`，成功；A/B/C 均 finite 且梯度通过。

## 5. Current-model exploratory 结果

范围：历史 official-404 view；不是 clean model-unseen。

| Arm | 状态 | AEPE | CMR@1 | CMR@2 | CMR@5 | CMR@10 | CMR@16 |
|---|---|---:|---:|---:|---:|---:|---:|
| Protected M4 | control | 2.3613 | 56.6832% | 78.9604% | 88.8614% | 94.8020% | 97.0297% |
| A step500 | COST_GATE_STOP | 6.1013 | 25.7426% | 54.7030% | 71.2871% | 79.9505% | 86.8812% |
| B step500 | COST_GATE_STOP_NO_SIGNAL | 2.3617415 | 56.6832% | 78.9604% | 88.8614% | 94.8020% | 97.0297% |
| C step500 | COST_GATE_STOP_NO_SIGNAL | 2.3615115 | 56.4356% | 78.9604% | 88.8614% | 94.8020% | 97.0297% |

B/C 各有 404 prediction、404 manifest 行和 404 metric-join 行，均通过 prediction-before-GT。没有 arm 有长训信号。

本地结果：

- `research/results/LUCKYR1_CURRENT_MODEL_EXPLORATORY_A_STEP500_RESULT.md`
- `research/results/LUCKYR1_CURRENT_MODEL_EXPLORATORY_B_STEP500_RESULT.md`
- `research/results/LUCKYR1_CURRENT_MODEL_EXPLORATORY_C_STEP500_RESULT.md`
- `research/results/LUCKYR1_CURRENT_MODEL_V3_G0_RESULT.json`

## 6. Clean 基座与 raw control

clean safe F2/F3 均已训练到 120k。

clean B/C 在 step117000 同时 rc137，结束时间同为 `2026-09-14 16:31:54+08:00`。可访问 journal 中没有 OOM、GPU Xid、reboot 或 shutdown 证据。共同时间戳支持共享事件，但不能归因给任何 actor。

按预声明顺序 `CMR@5 -> CMR@2 -> CMR@1 -> AEPE`：

- clean B 选择 step80k：AEPE 3.665468，CMR@1/2/5/10/16 = 18.9055/51.2438/77.1144/93.5323/97.5124%。
- clean C 选择 step40k：AEPE 3.817065，CMR@1/2/5/10/16 = 18.40796/39.3035/77.1144/93.5323/97.5124%。
- `bc_reduce` job：`20260915_093640_13a1739d`，成功。

clean M4 raw control 已完成 120k，选择 step120k：

- checkpoint SHA256：`0b0c248ca66b20c8b60ef36f131417f2ee573f6385586d97d0e4a4bb874bf8b2`
- AEPE 3.3589399
- CMR@1/2/5/10/16 = 22.8856/55.2239/79.1045/94.0299/98.5075%
- `m4_reduce` job：`20260916_085257_db358748`，成功。

## 7. Clean Lucky A/B/C 当前结果

所有数值均为 clean 201-dev，必须只和上一节的 clean M4 raw control 比较。

### Lucky-A

- job：`20260916_085359_a46af229`，成功完成 20k。
- step2000/5000/10000/20000 的所有 CMR 与 raw control 完全一致。
- reducer 顺序下选择 step2000，AEPE 3.3587850；相对 raw 仅约 0.000155 px 改善，没有 CMR 信号。
- checkpoint SHA256：`827dd8fbed10797f1b6642572409ca5df7cc9d6c6290b06a0cb9d3b0c384c401`。

### Lucky-B

- 8 次 attempt 已耗尽；orchestrator 状态 `failed`。
- 有效 checkpoint：step2000，SHA256 `3c01d0f2cb65509c430517d2410d68da8963e2ffcdff7a1667aaafe1443ff386`。
- step2000：AEPE 3.3589616；所有 CMR 与 raw control 完全一致，AEPE 略差。
- 第一次有效 attempt 到 step3375 后因外部 GPU 冲突 SIGTERM；第二次也有明确 collision。后续多次 rc2 卡均记录 `InterruptedError(signal 15)`，但没有足够证据统一归因。

### Lucky-C

- 8 次 attempt 已耗尽；orchestrator 状态 `cancelled`。
- 有效 checkpoint：step2000、5000、10000。
- step10000 最优：AEPE 3.3587790；所有 CMR 与 raw control 完全一致，仅约 0.000161 px AEPE 改善。
- step10000 SHA256：`44256d6d0eec207b64fa7726152917a4c69d289d1c79fe770c964231696a0788`。
- 首次 attempt 到 step11423 后被外部 GPU 冲突取消；最后 attempt `20260916_164053_dd28c56a` 也有明确 collision。其余快速 SIGTERM 不应无证据归因。

## 8. 当前阻塞与建议接管动作

当前状态：

- `lucky_reduce`：waiting，未提交。
- clean official-404 `finalizer`：waiting，未提交。
- watchdog：running=0、queued=0。
- orchestrator：健康，但 B/C attempts 已耗尽，所以不会自行推进。

建议按以下顺序处理：

1. 不要直接把 B/C max_attempts 从 8 调大并重训。先为 B step2000、C step10000 生成终态成本门卡，保留所有旧 attempt。
2. 核验两个 checkpoint SHA、201 prediction/manifest/join 数量与 `prediction_materialized_before_gt_join=true`。
3. 以 raw M4 step120k 为唯一 clean control。B 无信号；C/A 只有约 `1e-4 px` AEPE 微差且 CMR 完全不变，应按成本控制优先判为无实质信号，除非方法决策明确给出继续标准。
4. 新建 CPU finalizer task，而不是伪造 `lucky_b/lucky_c=succeeded`。让新的 reducer 依赖 finalizer task，并从已有有效 cards/checkpoints 选取 A/B/C。
5. 如果三臂均判 no-signal，则不要运行 untouched official 404 finalizer；写明 clean method gate stop，并保持 E075A-M4 为当前最佳。
6. 如果方法决策明确认为微小 AEPE 改善足以进入 finalizer，必须先说明统计与成本依据，再只对预声明 winner 运行一次 untouched official 404。
7. 继续保持 WHU 停止。

## 9. 结果卡与代码索引

- 最新已完成恢复卡：`research/results/LUCKYR1_RECOVERY_20260915_RESULT_CARD.md`
- 当前 handoff：`research/results/LUCKYR1_NEW_EXPERIMENT_ENGINEER_HANDOFF_20260919.md`
- clean B/C reducer：`research/results/LUCKYR1_CLEAN_BC_REDUCE_RESULT.json`
- protected M4/current best：`research/results/CURRENT_BEST_OS_METHOD_20260913.md`
- 远端每个 task 的权威卡：`<parent>/clean_1810_201/<task>/RESULT_CARD.json`
- 远端 scheduler attempt 卡：`/mnt/sda/zhushihao/experiment_watchdog_luckyr1/attempt_cards/<task>/<job>/`
- GPU collision evidence：`/mnt/sda/zhushihao/experiment_watchdog_luckyr1/collisions/`

## 10. 新工程接管时的首轮检查

```bash
ssh zzvr
cd /mnt/sda/zhushihao/experiment_watchdog_luckyr1
ps -fp "$(cat orchestrator_launcher.pid)"
python3 - <<'PY'
import json
s=json.load(open('lucky_orchestrator_state.json'))
print(s['heartbeat'], s['pid'])
for k in ['m4_raw','m4_reduce','lucky_clean_v3_g0','lucky_a','lucky_b','lucky_c','lucky_reduce','finalizer']:
    print(k, s['tasks'].get(k))
PY
```

不要因为 orchestrator heartbeat 正常就假定 DAG 会继续；必须显式处理 B/C exhausted state。
