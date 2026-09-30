# E-035D-E028B-PAPER-REFINER-STRONG-TRAINING：Resource-Corrected Refiner Training

方法版本：`M001 v3.16 resource recovery`

状态：`USER_APPROVED / READY_FOR_ENGINEERING`

## 1. 目标

E-035C 的 paper-style refiner 训练实际落在 CPU、`1` epoch、`8` tokens/pair，因此不能充分否定可训练
local refinement。E-035D 只改训练强度/资源协议，不改变方法语义：

> 在 CUDA 可用、多 epoch、更多 tokens/pair 下，验证 E-035C 的论文式 refiner 是否能超过 E-034C coarse
> reference。

## 2. 边界

- E028B safe dual adapter checkpoint 固定只读；
- 不训练/finetune E028B descriptor；
- 不接 GDROS backend adapter，不复用 GDROS dense GRU/backend；
- 允许训练 local fine encoder / local refinement head / confidence or uncertainty head；
- CUDA available 是 hard gate；CUDA 不可用则停止为 `E035D_RESOURCE_BLOCKED`；
- 允许多卡或单卡 GPU；必须记录 GPU 数、batch、accumulation、epochs、tokens/pair、时长、显存；
- primary coarse input 固定为 E-034C best affine/CMR variant `rank_growth_dual_256_irls`；
- 不使用 official testing.zip / confirm / OS424；
- selected/refined outputs 必须先落盘，再读取 truth-flow 做 metric join；
- 完成后停止，不自动启动 E035E、descriptor finetune、full sparse feedback 或 rescue。

## 3. Tracks

### A. `RESOURCE_AND_REPLAY_GATE`

必须：

- 记录 torch CUDA availability、GPU count、device names；
- 复现 E-034C/E-035C coarse reference；
- 复现 E-035A/E-035C window oracle；
- 若 CUDA unavailable，不训练，直接写 resource-blocked card。

### B. `STRONG_EFFICIENT_LOFTR_STYLE`

- 以 E-035C best track 为起点；
- epochs 至少 `8`；
- tokens/pair 至少 `64`，资源允许则 `128`；
- heatmap CE/focal + offset Huber + confidence loss；
- 保存 checkpoint，strict load 后 dev404 formal eval。

### C. `STRONG_ROMA_STYLE_RBC`

- specialized fine ConvNet from raw/structure inputs；
- regression-by-classification + robust residual；
- uncertainty/confidence head；
- epochs 至少 `8`；
- tokens/pair 至少 `64`。

### D. `UNCERTAINTY_CLEANUP`

- 使用 confidence/uncertainty/entropy 过滤或降权 refined matches；
- refined endpoints 再做 weighted affine / IRLS；
- 报告 cleanup 是否提升 AEPE/CMR。

## 4. Gates

### P0：resource/setup

必须 PASS：

- official training/testing archive SHA 与 pair count verified；
- CUDA available true；
- E028B checkpoint SHA、feature hook、feature shape recorded；
- E-034C primary coarse artifacts located/replayed；
- E-035C resource caveat recorded；
- no E028B descriptor mutation；
- no GDROS backend adapter / dense GRU reuse；
- baseline artifacts readonly；
- testing.zip / confirm / OS424 not used。

P0 fail：`E035D_RESOURCE_BLOCKED` or `E035D_SETUP_BLOCKED`。

### G0：strong training sanity

必须 PASS：

- epochs/tokens-pair meet frozen minimums or typed resource failure；
- losses finite/nonzero；
- gradients reach fine encoder/refinement/confidence heads；
- train-small overfit and train-full local endpoint error/heatmap mass improve over E-035C；
- strict checkpoint load verified；
- refined outputs written before truth-flow metric join；
- reviewer approves no leakage, coordinate mapping, resource equivalence, and baseline isolation。

G0 fail：`E035D_TRAINING_BLOCKED`。

### G1：dev404 refinement diagnosis

主对照：E-034C/E-035A coarse reference AEPE `8.550`、CMR@2 `8.996%`、CMR@5 `38.783%`、
CMR@16 `87.942%`。

判读：

- `STRONG_REFINER_GAIN_FOUND`：best refined track improves CMR@5 over `38.783%` and AEPE under `8.550`
  without CMR@16 collapse；
- `CMR5_GAIN_ONLY`：CMR@5 improves but AEPE/CMR@2 remain weak；
- `STRONG_REFINER_STILL_WEAK`：GPU strong training still cannot beat coarse replay；
- `UNCERTAINTY_CLEANUP_LIMITED`；
- `E035D_RESOURCE_BLOCKED`。

## 5. 输出

- `RESULT_CARD.md`；
- `P0.json`、`G0.json`、`G1.json`；
- `resource_gate.json`；
- `coarse_oracle_replay.json`；
- `strong_efficientloftr_metrics.json`；
- `strong_roma_rbc_metrics.json`；
- `uncertainty_cleanup_report.json`；
- `training_resource_report.json`；
- reviewer conclusion；
- `RUN_MANIFEST.json`；
- watchdog job/log/report。

完成后只回传精简结果和核心路径，等待方法任务和用户决策。不得自动启动 E035E、descriptor finetune、
full sparse feedback、OS424/confirm 或 rescue。
