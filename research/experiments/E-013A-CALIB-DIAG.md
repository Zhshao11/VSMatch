# E-013A-CALIB-DIAG：Calibration / Ranking Diagnostic

## 目的

E-013A 训练链路干净，但正式 dev G1 因 over-abstention 失败。本诊断回答：

> E-013A 是阈值校准失败，还是模型本身没有可用 ranking/purification 信号？

本诊断只读 E-013A checkpoint、outputs、labels 和 dev/train split；不训练、不改 checkpoint、不启动救援。

## 输入

- E-013A checkpoint：只读。
- E-013A formal train/dev candidate outputs 与 inference traces：只读。
- E-011A top128 candidates：只读，仅用于 parity 或补充 score collection。
- raw truth_flow：仅用于 train/dev diagnostic label join 和指标计算。
- 禁止：datum、OS424、confirm、测试集 GT。

## 必做检查

### P0：parity/provenance

- 复现 E-013A G1：
  - active `0.658/0/0/2`
  - correct@2px `0.017/0/0/0`
  - precision `1.246%/0/0/0`
  - coverage `0%`
- E-013A checkpoint SHA、E-011A input SHA、split SHA、candidate-before-label 顺序 PASS。
- no training / no checkpoint write / no result overwrite PASS。

P0 FAIL 时停止，状态 `COMPLETED / DIAG_PARITY_FAIL`。

### D1：score separability

Train/dev 分别报告：

- candidate-level AUROC、AP、AUPRC；
- Brier、ECE；
- positive/negative score quantiles：P1/P10/P50/P90/P99；
- positive rate、per-pair positive count P10/P50/P90、zero-positive pair count。

### D2：threshold sweep

在 train/dev 分别扫：

- threshold uniform grid `[0,1]`，至少 101 个点；
- score quantile thresholds：top `0.1/0.5/1/2/5/10/20%`；
- 对每个阈值报告 active count、correct@2px count、precision、recall、coverage、condition proxy；
- 若已有 affine evaluator 可安全复用，报告 affine AEPE/CMR proxy；否则明确 N/A。

注意：dev 最优阈值只能作诊断，不能追认 E-013A 成功。

### D3：topK/quota sweep

无需阈值，按 score 选：

- per-pair global topK：K=`8/16/32/64/128/256`；
- per-source best candidate 后 global topK；
- per-region quota：每 8x8 region quota=`1/2/4/8`；
- 报告 active count、correct@2px、precision、coverage、recall。

### D4：score source ablation

分别报告：

- endpoint logit only；
- basin logit only；
- final score；
- residual feedback 前；
- residual feedback 后。

### D5：诊断判读

按以下规则给唯一主判读，可附 secondary：

- `CALIBRATION_ONLY`：dev threshold/quantile sweep 满足 E-013A G1，而 train threshold 不满足。
- `RANKING_SIGNAL_PRESENT`：dev topK/quota 至少满足 precision mean `>=20%`、correct mean `>=24`、
  coverage `>=90%` 中两项，且显著超过 E-012A。
- `TRAIN_DEV_SHIFT`：train sweep/topK 可满足 G1，dev 明显失败。
- `NO_USABLE_RANKING_SIGNAL`：dev threshold/topK/quota 均无法接近 G1。
- `LABEL_OR_BASE_RATE_LIMITED`：per-pair positives 不足或极端失衡，使 G1 在相当比例 pair 上不可达。

## 禁止

- 不训练或 finetune。
- 不修改 E-013A checkpoint、formal results、baseline outputs。
- 不把 threshold/topK sweep 作为正式成功。
- 不启动 E-013B、descriptor retraining、完整 sparse affine feedback、OS424/confirm 或 rescue。

## 产物

- `RESULT_CARD.md`，不超过 100 行；
- `diagnostic_summary.json`；
- `threshold_sweep.json/csv`；
- `topk_quota_sweep.json/csv`；
- `score_separability.json`；
- reviewer conclusion；
- RUN_MANIFEST；
- watchdog log/report。

完成后只回传 RESULT_CARD、diagnostic_summary、reviewer、run root。等待方法决策。
