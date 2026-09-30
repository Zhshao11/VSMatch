# E-011A Baseline Isolation Audit

Conclusion: PASS before formal watchdog submission.

- GDROS baseline source, checkpoint, baseline caches, and prior run/results are read-only inputs for this experiment.
- E-011A code is isolated under `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e011a`.
- E-011A checkpoints, logs, summaries, reviewer artifacts, and provenance are written only under `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-011A/run-01`.
- The train/eval path loads raw optical/SAR images from the official extracted training archive through `record["files"]["opt"]["path"]` and `record["files"]["sar_warped"]["path"]`.
- The E-008A cache is used only for fixed S0 source proposals, S1 matched-region union identity, labels, and evaluation join. It is not used as raw/structure/local-tail descriptor input, teacher score, fallback, or verifier-after-frozen-top64.
- No GDROS backbone or FeatureTransformer parameters are trained or loaded by the E-011A retriever.
- The learned score is computed before top64 inside the matched S1 union; no hidden 4096x4096 full-image blind retrieval is materialized.

Evidence paths:

- Implementation: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e011a/retriever.py`
- Train entry: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e011a/train.py`
- Eval entry: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e011a/evaluate.py`
- Run manifest: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-011A/run-01/provenance/RUN_MANIFEST.json`
