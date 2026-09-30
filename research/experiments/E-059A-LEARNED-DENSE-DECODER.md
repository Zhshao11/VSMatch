# E-059A Learned Dense Decoder From Top-K Correlation Anchors

Status: `ENGINEERING_STARTED`

Frozen E046A-R2 frontend produces a full 1024-target distribution. E-059A
learns decoder-side anchor assignment, matchability and residuals from top-K
anchors, with independent RoMa-like probability/residual, DKM-like kernel
certainty, and sparse recurrent distribution-feedback tracks. Baseline and
frozen checkpoints are read-only.
