# E-006A P0 Protocol Review

Status: `APPROVE`

- Frozen specification, split, checkpoint and per-cache identities are enforced.
- Formal mode requires the exact 404-pair dev/cache set; smoke cannot report formal PASS.
- Live score traces record global `4 x [1,256,1024]`, region `8 x [1,32,512]`, and affine-local `64 x [1,32,81]`; maximum live score tensor is 262,144 elements.
- The inference source closure passes AST/forbidden-token auditing with source SHA evidence; no hidden 4096-square score tensor, dense state/GRU or K128 cap was found.
- The evaluator passed a nontrivial scale/shear/translation plus residual-truth differential against the frozen E-005A evaluator for AEPE and all geometry/PLR fields.
- Remote automatic tests passed `12/12`; formal 404 P0 was approved for watchdog execution.

Reviewer scope: provenance, budget/leakage, evaluator differential, frozen snapshots, tests, and `p0-smoke-v3.json`.
