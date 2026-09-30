# E-008A-STAGE-DIAG targeted review R1

Conclusion: `CHANGES_REQUIRED`

- Formal mode did not enforce exact dev404 and could emit a non-final subset as completed.
- A runtime invariant made `DEDUP_LOSS_DIAGNOSED` unreachable instead of detecting an unexpected loss.
- S0 candidate base rates and explicit source token xy were missing.

The reviewer approved the real E-008A stage mapping, GT-after-candidate join order, stride-8
coordinates, final candidate construction, and conservative conditional-plus-absolute attribution.

Resolution pending R2: add explicit smoke/formal mode, enforce formal dev404, retain the expected
dedup invariant as a test while permitting the diagnostic to measure a violation, and add the
missing fields.
