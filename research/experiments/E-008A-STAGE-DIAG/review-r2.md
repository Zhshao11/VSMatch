# E-008A-STAGE-DIAG targeted review R2

Conclusion: `APPROVE`

R1 findings are closed:

- formal mode requires exact dev404 and rejects `--max-pairs`;
- an unexpected S4-to-S5 recall loss remains measurable and attributable, while the normal
  target-id dedup recall invariant is unit-tested;
- S0 candidate base rates and explicit source token xy are persisted.

Evidence: remote `6 passed`; formal subset rejection passed; 2-pair S5 counts exactly match the
frozen E-008A P0 pair records.
