# E074A best freeze

Frozen from E074A closed loop on 2026-09-04.

## Recommended bases

- Balanced base: `B_step180000.pt`
- High-precision base: `C_step180000.pt`
- No-GT coverage diagnostic: A no-GT chooser

## Best verified results

| branch | AEPE | CMR@1 | CMR@2 | CMR@5 | CMR@10 | CMR@16 | note |
|---|---:|---:|---:|---:|---:|---:|---|
| B 180k | 2.5053 | 44.8020 | 72.7723 | 89.3564 | 95.2970 | 97.5248 | best AEPE / balanced |
| C 180k | 2.5697 | 48.7624 | 76.9802 | 87.8713 | 93.5644 | 96.7822 | best CMR@1 / high-precision |
| A no-GT chooser | 2.6361 | 36.8812 | 67.8218 | 90.0990 | 94.8020 | 97.5248 | best CMR@5 among no-GT diagnostics |

## Frozen provenance

- E074A script SHA256: `e331845b07cd3d1ef99fa91795c3bae8493def04ff3c771148f33c1b8abd8806`
- B checkpoint SHA256: `f40d80f2dca62e74ec86c1823963282a490d47f336f2cfce4a28fc211ea33218`
- C checkpoint SHA256: `c3c67e6d65aca0dd55cf7af2c2a9eedccadadc0fec5ad11fe6e493e08ecc85b9`
- E074A run manifest SHA256: `de27c2d5d2bce55396e2ec8d56160c340cd9768e950bff21d64522eaed38425c`

## Recommendation

- Use `C_step180000.pt` when preserving CMR@1 / CMR@2 is the main priority.
- Use `B_step180000.pt` when preserving AEPE / CMR@5 balance is the main priority.
- Do not overwrite E074A; treat these as frozen baselines for E075A.
