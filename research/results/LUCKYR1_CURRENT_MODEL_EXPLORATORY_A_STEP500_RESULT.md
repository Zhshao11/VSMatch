# LuckyR1-A current-model exploratory result

Status: `COST_GATE_STOP`

LuckyR1-A was connected to the real historical M4 B/C recurrent matchers and trained for 500 steps. This is the current-model exploratory track, not the clean 1810/201 model-unseen lineage.

| Output | AEPE | CMR@1 | CMR@2 | CMR@5 | CMR@10 | CMR@16 |
|---|---:|---:|---:|---:|---:|---:|
| LuckyR1-A step500, iter24 post-LSR | 6.1013 | 25.7426% | 54.7030% | 71.2871% | 79.9505% | 86.8812% |
| Protected E075A-M4 step80K | 2.3613 | 56.6832% | 78.9604% | 88.8614% | 94.8020% | 97.0297% |

The cost gate stopped A because AEPE was greater than 1.25 times the control and CMR@5 did not improve.

All 404 prediction tensors were persisted before GT metric joining. The prediction manifest SHA256 is `89dfa7b54a17f6082abc4bc514edaf89406e5b33ae2fa2185664e29532afa332`; the metric join SHA256 is `d07f1287a7e3443e5d000054e22e17f870d1aada05e24eaf652ba07437f579ae`.

This result cannot replace the protected E075A-M4 record and must not be reported as model-unseen.
