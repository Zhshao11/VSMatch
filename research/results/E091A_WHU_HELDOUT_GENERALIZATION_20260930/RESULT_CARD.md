# E091A WHU Held-Out Generalization

Status: `STOP_HELDOUT_GATE_FAILED`

E091A trained from the protected E081A-M4 plus a zero-initialized E090D adapter
on 200 source groups (1600 pairs). Candidate/evidence training used 6400 pair
presentations and recurrent training used 4800. Model selection used a disjoint
100-group, 800-pair dev set in the preregistered order
`CMR@1 -> AEPE -> CMR@2 -> CMR@5`.

The selected recurrent epoch 3 changed pair-level metrics as follows:

| Method | AEPE | CMR@1 | CMR@2 | CMR@5 | >=5 px failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen E081A-M4 | 2.480299 | 22.500% | 69.875% | 97.125% | 23 |
| E091A main | 2.465468 | 24.750% | 70.000% | 97.375% | 21 |
| Shuffled evidence | 2.465170 | 24.500% | 69.625% | 97.375% | 21 |
| Zero evidence | 2.995292 | 8.250% | 50.500% | 95.875% | 33 |

CMR@1 improved by 2.25 pp and AEPE improved by 0.01483 px. The bootstrap 95%
CI for AEPE improvement was `[-0.00809, 0.03618]`, and the main result was
0.00030 px worse than shuffled evidence. The preregistered CMR@1, AEPE,
bootstrap, and evidence-specificity gates therefore failed. Adapter-off parity
was exact (`max_abs=0`).

WHU 700-test was not accessed, no full-data expansion was started, and no
existing checkpoint was overwritten. Full pair-level artifacts remain on the
remote experiment host and are intentionally excluded from Git.
