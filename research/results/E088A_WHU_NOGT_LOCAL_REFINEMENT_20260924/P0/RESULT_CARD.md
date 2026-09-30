# E088A WHU no-GT local refinement P0

Status: `STOP_NO_CANDIDATE_DISCERNIBILITY`

Anchor/oracle-r2/oracle-r4/score-top1/score-top5 AEPE: `1.905052` / `0.807883` / `0.641615` / `4.968580` / `3.682175`.

Score oracle-near top1/top5/top8: `1.076%` / `4.974%` / `7.985%`.

Gate: `{'capacity_oracle_r4_gain_ge_0_5px': True, 'score_top1_near_ge_2x_random': False, 'score_top5_near_ge_2x_random': False, 'score_top5_candidate_aepe_improves_anchor_by_0_1px': False, 'score_top1_does_not_worsen_anchor': False}`. Next stage: `STOP_BEFORE_P1`.

Frozen E081A-M4 and backbone; prediction-before-GT enforced. No LuckyR2 or local head training was run in P0.
