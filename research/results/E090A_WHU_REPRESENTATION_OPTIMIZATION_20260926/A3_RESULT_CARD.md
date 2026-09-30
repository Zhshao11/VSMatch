# E090A A3 optimization ablation

Status: `COMPLETE_DIAGNOSTIC`

Candidate-only optimization improved CE substantially compared with raw E089 joint training, and removing the clamped scalar allowed the oracle gate to overfit. Candidate top1 nevertheless missed every A1 threshold, so representation remains the primary blocker. The candidate-to-gate staged arm was gate-blocked by A1 and not run.
