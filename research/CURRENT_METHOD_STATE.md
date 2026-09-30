# Current Method State

Updated: 2026-09-30 (Asia/Shanghai)

This file is the canonical starting point for a new researcher or Codex task.
Result cards and source snapshots remain authoritative for exact metrics and
hashes; this page states which results are current and which experiments did
not replace them.

## Current protected methods

### OS: E075A-M4 Region Gate

- Checkpoint: `M4_step80000.pt`
- Inference output: `iter24_post_lsr`
- Pair-level metrics on the historical 404-pair protocol:
  - AEPE: `2.3613`
  - CMR@1/2/5/10/16: `56.6832 / 78.9604 / 88.8614 / 94.8020 / 97.0297%`
- Main advantage: substantially higher CMR@1 than the released GDROS replay.
- Main weakness: model-unseen generalization and the error tail beyond 5 px.
- Source: `remote_stage/os/e075a_preserve_high_precision_recover_c2c5.py`
- Detailed snapshot: `results/CURRENT_BEST_OS_METHOD_20260913.md`

E075A-M4 remains the current OS method. LuckyR1, E075 diagnostics, and later
candidate generators did not produce a valid replacement.

### WHU: E081A-M4

- Checkpoint: `M4_step120000.pt`
- Pair-level metrics on the 700-pair WHU test protocol:
  - AEPE: `1.9055`
  - CMR@1/2/5/10/16: `28.0000 / 72.0000 / 98.1429 / 99.5714 / 99.5714%`
- Main strength: nearly all pairs are within 5 px.
- Main weakness: fine localization below 1-2 px.
- Source: `../e081a_whu_our_native_train.py`
- Diagnostic: `results/E081A_M4_ISOMORPHIC_DIAGNOSTIC_20260921/RESULT_CARD.md`

E081A-M4 remains the current WHU method. No E088-E091 experiment replaced it.

## Latest experiment: E091A

E090D first demonstrated an identity-safe late recurrent residual path on
1/4/8 training pairs. E091A then tested that path at a meaningful scale:

- Train: 200 source groups / 1600 pairs.
- Dev: 100 disjoint source groups / 800 pairs.
- Candidate presentations: 6400.
- Recurrent presentations: 4800.
- WHU 700-test: sealed and unused.

The selected recurrent epoch 3 changed dev AEPE from `2.4803` to `2.4655` and
CMR@1 from `22.50%` to `24.75%`. It did not pass the preregistered gate:

- CMR@1 gain was `+2.25 pp`, below `+3 pp`.
- AEPE gain was `0.0148 px`, below `0.05 px`.
- AEPE improvement 95% CI crossed zero.
- Shuffled evidence matched the main evidence, so evidence-specific benefit was
  not established.

Status: `STOP_HELDOUT_GATE_FAILED`. Do not describe E091A as the best method or
evaluate it on WHU 700-test without a new method decision.

## Closed directions

- LuckyR1 A/B/C: `NO_GAIN` under current-model and clean-lineage evaluation.
- Frozen stride-8 local candidate scoring: below random candidate selection.
- Lightweight high-resolution post-hoc descriptor: failed actionability.
- Tested relation descriptor: failed progressive overfit.
- E090D post-hoc evidence at E091 scale: small non-specific dev gain.

See `FAILED_METHODS.md`, `DECISION_LOG.md`, and individual result cards before
reopening a direction.

## Recommended next research question

The next WHU proposal should explain why shuffled structured evidence performs
like real evidence. A justified direction must separate useful cross-modal
evidence from a generic learned flow prior, for example by enforcing
evidence-conditional counterfactuals or updating the shared feature/correlation
representation rather than adding another terminal selector.

For OS, retain the separate diagnosis: the dominant issue is B/C-shared
generalization failure and the >=5 px tail, not the WHU fine-localization gap.
Do not transfer a WHU local head to OS without a clean OS lineage experiment.

## Before running anything

1. Read `REPRODUCIBILITY.md`.
2. Run `python scripts/preflight.py --scope all`.
3. Verify checkpoint hashes from `CHECKPOINTS.json`.
4. Keep prediction-before-GT evaluation and pair-level metrics.
5. Use a new experiment ID and preserve all terminal result cards.
