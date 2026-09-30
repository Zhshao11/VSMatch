# E-005A Implementation Map

Status: `IMPLEMENTING`

Binding: `BIND-STATE-001`. The `CURRENT_STATE.md` SHA change only records that E-005A is assigned and
does not change method semantics, gates, or scope.

| Frozen requirement | Implementation | Automatic evidence |
|---|---|---|
| M256 proposal and exact cosine top8 once | `candidate_pool.py` | E-004A proposal differential, E-003A/E-004A top8 differential, score count/trace |
| Sparse 5x5 quarter-token correlation | `local_corr.py` | coordinate/value/zero-padding/gradient/sample-count tests |
| G16 setwise consensus and T4 shared recurrent update | `consensus.py`, `model.py` | persistent-state, weight-sharing, first-round-zero-residual, tensor-shape and gradient tests |
| Per-round coverage-constrained K128 | `active_set.py` | deterministic ties, exact K, region [1,3], reactivation property |
| FP32 normalized weighted affine | `affine.py` | synthetic affine/noise/outlier/condition/ESS/flow-direction tests |
| Raw truth-flow supervision and fixed losses | `data.py`, `train.py` | label authority, path-negative, loss-weight and toy-overfit tests |
| Dev metrics and dense baseline parity | `evaluate.py` | full-image no-mask metric and GT-after-output tests |
| G2 timing and memory | `benchmark.py`, `aggregate_resources.py` | fresh PID/UUID, synchronization, raw rows, hidden-N2 shape audit |

## Execution gates

1. Freeze third-party references and provenance.
2. Implement and pass all automatic tests.
3. Run complete 404-pair P0; stop if the reviewed gate fails.
4. Obtain two independent incremental reviewer approvals.
5. Build the frozen-feature cache and train seeds `20260808` and `20260809` under watchdog.
6. Run G1; run G2 only if G1 fully passes.
7. Write `RESULT_CARD.md`, update the manifest/index, and stop.
