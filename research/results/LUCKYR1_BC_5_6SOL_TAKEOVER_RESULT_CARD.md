# LuckyR1-B/C 5.6sol takeover

Status: `SUPERSEDED_BY_20260915_RECOVERY_CARD`

Current authoritative status: [LUCKYR1_RECOVERY_20260915_RESULT_CARD.md](LUCKYR1_RECOVERY_20260915_RESULT_CARD.md).

## Audit result

- B passes the source audit. It modifies the real CorrBlock tensor immediately before the real BasicUpdateBlock, retains raw correlation additively, matches the official 81-channel candidate order, and uses the local full mapping Jacobian.
- C v1 is rejected. It collapses all candidates to one expected target before spatial smoothing and carries one expected coordinate across recurrent iterations. That contradicts the LuckyR1 requirement to preserve separated modes and reason over candidate-to-candidate compatibility.
- Official-404 mapping, prediction-before-GT join, pair-level AEPE/CMR aggregation, trainable-delta resume, and the step500 cost gate pass the audit.

## Versioned repair

Implementation `luckyr1-bc-v2-sparse-multimode-20260913` keeps several absolute-target modes, computes bounded neighbor candidate messages under the local Jacobian, retains unmatched mass, detaches recurrent modes between iterations, and preserves raw correlation as the additive base.

Local and zzvr CPU contracts pass: raw preservation with disabled evidence, target-translation equivariance, recurrent mode translation, finite outputs, and finite nonzero gradients for all C evidence parameters over two iterations.

The v2 files were deployed with matching SHA256 values. Historical-M4 v2 G0 job `20260913_214418_b3a88a40` is queued; final B/C v4 tasks wait on that gate. The old queued B/C attempts were cancelled and retired without deleting their attempt cards. Clean F2/F3 and the protected E075A-M4 checkpoint have not been touched; WHU has not been started.

No B/C endpoint metric is available yet. All eight GPUs currently have compute PIDs; clean F3 priority 20 precedes v2 G0 priority 10, and B/C remain priority 5. The dedicated zzvr orchestrator remains the sole long-queue authority and will run the G0 and then submit B/C with `candidate_gpus=null` and logical `cuda:0` when a genuinely exclusive card is available.
