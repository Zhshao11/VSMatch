# Proposed A: Global Multimodal Geometry Posterior

Revision: `2`

Status: `METHOD_SELECTED / EVIDENCE_EXPORT_IMPLEMENTED / CPU_VALIDATED / GPU_INFERENCE_NOT_AUTHORIZED`

Latest corrected diagnostic: `PROPOSED-A-D1-CORRECTED-R3`.

- Residual affine family capacity: supported.
- Current global cosine-edge candidate pool: no gain over base coarse.
- Held-out-region scoring: not causally testable because the pool contains no beneficial alternative mode.
- Trainable selector/posterior: not authorized on the current pool.

This is a design and diagnostic package only. It does not assign a formal experiment number, start GPU work, cancel a queue entry, or modify a remote run. The experiment engineering task must assign the next registered ID after checking `research/EXPERIMENT_INDEX.md`.

## Decision

Select A as the first new method family:

> Generate multiple whole-image geometric explanations from cross-image evidence, score each explanation using spatial regions that were not used to fit it, and perform fine refinement only inside the selected explanation.

Do not treat the current E087 run as an implementation of A. E087's current `forward` still applies independent F4/F2 local residual softmax stages and does not construct, score, or select a global affine hypothesis.

B remains the next candidate for WHU after A's zero-training diagnostics. C is not selected because the current evidence does not yet prove that the problem is missing cost evidence rather than invalid geometric decoding/candidate structure.

## Evidence Table

### Consistent with the independent report

- E082's OS candidates are a shared-residual plus global-translation family, not independent rotation/scale/shear hypotheses. Its oracle is therefore an upper bound for that restricted pool only.
- E084's high F4 oracle is a per-location lattice coverage diagnostic, not a full upsampled/LSR pipeline witness.
- E084's current fine decoder is vulnerable to mode averaging and invalid early centers; its output is slightly worse than its own coarse input.
- E086's prior code had an unnormalized `grid_sample` patch-window path and a Sinkhorn score equal to a conserved total-mass mean.
- E087's code does not contain the claimed global affine vote/solve in the inference path.
- E085 r1 ended with SIGTERM/143 and no valid milestone checkpoint; it is not evidence that CARF failed after training.

### New or changed evidence after the report

- E084A-A1 is now complete: test CMR@1/2/5=`12.857/58.571/97.429%`, AEPE=`2.275`; status `TRAINING_SUFFICIENT_NO_GAIN`.
- E084A-A2 is complete: test CMR@1/2/5=`9.429/56.857/97.143%`, AEPE=`2.356`; status `TRAINING_SUFFICIENT_NO_GAIN`.
- E086A-C2 has a 60K validation curve: CMR@1/2/5=`5.286/46.571/94.571%`, AEPE=`2.725`; no final RESULT_CARD was present at audit time and the process was still running. This is an interim curve, not a final method judgment.
- E087A-D2 has a validation affine preflight with GT: all 700 pairs pass the affine residual threshold with near-zero residual. This proves affine capacity under the GT-assisted audit, not no-GT affine consensus learnability.
- E085A-B2 r2, E086A-C2, and E087A-D2 are currently running. This package does not modify or cancel them.

Later stage snapshots, still not final method comparisons:

- E085A-B2 at 60K is already catastrophically wrong at coarse: coarse/final CMR@1/2/5=`0/0/0%`, AEPE approximately `965.2/687.4`. Treat as an implementation/provenance anomaly, not ordinary low accuracy.
- E086A-C2 at 80K: coarse CMR@1/2/5=`8.86/53.29/95.43%`, final=`7.29/52.14/95.29%`. Fine has negative net change at the same checkpoint.
- E087A-D2 at 20K: coarse CMR@1/2/5=`0.43/27.71/84.29%`, final=`0.43/27.00/84.00%`. Fine has negative net change, and the current forward still lacks the claimed global affine inference mechanism.

### Conflicts resolved by source/card evidence

- The report's statement that E086 has no formal progress is stale; the 20K/40K/60K curve exists. The report's implementation warning remains valid because the deployed source still contains the unnormalized `sample_window` and the conserved Sinkhorn mean.
- The report's statement that E087 has no affine preflight evidence is stale; `preflight_affine_audit_validation.json` exists and passes. The report's implementation warning remains valid because the deployed `FeatureConditionedFineMatcher.forward` has only `LocalDistributionStage` F4/F2 calls and no global affine state.
- The report's statement that E084's high oracle is not a full-pipeline witness remains valid. The E084 result cards explicitly report raw coarse/F4/F2 outputs and do not supply an equivalent full-pipeline GT-assisted witness.
- Historical OS baseline numbers remain protocol-sensitive: E075/E082 use 404-pair exploratory evaluation, while the released GDROS baseline is tied to the 424-pair official protocol. The 6.21-point cross-protocol oracle gap must not be used as a strict same-sample ceiling.

### Not yet verified

- Whether the OS candidate ceiling remains below the baseline after pair-ID and full-pipeline protocol alignment.
- Whether a no-GT whole-image cross-image correlation can generate affine modes absent from E082's translation-only pool.
- Whether WHU's local correct peaks are improved by global target-coordinate competition rather than by output-operator alignment alone.
- Whether a selected affine mode can preserve non-affine/local residuals without introducing new >5px failures.

## Minimum Implementation

Implemented locally in `image_matching_part2_migrated/.codex_tmp/global_multimodal_geometry_posterior.py`:

- full-image normalized correlation and mutual top-k edge generation;
- multiple affine hypotheses from displacement-mode clustering and weighted least squares;
- differentiable geometry-conditioned evidence scoring for fixed hypotheses;
- leave-one-region-out held-out scoring;
- single-mode decoding with residual correction restricted to that mode;
- explicit dataclasses for edges and posterior output.

This is a diagnostic core, not a production matcher. Hard top-k, mutual filtering, clustering, and hypothesis generation are non-differentiable diagnostic operations. The fixed-hypothesis evidence score is differentiable and has a CPU gradient test. The future learned implementation must replace hard generation with a documented soft or straight-through path if gradient training is required.

The module does not accept ground truth in its inference API.

Revision 2 adds a complete evidence-rebuild path instead of depending on historical caches:

- versioned evidence bundle writer/reader and validation;
- frozen GDROS provider with `eval()` and `inference_mode()`;
- deterministic OS/WHU 32-pair training-diagnostic manifests;
- translation-family and affine-family controls using identical exported edges;
- self, robust, held-out-region, and offline oracle scoring;
- CPU-equivalent GDROS LSR reassessment for existing flow tensors;
- separate no-GT evidence and GT evaluation sidecars.

## CPU Validation

Test file: `image_matching_part2_migrated/.codex_tmp/test_global_multimodal_geometry_posterior.py`.

- Python compilation passed locally.
- Existing zzvr PyTorch environment ran the tests with `CUDA_VISIBLE_DEVICES=""`.
- Identity and affine coordinate units: PASS.
- Correct versus incorrect geometry-conditioned evidence: PASS.
- Gradient through fixed-hypothesis evidence score: PASS.
- Incompatible modes are not averaged by mode-local decoding: PASS.
- Full-image mutual edge generation: PASS.
- Degenerate affine input remains finite: PASS.

This is only a synthetic/CPU implementation check. It is not an OS/WHU accuracy result and not evidence that A improves GDROS.

## D0/D1 Before Training

1. Align pair IDs, masks, coordinate units, image sizes, iterations, and raw/post-LSR output operators.
2. Recompute E084 raw/post-LSR flow metrics directly from saved flow arrays. Regenerate missing candidate lattices from a frozen checkpoint only under a separately approved inference-only budget.
3. Compare at fixed candidate budget and identical image evidence: E082 translation-only candidates, A's whole-image multi-affine candidates, self-scoring, robust consensus, and held-out-region predictive scoring.
4. Report pool reachability, held-out score ranking, candidate diversity, affine condition numbers, and full-pipeline witness metrics.

Do not train if the new family does not improve candidate reachability or if all improvement exists only under GT-assisted selection. Historical cache absence is no longer a stopping condition: generate a new common frozen evidence bundle after inference-only approval.

R3 has now generated and evaluated that common evidence. The next permissible method task is to improve the no-GT edge/candidate generator on the frozen evidence, not to train the held-out selector or repeat the same cosine top-k pool.

## Stop and Promote Rules

Stop before training if geometry-family capacity does not improve fixed-pool reachability, cross-region scoring does not improve no-GT ranking, synthetic geometry cannot be distinguished, or the full-pipeline witness cannot be reconstructed without changing protocol.

Promote to a short trainable probe only if D0 passes, at least one no-GT candidate-recall or held-out-ranking signal improves at fixed budget, all coordinate/mode tests pass, and the module instantiates for both OS and WHU.

## Non-Goals

- No threshold rescue, gate-only change, or larger-K-only experiment.
- No claim that global affine is sufficient for every pair.
- No RANSAC implementation in the primary path.
- No testing-set tuning or checkpoint selection.
- No cancellation or modification of E085/E086/E087 currently running.
