# Strong-Model Handoff: Unified OPT-SAR Matching Bottlenecks and Method Innovation

## Role

Act as the principal research scientist for an OPT-SAR image matching project. Diagnose the verified bottlenecks causally, inspect the actual experiment artifacts when useful, review relevant recent literature, and propose structural method innovations. Do not respond with generic suggestions such as "use a stronger backbone", "tune the threshold", "add a gate", or "train longer" unless the proposal includes a falsifiable mechanism and a controlled experiment.

Reason internally in English if helpful. Discuss conclusions with the user in Chinese.

## Research Policy

- Do not use SourceMux unless the user explicitly requests SourceMux in the current message.
- Search recent computer-vision papers directly through arXiv and Semantic Scholar, sorted by submission/publication date.
- Verify metadata against the original arXiv, DOI, publisher, or official project page.
- Use official GitHub repositories or Papers with Code for implementation details.
- Prioritize dense correspondence, optical flow, detector-free matching, cross-modal matching, remote-sensing registration, differentiable geometric estimation, structured prediction, and multimodal representation learning.

## Non-Negotiable Experimental Definition

- A new method must use the same architecture family on OS and WHU.
- OS and WHU are trained independently on their own training data.
- Dataset-specific losses, loss weights, schedules, augmentations, and physical-scale interpretations are allowed.
- Never compare OS and WHU absolute metric values to infer which dataset is easier. Compare each method only with the baseline on the same dataset.
- OS imagery has approximately 1-meter ground resolution; WHU-OPT-SAR imagery has approximately 5-meter ground resolution. Use this fact only to design scale-aware supervision and receptive fields.
- No ground truth may enter inference. Ground truth is allowed for training labels and offline oracle diagnostics only.

## Metric Definition

For the current evaluation implementation, one mean endpoint error is calculated for every image pair. `CMR@t` is the percentage of image pairs whose pair-level mean endpoint error is below `t` pixels. `AEPE` is the average of the pair-level mean endpoint errors.

An oracle metric is not a network result. An oracle uses ground truth offline to choose the best candidate or hypothesis from a candidate set. A low oracle proves that downstream selection or refinement cannot reach the target with the current candidate set. A high oracle proves only reachability, not no-GT separability or learnability.

## OS Dataset: Verified State

Strong reproducible GDROS baseline:

- AEPE: 1.4347
- CMR@1: 37.0283%
- CMR@2: 83.4906%
- CMR@5: 99.5283%

Current strongest method family reference, E075A-M4:

- AEPE: 2.3613
- CMR@1: 56.6832%
- CMR@2: 78.9604%
- CMR@5: 88.8614%
- CMR@10: 94.8020%
- CMR@16: 97.0297%

The current method has a real high-precision advantage at 1 pixel, but it has a severe pair-level tail: approximately 11.14% of pairs fail the 5-pixel threshold, while only approximately 0.47% fail for GDROS.

E082A dense/multiscale basin diagnostics:

- Best selected CMR@5: 88.8614%
- Best oracle CMR@5: approximately 93.32%
- Oracle-selection gap: approximately 4.46 percentage points
- Oracle-to-GDROS gap: approximately 6.21 percentage points
- Selected error P90/P95: approximately 5.92/11.01 pixels
- Increasing training from 20k to 40k did not materially improve the result.

OS bottleneck decomposition:

1. Candidate-generation ceiling: even ground-truth selection among the generated hypotheses reaches only approximately 93.3% CMR@5, below the 99.53% baseline. Therefore at least approximately 6.2 percentage points are inaccessible to any selector or refiner operating only on the current hypotheses.
2. Candidate-selection gap: the no-GT method reaches 88.86%, approximately 4.46 percentage points below the candidate oracle.
3. Heavy-tail behavior: the method is excellent on easy pairs and catastrophically wrong on a minority of pairs. Optimizing average local precision or CMR@1 alone worsens the research objective.
4. Hypothesis correlation: existing multiscale/local modes are generated around related predictions and are not sufficiently diverse to recover a missing global basin.
5. Cross-modal global ambiguity: repeated SAR structures, modality-specific appearance, and large displacement can suppress the correct global correlation mode before refinement begins.

The primary OS research problem is therefore not subpixel refinement. The primary OS problem is generating at least one globally correct, no-GT-reachable basin for nearly every pair, while preserving the existing CMR@1 advantage.

## WHU Dataset: Verified State

Paper GDROS target:

- CMR@1: 72.05 +/- 1.06%
- CMR@2: 96.86 +/- 0.65%
- CMR@5: 99.57 +/- 0.11%
- All AEPE: 0.90 +/- 0.04
- RMSE: 0.62 +/- 0.15

Clean E080A official-recipe reproduction:

- CMR@1: 66.0%
- CMR@2: 94.4286%
- CMR@5: 99.5714%
- AEPE: 0.97335

Current WHU method reference, E081A-M4:

- CMR@1: 28.0%
- CMR@2: 72.0%
- CMR@5: 98.1429%
- AEPE: 1.9054

E081A-M4 error bands:

- Below 1 pixel: 28.00%
- Between 1 and 2 pixels: 44.00%
- Between 2 and 5 pixels: 26.14%
- At least 5 pixels: 1.86%

E084A-A2 replaced the old geometry-only gate with F4/F2 feature-conditioned local-correlation refinement and was trained sufficiently. E084A-A2 failed:

- Final test CMR@1/2/5: 9.4286/56.8571/97.1429%
- Final test AEPE: 2.3557
- Coarse input CMR@1/2/5: 10.0/57.2857/97.1429%
- The learned fine matcher slightly degraded the coarse input.

E084A-A2 candidate diagnostics:

- F4 lattice oracle CMR@1: 97.7143%
- F4 lattice oracle CMR@2: 99.2857%
- F4 lattice oracle CMR@5: 99.4286%
- Mean GT-neighborhood rank: approximately 6.90
- F2 lattice oracle CMR@1/2/5: 78.7143/93.4286/99.1429%

Interpret the F4 oracle carefully. The F4 local lattice spatially covers the correct residual for almost every pair, but the learned decoder cannot identify the correct residual. The mean rank near 6.9 indicates weak but nontrivial correlation evidence; it does not imply that top-1 inference is solved.

WHU bottleneck decomposition:

1. Fine-scale discriminability: the correct local offset is often present but is not the strongest response under the learned OPT-SAR fine features.
2. Multimodal decoding error: E084A computes an expectation over all residual bins. Averaging incompatible modes can produce a coordinate located at no valid mode.
3. Missing bidirectional constraints: independent per-pixel matching does not enforce mutual local correspondence and is vulnerable to repeated SAR responses.
4. Missing structured consensus: E084A decodes pixels independently and does not use global affine consistency or sufficiently strong neighborhood consensus to amplify weak correct modes.
5. Training-distribution mismatch: E084A trains the fine head from the first step around an untrained coarse prediction. When ground truth is outside the local support, fine supervision is invalid or becomes a boundary target. The fine decoder receives contaminated supervision before the coarse path is ready.
6. Objective mismatch: dense pixel losses and large-error terms do not directly optimize the pair-level transitions `2-5 px -> below 2 px` and `1-2 px -> below 1 px` that determine WHU CMR@2 and CMR@1.
7. The WHU problem is not primarily search-window coverage. The WHU problem is conditional local correspondence ranking, structured decoding, and valid curriculum training.

## Important Architectural Fact

E081A-M4 is not simply the OS E075A frontend trained on WHU. E081A-M4 uses a GDROS/RAFT-initialized backbone plus a region-level geometry-only gate/refiner and threshold/tail losses. The region refiner observes predicted flow, LSR affine flow, their difference, and magnitude; it does not observe high-resolution OPT/SAR appearance. Do not conflate E081A-M4 with a fully unified E075A architecture.

The next proposed architecture must explicitly define the shared OS/WHU architecture, then specialize training objectives per dataset.

## Failed or Insufficient Directions to Avoid Repeating

- Frozen cosine/region-score candidate selection.
- Lightweight learned top-K selectors over weak cached evidence.
- Larger K alone: increasing K improves oracle recall but worsens no-GT contamination.
- Threshold, gate, or confidence tuning without changing evidence or structured inference.
- Geometry-only refinement without OPT/SAR fine features.
- Independent local softargmax/global expectation over multimodal residual distributions.
- Fine-head training around random or out-of-support coarse centers.
- Pure local window enlargement: a larger oracle does not imply that a network can select the correct target.
- Post-hoc isolated refiners that are not jointly optimized with the feature extractor and matcher.

## Already Queued, Not Yet Validated

Do not unknowingly duplicate these ideas. Critique or improve them explicitly:

1. E085A CARF: group-wise local cost evidence, spatial/offset cost aggregation, ConvGRU recurrent recentering at F4/F2. The first formal attempt was externally terminated; the stable retry is queued.
2. E086A MPBF: valid-center curriculum, GT-jitter fine pretraining, scheduled predicted centers, local dual-softmax or Sinkhorn, discrete top mode followed by mode-local softargmax.
3. E087A GACF: whole-image residual affine consensus over local candidate votes followed by lightweight F2 refinement, conditioned on a GT-affinity preflight.

## Research Assignment

Develop a coherent architecture that can address both asymmetric bottlenecks:

- OS requires globally robust basin generation and heavy-tail reduction before refinement.
- WHU requires precise structured decoding of already reachable local residuals.
- The architecture family must be the same across both datasets, while the losses and schedules may differ.

The response must include:

1. A causal bottleneck graph from input features to global candidates, structured matching, geometric estimation, local refinement, and final metrics.
2. A literature matrix focused on recent methods and the exact transferable mechanism, not paper-name listing.
3. Three to five structurally distinct method proposals. Each proposal must specify tensors, resolution, information flow, inference algorithm, training losses, computational complexity, and why the proposal addresses OS and WHU differently without changing the architecture.
4. A theoretical comparison with E085A, E086A, and E087A.
5. A ranked recommendation of the best one or two methods, including expected upside, main failure mode, and the cheapest falsification experiment.
6. A staged training protocol that never trains a local refiner on invalid centers and that separates candidate-generation learnability from decoder learnability.
7. Dataset-specific success gates:
   - OS: raise oracle CMR@5 first, then close the oracle-selection gap, preserve CMR@1 above the GDROS baseline.
   - WHU: improve CMR@2 and CMR@1 while preserving CMR@5, targeting E080A and then the paper baseline.
8. An implementation-audit checklist to prevent an experiment wrapper from silently reusing an old method.
9. A cost-control policy with smoke, minimum sufficient training, matched-step comparison, early stopping, and criteria for longer training.

Challenge the current decomposition if evidence supports a better explanation. Clearly distinguish verified facts, plausible hypotheses, and unknowns requiring diagnostics. Prefer a novel, defensible mechanism over a collection of auxiliary losses.

## Artifact References

- Historical local state summary, incomplete after the earlier experiment stages; the verified facts in this handoff and the listed formal result cards take precedence: `/Users/shihao/Documents/图像匹配之二/research/CURRENT_STATE.md`
- OS E075A-M4 result: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/formal/RESULT_CARD.md`
- OS E082A-S1 result: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-082A-ZZVR-OS-MULTISCALE-DENSE-BASIN-GENERATOR/run-s1-formal-r1/formal/RESULT_CARD.md`
- WHU E080A baseline reproduction: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-080A-ZZVR-WHU-OFFICIAL-RECIPE-FROM-SCRATCH/formal/RESULT_CARD.json`
- WHU E081A-M4 result: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-081A-ZZVR-WHU-OUR-METHOD-NATIVE-TRAIN/run-m4-native-r2/formal/RESULT_CARD.json`
- WHU E084A-A2 result: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-084A-ZZVR-WHU-FEATURE-CONDITIONED-FINE-MATCHER/run-a2-formal-r2/formal/RESULT_CARD.json`
