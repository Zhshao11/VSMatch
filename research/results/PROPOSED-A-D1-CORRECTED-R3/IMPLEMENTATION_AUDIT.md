# Implementation Audit

- `best_translation_capacity` and `best_affine_capacity` now operate on the same `GT - coarse` residual state.
- `residual_modes` fits either constant or affine residual displacement from identical global edges.
- The base zero-residual mode is included before scoring.
- Candidate flow is `coarse + upsample(feature_residual) * stride`.
- Translation and affine pools each contain 12 modes.
- Self, robust, held-out and oracle selectors reuse exactly the same pool per family.
- Held-out hypotheses are fitted outside the region and evaluated only on the held-out region.
- GT is loaded only after the frozen evidence bundle is read; inference evidence remains unchanged.
- Raw and post-LSR use identical candidate indices; the output operator changes only candidate evaluation.
- Hard top-k, clustering and mode proposal remain non-differentiable and are not described as trainable.
