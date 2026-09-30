# E-044A Method Divergence Plan

## Locked references

- Primary best overall reference: `E-041A-ZZVR-OPT-ONLY-SALIENT-FUSION-TUNING/grid-high-r2`
  - AEPE 6.844, CMR@2 11.704, CMR@5 48.237, CMR@16 93.235
- Auxiliary high-CMR@2 reference: `E-042A-ZZVR-LEARNED-SAR-SALIENCY/main`
  - AEPE 7.019, CMR@2 12.680, CMR@5 47.884, CMR@16 92.488
- Remote locked copies:
  - `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LOCKED_REFERENCES/E041A_best_gated_manual_salient_fusion/RESULT_CARD.md`
  - `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LOCKED_REFERENCES/E042A_best_hybrid_saliency/RESULT_CARD.md`

## E044A question

Can any lightweight, matcher-facing module beat the locked E041A backbone without touching GDROS baseline, DINO, or E028B recovery features?

## Divergence axes

1. SAR salient memory signal
   - `manual_no_energy`: remove projected energy from manual saliency.
   - `contrast_only`: test whether local contrast alone is the useful part.
   - `manual_minus_cross`: penalize globally over-attractive SAR tokens.
   - `manual_strong_minus_cross`: stronger penalty for repeated/global attraction.

2. Spatial distribution
   - `saliency_quota_grid=4`: region-balanced salient SAR memory, to avoid collapse onto repeated/high-energy SAR structures.

3. Interaction/context
   - `local_k=0`: remove local topK context and use salient memory only.
   - `alpha=0.08`: test slightly stronger residual fusion.

4. Training objective
   - `hard_ce`: margin-enhanced CE with learnable temperature.

## First-batch configs added locally

- `e044a_k64_l4_noenergy`
- `e044a_k64_l4_contrast`
- `e044a_k64_l4_minuscross`
- `e044a_k64_l4_strongminuscross`
- `e044a_k64_l4_quota4`
- `e044a_k96_l8_quota4`
- `e044a_k64_l0_a05`
- `e044a_k64_l4_a08`
- `e044a_k64_l4_hard`

## Decision rule

- If any config exceeds E041A on AEPE/CMR@5/CMR@16 without collapsing CMR@2, keep that branch and tune it.
- If no config exceeds E041A, keep E041A as the feature-interaction reference and move to the next pipeline stage.
- E042A remains useful only if the next stage specifically benefits from higher CMR@2.

## Paper-derived rationale

- LoFTR/XoFTR: pair-conditioned coarse interaction is useful, but full interaction can destabilize cross-modal features.
- RoMa/DKM-style dense matchers: foundation features help coarse robustness, but fine/matcher-facing decoders and confidence matter.
- LightGlue/SuperGlue family: confidence/adaptive sparse interaction is better than blind all-token mixing.
