# E-056A Region-Probability Fine Window Generator

Status: `ENGINEERING_STARTED`

## Objective

Replace fixed coarse-point / fixed-hypothesis fine windows with a target-side
region probability representation. Each source patch receives a multi-modal
target-region distribution, followed by dense or continuous refinement inside
the predicted region.

## Frozen Inputs

- E046A-R2 frontend/checkpoint: read-only.
- E046A-R2 strict replay: AEPE 6.474331, CMR@2/5/10/16
  12.344995/50.121105/83.559723/94.438789.
- No GDROS/E044A/E046A/E052-E055 writes; no testing.zip, OS424, confirm, or
  rescue.

## Tracks

1. `COARSE_CORR_AGG_WINDOW`: aggregate the complete selected-source coarse
   correlation distribution into target-region mass, retaining multiple modes.
2. `REGION_HEATMAP_16x16`: 16x16 local region logits and dense decoder.
3. `REGION_HEATMAP_32x32`: 32x32 local region logits and dense decoder.
4. `MIXTURE_WINDOWS_K3`, `MIXTURE_WINDOWS_K5`: K centers plus calibrated
   radii, not a fixed list of proposal points.
5. `AFFINE_CONDITIONED_REGION`: low-dimensional no-GT affine state conditions
   region logits but does not produce labels or RANSAC decisions.
6. `GT_CENTERED_CONTROL`: diagnostic-only fine feature upper-bound.

## Protocol

- Materialize correlation aggregates, region maps, mixture centers/radii,
  dense candidates and decoder logits before reading truth flow.
- Train window coverage masks and dense heatmap decoders only on train split.
- Each train track has independent seed/checkpoint/log and strict checkpoint
  replay. Dev is fixed evaluation only.
- Dev404 shards use deterministic pair-id partitioning and aggregate only if
  expected=404, unique=404, missing=0, duplicates=0.
