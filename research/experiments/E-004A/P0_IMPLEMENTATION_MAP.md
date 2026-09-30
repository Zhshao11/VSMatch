# E-004A P0 Specification-to-Code/Test Map

Status: `IMPLEMENTATION IN PROGRESS / P0 NOT AUTHORIZED UNTIL REVIEW`

## Frozen identity

- `CURRENT_STATE.md`: `4c8b6d1edb7791bd4f2c8de258c1888aee1fad98962419bee17e06f6c6220997`
- `METHOD_SPEC.md`: `e1c6b14664214018c622f17de54baabafc7b0f3674cc03e49397ad89ab9847d7`
- `METHOD_DELTA.md`: `d7410557e30378707a93731a15f0d4aa43ebfe39f4d340d3982ce9e3f746dd5a`
- `INNOVATION_CONTRACT.yaml`: `c01dcfaf170778687f00fa3b6ec7d74a35ccce6f0252f847dc8ca0297298359a`
- `E-004A.md`: `6110948722ff841257a938e59ac8e7682666491944b57c75eadfefb42b9ccdc9`
- `BIND-001`: uniform-region-K128 uses exactly `(2,2)` and `(6,6)` in every region.

## Mapping

| Frozen P0 item | Code | Automatic evidence |
|---|---|---|
| 8x8 regions, four fixed unique proposals | `proposals.build_region_proposals` | exact order/offset/region/unique property tests |
| GT-prioritized constrained K128 | `proposals.allocate_oracle_k128` | K128, every region 1..3, deterministic tie/property tests |
| uniform-region K128 | `proposals.uniform_region_k128` | exact two per region test for frozen `(2,2)`, `(6,6)` offsets |
| frozen 512->64 frontend | `proposal_cache.build_pair_cache` | real checkpoint no-grad integration test |
| exact M256/public-K128 cosine top8 | `proposal_cache.build_pair_cache` | online/cache differential and 256x4096 live-shape audit |
| raw-flow recoverable labels, datum excluded | `proposal_cache.build_labels` | direction/bounds/radius and GT isolation tests |
| selected-anchor GT oracle only | `oracle_gate.evaluate_anchor_set` | GT loaded after cached candidates; trace flag test |
| >=12, >=3 quadrants, cond(XtX)<=20 | `oracle_gate.fit_selected_oracle` | synthetic recovery and invalidity tests |
| ordinary FP32 LS and 512x512 no-mask metrics | `oracle_gate.fit_selected_oracle`, `evaluate_anchor_set` | affine recovery/evaluator parity tests |
| fixed/public, uniform, constrained oracle comparisons | `oracle_gate.run_p0` | full-dev P0 JSON and per-pair JSONL |
| no N2/consensus/hypothesis/test access | `test_provenance.py` | source/trace negative tests |

## Staged stop

Only proposal/cache/oracle/evaluator code is permitted before P0. No `anchor_heads.py`, training,
replacement implementation or resource benchmark may be created until P0 passes and the staged branch is
explicitly advanced.
