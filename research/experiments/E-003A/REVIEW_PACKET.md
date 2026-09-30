# E-003A Incremental Review Packet

Status: `REVIEW FIXES TESTED / TARGETED REREVIEW PENDING`

## Frozen identity

- Method: `M001 v0.6 / E-003A Learned Sparse Geometry Selector`
- `METHOD_SPEC.md`: `6d259b05900e4fea404841d35584492da4cfa8ec92391bcb8fe8bced6f51b3a2`
- `METHOD_DELTA.md`: `07b245e9ef803e76ced26b676cee84bc6e0f57b6c0a2fe5a3361bffcba7fe3e8`
- `INNOVATION_CONTRACT.yaml`: `f764876bed16ddf57c11aad7a25039d3358fdbd9163aa0e8200a6159995963e4`
- `E-003A.md`: `6d9dd55c2ecc03c8c8e0379272a8cc1c030d8270464453bb266001b154093791`
- `CURRENT_STATE.md`: `2aa64e7e9ade15e8f19d31b865eb275764977f392a508abb36337078e9143e57`

Remote code root:
`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e003a`

Run root:
`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-003A/run-01`

The existing GDROS baseline modifications outside this module are user-owned and out of scope. Frozen
E-002A/E-002B helpers are dependencies only. Review their recorded SHA and the E-003A call boundary; do not
request unrelated baseline cleanup.

## Specification-to-code/test mapping

| Frozen requirement | Implementation | Direct evidence |
|---|---|---|
| Official 512x512 data, grouped train/dev isolation | `data_protocol.py`; `candidate_cache._load_image`; `enforce_split_access` | `test_data_protocol.py`; `data_protocol-v3/data_protocol_gate.json` |
| Frozen post-Transformer GDROS, no gradient | `candidate_cache.build_frozen_candidate_table`; `load_slim_feature`; `torch.inference_mode` | `test_core.test_candidate_matching_never_materializes_n_by_n`; 2-pair parity |
| Public-stratified K128, cosine top8, chunked 128x4096 scores | `candidate_cache.build_cache`; `e002a.anchors`; `e002b.candidates.extract_candidates` | shape/complexity tests; cache validation |
| Candidate features contain no GT/datum | `build_candidate_features`; `_candidate_record` | provenance tests; candidate-cache flags and validation |
| Raw truth_flow is the only label authority | `read_flow`; `_sample_truth_flow`; `build_training_labels` | raw-flow direction/label test; separate label cache |
| Width64 shared encoder, T=2 feedback, dustbin | `selector.CandidateEncoder`; `LearnedGeoSelector.forward` | parameter/shape/gradient and toy-overfit tests |
| FP32 normalized weighted affine, ridge 1e-4, cond 1e6 | `selector.weighted_affine_solve` | affine recovery, degeneracy and non-finite tests |
| Fixed candidate/grid/confidence losses | `losses.py`; `train_selector._epoch` | finite/nonzero gradient, toy overfit, real-data smoke |
| Fixed 5px predicted-error abstention | `selector.LearnedGeoSelector.forward` | evaluator rows and G1 coverage/risk checks |
| Main, independent unary, soft-EM, C8 and GT oracle reporting | `evaluate_selector.evaluate_dev` | `g1.json`, per-pair CSV/JSON after formal dev |
| Selector and slim end-to-end resource gates | `benchmark_worker.py`; `benchmark_resources.py` | five fresh-process raw JSON and `g2.json` |
| Frozen source/spec/checkpoint provenance | `spec_binding.py`; `provenance.py` | `gates/provenance.json`, final source manifest |

## Completed automatic evidence

- Frozen SHA validation: PASS.
- Python compilation: PASS.
- Unit/provenance tests: `16/16 PASS`, including a real-checkpoint frozen/no-grad frontend test and
  NaN/Inf structured-abstention test.
- Two-pair online/cache exact differential: PASS, all reported max absolute differences `0.0`.
- Real data smoke: train/dev 4 pairs each, two epochs, nonzero finite gradients, checkpoint written,
  parameter count `18,692`, peak allocated `25.775 MiB`.
- Split v3: 1,607 train / 404 dev, zero group overlap, zero exact testing-optical overlap.
- Full cache validator v2: 2,011/2,011 PASS with opt/SAR/truth-flow SHA binding and complete tensor checks.
- Recursive AST source closure: 41 files, SHA `daaf3fea395180560109d74154bb1078e50958cbaacb337738fe9c1c86518482`.

## Review round 1: method semantics and training safety

Read the actual source manifest and call paths from `candidate_cache.py` through `train_selector.py`.
Check source/target and target-source flow direction, pixel/token normalization, truth-flow sampling,
candidate/dustbin labels, FP32 weighted solve, T=2 meaning, residual feedback, solver abstention,
confidence target, gradient isolation, and absence of GT/datum in inference candidates. Confirm the fixed
model/optimizer/data budget and no hidden N2/hypothesis/fallback path.

Output: `reviews/round-1-method.md`, ending in exactly `APPROVE` or `CHANGES_REQUIRED` with file/line findings.

## Review round 2: metrics, resources and provenance

Read `evaluate_selector.py`, `benchmark_worker.py`, `benchmark_resources.py`, `validate_cache.py`,
`provenance.py`, the final source manifest and formal commands. Check full-image unmasked AEPE/CMR,
PairLocalReady, abstention coverage and fixed-90%-coverage risk, oracle timing/GT isolation, control identity,
bootstrap/failure metrics, synchronized timing, cold/warm memory, five-process aggregation, baseline speedup,
score/selector element counts, checkpoint/split/source hashes and artifact paths.

Output: `reviews/round-2-eval-resources.md`, ending in exactly `APPROVE` or `CHANGES_REQUIRED` with file/line findings.

Any high-risk fix invalidates the affected tests and source SHA and requires a targeted rereview before formal
training. Formal training/resource jobs may start only after both final conclusions are `APPROVE`.
