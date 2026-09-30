# E-001 Implementation Review

Date: `2026-08-04`

Method: `M001 v0.2`

Status: `APPROVED FOR FORMAL RUN AFTER E-000`

## Final Reviewed SHA256

- `evaluate_sparse_lsr.py`:
  `9628c826b42bc48ef5ab8023fcb15da36bd6410e5a613cb584b86326ea13d94b`
- `benchmark_sparse_lsr.py`:
  `3dbcb88f2a443018fdc7c7312b81051f6d522173140c5900cebd8dc9ba3f0473`
- `test_sparse_lsr.py`:
  `081aca6b28771c7c9d5d30d8b5bcf84080325dc9a0661b80f78bc8811f60512d`

## Review Coverage

- Tensor shapes, row-major point indices, flow units/direction and dense affine generation.
- Exact public LSR parity and the public component-error metric versus vector EPE diagnostic.
- No valid-mask evaluation, including black borders.
- Deterministic/unique/in-bounds sampling, rank and condition diagnostics, GT isolation and
  absence of test-label selection leakage.
- Diagnostic/formal gate separation and exact-public-pool versus full-image attribution.
- CUDA synchronization, output lifetime, CPU/GPU output placement, peak-memory scope and
  end-to-end timing interpretation.
- Baseline dependency hashes/diff and preservation of the uncommitted reproduction patch.

## Closed High-Risk Findings

1. Formal provenance now checks the fixed official ZIP SHA/size and exact member set, hashes
   every ZIP member, hashes the current 2120-file published root, and requires archive/root/
   provenance manifests to match.
2. Formal mode requires post-publication validation flags and rehashes the dataset again after
   accuracy or benchmark execution.
3. Formal mode fixes 424 pairs, `testing`, released checkpoint SHA, `512x512`, K values,
   random seeds, seed 1234, parity tolerance and oracle behavior.
4. The loader's ordered optical/SAR/root-flow paths must exactly match the verified published
   root, closing absolute/parent split-path escapes.
5. Both parsers use list-valued `num_classes=[8]`; real `parse_args()` default tests cover
   accuracy and benchmark.

## Verification

- Unit/invariant suite: `20/20` passed independently on RTX 4090.
- Full public-pool synthetic parity and two-pair smoke passed.
- Latest smoke:
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/validation/E-001/smoke-10-parser-default`
- Latest smoke dense-flow parity: `6.1035e-05 px`; primary pair-error parity:
  `1.1921e-07 px`.
- Diagnostic correctly records `formal_gate_applicable=false` and null gate decisions.

Non-blocking limitations remain: the resource benchmark keeps prepared designs resident in one
process, and only a fresh formal run may be used for the E-002 decision.
