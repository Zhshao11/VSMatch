# E-000 Formal Metrics Gate Review

- Date: `2026-08-04`
- Method: `M001 v0.2`
- Scope: formal baseline wrapper, metrics checker, and the imported E-001 provenance gate
- Decision: `APPROVE`
- Status limitation: this approves the implementation gate only; the official-archive E-000 run is not yet complete

## Reviewed call path

1. Watchdog launches
   `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e000/run_gdros_e000_formal.sh`.
2. The wrapper runs the unchanged baseline evaluation entry
   `/mnt/sda/zhushihao/VS_Match/GDROS/test.py` on the independently published official dataset root.
3. Only after `test.py` succeeds, the wrapper runs
   `check_gdros_e000_metrics.py`.
4. The checker imports `load_dataset_provenance` from the reviewed E-001 evaluator and binds the
   published directory, official archive, and recorded manifests by content hashes.
5. Any command failure propagates to watchdog through `set -euo pipefail`.

## Final deployed hashes

- Formal checker: `cb331ce362be8a6bfcc66eb2b338c41cb3b0eb2a2cd3a12fe46fd74c6a91e870`
- Formal wrapper: `c22c9545b4663d7ccfb3e578c955b0f298ad9126bf1a2db10f6956621b5bb938`
- E-000 checker tests: `9d4f01b5c54a54b5cf8099c20166734b4a236ef42047f8e28b0a5e3e680e6699`
- Imported E-001 provenance evaluator: `9628c826b42bc48ef5ab8023fcb15da36bd6410e5a613cb584b86326ea13d94b`
- E-001 provenance tests: `081aca6b28771c7c9d5d30d8b5bcf84080325dc9a0661b80f78bc8811f60512d`
- Baseline `test.py`: `5f5a60c4654bc2f81ccd3e27e11d6e2623271bd3e1482223022c22eef277367c`
- Fixed equivalent-precheck reference: `fcafa4d9b077b2b7e88e43e7efe2bfd638f6a4af2b1007961f8f056efb6cc6db`

## Closed review findings

- The reference JSON is fixed by SHA and its hash is recorded in every formal environment record.
- Both formal and reference summaries are recomputed from all 424 per-pair EPE values.
- AEPE, population standard deviation, legacy variance, strict-threshold matched counts, CMR,
  and matched-subset statistics are checked for internal consistency.
- JSON rejects `NaN` and `Infinity` globally; all checked numeric values must be finite.
- `max-pair-delta` must be finite and no greater than `1e-4`.
- Provenance must be `DATASET_ROOT/E000_PROVENANCE.json`; the official ZIP and published data are
  both rehashed, and archive/root/provenance manifests must agree.
- The checker cannot overwrite its metrics, reference, provenance, or checkpoint inputs.
- The wrapper fixes the official dataset root, released checkpoint, `testing` split, 200,000-point
  public LSR, seed 1234, and an empty output directory below the E-000 run root.
- The final wrapper records `CUDA_VISIBLE_DEVICES` and hashes the dynamic provenance-gate module.

## Verification evidence

- Local E-000 checker tests: `8/8 PASS`.
- Remote E-000 checker tests in the GDROS conda environment: `8/8 PASS`, no skips.
- Remote E-001 implementation/provenance regression: `20/20 PASS`, no skips.
- Local and remote wrapper syntax checks: pass.
- Negative CLI tests: infinite tolerance and output/input path collision both exit nonzero.
- The real fixed reference recomputes to AEPE `1.4612304443177186`; strict matched counts for
  thresholds 1 through 5 are `143/346/393/416/422`.

The independent reviewer found no remaining high- or medium-risk issue and re-approved the final
wrapper after the two low-risk audit fields were added.

## Final Provenance Increment And Run Outcome

The pre-run review above is historical. Before the actual formal run, the imported provenance
gate was tightened to distinguish loader-equivalent content from raw file bytes. The executed
versions were:

- Imported E-001 provenance evaluator:
  `a16fe7586053ea8fb6ea33a9607fdff4b75eee215b264af13d4ebcfbca8a226e`
- E-001 provenance tests:
  `39e00d837376cd74e701e549cc1715d9751985ce9ec01a2e46168dcf91d9b3d0`
- E-000 validator:
  `da6d59041a926eba98bb59bdbc85aa1302c7744bc23bc6b4a58680df2a6ba609`

The new E-000 tests passed `9/9`, the full E-001 regression passed `21/21`, and two independent
incremental reviews returned `APPROVE`. Formal watchdog job `20260804_040000_19368569` succeeded
on GPU 2. Its checker returned `passed=true`, AEPE `1.4612304443177186`, and maximum absolute
pair delta `0.0` against the fixed reference.
