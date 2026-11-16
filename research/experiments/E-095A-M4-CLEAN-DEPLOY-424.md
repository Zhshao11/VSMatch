# E095A: remove inference redundancy from protected M4

Requested on 2026-11-16: identify computation and parameter storage that do
not contribute to the final M4 inference output, remove them from a separate
deployment path, and rerun all 424 official OS test pairs without training.

## Scope and control

The protected OS step80K checkpoint remains unchanged, SHA256
`55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b`.
The paired control is E094A's verified five-LSR implementation with
`c_skip_teachers=True`, not a newly trained model. Both paths retain all
three 32-step recurrent matchers, five four-level correlation pyramids,
active adapters and learned gates, and five official LSR calls per pair.

The deployment implementation is `../remote_stage/os/m4_clean_deploy.py`.
It extracts only active modules from a strictly loaded protected architecture.
It is inference-only; historical training and sequence-supervision paths are
preserved. No untested deletion is applied to the original checkpoint.

## Removed work and storage

| Component | Control | Clean deployment | Reason |
|---|---|---|---|
| Intermediate mask head and 8x upsampling | 96 each per pair | 3 each | Only final high-resolution output is consumed; recurrent coordinates stay at low resolution |
| Nested V1/V2 context CNN | 6 calls | 0 | Actual FeatureEncoder ignores its context argument; the 3 active GRU context encoders remain |
| Frozen DINO forward | 6 calls | 2 | Three backbone state/config groups must prove identical before reuse; trained FiLM projections remain separate |
| Frozen optical/SAR descriptor adapters and resize | 3 sets | 1 set | Identical frozen state/config, same images and output sizes; active FiLM projections remain separate |
| Correlation lookup offsets | Reconstructed/transferred every lookup | One registered buffer | Same numeric grid and lookup arithmetic |
| F3 blended feature outputs and diagnostic scalar transfers | Computed and discarded | Omitted | Final inference consumes recurrent flow only |
| C teachers, nested V1/V2 update blocks, unused cost perceivers, unused heads/scalars | Registered but unused | Omitted from deployment model | Storage-only reduction; C teachers were already skipped by the control |

The recurrent flow head runs 96 times in both paths. Its outputs and hidden
states participate in subsequent iterations and cannot be removed by this
dependency argument. DINO itself, its learned transformer layers, and active
optical/SAR adapters are retained through verified shared frozen computation.

## Execution and verification

Run the staged evaluator from the migrated project root on zzvr, with the
existing frozen E092A loader and dependencies available:

```bash
CUDA_VISIBLE_DEVICES=<available_gpu> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 <python> .codex_tmp/e095a_clean_deploy_eval.py \
  --output-dir <fresh_run_directory>
```

The evaluator verifies the checkpoint and E094A prediction manifest, hashes
and snapshots sources, strictly loads the reference on CPU, checks frozen
DINO state/config equality, and warms each path on 3 pairs. Parameterized leaf
modules in the clean graph must execute in an additional coverage pass.

Every test pair is evaluated in both paths with alternating order. Timing
uses synchronized GPU wall clock, includes forward and terminal LSR, and
excludes loading, decoding, H2D, file archival and GT scoring. Both TF32 flags
and cuDNN benchmarking are disabled. Evaluation stops if raw output maximum
absolute difference exceeds 1e-5 or post-LSR difference exceeds 1e-4.
GT is joined only after both variants save all 424 predictions. Metrics use
the same public component-error and vector-EPE definitions as E094A.

The compact state artifact is saved remotely after validation. Its current
loader first constructs CleanM4 from the protected legacy architecture and
then strictly loads the compact state; it is not an independent cold-start
architecture factory. Dense predictions and model files are excluded from Git.

For inference integration, construct on CPU before moving to the GPU:

```python
legacy, _ = base.load_m4(torch.device("cpu"))
clean = CleanM4(legacy).eval()
del legacy
clean = clean.to(device)
raw_flow = clean(source, target, iters=32, lsr_fn=official_lsr)
final_flow = official_lsr(raw_flow)
```

`source` and `target` use the frozen OS evaluator's float32 RGB 0..255,
`[1, 3, 512, 512]` tensors, optical first and SAR second. `official_lsr` must
be the same wrapper `LSR(1, 512, 512, flow, 200000)` used in the experiment.
Each call has a fresh descriptor cache; no image features are reused between
different image pairs. Sharing is conditional on exact frozen state/config
equality, so independently trained adapters will form different cache groups.

Results: `../results/E095A_M4_CLEAN_DEPLOY_424_20261116/REPORT.md`.

Final status: COMPLETE. The final `E-095A-M4-CLEAN-DEPLOY-424-20261116-r2`
run passed all 424 raw/post bitwise comparisons. Mean latency was
621.689 ms (reference5) versus 340.450 ms (clean), paired on shared GPU6.
Registered parameters were 1,692,064,147 versus 347,975,753. Public component
mean was 1.4744179696 px for both; CMR@1/2/5 was
72.40566038/84.90566038/94.10377358%. Independent compact-statistics and
all 848 dense-prediction SHA256 checks passed. A preliminary DINO-only-sharing
run is preserved separately on the host; the final result uses the `-r2` code
snapshot, including frozen-adapter sharing.
