# E094A: omit unused intermediate M4 fusion at inference

## Question and scope

With the protected OS E075A-M4 step80K checkpoint, keep all 32 recurrent
updates and the currently loaded C-teacher setting, but compute the B and
outer M4 gates only for the requested final iteration. This reduces official
LSR calls per image pair from 129 to 5. No training, checkpoint selection,
weight change, correlation change, or terminal-LSR change is permitted.

The optimization removes the intermediate outer gates as well as their LSR
features. Any measured speedup must not be described as LSR-only profiling.

## Prediction equivalence

- Evaluate all 424 author-public OS pairs in the existing E092A order.
- Same model instance and weights for original129 and final5; alternate
  their execution order by pair index.
- Compare final raw flow (maximum absolute tolerance 1e-5 pixels) and final
  post-LSR flow (1e-4 pixels), also report exact equality counts.
- Stop on a raw/post equivalence failure rather than change tolerances.
- Compare the current original post-LSR outputs to the hash-verified E092A
  archived 32-iteration outputs. Report archive differences separately.
- Materialize predictions before reading GT. Join public component error
  and standard vector EPE separately, using the existing E092A formulas.
- Record actual LSR invocation counts, not only static estimates.

## Timing

Same physical GPU and process; batch=1, 512x512, FP32, AMP/TF32 disabled,
one CPU compute thread. Three warmups per variant precede all 424 paired
measurements. CUDA-synchronized wall-clock timing includes the forward and
one terminal official LSR, all internal CPU/GPU transfers and gates. It
excludes checkpoint loading, image decoding, input transfer, output
validation, prediction archival, GT join, and nvidia-smi snapshots.

Record the GPU UUID and external processes. This run uses a shared GPU;
paired alternating order reduces drift but does not establish exclusive
latency. Do not compare its absolute times directly with the earlier quick
10-pair measurement taken on a different GPU/load.

## Implementation

- `../remote_stage/os/m4_final_only_lsr.py`: inference-only final raw flow.
- `../remote_stage/os/e094a_final_only_lsr_eval.py`: paired evaluation and
  timing, strict frozen-checkpoint load, source snapshots and result cards.
- Requires the existing E092A loader, quick timing snapshot utility, legacy
  model dependencies, checkpoints, official data and archived predictions.
- The E074/E075 remote inference interfaces already default to skipping the
  unused C teachers. E094A records and preserves that setting in both paths.
  The older local source snapshot may execute them; do not conflate those
  implementations when interpreting timing.

Stage the scripts beside `e092a_official_eval.py` in the migrated project's
`.codex_tmp` and run from that project's root with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES="$BENCH_GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 "$BENCH_PYTHON" -u \
  .codex_tmp/e094a_final_only_lsr_eval.py --output-dir "$BENCH_OUTPUT"
```

Large predictions and runtime state stay on the experiment host. Compact
result cards, comparisons, timing records and reports are synchronized after
completion. Running this protocol is not evidence of equivalence or speedup
until the result card reports completion.
