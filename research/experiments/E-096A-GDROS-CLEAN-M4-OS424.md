# E096A: official GDROS versus clean M4 on all OS424 pairs

User request on 2026-11-16: rerun full GDROS and compare with the newly
validated clean M4 inference implementation. No training or weight changes.

## Fixed comparison

| Setting | GDROS | M4 |
|---|---|---|
| Weights | Released GDROS OS512 | Protected OS step80K |
| Implementation | Unmodified official GDROS, test_mode=True | E095A CleanM4 |
| Iterations per matcher | 32 | 32, for each of three matchers |
| Terminal LSR | Official LSR | Same official LSR |
| Total LSR calls per pair | 1 | 5 |
| Dataset | Full official OS 424 test pairs | Same pairs and input tensors |
| Precision | FP32, both TF32 flags off | Same |

Image loading, checkpoint paths and strict loading reuse the audited E092A
loader. Inputs are whole 512x512 RGB float32 0..255 tensors, optical first,
SAR second, batch=1, without resizing/cropping/valid masks. Official LSR is
`LSR(1, 512, 512, flow, 200000)`.

Each variant receives three warmup pairs (0,211,423). Both are then rerun on
every pair on the same GPU, alternating which runs first. Synchronized wall
clock includes forward and terminal LSR, excludes load/decode/H2D/output
validation/save/GT scoring, and records actual LSR calls. GPU process/load
snapshots document sharing; this is not an exclusive GPU benchmark. GDROS
does not receive the inference-cleanup transformations applied to M4.

The run verifies protected checkpoint SHA256 and the saved E095A clean
prediction manifest. Current M4 outputs are compared with E095A pair by pair
and must stay within 1e-4 px maximum absolute difference. Prediction does not
read GT. Only after all 848 predictions are saved does the CPU GT join compute
public component error and vector EPE independently. CMR thresholds apply
strictly to per-pair mean errors. Sources are snapshotted and hashed before
and after the run.

## Reproduction

Stage `e096a_gdros_clean_m4_eval.py` and the E092/E095 loader/helper chain in
the experiment host's `.codex_tmp/`, then run from the migrated project root:

```bash
CUDA_VISIBLE_DEVICES=<available_gpu> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 <python> .codex_tmp/e096a_gdros_clean_m4_eval.py \
  --output-dir <fresh_empty_directory>
```

Final result directory:
`research/results/E096A_GDROS_CLEAN_M4_OS424_20261116/`.
Dense NPY files stay on zzvr outside Git. Compact CSV-equivalent JSON rows
can be independently checked without data, weights or a GPU:

```bash
python3 research/remote_stage/os/verify_e096a_results.py \
  research/results/E096A_GDROS_CLEAN_M4_OS424_20261116
```

On zzvr, add `--verify-predictions` to rehash all 848 dense predictions.

Final status: COMPLETE. The full 424-pair comparison ran on shared RTX 5090
GPU3. GDROS / clean-M4 mean latency was 221.343 / 571.037 ms (M4 2.580x).
Public component mean was 1.461660750 / 1.474417970 px, CMR@1 was
33.72641509 / 72.40566038%, CMR@2 was 81.60377358 / 84.90566038%, and
CMR@5 was 99.52830189 / 94.10377358%. M4 matched E095A bitwise on all
424 pairs. Both remote dense-file hash verification and local compact
statistics verification passed; see the full report and VERIFICATION.json.
