# E097A: retain only LSR slot 5, then slots 4 and 5

Requested 2026-11-16. No training, all protected M4 step80K parameters,
three 32-step matchers, shared frozen features and gates are preserved.
This is a fixed-weight inference ablation of geometric gate inputs.

## Dependency analysis and explicit missing-input rule

| Logical slot | Flow fitted | Consumer |
|---|---|---|
| 1 | F2 final flow | B gate's scalar norm of F2 geometric residual |
| 2 | F3 final flow | B gate's scalar norm of F3 geometric residual |
| 3 | B fused flow | Final M4 gate's fitted B flow and componentwise absolute residual |
| 4 | C/student final flow | Final M4 gate's fitted C flow and componentwise absolute residual |
| 5 | M4 fused flow | Final registered output |

The B gate takes 9 channels: F2(2), F3(2), their signed difference(2),
difference norm(1), and two geometric-residual norms(1 each). The M4 gate
takes 16 channels: B(2), C(2), signed difference(2), absolute difference(2),
fitted B(2), fitted C(2), absolute B residual(2), absolute C residual(2).

Removing fits requires defining the missing inputs. In this experiment,
a skipped fit returns its input flow: `fit(f) := f`, so its residual is
zero. Dimensions and weights remain unchanged. This is an identity-flow
substitution, not setting fitted flow to zero or removing a matcher.

| Variant | Actual official LSR slots | B-gate change | M4-gate change |
|---|---|---|---|
| reference5 | 1,2,3,4,5 | None | None |
| final1 | 5 | Both residual norms zero | Fitted B/C replaced with raw B/C; both residuals zero |
| last45 | 4,5 | Both residual norms zero | Fitted B replaced with raw B and its residual zero; real C fit/residual retained |

Both ablations produce the same B flow for a fixed pair; C's matcher is
also unchanged. Their difference isolates retaining C geometry at the final
gate, conditional on the same ablated B gate. Neither is guaranteed to
improve or preserve accuracy: zero residual can look like perfect geometric
consistency to a trained gate, and fitted-flow channels now have a different
distribution. A worse result does not establish that retraining a smaller
gate without those inputs would fail. A better result applies to this frozen
checkpoint and this evaluation set, not automatically to other datasets.

## Execution and verification

Implementation: `remote_stage/os/m4_lsr_ablation.py:SelectedLSR`. It wraps
the validated CleanM4 callback without modifying CleanM4 or the checkpoint.
Every inference must traverse exactly five logical slots and execute exactly
the declared subset. The original official LSR is counted separately, so
identity callback visits are never reported as fits. Terminal slot 5 remains
enabled in every policy.

The full runner is `remote_stage/os/e097a_lsr_slot_ablation_eval.py`. It
uses the same E092A strict CPU loader, official OS424 input format and LSR
wrapper as E095A/E096A. Both TF32 flags are off; FP32, batch=1, 512x512,
three warmups per variant. Warmups run final1 first, then last45, then
reference5. Measurement cycles all six variant orders across the same
424 pairs, to compare timing under similar GPU load.

Synchronized wall-clock timing includes all three recurrent matchers,
fusion and terminal LSR, excludes initialization, decoding, H2D, output
validation, saving and GT scoring. Baseline final predictions must reproduce
the E096A M4 archive within 1e-4 px maximum difference. Experimental variants
may differ; their raw and final output differences are recorded.

GT is read only after all 1,272 final predictions are saved. Public component
error is `mean_pixels((abs(dx_error)+abs(dy_error))/2)`, CMR uses strict
per-pair mean thresholds, and vector EPE is reported separately. All source
files are snapshotted and hashed before/after the run. Dense NPY stays on
zzvr outside Git.

```bash
CUDA_VISIBLE_DEVICES=<available_gpu> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 <python> .codex_tmp/e097a_lsr_slot_ablation_eval.py \
  --output-dir <fresh_empty_directory>
```

Stage the runner, selection helper, validated CleanM4 and existing E092A
loader chain in `.codex_tmp/`; run from the migrated project root.

Final report and compact evidence will be under
`research/results/E097A_M4_LSR_SLOT_ABLATION_OS424_20261116/`.

Final status: COMPLETE. All three policies ran all 424 pairs on shared RTX
5090 GPU7. Five-fit / final1 / last45 public means were
1.474417970 / 1.510789589 / 1.482544107 px; CMR@1 was
72.40566038 / 71.46226415 / 70.51886792%, and CMR@5 was
94.10377358 / 93.63207547 / 93.63207547%. Mean latencies were
354.466 / 337.404 / 343.778 ms. The full control matched E096A bitwise
on all pairs. Statistics and all 1,272 dense-file hashes passed independent
verification. Default inference remains five fits.

```bash
python3 research/remote_stage/os/verify_e097a_results.py \
  research/results/E097A_M4_LSR_SLOT_ABLATION_OS424_20261116
```
