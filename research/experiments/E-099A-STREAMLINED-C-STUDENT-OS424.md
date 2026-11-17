# E099A: genuinely streamlined C/student standalone inference

Requested 2026-11-16: extract and streamline C/student, measure its own
parameter count, standalone latency and all official OS424 metrics. This is
not a timing comparison against full M4-streamlined and not the old unpruned
C-only wrapper. No training, checkpoint selection or changes to active weights.

## Model and fixed protocol

`remote_stage/os/m4_streamlined_c_student.py:M4StreamlinedCStudent` extracts
only the live protected `c.student` graph, using E095A's verified
`RecurrentMatcher`, `FeatureFrontend` and `DescriptorAdapters`. It registers
one frozen DINO, one frozen descriptor-adapter pair, two feature frontends,
two four-level correlations, one context/update/GRU stack and one final
mask/upsampling. It registers no B matcher, C teacher or external fusion gate.
Lookup offsets are cached as a buffer; descriptors are cached within a pair.

Weights: protected OS M4 step80K,
SHA256 `55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b`.
Read the protected payload on CPU, use E093A's strict C-prefix constructor
to build only C/student (no B or C teachers), then release the payload.
Keep an unclean C/student reference only for three untimed raw/post checks, then
delete it and clear the allocator before standalone timing. No full M4
inference is required.

Fixed: all 424 author-public OS pairs, optical first, SAR second, 512x512,
float32 0..255 input, batch1, 32 recurrent steps, both TF32 flags off, cuDNN
benchmark off. Only one official terminal `LSR(1,512,512,flow,200000)`,
zero internal LSR. No cropping, mask or hard-pair filtering.

Fresh predictions are checked against the hashed historical C-only 32-step
predictions revalidated by E098A. This establishes cleanup parity only; it
does not treat cached latency as a current benchmark. Three direct C checks
use indices 0/211/423. Raw tolerance 1e-5, post tolerance 1e-4; exact counts
and maximum differences must be reported rather than assumed.

## Required evidence

- Parameter counts for unique registered tensors, frozen DINO, descriptor
  adapters, feature frontends and recurrent/context/FiLM/scalar parts.
  `requires_grad` flags from the frozen training checkpoint are reported
  separately; inference does not train them.
- One instrumented coverage call checks every parameterized leaf and actual
  GRU32, motion32, flow32, mask1, context1, two feature calls, DINO2,
  correlation-build2 and correlation sampling256, terminal LSR1. Hooks and
  wrappers are removed before timing.
- Delete the unpruned reference before timing. Three warmups, synchronized
  GPU wall-clock network plus terminal LSR; exclude load/decode/H2D, validation,
  prediction saves and GT scoring. Report mean/median/P95/std and FPS.
- Record GPU index/load snapshots and CUDA live/peak allocated memory after
  cleanup; do not claim exclusive execution without evidence.
- Save all predictions before GT metric join. Compute public component-error
  means/CMR and separately standard vector EPE. Hash all predictions and all
  relevant sources before/after; verify no source changed mid-run.
- Save compact state outside Git; independent offline verification must
  recompute statistics, all424 cached-control comparisons and call counts.

Default M4-streamlined stays unchanged. No training or Git publication is
part of this experiment request.

## Final status: COMPLETE

424 fresh predictions match the accepted C cache bitwise, maximum difference
zero. The three direct raw/post comparisons are also exact. Coverage passed,
including 32 GRUs, two correlations and one LSR. Parameters: 319,303,309,
including 304,367,616 DINO parameters. Standalone mean/median/P95 on GPU3:
143.4758/143.2257/145.7949 ms. Public-component mean 1.5929032520 px,
CMR@1/2/5 69.5755/85.1415/92.9245%, failures>=5 30/424. Independent local
statistics and remote full prediction/state hashing passed. See
`results/E099A_STREAMLINED_C_STUDENT_OS424_20261116/REPORT.md`.
