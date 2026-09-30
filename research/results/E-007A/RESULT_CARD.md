# E-007A Result Card

Experiment: `M001 v1.0 / E-007A Patch / Structure / Descriptor Evidence Diagnostic`

Status: `COMPLETED`

Interpretation: `NO_USEFUL_OBSERVATION_GAIN`

## Outcome

The frozen E-006A baseline was reproduced exactly at topK-32 precision `2.2587%`, AEPE `63.5157 px`, and CMR@2 `0%`.
No patch, structure, descriptor, or fusion track passed the preregistered STRONG, USEFUL, or WEAK gate.

The best affine result was `all_evidence`: AEPE fell by `27.4396 px` to `36.0761 px`, but topK-32 precision fell to
`1.7868%`, AP fell to `0.00984`, and CMR@2 remained `0%`. It therefore does not establish useful candidate
selectability. The review seed gave the same conclusion (`2.0808%`, `36.6698 px`, CMR@2 `0%`).

## Main Dev Results

Official group-isolated dev split, 404 pairs, full 512x512 no-mask affine metric. Precision is global topK-32.

| Track | AUROC | AP | ECE | P@32 | AEPE px | CMR@2 | CMR@5 | PLR | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E-006A D2 baseline | 0.9171 | 0.01386 | 0.3133 | 2.2587% | 63.5157 | 0% | 0.9901% | 11.8812% | 93.8119% |
| raw patch only | 0.5337 | 0.00166 | 0.2768 | 0.2321% | 138.0443 | 0% | 0% | 0% | 99.5050% |
| frozen feature patch only | 0.8135 | 0.00586 | 0.1917 | 1.2067% | 90.5367 | 0% | 0.2475% | 1.2376% | 96.2871% |
| patch combined | 0.8142 | 0.00601 | 0.1798 | 1.2531% | 87.7608 | 0% | 0.2475% | 1.9802% | 97.0297% |
| handcrafted structure | 0.4868 | 0.00129 | 0.4538 | 0.1238% | 180.9078 | 0% | 0% | 0% | 89.3564% |
| learned structure | 0.5505 | 0.00159 | 0.2756 | 0.3094% | 140.5526 | 0% | 0% | 0% | 99.7525% |
| descriptor only | 0.6072 | 0.00240 | 0.2213 | 0.8663% | 127.8814 | 0% | 0% | 0% | 100% |
| descriptor + patch | 0.7799 | 0.00495 | 0.1370 | 1.3072% | 99.9055 | 0% | 0% | 0.2475% | 99.5050% |
| descriptor + structure | 0.6082 | 0.00239 | 0.2041 | 0.8509% | 128.2480 | 0% | 0% | 0% | 100% |
| all evidence | 0.8997 | 0.00984 | 0.0738 | 1.7868% | 36.0761 | 0% | 0.2475% | 7.9208% | 99.5050% |

For all-evidence, topK-16/32/64 precision was `2.3360% / 1.7868% / 1.4929%`; AEPE was
`57.9356 / 36.0761 / 33.0043 px`. The uncapped 0.5-threshold and region-balanced-32 diagnostics reached
`34.2085 px` and `34.9575 px`, respectively, but both retained CMR@2 `0%`.

## Groups And Resources

- All-evidence AP was `0.01062` in-bounds and `0.00057` out-of-bounds. By candidate source type, AP was
  `0.01440` global, `0.01132` region, and `0.00828` local.
- TopK-32 used exactly 32 anchors. Region count p10/p50/p90 was `20/24/26`; 403/404 pairs fell in the
  `16-31` region bin, where AEPE was `34.8951 px`, CMR@2 `0%`, PLR `7.9404%`, and valid `99.7519%`.
- Frozen observation extraction p50/p95 was `81.36/173.75 ms`; joint candidate scoring was `3.73/4.48 ms`.
  The isolated all-evidence verifier was `0.762/0.807 ms`. Maximum end-to-end allocated memory was
  `315,151,872 bytes` (`300.55 MiB`). No dense retrieval was used.

## Integrity And Reviews

- Automatic tests: `14/14 PASS`.
- Observation manifests: `10/10 PASS`, exact `1607/404`, candidate parity true, finite evidence tensors.
- Label manifests: `10/10 PASS`; label join occurred after observation materialization.
- Formal training: both seeds PASS, 1,607 records, 104,608 samples, fixed 10 epochs.
- Hard negatives: same-region `25,124`, same-basin `29,888`, high-cosine `29,888`, explicit fallback `4,764`.
- Independent reviews: feature `APPROVE`, descriptor/training `APPROVE`, evaluator `APPROVE`.
- No OS424, confirm, datum labels, candidate expansion, GDROS training, dense retrieval, RANSAC, IRLS, or baseline fallback.

## Provenance

Frozen SHA256:

- `CURRENT_STATE.md`: `6eca5e85906c6e42f78bc27a48378903f3fa248c220f68f1cdee72be45e7e1be`
- `METHOD_DELTA.md`: `5c9125b147db5828e85e5eb122080d08b69ea1666a9388e1680cb4b9fd022cf8`
- `INNOVATION_CONTRACT.yaml`: `04ef30c09daf70535be35fc20f65c5c27c0cf6b53e3809d77af16e532f9bbbb3`
- `E-007A.md`: `6b47bcf3b78e975a989772cb4062387020197b4aa7de0c674b72c33736cbc3d6`
- split manifest: `e8c990888123896af425923970215eb1a9ec8aca89eae0421998d2bdc13a99d3`
- GDROS checkpoint: `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- E-006A corrected source: `a26f9619ad395bdbc1d45b3ec09ae256d603512042681259efdc8aac4e486d2a`
- seed checkpoints: `9eaf3e6b...14201f15`, `b954b1fb...ec44490`
- formal summary/per-pair: `a9b7f44c...f8ace47`, `de26d3c6...f246e9`

## What This Establishes

On the fixed corrected E-006A candidate set, these first-version raw/feature patch verifiers, robust structure channels,
and train-only local descriptor adapter do not make 2px-correct candidates selectable at the preregistered purity.
Fusion can substantially stabilize the ordinary affine fit without increasing point correctness, so coordinate correctness
remains the limiting factor rather than verifier runtime.

This result does not reject patch, structure, or descriptors in general, nor does it test proposal expansion, a different
candidate generator, a robust multi-basin solver, or OS424 generalization. No follow-up experiment is authorized here.

## Artifacts

- Remote run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-007A/run-01`
- Remote summary: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-007A/run-01/eval-formal-v1/summary.json`
- Remote per-pair rows: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-007A/run-01/eval-formal-v1/per-pair.jsonl`
- Remote checkpoints: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-007A/run-01/model-formal-v1`
- Remote watchdog log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260807_122907_759127e6.log`
- Local summary/per-pair: `research/results/E-007A/raw/summary.json`, `research/results/E-007A/raw/per-pair.jsonl`
- Gate: `research/results/E-007A/gate.json`
- Reviews: `research/results/E-007A/reviews/`
- Manifest: `tmp/e007a_stage/RUN_MANIFEST.json`
