# GDROS and CRFT Reproduction Status

Date: 2026-08-04 (Asia/Shanghai)

Remote root: `/mnt/sda/zhushihao/VS_Match`

## Status

Both official repositories, checkpoints, Python environments, training smoke
tests, and evaluation paths are working on `bl_tunnel`. The GDROS OS baseline
is now formally reproduced on the authors' complete processed `testing.zip`.
CRFT formal reproduction remains pending on its downloaded official archive.

Results on locally regenerated data are diagnostic only and must not be cited
as paper reproduction results.

## GDROS Formal OS Reproduction

Completed on 2026-08-04 with the official 1,973,635,350-byte archive.

- Archive SHA256: `e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218`
- Published dataset: `/mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official_e000_archive`
- Samples: 424, full-image evaluation without a valid mask
- AEPE: `1.461230444`
- CMR@1/2/5: `33.726415/81.603774/99.528302%`
- Formal result: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-000/run-01/formal-official-archive`
- Validation: `passed=true`; maximum per-pair delta from the independently checked equivalent
  materialization is `0.0`

## Remote Setup

- Hardware: 7 x NVIDIA RTX 4090 24 GB
- GDROS commit: `ee6b6216fe780bc03bfb4759ae094daa0cfddc60`
- CRFT commit: `4211ec7ca5375b0b34cf22100cf6bf26feb31470`
- GDROS environment: `/mnt/sda/zhushihao/VS_Match/.conda/gdros`
- CRFT environment: `/mnt/sda/zhushihao/VS_Match/.conda/crft`
- GDROS branch: `codex/reproduction`
- CRFT branch: `codex/reproduction`

The reproduction patches in both repositories are intentionally uncommitted.
Do not reset or overwrite them.

## Completed Checks

| Method | Evaluation data | Measured | Paper target | Conclusion |
| --- | --- | --- | --- | --- |
| GDROS | official processed OS, 424 pairs | AEPE 1.46123; pair CMR@1/2/5 33.73/81.60/99.53% | AEPE 1.48; variance about 0.75; CMR 33.88/80.19/99.45% | Formal reproduction passed |
| GDROS | regenerated OS, paper-style, 424 pairs | AEPE 4.11646; RMSE 8.40548; pair CMR@1/2/5 16.75/50.24/79.01% | AEPE 1.48; RMSE 0.75; CMR 33.88/80.19/99.45% | Not a formal reproduction |
| CRFT | regenerated OS, release-style, 424 pairs | AEPE 1.76162; pair CMR@3/1/0.7 93.63/17.22/8.49% | AEPE 0.65; CMR 99.0/95.1/89.9% | Not a formal reproduction |

CRFT loaded all `275/275` checkpoint tensors with zero missing or unexpected
keys. Its released checkpoint records validation AEPE `0.6714`, close to the
paper, which supports a fixed-data protocol mismatch rather than a damaged
checkpoint.

CRFT flow-convention diagnostics ruled out direction and file-format errors:
the regenerated affine flow and loader flow have maximum EPE residual `0.0`.
Translation magnitude is the main observed factor. The `<5 px` subset has AEPE
`0.69038`, while the full regenerated set has AEPE `1.76162`. Selecting this
easy subset would be metric chasing and is not an acceptable reproduction.

One-step training smoke tests completed successfully:

- GDROS: about 1.565 s, peak 1861 MiB
- CRFT: about 1.667 s, peak 1866 MiB

These are functional smoke measurements, not full-training time estimates.

## Official Archive Status

Both archives were downloaded locally with the expected byte counts. GDROS has
completed integrity checks, transfer, extraction, provenance validation, and
official-checkpoint evaluation. CRFT still needs its independent formal archive
workflow before its reproduction block can be closed.

- GDROS: `/Users/shihao/Downloads/os_dataset/testing.zip`
  (`1,973,635,350` bytes), complete and formally evaluated
- CRFT: `/Users/shihao/Downloads/OSdataset.zip`
  (`4,723,757,478` bytes), formal remote evaluation pending

Official archive records:

1. GDROS OS `testing.zip`
   - Official share: <https://pan.baidu.com/s/1Tn6bkfgXeKQg79rvuUswmg?pwd=7ewa>
   - Exact size: `1,973,635,350` bytes
   - Destination: `datasets/official_archives/gdros/testing.zip`
2. CRFT `OSdataset.zip`
   - Official share: <https://pan.baidu.com/s/1UyCt8oPjN9b0YDfmRjtOWw?pwd=qjxy>
   - Exact size: `4,723,757,478` bytes
   - Destination: `datasets/official_archives/crft/OSdataset.zip`

No verified non-Baidu mirror was found on GitHub Releases, Hugging Face,
Kaggle, Zenodo, Figshare, ModelScope, or Papers with Code. The similarly named
Kaggle archive already present under `official_archives` is the raw aligned
OSdataset and is not a substitute for either processed archive.

The authors published no trustworthy checksum. After download, verify the
exact byte count, run `unzip -t`, and record a local SHA256 before extraction.

For full GDROS retraining, `training.zip` has exact size `5,727,075,338` bytes.
It is not needed for the current checkpoint-based E-001/E-002 feasibility path
and should only be fetched if a later approved experiment requires training.

## Minimum User Action

No user action is needed for GDROS. CRFT's local `OSdataset.zip` still needs a
separate verified transfer/extraction/evaluation workflow when CRFT reproduction
is resumed. Do not send passwords, cookies, BDUSS, or STOKEN through chat.

## Diagnostic Artifacts

- GDROS metrics: `GDROS/results/os_regenerated_paper_seed20260803_diagnostic.json`
- CRFT metrics: `CRFT/results/os_regenerated_release_seed20260803/metrics.json`
- CRFT flow diagnostics: `CRFT/results/os_regenerated_release_seed20260803/flow_convention_diagnostic.json`
- CRFT grouped diagnostics: `CRFT/results/diagnostics/crft_os_release_seed20260803_grouped.json`
- CRFT grouped JSON SHA256: `336105b687512d104d68fa64dfd2ac467d5e0602c2f25e8c4ed2a17c3fe5fd28`

No reproduction process was left running after these checks.
