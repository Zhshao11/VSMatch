# Git Sync Audit - 2026-09-30

## Repository facts

- Repository root: `/Users/shihao/Documents/图像匹配之二`
- Branch: `main`
- Commit history: none (`No commits yet on main`)
- Configured remote: none
- Consequence: Git cannot distinguish old and new work. This audit uses file
  modification dates, experiment lineage, source content, result cards, and
  remote SHA/checkpoint provenance.

## Work completed

### Baseline and protocol

- Reproduced the released GDROS OS checkpoint under the official processed-data
  protocol and added independent dataset/metric validators.
- Added pair-level AEPE and CMR@1/2/5/10/16 contracts, prediction-before-GT
  materialization, split/provenance checks, and watchdog result cards.

### OS method line

- E072/E073/E074 established the F2/F3 branch and two-stage fusion chain.
- E075A-M4 introduced a 4x4 region gate over B/C flows and remains the protected
  OS best at step 80K.
- E075B/C/D diagnosed the OS generalization tail and ruled out simple iteration,
  final-LSR, and query-window explanations as primary bottlenecks.
- E082A explored a multiscale dense-basin generator without replacing E075A.
- The exact E072-E075 source chain has been synced to `remote_stage/os/`, with
  original remote SHA256 values recorded in `SOURCE_MANIFEST.json`.

### WHU method line

- E081A-M4 trained the common GDROS/RAFT backbone plus region refiner on WHU.
- E081A diagnostics separated backbone, M4, LSR, local oracle, and recurrent
  evidence behavior.
- E088A and E089A rejected frozen stride-8 and lightweight high-resolution
  post-hoc candidate selection.
- E090A separated representation failure from gate/action-path optimization;
  E090B rejected the tested structural relation descriptor.
- E090D added identity-safe structured evidence to the final recurrent updates
  and passed 1/4/8-pair actionability gates.
- E091A ran a group-disjoint 1600-train/800-dev validation. It improved dev
  CMR@1 by 2.25 pp but failed preregistered AEPE, confidence, and
  evidence-specificity gates. WHU 700-test remained sealed.

### LuckyR1

- Implemented belief, relation, and combined arms with clean lineage auditing.
- Current-model and clean-lineage evidence showed no substantive gain; LuckyR1
  was formally closed as `NO_GAIN`.

## Classification

### A. Ready to commit

- Repository hygiene and entry points: `.gitignore`, `README.md`,
  `requirements-research.txt`, `REPRODUCTION_STATUS.md`.
- Dataset and metric tools under `tools/`.
- Research protocol and method documentation under `research/*.md` and
  `research/*.yaml`.
- Experiment specifications, implementation modules, tests, and manifests under
  `research/experiments/`.
- Current method implementations under `research/remote_stage/`, including the
  OS E072-E075 source chain and WHU E081-E091 source/DAG.
- `e081a_whu_our_native_train.py` and
  `e082a_os_multiscale_dense_basin_generator.py`.
- Compact result cards and audit summaries under `research/results/`; generated
  `.npz`, `.npy`, `.jsonl`, `.csv`, raw predictions, and per-pair payloads are
  excluded by `.gitignore`.

### B. Valuable but needs organization before commit

- Root `e075b_*`, `e075c_*`, and `e075d_*` scripts: one-off remote orchestration
  helpers with hard-coded deployment paths and overlapping stage-specific code.
  Their durable method logic/results are already represented in
  `research/remote_stage/os/` and result cards.
- Root `lucky_*` scripts: historical scheduler repair and migration helpers.
  The maintained LuckyR1 implementation is under `research/remote_stage/luckyr1/`;
  root copies should be consolidated before inclusion.
- `image_matching_part2_migrated/.codex_tmp/`: older Proposed-A source and tests
  are potentially useful but remain in a temporary-named directory. They should
  be moved to a named legacy experiment module and supplied with a manifest
  before a later commit.
- Historical compact JSON summaries that do not have a result card or manifest
  should be reviewed experiment-by-experiment before broad inclusion.

### C. Excluded from Git

- `tmp/` (about 3.8 GB): archive chunks, worker logs, caches, and generated runs.
- Root `.codex_tmp/`: local staging, cached official source, and an embedded Git
  repository.
- `output/`: generated PDF figures.
- Datasets, checkpoints, model weights, archives, logs, and run directories.
- Dense result payloads: predictions, pair-level CSV/JSONL, NPZ/NPY arrays,
  train curves, and scheduler runtime cards.
- Python/editor/OS caches.

## Reproducibility assessment

The repository now explains the original methods, the E075/E081 baselines, the
subsequent changes, commands, data contract, metric definitions, checkpoint
selection, and held-out conclusions. Current scripts accept `SPARSEGEO_ROOT`
and related environment overrides while retaining the historical zzvr defaults.

It is not standalone: upstream GDROS, WHU/OS datasets, and checkpoints are
external and intentionally excluded. With those assets present at the documented
paths, the current E081-E091 workflow and watchdog DAG are reproducible. Exact
checkpoint/source hashes are recorded where available.

## Verification

- Python syntax compile: pass for tools, current remote-stage scripts, E081A,
  and E082A.
- Tool unit tests: 15/15 pass using the bundled workspace Python.
- E003A/E004A historical research tests: not runnable locally because the
  available bundled Python lacks PyTorch and their historical package layout
  expects an upstream `experiments` package. They remain documented remote tests,
  not claimed as locally passing.
- Secret scan: no private-key, AWS-key, API-key, client-secret, or access-token
  assignment found in the commit candidate scope.
- E091A compact result JSON and watchdog plan JSON parse successfully.

## Proposed commit structure

1. `chore: establish reproducible image matching research workspace`
   - Git hygiene, top-level documentation, dependencies, baseline tools,
     protocols, experiment specifications, and tests.
2. `feat: add SparseGeo OS and WHU method improvements`
   - E072-E075 OS chain, E081-E091 WHU chain, LuckyR1 maintained source,
     watchdog DAG, and compact result cards.

Do not include category B or C files in these commits. Do not push until a
remote URL and push authorization are provided.

## User confirmation needed

- Approve the two-commit split and commit messages.
- Decide whether category-B root orchestration scripts should be refactored and
  included now or deferred.
- Provide/confirm the Git remote URL before any later push.
