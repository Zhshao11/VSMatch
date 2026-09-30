# E-004A Result Card

Status: `COMPLETED / P0 PASS / G0 PASS / G1 FAIL / G2 NOT RUN`

Method: `M001 v0.7 / Coverage-Constrained Adaptive Anchor Budgeting`

Decision: stop at the frozen G1 gate and return to the ACTIVE method task. G2 was not run because G1 did
not fully pass. Do not start E-004B, RG-SAC, a rescue variant, or post-result threshold/budget tuning.

## Frozen identity

- E-004A specification SHA: `6110948722ff841257a938e59ac8e7682666491944b57c75eadfefb42b9ccdc9`
- Final G0 source SHA: `76c735d1fbe34543903148bd69c1f8f43a0f9ed8f855d44334565256a92b9453`
- Head-cache content SHA: `163d2a7dbe0ece96d5a1991f908ecda63521e2e60e749774e75ad813892f1197`
- Split manifest SHA: `e8c990888123896af425923970215eb1a9ec8aca89eae0421998d2bdc13a99d3`
- Official GDROS checkpoint SHA: `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- Train/dev: `1,607 / 404` pairs. No OS424 or confirm data was used.

## Completed gates

- P0: PASS on all 9 gates over 404 dev pairs. The GT-constrained adaptive oracle reached `61.7942%`
  recoverable anchors, median `78/128`, AEPE `2.08482 px`, and PairLocalReady `100%`.
- G0: `23 passed`; real-checkpoint guarded test also passed separately. Two independent final reviews were
  `APPROVE / APPROVE` on source SHA `76c735...9453`.
- Head cache: complete for all `2,011/2,011` manifest pairs (`1,607` train and `404` dev).
- Training: all three frozen seeds completed successfully in parallel and their checkpoint provenance matches
  the frozen source, split, cache and pair counts.

## Training

| Seed | Best pre epoch / dev BCE | Best post epoch / dev BCE | Checkpoint SHA |
|---:|---:|---:|---|
| 20260805 | 48 / 0.863661 | 33 / 0.852626 | `1a61c07a...a7113` |
| 20260806 | 21 / 0.866798 | 47 / 0.855277 | `70dd1163...faaa` |
| 20260807 | 37 / 0.865550 | 43 / 0.853971 | `73a1126b...cf0` |

Each checkpoint contains `1,473` pre-head and `1,601` post-head parameters. The three watchdog jobs ran on
GPUs 2, 3 and 4 and all returned code 0.

## G1 main result

The frozen main seed is `20260805`.

| Check | Required | Observed | Result |
|---|---:|---:|---|
| Recoverable fraction | >=50% | 41.7969% | FAIL |
| Median recoverable anchors | >=64 | 53 | FAIL |
| Gain over BIND-001 uniform | >=10 pp | +9.6767 pp | FAIL |
| Gain over no replacement | >=3 pp | +0.0522 pp | FAIL |
| Oracle AEPE | <=2.5 px | 2.45217 px | PASS |
| PairLocalReady | >=99% | 99.7525% | PASS |
| Valid affine | >=99% | 99.7525% | PASS |
| Fine searches / region / K / finite / deterministic | frozen invariants | all satisfied | PASS |

The other seeds fail the same four gates. Their recoverable fractions are `41.6209%` and `41.6460%`, with
medians `52/128`. All three seeds improve over uniform and no-replacement in the same direction, so the
result is stable rather than a seed anomaly.

For the main seed, the pre-head ranking signal is moderate (AUROC `0.69783`, AP `0.52816`), but it does not
close the P0 oracle headroom. The post-head accepted only `519/11,153` reserve attempts (`4.653%`) and added
just `0.0522 pp` recoverability over the learned primary set. This isolates the main failure to learned
anchor acquisition and especially ineffective reserve replacement, not affine solvability after good
anchors are available.

## G2

`NOT_RUN_DUE_TO_G1_FAIL`. No formal selector-only or slim end-to-end resource result may be claimed for
E-004A.

## Absolute artifacts

- Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-004A/run-01`
- G1 gate JSON: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-004A/run-01/gates/g1-dev-gate.json`
- G1 per-pair JSONL: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-004A/run-01/gates/g1-dev-per-pair.jsonl`
- Heads and training summaries: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-004A/run-01/heads`
- P0 gate JSON: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-004A/run-01/gates/p0-oracle-gate.json`
- G0 reviews: `/Users/shihao/Documents/图像匹配之二/research/experiments/E-004A/G0_REVIEW_APPROVE.md`
- Training watchdog IDs: `20260806_085528_279ade08`, `20260806_085528_0b6e46eb`,
  `20260806_085528_cb23a2dd`
- G1 watchdog ID: `20260806_085925_15474740`
- G1 watchdog log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260806_085925_15474740.log`
- G1 watchdog report: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260806_085925_15474740.json`
- G1 JSON SHA: `bf56e85acb1e7759c70c899c8709f9da89c872b221684966d2bf7704dac5d24d`
- G1 per-pair JSONL SHA: `ab3e8d6c054054671cca926fbc027815589e014c3632daeb445e94991cd314dd`

## Method decision needed

P0 proves that coverage-constrained adaptive anchors have sufficient oracle headroom, while G1 shows the
current lightweight pre/post heads recover only about two thirds of that headroom and the post-search
replacement contributes almost nothing. Any next experiment therefore requires an explicit method decision
about improving the recoverability representation and/or replacement decision. E-004A itself authorizes no
automatic rescue.
