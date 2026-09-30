#!/usr/bin/env python3
"""Finalize a complete step500 exploratory evaluation after external cost stop."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import luckyr1_pipeline as pipeline

JOBS = {
    "B": "20260915_093904_f13d89b4",
    "C": "20260915_093904_7b216495",
}
CONTROL = {
    "aepe": 2.3613,
    "cmr1": 56.683168316831676,
    "cmr2": 78.96039603960396,
    "cmr5": 88.86138613861387,
    "cmr10": 94.80198019801979,
    "cmr16": 97.02970297029702,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["B", "C"])
    args = parser.parse_args()
    variant = args.variant
    arm = f"exploratory_{variant.lower()}"
    finalize_task = f"exploratory_{variant.lower()}_finalize_v5"
    run_dir = pipeline.task_dir(arm)
    progress = json.loads((run_dir / "TRAIN_PROGRESS.json").read_text())
    rows = [row for row in progress.get("evaluations", []) if int(row["step"]) == 500]
    if len(rows) != 1:
        raise RuntimeError(f"expected one step500 evaluation, found {len(rows)}")
    selected = rows[0]
    checkpoint = Path(selected["checkpoint"])
    if pipeline.sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("step500 checkpoint SHA mismatch")
    metrics = selected["metrics"]
    prediction_dir = run_dir / "predictions/step500_iter24_post_lsr"
    manifest = Path(metrics["prediction_manifest"])
    joined = Path(metrics["metric_join"])
    prediction_count = len(list(prediction_dir.glob("*.pt")))
    manifest_count = len(manifest.read_text().splitlines())
    with joined.open() as handle:
        join_count = sum(1 for _ in csv.DictReader(handle))
    if (prediction_count, manifest_count, join_count) != (404, 404, 404):
        raise RuntimeError((prediction_count, manifest_count, join_count))
    if not metrics.get("prediction_materialized_before_gt_join") or metrics.get("metric_scope") != "pair-level":
        raise RuntimeError("invalid evaluation protocol")
    signal = (
        metrics["cmr5"], metrics["cmr2"], metrics["cmr1"], -metrics["aepe"]
    ) > (CONTROL["cmr5"], CONTROL["cmr2"], CONTROL["cmr1"], -CONTROL["aepe"])
    if signal:
        raise RuntimeError("finalizer may only stop a no-signal arm")
    arm_card = {
        "task": arm,
        "status": "COST_GATE_STOP_NO_SIGNAL",
        "last_completed_stage": "STEP500_FULL_404_EVALUATED_COST_GATE",
        "claim_scope": "CURRENT_MODEL_EXPLORATORY_ONLY",
        "implementation_version": "luckyr1-bc-v3-e074b-evidence-graph-20260915",
        "train_pairs": 2011,
        "evaluation_pairs": 404,
        "evaluations": rows,
        "selected": selected,
        "cost_gate_policy": "continue only if (CMR@5, CMR@2, CMR@1, -AEPE) lexicographically exceeds protected M4",
        "cost_gate_signal": False,
        "protected_m4": {"sha256": pipeline.HISTORICAL_M4_SHA, "metrics": CONTROL},
        "prediction_count": prediction_count,
        "manifest_count": manifest_count,
        "join_count": join_count,
        "prediction_materialized_before_gt_join": True,
        "source_job": JOBS[variant],
        "protected_m4_modified": False,
    }
    payload = json.dumps(arm_card, indent=2, sort_keys=True) + "\n"
    (run_dir / "RESULT_CARD.json").write_text(payload)
    (run_dir / "RESULT_CARD.md").write_text(
        f"# {arm}\n\nStatus: `{arm_card['status']}`\n\n```json\n{payload}```\n"
    )
    attempt_dir = run_dir / "attempts" / JOBS[variant]
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "RESULT_CARD.json").write_text(payload)
    (attempt_dir / "RESULT_CARD.md").write_text(
        f"# {arm} attempt {JOBS[variant]}\n\nStatus: `{arm_card['status']}`\n\n```json\n{payload}```\n"
    )
    pipeline.write_card(finalize_task, {
        "status": "COMPLETE",
        "last_completed_stage": "STEP500_NO_SIGNAL_ARM_FROZEN",
        "variant": variant,
        "selected": selected,
        "source_job": JOBS[variant],
        "candidate_gpus": None,
        "gpu_used": False,
    })


if __name__ == "__main__":
    main()
