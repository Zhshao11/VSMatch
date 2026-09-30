#!/usr/bin/env python3
"""Freeze interrupted clean B/C attempts and select valid checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import luckyr1_pipeline as pipeline

JOBS = {
    "B": "20260914_080445_ac5e3277",
    "C": "20260914_080445_c2da3f0b",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["B", "C"], required=True)
    args = parser.parse_args()
    variant = args.variant
    arm = f"bc_{variant.lower()}"
    finalize_task = f"bc_finalize_{variant.lower()}"
    run_dir = pipeline.task_dir(arm)
    progress = json.loads((run_dir / "TRAIN_PROGRESS.json").read_text())
    evaluations = progress["evaluations"]
    for row in evaluations:
        checkpoint = Path(row["checkpoint"])
        actual = pipeline.sha256(checkpoint)
        if actual != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}: {actual}")
        metrics = row["metrics"]
        if metrics.get("pairs") != 201 or metrics.get("metric_scope") != "pair-level":
            raise RuntimeError(f"invalid metric scope: {row}")
        if not metrics.get("prediction_materialized_before_gt_join"):
            raise RuntimeError(f"prediction/GT phase violation: {row}")
    selected = max(evaluations, key=pipeline.selection_key)
    interrupted = {
        "task": arm,
        "status": "INTERRUPTED_COST_GATE_SELECTED",
        "last_completed_stage": "TRAIN_STEP_117000_THEN_SHARED_RC137",
        "claim_scope": "CLEAN_1810_201_MODEL_UNSEEN_DEV",
        "train_pairs": 1810,
        "evaluation_pairs": 201,
        "last_recorded_step": 117000,
        "watchdog_job_id": JOBS[variant],
        "return_code": 137,
        "termination": {
            "signal_semantics": "shell rc137 is consistent with SIGKILL",
            "shared_timestamp": "2026-09-14T16:31:54+08:00",
            "peer_job": JOBS["C" if variant == "B" else "B"],
            "kernel_oom_evidence": "none found in accessible journal window",
            "gpu_xid_evidence": "none found in accessible journal window",
            "reboot_or_shutdown_evidence": "none; host boot time 2026-08-29T16:52:10+08:00",
            "attribution": "UNKNOWN_SHARED_TERMINATION_EVENT",
        },
        "evaluations": evaluations,
        "selected": selected,
        "selection_policy": "CMR@5, then CMR@2, then CMR@1, then AEPE; pair-level on clean 201 dev",
        "resume_decision": "STOP_NO_RESUME",
        "resume_reason": "valid 40k/80k curve plateaued or regressed; 117k has no checkpoint evaluation",
        "sources": json.loads((run_dir / "RESULT_CARD.json").read_text()).get("sources"),
    }
    payload = json.dumps(interrupted, indent=2, sort_keys=True) + "\n"
    (run_dir / "RESULT_CARD.json").write_text(payload)
    (run_dir / "RESULT_CARD.md").write_text(
        f"# {arm}\n\nStatus: `{interrupted['status']}`\n\n```json\n{payload}```\n"
    )
    attempt_dir = run_dir / "attempts" / JOBS[variant]
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "RESULT_CARD.json").write_text(payload)
    (attempt_dir / "RESULT_CARD.md").write_text(
        f"# {arm} attempt {JOBS[variant]}\n\nStatus: `{interrupted['status']}`\n\n```json\n{payload}```\n"
    )
    state = {
        "status": "COMPLETE",
        "last_completed_stage": "INTERRUPTED_ATTEMPT_FROZEN_AND_VALID_CHECKPOINT_SELECTED",
        "variant": variant,
        "arm_card": str(run_dir / "RESULT_CARD.json"),
        "selected": selected,
        "source_job": JOBS[variant],
        "candidate_gpus": None,
        "gpu_used": False,
    }
    pipeline.write_card(finalize_task, state)


if __name__ == "__main__":
    main()
