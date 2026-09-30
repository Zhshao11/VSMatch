#!/usr/bin/env python3
"""Freeze existing LuckyR1 clean-dev evidence and close the method as NO_GAIN."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path


ROOT = Path("/mnt/sda/zhushihao/VS_Match")
BASE = ROOT / "GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN/clean_1810_201"
EXPECTED = {
    "control": ("m4_raw", 120000, "0b0c248ca66b20c8b60ef36f131417f2ee573f6385586d97d0e4a4bb874bf8b2"),
    "A": ("lucky_a", 2000, "827dd8fbed10797f1b6642572409ca5df7cc9d6c6290b06a0cb9d3b0c384c401"),
    "B": ("lucky_b", 2000, "3c01d0f2cb65509c430517d2410d68da8963e2ffcdff7a1667aaafe1443ff386"),
    "C": ("lucky_c", 10000, "44256d6d0eec207b64fa7726152917a4c69d289d1c79fe770c964231696a0788"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluation(task: str, step: int, expected_sha: str) -> dict:
    run = BASE / task
    checkpoint = run / "checkpoints" / f"{task}_step{step}.pt"
    manifest = run / "predictions" / f"step{step}_iter24_post_lsr" / "PREDICTIONS.jsonl"
    joined = run / f"step{step}_iter24_post_lsr_METRIC_JOIN.csv"
    actual_sha = sha256(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint} {actual_sha}")
    manifest_rows = manifest.read_text().splitlines()
    with joined.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(manifest_rows) != 201 or len(rows) != 201:
        raise RuntimeError(f"incomplete evidence for {task}: manifest={len(manifest_rows)} join={len(rows)}")
    values = [float(row["pair_aepe"]) for row in rows]
    metrics = {
        "pairs": len(values),
        "aepe": sum(values) / len(values),
        **{f"cmr{threshold}": 100.0 * sum(value < threshold for value in values) / len(values) for threshold in (1, 2, 5, 10, 16)},
        "metric_scope": "pair-level",
        "prediction_manifest": str(manifest),
        "prediction_manifest_sha256": sha256(manifest),
        "metric_join": str(joined),
        "metric_join_sha256": sha256(joined),
        "prediction_materialized_before_gt_join": True,
    }
    return {
        "task": task,
        "step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "metrics": metrics,
    }


def main() -> None:
    started = time.time()
    records = {name: evaluation(*spec) for name, spec in EXPECTED.items()}
    control = records["control"]["metrics"]
    cmr_keys = ["cmr1", "cmr2", "cmr5", "cmr10", "cmr16"]
    for arm in ("A", "B", "C"):
        row = records[arm]
        metrics = row["metrics"]
        cmr_unchanged = all(metrics[key] == control[key] for key in cmr_keys)
        delta = metrics["aepe"] - control["aepe"]
        if not cmr_unchanged or abs(delta) >= 1e-3:
            raise RuntimeError(f"method decision assumptions no longer hold for {arm}: delta={delta}, metrics={metrics}")
        card = {
            "status": "COST_GATE_STOP_NO_SUBSTANTIVE_SIGNAL",
            "last_completed_stage": "EXISTING_CHECKPOINT_AND_201_DEV_EVIDENCE_FROZEN",
            "task": f"lucky_{arm.lower()}_terminal",
            "arm": arm,
            "claim_scope": "CLEAN_1810_201_MODEL_UNSEEN_DEV",
            "selected_existing_evidence": row,
            "clean_m4_raw_control": records["control"],
            "comparison": {
                "aepe_delta_arm_minus_control": delta,
                "all_cmr_unchanged": cmr_unchanged,
                "cmr_keys": cmr_keys,
                "decision": "NO_SUBSTANTIVE_SIGNAL",
                "reason": "All pair-level CMR values are identical to clean M4; approximately 1e-4 px AEPE variation is immaterial.",
            },
            "no_retraining": True,
            "no_additional_attempts": True,
            "old_attempt_cards_preserved": True,
        }
        out = BASE / f"lucky_{arm.lower()}_terminal"
        write_json(out / "RESULT_CARD.json", card)
        (out / "RESULT_CARD.md").write_text(
            f"# LuckyR1 {arm} terminal cost gate\n\n"
            "Status: `COST_GATE_STOP_NO_SUBSTANTIVE_SIGNAL`\n\n"
            f"Selected existing checkpoint: step `{row['step']}` (`{row['checkpoint_sha256']}`).\n\n"
            f"AEPE delta versus clean M4: `{delta:+.12f}` px. All CMR@1/2/5/10/16 values are unchanged.\n\n"
            "No retraining or additional attempt was run; all historical attempt cards are preserved.\n"
        )

    winner = min((records[name] for name in ("A", "B", "C")), key=lambda row: row["metrics"]["aepe"])
    reduced = {
        "status": "COMPLETE_NO_GAIN",
        "method_decision": "LUCKYR1_NO_GAIN",
        "last_completed_stage": "CLEAN_METHOD_COST_GATE_REDUCED",
        "claim_scope": "CLEAN_1810_201_MODEL_UNSEEN_DEV",
        "control": records["control"],
        "arms": {name: records[name] for name in ("A", "B", "C")},
        "numerical_best_arm_only": winner,
        "selection_for_downstream": records["control"],
        "decision_basis": {
            "all_arms_cmr_unchanged": True,
            "aepe_changes_order": "approximately 1e-4 px",
            "interpretation": "No substantive signal; numerical ordering is not a method gain.",
        },
        "untouched_official_404_finalizer": "NOT_RUN_BY_METHOD_DECISION",
        "protected_e075a_m4_unchanged": True,
        "lucky_r2_started": False,
        "elapsed_seconds": time.time() - started,
    }
    out = BASE / "lucky_reduce"
    write_json(out / "SELECTION.json", reduced)
    write_json(out / "RESULT_CARD.json", reduced)
    (out / "RESULT_CARD.md").write_text(
        "# LuckyR1 clean method reduction\n\n"
        "Status: `COMPLETE_NO_GAIN`\n\n"
        "Clean M4 step120K is the only control and remains the downstream selection. A step2K, B step2K, and C step10K have identical CMR@1/2/5/10/16; their roughly 1e-4 px AEPE differences are not substantive.\n\n"
        "No B/C retry, continuation, untouched official-404 finalizer, or LuckyR2 run was performed.\n"
    )
    print(json.dumps(reduced, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
