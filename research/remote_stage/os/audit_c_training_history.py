#!/usr/bin/env python3
"""Read-only C-stage time bounds and optional frozen C/M4 tensor identity."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    # Linux birth time is available through GNU stat although Python 3.10
    # does not expose st_birthtime. A missing birth time stays unknown.
    result = subprocess.run(["stat", "-c", "%W", str(path)], text=True, capture_output=True, check=True)
    birth = int(result.stdout.strip())
    modified = path.stat().st_mtime
    return dict(path=str(path), birth_unix=birth if birth > 0 else None,
                mtime_unix=modified, birth_local=datetime.fromtimestamp(birth).isoformat() if birth > 0 else None,
                mtime_local=datetime.fromtimestamp(modified).isoformat())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="SPARSEGEO_ROOT")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--verify-weights", action="store_true")
    args = ap.parse_args()
    runs = args.root / "GDROS/experiments/sparsegeo/runs"
    run = runs / "E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-c-stage180k"
    log = run / "logs/e074a_c_stage180k.log"
    report_path = run / "formal/checkpoint_curve_report.json"
    curves = json.loads(report_path.read_text())["curves"]
    full = [r["step"] for r in curves if r["phase"] == "full"]
    toy = [r["step"] for r in curves if r["phase"] == "toy"]
    assert min(full) == 120001 and max(full) == 180000 and max(toy) == 200
    files = [log, *(run / "checkpoints" / f"C_step{s}.pt" for s in (140000, 160000, 180000)),
             *(run / "formal/materialized" / f"C_step{s}_predictions_before_gt.jsonl" for s in (140000, 160000, 180000))]
    timings = {str(p.relative_to(run)): record(p) for p in files}
    begin = timings["logs/e074a_c_stage180k.log"]["birth_unix"]
    end = timings["logs/e074a_c_stage180k.log"]["mtime_unix"]
    resource_path = run / "formal/resource_report.json"
    resource = json.loads(resource_path.read_text())
    result = dict(status="AUDITED", run=str(run), local_timezone=subprocess.check_output(["date", "+%Z %z"], text=True).strip(),
                  update_counts=dict(full_additional=60000, toy=200, total_additional=60200,
                                     inherited_step_label=120000, final_step_label=180000),
                  time_records=timings, stage_wall_interval_seconds=None if begin is None else end - begin,
                  timing_scope="log creation through last write; includes initialization, toy/full updates, checkpoint saves and three milestone evaluations; not pure GPU training time",
                  original_resource_report=resource,
                  original_resource_elapsed_valid=False,
                  timer_bug="train() calls finalize(..., time.time()) after the loop; finalize subtracts that end timestamp, not the training start",
                  predecessor_costs_included=False,
                  streamlined_model_retrained=False,
                  historical_training_log_sha256=sha(log), historical_curve_report_sha256=sha(report_path),
                  training_curve_samples=[{k: row[k] for k in ("phase", "step", "loss", "distill")}
                                          for row in (curves[0], curves[-1])],
                  default_hyperparameters=dict(train_iters=12, lambda_seq=.5, lambda_geo=.5, distill_weight=.2, lr=3e-5),
                  hyperparameter_evidence="defaults in retained script; original launch command not recovered, so override absence is unverified")
    if args.verify_weights:
        import torch
        c = torch.load(run / "checkpoints/C_step180000.pt", map_location="cpu", weights_only=False)
        student = {k[len("student."):]: v for k, v in c["model"].items() if k.startswith("student.")}
        del c
        m4_path = runs / "E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt"
        m4 = torch.load(m4_path, map_location="cpu", weights_only=False)
        extracted = {k[len("c.student."):]: v for k, v in m4["model"].items() if k.startswith("c.student.")}
        assert student.keys() == extracted.keys()
        mismatches = [k for k in student if not torch.equal(student[k], extracted[k])]
        assert not mismatches
        result["c180k_vs_protected_m4_student"] = dict(status="PASS", keys=len(student), tensors_bitwise_equal=True,
                                                      historical_checkpoint_sha256=sha(run / "checkpoints/C_step180000.pt"),
                                                      protected_m4_sha256=sha(m4_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
