#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _run_worker(args, mode, replicate):
    output = args.output_dir / mode / f"replicate-{replicate}"
    command = [
        sys.executable,
        "-m",
        "experiments.sparsegeo.e003a.benchmark_worker",
        "--mode",
        mode,
        "--replicate",
        str(replicate),
        "--gdros-checkpoint",
        str(args.gdros_checkpoint),
        "--selector-checkpoint",
        str(args.selector_checkpoint),
        "--split-manifest",
        str(args.split_manifest),
        "--cache-root",
        str(args.cache_root),
        "--output-dir",
        str(output),
        "--warmup",
        "20",
        "--repeats",
        "200",
        "--repo-root",
        str(args.repo_root),
        "--run-manifest",
        str(args.run_manifest),
        "--formal-config",
        str(args.formal_config),
        "--cache-gate",
        str(args.cache_gate),
    ]
    subprocess.run(command, check=True)
    return json.loads((output / "result.json").read_text())


def _median(records, section, key):
    return float(np.median([record[section]["summary"][key] for record in records]))


def run(args):
    from experiments.sparsegeo.e003a.provenance import validate_runtime_bindings

    binding = validate_runtime_bindings(
        args.repo_root,
        args.run_manifest,
        args.formal_config,
        args.split_manifest,
        args.cache_gate,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty resource output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        mode: [_run_worker(args, mode, replicate) for replicate in range(5)]
        for mode in ("selector", "paired")
    }
    selector_p50 = _median(records["selector"], "selector", "p50_ms")
    selector_p95 = _median(records["selector"], "selector", "p95_ms")
    slim_p50 = _median(records["paired"], "slim", "p50_ms")
    slim_p95 = _median(records["paired"], "slim", "p95_ms")
    baseline_p50 = _median(records["paired"], "baseline", "p50_ms")
    process_speedups = np.asarray([row["same_process_speedup"] for row in records["paired"]])
    speedup = float(np.median(process_speedups))
    slim_p50s = np.asarray([row["slim"]["summary"]["p50_ms"] for row in records["paired"]])
    cv = float(100.0 * slim_p50s.std(ddof=0) / slim_p50s.mean())
    selector_peak = max(row["selector"]["warm_peak_allocated_mib"] for row in records["selector"])
    slim_cold_peak = max(row["slim"]["cold_peak_allocated_mib"] for row in records["paired"])
    slim_warm_peak = max(row["slim"]["warm_peak_allocated_mib"] for row in records["paired"])
    complexity = [row["complexity"] for row in records["paired"]]
    complexity_checks = {
        "observed_four_score_chunks": all(row["observed_score_chunk_count"] == 4 for row in complexity),
        "observed_score_chunk_shapes_exact": all(
            row["observed_score_chunk_shapes"] == [[1, 128, 1024]] * 4 for row in complexity
        ),
        "observed_scores_eq_128x4096": all(row["observed_score_elements"] == 128 * 4096 for row in complexity),
        "max_live_score_block_le_128x1024": all(row["max_live_score_elements"] <= 128 * 1024 for row in complexity),
        "selector_tokens_eq_128x8": all(row["selector_tokens"] == 128 * 8 for row in complexity),
        "no_hidden_n2": all(not row["hidden_n2_materialized"] for row in complexity),
    }
    checks = {
        "selector_p50_le_5": selector_p50 <= 5.0,
        "selector_p95_le_8": selector_p95 <= 8.0,
        "selector_warm_allocated_le_64": selector_peak <= 64.0,
        "slim_p50_le_15": slim_p50 <= 15.0,
        "slim_p95_le_20": slim_p95 <= 20.0,
        "slim_cold_allocated_le_284": slim_cold_peak <= 284.0,
        "slim_warm_allocated_le_284": slim_warm_peak <= 284.0,
        "baseline_speedup_ge_5": speedup >= 5.0,
        "slim_process_p50_cv_le_5": cv <= 5.0,
        **complexity_checks,
    }
    result = {
        "experiment": "E-003A",
        "gate": "G2_RESOURCES",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "selector": {
            "p50_ms": selector_p50,
            "p95_ms": selector_p95,
            "warm_peak_allocated_mib": selector_peak,
        },
        "slim_end_to_end": {
            "p50_ms": slim_p50,
            "p95_ms": slim_p95,
            "cold_peak_allocated_mib": slim_cold_peak,
            "warm_peak_allocated_mib": slim_warm_peak,
            "process_p50_values_ms": slim_p50s.tolist(),
            "process_p50_cv_percent": cv,
        },
        "baseline": {"p50_ms": baseline_p50},
        "baseline_speedup": speedup,
        "same_process_speedup_values": process_speedups.tolist(),
        "same_process_comparison": True,
        "complexity": complexity,
        "selector_parameter_count": records["selector"][0].get("selector_parameter_count", 18692),
        "selector_linear_macs": 128 * 8 * (13 * 64 + 64 * 64 + 64 + 68 * 64 + 64) + 128 * (128 * 64 + 64) + (10 * 64 + 64),
        "selector_linear_flops": 2 * (128 * 8 * (13 * 64 + 64 * 64 + 64 + 68 * 64 + 64) + 128 * (128 * 64 + 64) + (10 * 64 + 64)),
        "checks": checks,
        "fresh_process_records": records,
        "runtime_binding": binding,
    }
    path = args.output_dir / "g2.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks}, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdros-checkpoint", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--cache-gate", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    from experiments.sparsegeo.e003a.spec_binding import validate_frozen_specs

    validate_frozen_specs()
    run(parse_args())
