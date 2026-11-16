#!/usr/bin/env python3
"""Shared-GPU smoke benchmark of unchanged E092A GDROS/M4, not a paper benchmark.

Run each model in a separate process on the same CUDA_VISIBLE_DEVICES GPU.
Requires the existing E092A loader and its frozen checkpoints/data on zzvr.
"""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

import e092a_official_eval as base


def snapshot():
    queries = {
        "gpu": ["--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"],
        "processes": ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory"],
    }
    return {key: subprocess.check_output(["nvidia-smi", *args, "--format=csv"], text=True)
            for key, args in queries.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gdros", "m4"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    before = snapshot()
    checkpoint = base.GDROS_CKPT if args.model == "gdros" else base.M4_CKPT
    checkpoint_sha = base.sha256(checkpoint)
    if checkpoint_sha != base.EXPECTED[checkpoint]:
        raise RuntimeError("Frozen checkpoint hash mismatch")
    print(json.dumps({"stage": "loading", "model": args.model,
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    model, load_report = (base.load_gdros(device) if args.model == "gdros" else base.load_m4(device))
    lsr_module = importlib.import_module("core.LSRnet.LSmodel")
    original_lsr = lsr_module.LSR
    calls = 0

    def counted_lsr(*positional, **keywords):
        nonlocal calls
        calls += 1
        return original_lsr(*positional, **keywords)

    lsr_module.LSR = counted_lsr

    def lsr(flow):
        return counted_lsr(1, 512, 512, flow, 200000)

    indices = np.linspace(0, 423, 10, dtype=int).tolist()
    all_pairs = base.pairs(include_gt=False)
    pairs = [all_pairs[index] for index in indices]
    inputs = [(base.image_tensor(row["source"]).to(device),
               base.image_tensor(row["target"]).to(device)) for row in pairs]
    torch.cuda.synchronize(device)

    def execute(source, target):
        if args.model == "gdros":
            _, dense = model(source, target, iters=32, test_mode=True)
        else:
            sequence = model(source, target, iters=32, lsr_fn=lsr)
            dense = sequence[-1]
        return lsr(dense)

    with torch.inference_mode():
        for i in range(args.warmups):
            output = execute(*inputs[i % len(inputs)])
            del output
            print(json.dumps({"stage": "warmup", "model": args.model, "completed": i + 1}), flush=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        rows = []
        snapshots = [{"stage": "before_measurement", "snapshot": snapshot()}]
        for row, input_pair in zip(pairs, inputs):
            calls = 0
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = execute(*input_pair)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            expected_calls = 1 if args.model == "gdros" else 129
            if calls != expected_calls:
                raise RuntimeError(f"Expected {expected_calls} LSR calls, observed {calls}")
            if tuple(output.shape) != (1, 2, 512, 512) or not torch.isfinite(output).all():
                raise RuntimeError("Invalid output")
            result = {"pair_index": row["pair_index"], "pair_id": row["pair_id"],
                      "elapsed_ms": elapsed_ms, "lsr_calls": calls}
            rows.append(result)
            print(json.dumps({"stage": "measurement", "model": args.model, **result}), flush=True)
            del output
            snapshots.append({"stage": f"after_pair_{row['pair_index']}", "snapshot": snapshot()})
    timings = [row["elapsed_ms"] for row in rows]
    card = {
        "status": "COMPLETE", "model": args.model, "training": False,
        "implementation": "unchanged E092A original M4 including C teachers" if args.model == "m4" else "official GDROS",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "load_report": load_report, "pid": os.getpid(),
        "environment": base.environment(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device), "iterations": 32,
        "warmup_pairs": args.warmups, "measured_pairs": 10, "indices": indices,
        "shared_gpu": True, "exclusive_gpu": False,
        "timing_scope": "synchronized wall-clock model forward plus one terminal official LSR, including internal transfers; excludes loading, image decode, input H2D, output validation and nvidia-smi",
        "fp32": True, "autocast": False, "tf32": False, "cpu_threads": 1,
        "summary": {"mean_ms": statistics.fmean(timings), "median_ms": statistics.median(timings),
                    "std_ms": statistics.pstdev(timings), "min_ms": min(timings), "max_ms": max(timings),
                    "total_seconds": sum(timings) / 1000,
                    "fps": 1000 / statistics.fmean(timings)},
        "lsr_calls_per_pair": rows[0]["lsr_calls"],
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024 ** 2,
        "smi_before_load": before, "smi_snapshots": snapshots, "rows": rows,
        "sources": {str(path): base.sha256(path) for path in [Path(__file__), Path(base.__file__),
                    base.TMP / "e075a_preserve_high_precision_recover_c2c5.py",
                    base.TMP / "e074a_f2_f3_twostage_fusion.py",
                    base.OFFICIAL / "core/LSRnet/LSmodel.py"]},
        "limitation": "Ten pairs, one pass, shared GPU with external jobs; this does not isolate LSR cost and is not an exclusive deployment benchmark.",
    }
    base.write_json(args.output_dir / f"{args.model}.json", card)
    with (args.output_dir / f"{args.model}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"stage": "COMPLETE", "model": args.model, "summary": card["summary"]}), flush=True)


if __name__ == "__main__":
    main()
