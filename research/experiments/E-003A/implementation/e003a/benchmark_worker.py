#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.sparsegeo.e002a.anchors import public_stratified_anchor_set
from experiments.sparsegeo.e002a.benchmark_worker import BaselineContract
from experiments.sparsegeo.e002a.feature import load_slim_feature
from experiments.sparsegeo.e003a.candidate_cache import _load_image, build_frozen_candidate_table
from experiments.sparsegeo.e003a.candidate_cache import CHECKPOINT_SHA256
from experiments.sparsegeo.e003a.data_protocol import load_split_manifest, sha256_file
from experiments.sparsegeo.e003a.provenance import runtime_identity, validate_runtime_bindings
from experiments.sparsegeo.e003a.selector import LearnedGeoSelector, dense_affine_flow


def _timed(function, warmup, repeats, metadata):
    for _ in range(warmup):
        function()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for iteration in range(repeats):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter_ns()
        start_event.record()
        output = function()
        stop_event.record()
        stop_event.synchronize()
        stopped = time.perf_counter_ns()
        if not torch.isfinite(output).all():
            raise RuntimeError("benchmark output is non-finite")
        rows.append(
            {
                **metadata,
                "iteration": iteration,
                "wall_ms": (stopped - started) / 1e6,
                "cuda_ms": float(start_event.elapsed_time(stop_event)),
            }
        )
    return rows, torch.cuda.max_memory_allocated() / 1024**2


def _summary(rows):
    values = np.asarray([row["wall_ms"] for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "p50_ms": float(np.quantile(values, 0.5)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=0)),
    }


def _measure(contract, mode, replicate, warmup, repeats):
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        cold = contract()
        torch.cuda.synchronize()
    cold_peak = torch.cuda.max_memory_allocated() / 1024**2
    if not torch.isfinite(cold).all():
        raise RuntimeError("cold benchmark output is non-finite")
    rows, warm_peak = _timed(
        contract,
        warmup,
        repeats,
        {"mode": mode, "replicate": replicate},
    )
    return {
        "summary": _summary(rows),
        "cold_peak_allocated_mib": cold_peak,
        "warm_peak_allocated_mib": warm_peak,
        "output_shape": list(cold.shape),
        "rows": rows,
    }


class SelectorContract:
    def __init__(self, checkpoint, candidate, device):
        content = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = LearnedGeoSelector().to(device).eval()
        self.model.load_state_dict(content["model_state"], strict=True)
        self.parameter_count = self.model.trainable_parameter_count
        self.features = candidate["candidate_features"].unsqueeze(0).to(device)
        self.source = candidate["source_xy_norm"].unsqueeze(0).to(device)
        self.target = candidate["target_xy_norm"].unsqueeze(0).to(device)

    @torch.inference_mode()
    def __call__(self):
        return self.model(self.features, self.source, self.target, use_feedback=True).affine_normalized


class SlimLearnedContract:
    def __init__(self, checkpoint, selector_checkpoint, optical, sar, device):
        self.feature, _, _ = load_slim_feature(checkpoint, device)
        self.feature.requires_grad_(False).eval()
        content = torch.load(selector_checkpoint, map_location="cpu", weights_only=True)
        self.selector = LearnedGeoSelector().to(device).eval()
        self.selector.load_state_dict(content["model_state"], strict=True)
        self.optical = optical
        self.sar = sar
        self.anchors = public_stratified_anchor_set(128, 512, 512, 8, (0, 0))

    @torch.inference_mode()
    def __call__(self):
        table = build_frozen_candidate_table(
            self.feature, self.optical, self.sar, self.anchors, to_cpu=False
        )
        output = self.selector(
            table["candidate_features"].unsqueeze(0),
            table["source_xy_norm"].unsqueeze(0),
            table["target_xy_norm"].unsqueeze(0),
            use_feedback=True,
        )
        return dense_affine_flow(output.affine_normalized)

    @torch.inference_mode()
    def audit_complexity(self):
        live_shapes = []
        table = build_frozen_candidate_table(
            self.feature,
            self.optical,
            self.sar,
            self.anchors,
            to_cpu=False,
            live_shapes=live_shapes,
        )
        score_records = [row for row in live_shapes if "score" in row.get("name", "")]
        max_live_score_elements = max((row["numel"] for row in score_records), default=0)
        observed_score_elements = sum(row["numel"] for row in score_records)
        return {
            "expected_logical_score_shape": [128, 4096],
            "observed_score_chunk_count": len(score_records),
            "observed_score_chunk_shapes": [row["shape"] for row in score_records],
            "observed_score_elements": observed_score_elements,
            "target_chunk": 1024,
            "max_live_score_elements": max_live_score_elements,
            "selector_shape": list(table["candidate_features"].shape[:2]),
            "selector_tokens": int(table["candidate_features"].shape[0] * table["candidate_features"].shape[1]),
            "hidden_n2_materialized": any(
                row.get("shape", [])[-2:] == [4096, 4096] for row in live_shapes
            ),
            "live_shapes": live_shapes,
        }


def _environment_record():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cudnn": torch.backends.cudnn.version(),
    }


def run(args):
    device = torch.device("cuda")
    records = load_split_manifest(args.split_manifest, "dev")
    record = records[0]
    binding = validate_runtime_bindings(
        args.repo_root,
        args.run_manifest,
        args.formal_config,
        args.split_manifest,
        args.cache_gate,
        cache_root=args.cache_root,
        cache_pair_ids=[record["pair_id"]],
    )
    if sha256_file(args.gdros_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("resource benchmark requires the official GDROS checkpoint")
    selector_content = torch.load(args.selector_checkpoint, map_location="cpu", weights_only=True)
    if selector_content.get("seed") != 20260805 or selector_content.get("variant") != "main":
        raise RuntimeError("resource benchmark requires formal main seed checkpoint")
    if selector_content.get("split_manifest_sha256") != binding["split_manifest_sha256"]:
        raise RuntimeError("resource selector checkpoint split mismatch")
    if not args.smoke and selector_content.get("runtime_provenance_identity") != runtime_identity(binding):
        raise RuntimeError("resource selector checkpoint provenance mismatch")
    if args.smoke is False and (args.warmup != 20 or args.repeats != 200):
        raise ValueError("formal resource worker requires warmup=20 and repeats=200")
    optical = _load_image(Path(record["files"]["opt"]["path"])).to(device)
    sar = _load_image(Path(record["files"]["sar_warped"]["path"])).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_records = []
    if args.mode == "selector":
        candidate = torch.load(
            args.cache_root / "candidates/dev" / f"{record['pair_id']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        contract = SelectorContract(args.selector_checkpoint, candidate, device)
        measured = _measure(contract, "selector", args.replicate, args.warmup, args.repeats)
        raw_records.extend(measured.pop("rows"))
        payload = {"selector": measured, "selector_parameter_count": contract.parameter_count}
    else:
        slim = SlimLearnedContract(
            args.gdros_checkpoint, args.selector_checkpoint, optical, sar, device
        )
        complexity = slim.audit_complexity()
        slim_measured = _measure(slim, "slim", args.replicate, args.warmup, args.repeats)
        raw_records.extend(slim_measured.pop("rows"))
        del slim
        gc.collect()
        torch.cuda.empty_cache()
        baseline = BaselineContract(
            args.gdros_checkpoint, device, optical, sar, contract="dense-contract"
        )
        baseline_measured = _measure(
            baseline, "baseline", args.replicate, args.warmup, args.repeats
        )
        raw_records.extend(baseline_measured.pop("rows"))
        payload = {
            "slim": slim_measured,
            "baseline": baseline_measured,
            "same_process_speedup": (
                baseline_measured["summary"]["p50_ms"] / slim_measured["summary"]["p50_ms"]
            ),
            "complexity": complexity,
        }
    raw_path = args.output_dir / "timing_raw.jsonl"
    with raw_path.open("w") as stream:
        for row in raw_records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "status": "PASS",
        "mode": args.mode,
        "replicate": args.replicate,
        **payload,
        "same_process_comparison": args.mode == "paired",
        "runtime_binding": binding,
        "environment": _environment_record(),
        "gdros_checkpoint_sha256": sha256_file(args.gdros_checkpoint),
        "selector_checkpoint_sha256": sha256_file(args.selector_checkpoint),
        "optical_sha256": sha256_file(Path(record["files"]["opt"]["path"])),
        "sar_sha256": sha256_file(Path(record["files"]["sar_warped"]["path"])),
        "raw_timing": str(raw_path.resolve()),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("selector", "paired"), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--gdros-checkpoint", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--cache-gate", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
