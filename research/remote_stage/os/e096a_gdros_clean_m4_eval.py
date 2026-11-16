#!/usr/bin/env python3
"""Paired official OS424: unchanged released GDROS versus inference-clean M4.

Run from the migrated experiment project with the existing E092/E095 helpers.
Dense predictions stay outside Git in a fresh --output-dir.
"""
import argparse
import csv
import gc
import importlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import e092a_official_eval as base
from m4_clean_deploy import CleanM4, parameter_stats
from quick_timing_10pairs_20261115 import snapshot


VARIANTS = ("gdros", "m4_clean")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError("Use an empty output directory")
    os.chdir(base.TMP.parent)
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows = base.pairs(include_gt=False)
    previous = base.ROOT / "GDROS/experiments/sparsegeo/runs/E-095A-M4-CLEAN-DEPLOY-424-20261116-r2"
    archived = {r["pair_index"]: r for r in json.loads((previous / "PREDICTION_MANIFEST.json").read_text())
                if r["variant"] == "clean"}
    if len(rows) != 424 or len(archived) != 424:
        raise RuntimeError("Expected 424 pairs and E095A clean predictions")
    for row in rows:
        record = archived[row["pair_index"]]
        if record["pair_id"] != row["pair_id"] or base.sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("E095A archive integrity mismatch")
    checkpoints = {name: base.sha256(path) for name, path in
                   (("gdros", base.GDROS_CKPT), ("m4_clean", base.M4_CKPT))}
    if checkpoints["gdros"] != base.EXPECTED[base.GDROS_CKPT] or checkpoints["m4_clean"] != base.EXPECTED[base.M4_CKPT]:
        raise RuntimeError("Protected checkpoint hash mismatch")
    source_paths = [Path(__file__).resolve(), Path(base.__file__).resolve(),
                    Path(importlib.import_module("m4_clean_deploy").__file__).resolve(),
                    base.TMP / "e075a_preserve_high_precision_recover_c2c5.py",
                    base.TMP / "e074a_f2_f3_twostage_fusion.py",
                    base.TMP / "e073a_v1_v2_fusion_longtrain.py",
                    base.TMP / "e072a_layerwise_frontend_alignment.py",
                    base.OFFICIAL / "core/GDROSnet/FeatureFormer/gdros.py",
                    base.OFFICIAL / "core/GDROSnet/FeatureFormer/encoder.py",
                    base.OFFICIAL / "core/update.py", base.OFFICIAL / "core/corr.py",
                    base.OFFICIAL / "core/LSRnet/LSmodel.py",
                    base.TMP.parent / "experiments/sparsegeo/e045a/model.py",
                    base.TMP.parent / "experiments/sparsegeo/e027a/dino_transfer.py"]
    sources = {str(path): base.sha256(path) for path in source_paths}
    (out / "source_snapshot").mkdir()
    for path in source_paths:
        (out / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    protocol = {"pairs": 424, "iterations_per_matcher": 32, "batch_size": 1,
                "image_shape": [512, 512], "fp32": True, "tf32": False,
                "warmups_per_variant": 3, "shared_gpu": True,
                "variants": {"gdros": "unchanged official model and OS released checkpoint; test_mode=True",
                             "m4_clean": "E095A equivalent inference-clean OS step80K checkpoint"},
                "lsr_calls_per_pair": {"gdros": 1, "m4_clean": 5},
                "timing": "paired alternating order; synchronized wall-clock forward plus terminal official LSR; excludes loading/decode/H2D/output validation/archival/GT scoring",
                "metric": "official public component mean and separate vector EPE; whole image, no mask; CMR thresholds on per-pair means",
                "prediction_before_gt_join": True, "checkpoint_sha256": checkpoints,
                "sources": sources, "training": False}
    base.write_json(out / "PROTOCOL.json", protocol)
    snapshots = [{"stage": "before_load", "snapshot": snapshot()}]
    print(json.dumps({"stage": "LOADING", "checkpoint_sha256": checkpoints}), flush=True)
    legacy, m4_load = base.load_m4(torch.device("cpu"))
    m4 = CleanM4(legacy).eval()
    del legacy
    gc.collect()
    gdros, gdros_load = base.load_gdros(torch.device("cpu"))
    stats = {"gdros": parameter_stats(gdros), "m4_clean": parameter_stats(m4)}
    gdros, m4 = gdros.to(device).eval(), m4.to(device).eval()
    print(json.dumps({"stage": "MODELS_LOADED", "parameters": stats}), flush=True)
    lsr_module = importlib.import_module("core.LSRnet.LSmodel")
    original_lsr, calls = lsr_module.LSR, 0

    def counted_lsr(*positional, **keywords):
        nonlocal calls
        calls += 1
        return original_lsr(*positional, **keywords)

    lsr_module.LSR = counted_lsr

    def lsr(flow):
        return counted_lsr(1, 512, 512, flow, 200000)

    def execute(variant, source, target):
        nonlocal calls
        calls = 0
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        if variant == "gdros":
            _, raw = gdros(source, target, iters=32, test_mode=True)
        else:
            raw = m4(source, target, iters=32, lsr_fn=lsr)
        post = lsr(raw)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - begin) * 1000
        if calls != protocol["lsr_calls_per_pair"][variant]:
            raise RuntimeError(f"Unexpected LSR count: {variant} {calls}")
        array = post.detach().cpu().numpy().copy()
        if array.shape != (1, 2, 512, 512) or not np.isfinite(array).all():
            raise RuntimeError("Invalid prediction")
        return array, elapsed_ms, calls

    for variant in VARIANTS:
        (out / "predictions" / variant).mkdir(parents=True)
    timings, manifest, parity = [], [], []
    with torch.inference_mode():
        for index in (0, 211, 423):
            source, target = (base.image_tensor(rows[index][key]).to(device) for key in ("source", "target"))
            for variant in VARIANTS:
                execute(variant, source, target)
            print(json.dumps({"stage": "WARMUP", "pair_index": index}), flush=True)
        snapshots.append({"stage": "before_measurement", "snapshot": snapshot()})
        with (out / "PROGRESS.jsonl").open("w", buffering=1) as log:
            for row in rows:
                index = row["pair_index"]
                source, target = (base.image_tensor(row[key]).to(device) for key in ("source", "target"))
                order = VARIANTS if index % 2 == 0 else VARIANTS[::-1]
                for variant in order:
                    pred, ms, n_calls = execute(variant, source, target)
                    timings.append({"pair_index": index, "pair_id": row["pair_id"], "variant": variant,
                                    "order": order.index(variant), "elapsed_ms": ms, "lsr_calls": n_calls})
                    path = out / "predictions" / variant / f"{index:06d}.npy"
                    np.save(path, pred, allow_pickle=False)
                    manifest.append({"pair_index": index, "pair_id": row["pair_id"], "variant": variant,
                                     "path": str(path), "sha256": base.sha256(path)})
                    if variant == "m4_clean":
                        old = np.load(archived[index]["path"], allow_pickle=False)
                        delta = np.abs(pred - old)
                        comparison = {"pair_index": index, "pair_id": row["pair_id"],
                                      "max_abs": float(delta.max()), "bitwise_equal": bool(np.array_equal(pred, old))}
                        parity.append(comparison)
                        if comparison["max_abs"] > 1e-4:
                            base.write_json(out / "M4_PARITY_FAILURE.json", comparison)
                            raise RuntimeError("Current M4 differs from validated E095A archive")
                progress = {"completed_pairs": index+1, "timings": timings[-2:], "m4_vs_e095a": comparison}
                log.write(json.dumps(progress)+"\n")
                if index % 32 == 0 or index == 423:
                    base.write_json(out / "PROGRESS.json", progress)
                    print(json.dumps({"stage": "PAIRED_PREDICTION", **progress}), flush=True)
                    snapshots.append({"stage": f"after_{index+1}_pairs", "snapshot": snapshot()})
                del source, target, pred
    write_csv(out / "TIMINGS.csv", timings)
    base.write_json(out / "PREDICTION_MANIFEST.json", manifest)
    base.write_json(out / "M4_ARCHIVE_COMPARISONS.json", parity)
    print(json.dumps({"stage": "ALL_PREDICTIONS_SAVED_START_CPU_GT_JOIN"}), flush=True)
    metric_rows = []
    metrics = {v: {"public_component": [], "vector_epe": []} for v in VARIANTS}
    for row in base.pairs(include_gt=True):
        gt = base.read_flo(row["gt"]).astype(np.float32, copy=False)
        result = {"pair_index": row["pair_index"], "pair_id": row["pair_id"]}
        for variant in VARIANTS:
            pred = np.load(out / "predictions" / variant / f"{row['pair_index']:06d}.npy", allow_pickle=False)[0]
            delta = pred - gt
            values = {"public_component": float(np.mean((np.abs(delta[0])+np.abs(delta[1]))/2)),
                      "vector_epe": float(np.mean(np.sqrt(np.square(delta[0])+np.square(delta[1]))))}
            for metric, value in values.items():
                metrics[variant][metric].append(value)
                result[f"{variant}__{metric}"] = value
        metric_rows.append(result)
    write_csv(out / "METRICS.csv", metric_rows)
    summaries = {}
    for variant in VARIANTS:
        values = [r["elapsed_ms"] for r in timings if r["variant"] == variant]
        summaries[variant] = {"samples": len(values), "mean_ms": statistics.fmean(values),
                              "median_ms": statistics.median(values), "p95_ms": float(np.quantile(values, .95)),
                              "std_ms": statistics.pstdev(values), "total_seconds": sum(values)/1000}
    sources_after = {str(path): base.sha256(path) for path in source_paths}
    card = {"status": "COMPLETE", "pairs": 424, "iterations": 32, "training": False,
            "checkpoint_sha256": checkpoints, "load_reports": {"gdros": gdros_load, "m4_clean": m4_load},
            "model_parameters": stats, "m4_pruning_audit": m4.audit,
            "metrics": {v: {name: base.summarize(values) for name, values in by.items()} for v, by in metrics.items()},
            "timing": summaries, "m4_over_gdros_latency_ratio": summaries["m4_clean"]["mean_ms"]/summaries["gdros"]["mean_ms"],
            "m4_vs_e095a": {"max_abs": max(r["max_abs"] for r in parity), "bitwise_equal_pairs": sum(r["bitwise_equal"] for r in parity)},
            "lsr_calls_per_pair": protocol["lsr_calls_per_pair"], "warmups_per_variant": 3,
            "shared_gpu": True, "exclusive_gpu": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device), "environment": base.environment(),
            "source_hashes_start": sources, "source_hashes_end": sources_after,
            "source_files_unchanged_during_run": sources == sources_after, "smi_snapshots": snapshots,
            "limitation": "Paired shared-GPU wall-clock measurements; official GDROS implementation is unchanged while M4 uses its validated clean inference path; no GDROS inference cleanup is applied."}
    if sources != sources_after:
        raise RuntimeError("Source changed during evaluation")
    base.write_json(out / "RESULT_CARD.json", card)
    print(json.dumps({"stage": "COMPLETE", "timing": summaries, "metrics": card["metrics"],
                      "m4_over_gdros_latency_ratio": card["m4_over_gdros_latency_ratio"]}), flush=True)


if __name__ == "__main__":
    main()
