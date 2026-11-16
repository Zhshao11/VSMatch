#!/usr/bin/env python3
"""Full OS424 fixed-weight M4 LSR ablations, preserving terminal slot five.

Three warmup variants are first checked in requested order: final1, last45,
reference5. Full-run timings use all six orders cyclically to reduce drift.
Skipped fits use identity flow and zero residual, with all learned weights
and 32-step recurrent matchers unchanged. Dense outputs remain outside Git.
"""
import argparse
import csv
import gc
import importlib
import itertools
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import e092a_official_eval as base
from m4_clean_deploy import CleanM4
from m4_lsr_ablation import POLICIES, SLOT_NAMES, SelectedLSR
from quick_timing_10pairs_20261115 import snapshot


VARIANTS = ("final1", "last45", "reference5")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def difference(left, right):
    delta = np.abs(left-right)
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean(dtype=np.float64)),
            "bitwise_equal": bool(np.array_equal(left, right))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError("Use a fresh empty output directory")
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
    previous = base.ROOT / "GDROS/experiments/sparsegeo/runs/E-096A-GDROS-CLEAN-M4-OS424-20261116"
    archive = {r["pair_index"]: r for r in json.loads((previous / "PREDICTION_MANIFEST.json").read_text())
               if r["variant"] == "m4_clean"}
    if len(rows) != 424 or len(archive) != 424:
        raise RuntimeError("Expected all 424 pairs and reference archive")
    for row in rows:
        record = archive[row["pair_index"]]
        if record["pair_id"] != row["pair_id"] or base.sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("Archived E096A M4 integrity mismatch")
    checkpoint_sha = base.sha256(base.M4_CKPT)
    if checkpoint_sha != base.EXPECTED[base.M4_CKPT]:
        raise RuntimeError("Protected checkpoint hash mismatch")
    source_paths = [Path(__file__).resolve(), Path(base.__file__).resolve(),
                    Path(importlib.import_module("m4_clean_deploy").__file__).resolve(),
                    Path(importlib.import_module("m4_lsr_ablation").__file__).resolve(),
                    base.TMP / "e075a_preserve_high_precision_recover_c2c5.py",
                    base.TMP / "e074a_f2_f3_twostage_fusion.py",
                    base.TMP / "e073a_v1_v2_fusion_longtrain.py",
                    base.TMP / "e072a_layerwise_frontend_alignment.py",
                    base.OFFICIAL / "core/LSRnet/LSmodel.py",
                    base.OFFICIAL / "core/GDROSnet/FeatureFormer/encoder.py",
                    base.OFFICIAL / "core/update.py", base.OFFICIAL / "core/corr.py",
                    base.TMP.parent / "experiments/sparsegeo/e045a/model.py",
                    base.TMP.parent / "experiments/sparsegeo/e027a/dino_transfer.py"]
    sources = {str(path): base.sha256(path) for path in source_paths}
    (out / "source_snapshot").mkdir()
    for path in source_paths:
        (out / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    orders = list(itertools.permutations(VARIANTS))
    protocol = {"pairs": 424, "iterations": 32, "batch_size": 1,
                "warmups_per_variant": 3, "shared_gpu": True, "training": False,
                "checkpoint_sha256": checkpoint_sha, "policies": POLICIES, "slot_names": SLOT_NAMES,
                "skipped_fit": "identity flow; fitted-flow channels use raw flow and corresponding residual channels become zero",
                "interpretation": "fixed-weight gate-input ablation; not an exact redundancy deletion or evidence of retrained architectural necessity",
                "timing": "synchronized wall-clock forward plus terminal LSR; all six variant orders cycle by pair; excludes load/decode/H2D/validation/archival/GT scoring",
                "orders": orders, "warmup_order": VARIANTS, "fp32": True, "tf32": False,
                "prediction_before_gt_join": True, "sources": sources}
    base.write_json(out / "PROTOCOL.json", protocol)
    snapshots = [{"stage": "before_load", "snapshot": snapshot()}]
    print(json.dumps({"stage": "LOADING", "checkpoint_sha256": checkpoint_sha}), flush=True)
    legacy, load_report = base.load_m4(torch.device("cpu"))
    model = CleanM4(legacy).eval()
    del legacy
    gc.collect()
    model = model.to(device).eval()
    lsr_module = importlib.import_module("core.LSRnet.LSmodel")
    original_lsr, actual_calls = lsr_module.LSR, 0

    def counted_lsr(*positional, **keywords):
        nonlocal actual_calls
        actual_calls += 1
        return original_lsr(*positional, **keywords)

    lsr_module.LSR = counted_lsr

    def official_lsr(flow):
        return counted_lsr(1, 512, 512, flow, 200000)

    def execute(variant, source, target):
        nonlocal actual_calls
        selector = SelectedLSR(variant, official_lsr)
        actual_calls = 0
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        raw = model(source, target, iters=32, lsr_fn=selector)
        post = selector(raw)
        torch.cuda.synchronize(device)
        ms = (time.perf_counter()-begin)*1000
        selector.validate()
        if actual_calls != len(POLICIES[variant]):
            raise RuntimeError("Actual official LSR count differs from selected slots")
        raw_array, post_array = (x.detach().cpu().numpy().copy() for x in (raw, post))
        if any(x.shape != (1, 2, 512, 512) or not np.isfinite(x).all() for x in (raw_array, post_array)):
            raise RuntimeError("Invalid prediction")
        return raw_array, post_array, ms, actual_calls

    for variant in VARIANTS:
        (out / "predictions" / variant).mkdir(parents=True)
    timings, manifest, comparisons = [], [], []
    with torch.inference_mode():
        for index in (0, 211, 423):
            source, target = (base.image_tensor(rows[index][key]).to(device) for key in ("source", "target"))
            for variant in VARIANTS:
                execute(variant, source, target)
                print(json.dumps({"stage": "WARMUP", "pair_index": index, "variant": variant}), flush=True)
        snapshots.append({"stage": "before_measurement", "snapshot": snapshot()})
        with (out / "PROGRESS.jsonl").open("w", buffering=1) as log:
            for row in rows:
                index = row["pair_index"]
                source, target = (base.image_tensor(row[key]).to(device) for key in ("source", "target"))
                order = orders[index % len(orders)]
                outputs = {}
                for variant in order:
                    raw, post, ms, calls = execute(variant, source, target)
                    outputs[variant] = (raw, post)
                    timings.append({"pair_index": index, "pair_id": row["pair_id"], "variant": variant,
                                    "order": order.index(variant), "elapsed_ms": ms, "lsr_calls": calls})
                    path = out / "predictions" / variant / f"{index:06d}.npy"
                    np.save(path, post, allow_pickle=False)
                    manifest.append({"pair_index": index, "pair_id": row["pair_id"], "variant": variant,
                                     "path": str(path), "sha256": base.sha256(path)})
                comparison = {"pair_index": index, "pair_id": row["pair_id"],
                              "reference_vs_archive": difference(outputs["reference5"][1], np.load(archive[index]["path"], allow_pickle=False))}
                for variant in ("final1", "last45"):
                    comparison[variant] = {"raw": difference(outputs[variant][0], outputs["reference5"][0]),
                                           "post": difference(outputs[variant][1], outputs["reference5"][1])}
                comparisons.append(comparison)
                if comparison["reference_vs_archive"]["max_abs"] > 1e-4:
                    base.write_json(out / "CONTROL_FAILURE.json", comparison)
                    raise RuntimeError("Five-fit control no longer reproduces the archived M4")
                progress = {"completed_pairs": index+1, "comparison": comparison, "timings": timings[-3:]}
                log.write(json.dumps(progress)+"\n")
                if index % 32 == 0 or index == 423:
                    base.write_json(out / "PROGRESS.json", progress)
                    print(json.dumps({"stage": "PAIRED_PREDICTION", **progress}), flush=True)
                    snapshots.append({"stage": f"after_{index+1}_pairs", "snapshot": snapshot()})
                del source, target, raw, post, outputs
    write_csv(out / "TIMINGS.csv", timings)
    base.write_json(out / "PREDICTION_MANIFEST.json", manifest)
    base.write_json(out / "COMPARISONS.json", comparisons)
    print(json.dumps({"stage": "ALL_PREDICTIONS_SAVED_START_CPU_GT_JOIN"}), flush=True)
    metric_rows, metrics = [], {v: {"public_component": [], "vector_epe": []} for v in VARIANTS}
    for row in base.pairs(include_gt=True):
        gt = base.read_flo(row["gt"]).astype(np.float32, copy=False)
        result = {"pair_index": row["pair_index"], "pair_id": row["pair_id"]}
        for variant in VARIANTS:
            pred = np.load(out / "predictions" / variant / f"{row['pair_index']:06d}.npy", allow_pickle=False)[0]
            delta = pred-gt
            values = {"public_component": float(np.mean((np.abs(delta[0])+np.abs(delta[1]))/2)),
                      "vector_epe": float(np.mean(np.sqrt(np.square(delta[0])+np.square(delta[1]))))}
            for name, value in values.items():
                metrics[variant][name].append(value)
                result[f"{variant}__{name}"] = value
        metric_rows.append(result)
    write_csv(out / "METRICS.csv", metric_rows)
    summaries = {}
    for variant in VARIANTS:
        values = [r["elapsed_ms"] for r in timings if r["variant"] == variant]
        summaries[variant] = {"samples": len(values), "mean_ms": statistics.fmean(values),
                              "median_ms": statistics.median(values), "p95_ms": float(np.quantile(values, .95)),
                              "std_ms": statistics.pstdev(values), "total_seconds": sum(values)/1000}
    sources_after = {str(path): base.sha256(path) for path in source_paths}
    if sources != sources_after:
        raise RuntimeError("Source changed during evaluation")
    card = {"status": "COMPLETE", "pairs": 424, "iterations": 32, "training": False,
            "checkpoint_sha256": checkpoint_sha, "load_report": load_report, "policies": POLICIES,
            "pruning_audit": model.audit, "metrics": {v: {name: base.summarize(values) for name, values in by.items()} for v, by in metrics.items()},
            "timing": summaries, "speedup_vs_reference5": {v: summaries["reference5"]["mean_ms"]/summaries[v]["mean_ms"] for v in VARIANTS},
            "reference_vs_archive": {"max_abs": max(r["reference_vs_archive"]["max_abs"] for r in comparisons),
                                     "bitwise_equal_pairs": sum(r["reference_vs_archive"]["bitwise_equal"] for r in comparisons)},
            "output_changes": {v: {name: {"max_abs": max(r[v][name]["max_abs"] for r in comparisons),
                                           "bitwise_equal_pairs": sum(r[v][name]["bitwise_equal"] for r in comparisons)}
                                    for name in ("raw", "post")} for v in ("final1", "last45")},
            "warmups_per_variant": 3, "shared_gpu": True, "exclusive_gpu": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": torch.cuda.get_device_name(device),
            "environment": base.environment(), "source_hashes_start": sources, "source_hashes_end": sources_after,
            "source_files_unchanged_during_run": True, "smi_snapshots": snapshots,
            "limitation": protocol["interpretation"]+"; shared-GPU paired timings, no retraining or gate recalibration"}
    base.write_json(out / "RESULT_CARD.json", card)
    print(json.dumps({"stage": "COMPLETE", "timing": summaries, "metrics": card["metrics"],
                      "output_changes": card["output_changes"]}), flush=True)


if __name__ == "__main__":
    main()
