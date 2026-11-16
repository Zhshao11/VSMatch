#!/usr/bin/env python3
"""Paired full-424 audit of the verified five-LSR path and a clean deployment model.

Run in the migrated project root alongside the frozen E092A loader. Large
predictions stay in --output-dir on the experiment host, outside source Git.
"""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import statistics
import gc
import time

import numpy as np
import torch

import e092a_official_eval as base
from m4_final_only_lsr import forward_final_only
from m4_clean_deploy import CleanM4


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def difference(left, right):
    delta = np.abs(left - right)
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean(dtype=np.float64)),
            "bitwise_equal": bool(np.array_equal(left, right))}


def clear_caches(model):
    # Diagnostic tensors are never consumed by subsequent inference calls.
    for module, name in [(model, "last_branch_preds"), (model.c, "last_teacher_avg")]:
        if hasattr(module, name):
            delattr(module, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RESULT_CARD.json").exists():
        raise RuntimeError("Use a fresh directory; do not overwrite a completed experiment")
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
    previous = base.ROOT / "GDROS/experiments/sparsegeo/runs/E-094A-M4-FINAL-ONLY-LSR-424-20261115"
    old_dir = previous / "predictions/final5"
    manifest = [r for r in json.loads((previous / "PREDICTION_MANIFEST.json").read_text()) if r["variant"] == "final5"]
    if len(rows) != 424 or len(manifest) != 424:
        raise RuntimeError("Expected full 424 pairs and archived predictions")
    old_hashes = {}
    for row, record in zip(rows, manifest):
        path = old_dir / f"{row['pair_index']:06d}.npy"
        if record["pair_id"] != row["pair_id"] or base.sha256(path) != record["sha256"]:
            raise RuntimeError("Archived E094 five-call prediction integrity mismatch")
        old_hashes[row["pair_index"]] = record["sha256"]
    checkpoint_sha = base.sha256(base.M4_CKPT)
    if checkpoint_sha != base.EXPECTED[base.M4_CKPT]:
        raise RuntimeError("Frozen checkpoint hash mismatch")
    source_paths = [Path(__file__).resolve(), Path(forward_final_only.__code__.co_filename).resolve(),
                    Path(base.__file__).resolve(), Path(importlib.import_module("m4_clean_deploy").__file__).resolve(), base.TMP / "e075a_preserve_high_precision_recover_c2c5.py",
                    base.TMP / "e074a_f2_f3_twostage_fusion.py",
                    base.TMP / "e073a_v1_v2_fusion_longtrain.py",
                    base.TMP / "e072a_layerwise_frontend_alignment.py",
                    base.OFFICIAL / "core/LSRnet/LSmodel.py",
                    base.OFFICIAL / "core/GDROSnet/FeatureFormer/encoder.py",
                    base.OFFICIAL / "core/update.py",
                    base.TMP.parent / "experiments/sparsegeo/e045a/model.py",
                    base.TMP.parent / "experiments/sparsegeo/e027a/dino_transfer.py"]
    sources = {str(path): base.sha256(path) for path in source_paths}
    snapshot_dir = out / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    for path in source_paths:
        (snapshot_dir / path.name).write_bytes(path.read_bytes())
    import quick_timing_10pairs_20261115 as timing
    smi = [{"stage": "before_load", "snapshot": timing.snapshot()}]
    base.write_json(out / "PROTOCOL.json", {
        "pairs": 424, "iterations": 32, "warmups_per_variant": 3,
        "raw_tolerance": 1e-5, "post_tolerance": 1e-4,
        "timing": "paired alternating order, synchronized wall clock, model forward and terminal official LSR; excludes load/decode/H2D/validation/archival/GT join",
        "shared_gpu": True, "checkpoint_sha256": checkpoint_sha, "sources": sources,
        "change": "Remove dead module storage, ignored contexts, nonfinal mask/upsampling, unused diagnostics and repeated lookup offsets; reuse frozen DINO, descriptor adapters and resized outputs only after exact state/config grouping",
        "training": False, "prediction_before_gt_join": True,
    })
    print(json.dumps({"stage": "LOADING", "checkpoint_sha256": checkpoint_sha}), flush=True)
    # Avoid a second checkpoint-sized GPU allocation during strict loading.
    model, load_report = base.load_m4(torch.device("cpu"))
    clean = CleanM4(model).eval()
    base.write_json(out / "PRUNING_AUDIT.json", clean.audit)
    print(json.dumps({"stage": "PRUNING_AUDIT", **clean.audit}), flush=True)
    model = model.to(device).eval()
    clean = clean.to(device).eval()
    c_skip_teachers = getattr(model, "c_skip_teachers", False)
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

    def execute(variant, source, target):
        nonlocal calls
        calls = 0
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        if variant == "reference5":
            raw = forward_final_only(model, source, target, iters=32, lsr_fn=lsr)
        else:
            raw = clean(source, target, iters=32, lsr_fn=lsr)
        post = lsr(raw)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - begin) * 1000
        expected = 5
        if calls != expected:
            raise RuntimeError(f"{variant}: expected {expected} LSR calls, got {calls}")
        raw_array = raw.detach().cpu().numpy().copy()
        post_array = post.detach().cpu().numpy().copy()
        if not np.isfinite(raw_array).all() or not np.isfinite(post_array).all():
            raise RuntimeError("Nonfinite prediction")
        clear_caches(model)
        return raw_array, post_array, elapsed_ms, calls

    for variant in ("reference5", "clean"):
        (out / "predictions" / variant).mkdir(parents=True, exist_ok=True)
    timings, comparisons, prediction_manifest = [], [], []
    with torch.inference_mode():
        for index in (0, 211, 423):
            source, target = (base.image_tensor(rows[index][key]).to(device) for key in ("source", "target"))
            for variant in ("reference5", "clean"):
                execute(variant, source, target)
            print(json.dumps({"stage": "WARMUP", "pair_index": index}), flush=True)
        coverage = {}
        handles = []
        for name, module in clean.named_modules():
            if not list(module.children()) and sum(p.numel() for p in module.parameters(recurse=False)):
                coverage[name] = 0
                def record(_module, _inputs, key=name):
                    coverage[key] += 1
                handles.append(module.register_forward_pre_hook(record))
        execute("clean", source, target)
        for handle in handles:
            handle.remove()
        uncalled = [name for name, count in coverage.items() if count == 0]
        base.write_json(out / "LEAF_PARAMETER_COVERAGE.json", {"calls": coverage, "uncalled": uncalled})
        if uncalled:
            raise RuntimeError(f"Inspect remaining uncalled parameterized leaf modules before full run: {uncalled}")
        torch.cuda.reset_peak_memory_stats(device)
        smi.append({"stage": "before_measurement", "snapshot": timing.snapshot()})
        with (out / "PROGRESS.jsonl").open("w", buffering=1) as log:
            for row in rows:
                index = row["pair_index"]
                source, target = (base.image_tensor(row[key]).to(device) for key in ("source", "target"))
                order = ("reference5", "clean") if index % 2 == 0 else ("clean", "reference5")
                outputs = {}
                for variant in order:
                    raw, post, ms, n_calls = execute(variant, source, target)
                    outputs[variant] = (raw, post)
                    timings.append({"pair_index": index, "pair_id": row["pair_id"], "variant": variant,
                                    "order": order.index(variant), "elapsed_ms": ms, "lsr_calls": n_calls})
                    path = out / "predictions" / variant / f"{index:06d}.npy"
                    np.save(path, post, allow_pickle=False)
                    prediction_manifest.append({"pair_index": index, "pair_id": row["pair_id"],
                                                "variant": variant, "path": str(path), "sha256": base.sha256(path)})
                raw_diff = difference(outputs["reference5"][0], outputs["clean"][0])
                post_diff = difference(outputs["reference5"][1], outputs["clean"][1])
                old = np.load(old_dir / f"{index:06d}.npy", allow_pickle=False)
                archive_diff = difference(outputs["reference5"][1], old)
                compare = {"pair_index": index, "pair_id": row["pair_id"],
                           "raw": raw_diff, "post": post_diff, "reference_vs_archive": archive_diff}
                comparisons.append(compare)
                if raw_diff["max_abs"] > 1e-5 or post_diff["max_abs"] > 1e-4:
                    base.write_json(out / "PARITY_FAILURE.json", compare)
                    raise RuntimeError("Final-only output parity failed; stop before continuing evaluation")
                progress = {"completed_pairs": index + 1, "comparison": compare,
                            "timings": timings[-2:], "c_skip_teachers": c_skip_teachers}
                log.write(json.dumps(progress) + "\n")
                if index % 16 == 0 or index == 423:
                    base.write_json(out / "PROGRESS.json", progress)
                    print(json.dumps({"stage": "PAIRED_PREDICTION", **progress}), flush=True)
                    smi.append({"stage": f"after_{index+1}_pairs", "snapshot": timing.snapshot()})
                del source, target, outputs
    write_csv(out / "TIMINGS.csv", timings)
    base.write_json(out / "COMPARISONS.json", comparisons)
    base.write_json(out / "PREDICTION_MANIFEST.json", prediction_manifest)
    print(json.dumps({"stage": "ALL_PREDICTIONS_SAVED_START_CPU_GT_JOIN"}), flush=True)
    # GT is read only after both variants have completed all 424 predictions.
    metrics = {v: {"public_component": [], "vector_epe": []} for v in ("reference5", "clean")}
    metric_rows = []
    for row in base.pairs(include_gt=True):
        gt = base.read_flo(row["gt"]).astype(np.float32, copy=False)
        result = {"pair_index": row["pair_index"], "pair_id": row["pair_id"]}
        for variant in metrics:
            pred = np.load(out / "predictions" / variant / f"{row['pair_index']:06d}.npy", allow_pickle=False)[0]
            delta = pred - gt
            public = float(np.mean((np.abs(delta[0]) + np.abs(delta[1])) / 2.0))
            vector = float(np.mean(np.sqrt(np.square(delta[0]) + np.square(delta[1]))))
            for name, value in (("public_component", public), ("vector_epe", vector)):
                metrics[variant][name].append(value)
                result[f"{variant}__{name}"] = value
        metric_rows.append(result)
    write_csv(out / "METRICS.csv", metric_rows)
    summaries = {}
    for variant in metrics:
        values = [r["elapsed_ms"] for r in timings if r["variant"] == variant]
        summaries[variant] = {"samples": len(values), "mean_ms": statistics.fmean(values),
                              "median_ms": statistics.median(values), "p95_ms": float(np.quantile(values, .95)),
                              "std_ms": statistics.pstdev(values), "total_seconds": sum(values) / 1000}
    sources_after = {str(path): base.sha256(path) for path in source_paths}
    card = {
        "status": "COMPLETE", "training": False, "pairs": 424, "iterations": 32,
        "checkpoint_sha256": checkpoint_sha, "load_report": load_report,
        "pruning_audit": clean.audit, "leaf_parameter_coverage_uncalled": uncalled,
        "c_skip_teachers": c_skip_teachers, "lsr_calls_per_pair": {"reference5": 5, "clean": 5},
        "warmups_per_variant": 3, "shared_gpu": True, "exclusive_gpu": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "environment": base.environment(),
        "timing": summaries,
        "speedup_mean": summaries["reference5"]["mean_ms"] / summaries["clean"]["mean_ms"],
        "metrics": {v: {name: base.summarize(values) for name, values in by.items()} for v, by in metrics.items()},
        "equivalence": {"raw_max_abs": max(r["raw"]["max_abs"] for r in comparisons),
                        "post_max_abs": max(r["post"]["max_abs"] for r in comparisons),
                        "raw_bitwise_equal_pairs": sum(r["raw"]["bitwise_equal"] for r in comparisons),
                        "post_bitwise_equal_pairs": sum(r["post"]["bitwise_equal"] for r in comparisons),
                        "current_reference_vs_archive_max_abs": max(r["reference_vs_archive"]["max_abs"] for r in comparisons),
                        "current_reference_vs_archive_bitwise_equal_pairs": sum(r["reference_vs_archive"]["bitwise_equal"] for r in comparisons)},
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "source_hashes_start": sources, "source_hashes_end": sources_after,
        "source_files_unchanged_during_run": sources == sources_after,
        "smi_snapshots": smi, "pid": os.getpid(),
        "limitation": "Paired alternating order reduces drift, but this shared-GPU result is not an exclusive latency benchmark; multiple redundant computations are removed together, so speedup cannot be attributed to an individual component.",
    }
    card["both_models_live_allocated_mib"] = torch.cuda.memory_allocated(device) / 1024 ** 2
    clear_caches(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    card["clean_only_allocated_mib"] = torch.cuda.memory_allocated(device) / 1024 ** 2
    packed = {"model": {key: value.detach().cpu() for key, value in clean.state_dict().items()},
              "source_checkpoint_sha256": checkpoint_sha, "architecture": clean.audit,
              "loader": "Build CleanM4 from the protected legacy architecture, then strictly load this state"}
    packed_path = out / "clean_deploy_state.pt"
    torch.save(packed, packed_path)
    card["compact_checkpoint"] = {"path": str(packed_path), "bytes": packed_path.stat().st_size,
                                   "sha256": base.sha256(packed_path)}
    base.write_json(out / "RESULT_CARD.json", card)
    print(json.dumps({"stage": "COMPLETE", "timing": summaries, "equivalence": card["equivalence"],
                      "speedup_mean": card["speedup_mean"]}), flush=True)


if __name__ == "__main__":
    main()
