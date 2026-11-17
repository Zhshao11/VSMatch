#!/usr/bin/env python3
"""Fresh OS424 inference, standalone timing and parameter audit of clean C."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import statistics

import numpy as np
import torch

import e092a_official_eval as base
import e093a_bc_ablation_profile as branches
from m4_clean_deploy import parameter_stats
from m4_streamlined_c_student import M4StreamlinedCStudent
from quick_timing_10pairs_20261115 import snapshot


def compare(left, right):
    delta = np.abs(left - right)
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean(dtype=np.float64)),
            "bitwise_equal": bool(np.array_equal(left, right))}


def coverage_audit(model, execute, source, target):
    counts, leaf_calls, handles = {}, {}, []

    def hook(key):
        def record(_module, _inputs):
            counts[key] = counts.get(key, 0) + 1
        return record

    for name, module in (("gru", model.matcher.gru), ("motion", model.matcher.motion_encoder),
                         ("flow_head", model.matcher.flow_head), ("mask_head", model.matcher.mask_head),
                         ("context", model.matcher.context), ("frontend_v1", model.matcher.v1),
                         ("frontend_v2", model.matcher.v2)):
        handles.append(module.register_forward_pre_hook(hook(name)))
    for name, module in model.named_modules():
        if not list(module.children()) and sum(p.numel() for p in module.parameters(recurse=False)):
            leaf_calls[name] = 0

            def leaf(_module, _inputs, key=name):
                leaf_calls[key] += 1

            handles.append(module.register_forward_pre_hook(leaf))
    corr = importlib.import_module("core.corr")
    utils = importlib.import_module("core.utils.utils")
    old_init, old_sample = corr.CorrBlock.__init__, utils.bilinear_sampler
    old_dino = model.dino.forward_features
    had_dino_override = "forward_features" in model.dino.__dict__
    override = model.dino.__dict__.get("forward_features")

    def corr_init(*args, **kwargs):
        counts["correlation_pyramids"] = counts.get("correlation_pyramids", 0) + 1
        return old_init(*args, **kwargs)

    def sample(*args, **kwargs):
        counts["correlation_sampling"] = counts.get("correlation_sampling", 0) + 1
        return old_sample(*args, **kwargs)

    def dino(*args, **kwargs):
        counts["dino_forward_features"] = counts.get("dino_forward_features", 0) + 1
        return old_dino(*args, **kwargs)

    try:
        corr.CorrBlock.__init__, utils.bilinear_sampler = corr_init, sample
        model.dino.forward_features = dino
        reference, _, lsr_calls = execute(source, target)
    finally:
        corr.CorrBlock.__init__, utils.bilinear_sampler = old_init, old_sample
        if had_dino_override:
            model.dino.forward_features = override
        else:
            del model.dino.forward_features
        for handle in handles:
            handle.remove()
    counts["terminal_lsr"] = lsr_calls
    expected = {"gru": 32, "motion": 32, "flow_head": 32, "mask_head": 1,
                "context": 1, "frontend_v1": 1, "frontend_v2": 1,
                "correlation_pyramids": 2, "correlation_sampling": 256,
                "dino_forward_features": 2, "terminal_lsr": 1}
    assert counts == expected, (counts, expected)
    assert all(leaf_calls.values()), [k for k, v in leaf_calls.items() if not v]
    plain, _, _ = execute(source, target)
    parity = compare(reference, plain)
    assert parity["max_abs"] == 0, parity
    return {"counts": counts, "leaf_calls": leaf_calls, "uncalled_parameter_leaves": [],
            "instrumentation_removed": True, "instrumented_vs_plain": parity}


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
    assert len(rows) == 424
    checkpoints = {str(path): base.sha256(path) for path in (base.M4_CKPT, base.ARCHIVE)}
    assert all(checkpoints[str(path)] == base.EXPECTED[path] for path in (base.M4_CKPT, base.ARCHIVE))
    provenance = base.verify_extracted_dataset()
    assert provenance["all_match_archive"]
    accepted_root = base.ROOT / "GDROS/experiments/sparsegeo/runs/E-098A-M4-COST-CONTRIBUTION-20261116/cache_audit"
    accepted_card = json.loads((accepted_root / "RESULT_CARD.json").read_text())
    assert accepted_card["status"] == "COMPLETE"
    assert accepted_card["checkpoint_sha256"] == checkpoints[str(base.M4_CKPT)]
    metadata = accepted_card["manifests"]["c"]
    cached_manifest = Path(metadata["path"])
    assert base.sha256(cached_manifest) == metadata["sha256"]
    cached = [json.loads(line) for line in cached_manifest.read_text().splitlines()]
    assert len(cached) == 424
    for row, record in zip(rows, cached):
        assert row["pair_index"] == record["pair_index"] and row["pair_id"] == record["pair_id"]
        assert record["iters"] == 32 and record["post_lsr"] and not record["uses_truth_flow"]
        assert base.sha256(Path(record["prediction"])) == record["sha256"]
    sources = [Path(__file__).resolve(), Path(importlib.import_module("m4_streamlined_c_student").__file__).resolve(),
               Path(importlib.import_module("m4_clean_deploy").__file__).resolve(), Path(base.__file__).resolve(),
               Path(branches.__file__).resolve(),
               *[base.TMP / name for name in ("e072a_layerwise_frontend_alignment.py", "e073a_v1_v2_fusion_longtrain.py",
                                              "e074a_f2_f3_twostage_fusion.py", "e075a_preserve_high_precision_recover_c2c5.py")],
               base.OFFICIAL / "core/corr.py", base.OFFICIAL / "core/update.py",
               base.OFFICIAL / "core/LSRnet/LSmodel.py",
               base.TMP.parent / "experiments/sparsegeo/e027a/dino_transfer.py",
               base.TMP.parent / "experiments/sparsegeo/e045a/model.py"]
    start_hashes = {str(p): base.sha256(p) for p in sources}
    (out / "source_snapshot").mkdir()
    for path in sources:
        (out / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    protocol = {"method": M4StreamlinedCStudent.method_name, "pairs": 424, "iterations": 32,
                "input_shape": [1, 3, 512, 512], "fp32": True, "tf32": False,
                "warmups": [0, 211, 423], "raw_parity_tolerance": 1e-5, "post_parity_tolerance": 1e-4,
                "checkpoint_sha256": checkpoints, "dataset_provenance": provenance,
                "cached_c_manifest": metadata, "training": False, "prediction_before_gt_join": True,
                "timing": "standalone C only after deleting original C reference; synchronized wall-clock network plus one official terminal LSR; excludes load, decode, H2D, validation, save and GT scoring",
                "full_m4_comparison": False, "sources": start_hashes}
    base.write_json(out / "PROTOCOL.json", protocol)
    snapshots = [{"stage": "before_load", "snapshot": snapshot()}]
    print(json.dumps({"stage": "PREFLIGHT_PASS_LOAD_C_PREFIX_ONLY"}), flush=True)
    payload, state = branches.checkpoint_state()
    reference, load_report = branches.build_c_student(torch.device("cpu"), state)
    assert payload.get("step") == 80000
    del state, payload
    model = M4StreamlinedCStudent(reference).eval()
    assert set(dict(model.named_children())) == {"dino", "descriptor_adapters", "matcher"}
    stats = parameter_stats(model)
    groups = {name: parameter_stats(module) for name, module in model.named_children()}
    matcher_groups = {name: parameter_stats(module) for name, module in model.matcher.named_children()}
    matcher_direct = sum(p.numel() for p in model.matcher.parameters(recurse=False))
    assert sum(g["parameters"] for g in groups.values()) == stats["parameters"]
    assert sum(g["parameters"] for g in matcher_groups.values()) + matcher_direct == groups["matcher"]["parameters"]
    base.write_json(out / "MODEL_AUDIT.json", {"stats": stats, "groups": groups,
                    "matcher_groups": matcher_groups, "matcher_direct_parameters": matcher_direct,
                    "requires_grad_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "note": "Frozen inference flags are not an estimate of trainable capacity after retraining.",
                    "top_level_children": list(dict(model.named_children())), "pruning": model.audit})
    reference, model = reference.to(device).eval(), model.to(device).eval()
    _, _, official_lsr = base.official_imports()
    lsr_calls = 0

    def lsr(flow):
        nonlocal lsr_calls
        lsr_calls += 1
        return official_lsr(1, 512, 512, flow, 200000)

    direct_parity = []
    with torch.inference_mode():
        for index in (0, 211, 423):
            source, target = (base.image_tensor(rows[index][key]).to(device) for key in ("source", "target"))
            old_raw = reference(source, target, iters=32, return_features=False)[-1]
            new_raw = model(source, target, iters=32)
            raw_diff = compare(old_raw.cpu().numpy(), new_raw.cpu().numpy())
            post_diff = compare(lsr(old_raw).numpy(), lsr(new_raw).numpy())
            direct_parity.append({"pair_index": index, "raw": raw_diff, "post": post_diff})
            assert raw_diff["max_abs"] <= 1e-5 and post_diff["max_abs"] <= 1e-4, direct_parity[-1]
            del old_raw, new_raw, source, target
    base.write_json(out / "DIRECT_C_PARITY.json", direct_parity)
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    import time

    def execute(source, target):
        nonlocal lsr_calls
        lsr_calls = 0
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        raw = model(source, target, iters=32)
        post = lsr(raw)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - begin) * 1000
        assert lsr_calls == 1
        array = post.detach().cpu().numpy().copy()
        assert array.shape == (1, 2, 512, 512) and np.isfinite(array).all()
        return array, elapsed_ms, lsr_calls

    source, target = (base.image_tensor(rows[0][key]).to(device) for key in ("source", "target"))
    coverage = coverage_audit(model, execute, source, target)
    base.write_json(out / "COVERAGE.json", coverage)
    del source, target
    for index in (0, 211, 423):
        source, target = (base.image_tensor(rows[index][key]).to(device) for key in ("source", "target"))
        execute(source, target)
        del source, target
    gc.collect()
    torch.cuda.empty_cache()
    memory_baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    print(json.dumps({"stage": "START_STANDALONE_OS424", "parameters": stats,
                      "memory_baseline_bytes": memory_baseline, "calls": coverage["counts"]}), flush=True)
    snapshots.append({"stage": "before_timing", "snapshot": snapshot()})
    timings, manifest, comparisons = [], [], []
    pred_dir = out / "predictions"
    pred_dir.mkdir()
    for row in rows:
        index = row["pair_index"]
        source, target = (base.image_tensor(row[key]).to(device) for key in ("source", "target"))
        array, ms, calls = execute(source, target)
        timings.append({"pair_index": index, "pair_id": row["pair_id"], "elapsed_ms": ms, "lsr_calls": calls})
        old = np.load(cached[index]["prediction"], allow_pickle=False)
        diff = compare(array, old)
        comparisons.append({"pair_index": index, "pair_id": row["pair_id"], **diff})
        if diff["max_abs"] > 1e-4:
            base.write_json(out / "PARITY_FAILURE.json", comparisons[-1])
            raise RuntimeError("Fresh streamlined C differs from accepted C predictions")
        path = pred_dir / f"{index:06d}.npy"
        np.save(path, array, allow_pickle=False)
        manifest.append({"pair_index": index, "pair_id": row["pair_id"], "path": str(path),
                         "sha256": base.sha256(path), "uses_truth_flow": False})
        del source, target, array, old
        if index % 32 == 0 or index == 423:
            progress = {"completed_pairs": index + 1, "last_ms": ms, "max_parity_abs": max(c["max_abs"] for c in comparisons)}
            base.write_json(out / "PROGRESS.json", progress)
            print(json.dumps({"stage": "PREDICT", **progress}), flush=True)
            snapshots.append({"stage": f"after_{index+1}", "snapshot": snapshot()})
    memory = {"model_live_allocated_bytes": memory_baseline,
              "inference_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "inference_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "reference_deleted_before_timing": True}
    base.write_json(out / "TIMINGS_ROWS.json", timings)
    base.write_json(out / "PREDICTION_MANIFEST.json", manifest)
    base.write_json(out / "C_CACHE_COMPARISONS.json", comparisons)
    print(json.dumps({"stage": "ALL_424_PREDICTIONS_SAVED_START_CPU_GT_JOIN"}), flush=True)
    metrics = {"public_component": [], "vector_epe": []}
    metric_rows = []
    for row in base.pairs(include_gt=True):
        gt = base.read_flo(row["gt"]).astype(np.float32, copy=False)
        prediction = np.load(pred_dir / f"{row['pair_index']:06d}.npy", allow_pickle=False)[0]
        delta = prediction - gt
        values = {"public_component": float(np.mean((np.abs(delta[0]) + np.abs(delta[1])) / 2)),
                  "vector_epe": float(np.mean(np.sqrt(np.square(delta[0]) + np.square(delta[1]))))}
        for key, value in values.items():
            metrics[key].append(value)
        metric_rows.append({"pair_index": row["pair_index"], "pair_id": row["pair_id"], **values})
    base.write_json(out / "METRICS_ROWS.json", metric_rows)
    elapsed = [r["elapsed_ms"] for r in timings]
    timing = {"samples": len(elapsed), "mean_ms": statistics.fmean(elapsed),
              "median_ms": statistics.median(elapsed), "p95_ms": float(np.quantile(elapsed, .95)),
              "std_ms": statistics.pstdev(elapsed), "total_seconds": sum(elapsed) / 1000,
              "fps_from_mean": 1000 / statistics.fmean(elapsed)}
    state_path = out / "M4-streamlined-C-student-state.pt"
    torch.save(model.state_dict(), state_path)
    compact_state = {"path": str(state_path), "bytes": state_path.stat().st_size, "sha256": base.sha256(state_path)}
    end_hashes = {str(p): base.sha256(p) for p in sources}
    assert start_hashes == end_hashes
    card = {"status": "COMPLETE", "method": model.method_name, "pairs": 424, "iterations": 32,
            "fresh_inference": True, "training": False, "full_m4_comparison": False,
            "checkpoint_sha256": checkpoints[str(base.M4_CKPT)], "load_report": load_report,
            "model_parameters": stats, "parameter_groups": groups,
            "metrics": {k: base.summarize(v) for k, v in metrics.items()}, "timing": timing,
            "c_cache_parity": {"pairs": len(comparisons), "bitwise_equal_pairs": sum(c["bitwise_equal"] for c in comparisons),
                               "max_abs": max(c["max_abs"] for c in comparisons)},
            "actual_counts_per_pair": coverage["counts"], "memory": memory,
            "compact_state": compact_state, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device), "environment": base.environment(),
            "shared_host": True, "exclusive_gpu_guaranteed": False, "snapshots": snapshots,
            "source_hashes_start": start_hashes, "source_hashes_end": end_hashes,
            "limitation": "Standalone GPU3 timing, not same-session full-M4 comparison. No fixed ratio to earlier shared-GPU experiments is inferred."}
    base.write_json(out / "RESULT_CARD.json", card)
    print(json.dumps({"stage": "COMPLETE", "parameters": stats, "timing": timing,
                      "metrics": card["metrics"], "parity": card["c_cache_parity"]}), flush=True)


if __name__ == "__main__":
    main()
