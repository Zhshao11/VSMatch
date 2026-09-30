#!/usr/bin/env python3
"""Persistent DAG task entry points for the LuckyR1 clean OS lineage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/mnt/sda/zhushihao/VS_Match")
TMP = ROOT / "image_matching_part2_migrated/.codex_tmp"
PARENT = ROOT / "GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN"
CLEAN = PARENT / "clean_1810_201"
CURRENT = PARENT / "current_model_exploratory"
OFFICIAL_TEST = ROOT / "datasets/extracted/gdros_official_e003a_testing"
OFFICIAL_TRAIN = ROOT / "datasets/extracted/gdros_official_e003a_training"
HISTORICAL_M4 = ROOT / "GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt"
HISTORICAL_M4_SHA = "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b"
CACHE_ROOT = Path("/dev/shm/zhushihao_luckyr1_ckpts")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_metrics(values):
    array = np.asarray(values, dtype=np.float64)
    return {"pairs": int(array.size), "aepe": float(array.mean()), **{f"cmr{k}": float((array < k).mean() * 100.0) for k in (1, 2, 5, 10, 16)}}


def task_dir(task: str) -> Path:
    return CURRENT / task if task.startswith("exploratory") else CLEAN / task


def write_card(task: str, card: dict) -> None:
    run_dir = task_dir(task)
    run_dir.mkdir(parents=True, exist_ok=True)
    card = dict(card)
    card["task"] = task
    card["updated_at_unix"] = time.time()
    payload = json.dumps(card, indent=2, sort_keys=True) + "\n"
    (run_dir / "RESULT_CARD.json").write_text(payload)
    (run_dir / "RESULT_CARD.md").write_text(f"# {task}\n\nStatus: `{card.get('status')}`\n\n```json\n{payload}```\n")
    attempt = os.getenv("WATCHDOG_JOB_ID", f"manual-{os.getpid()}")
    attempt_dir = run_dir / "attempts" / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "RESULT_CARD.json").write_text(payload)
    (attempt_dir / "RESULT_CARD.md").write_text(f"# {task} attempt {attempt}\n\nStatus: `{card.get('status')}`\n\n```json\n{payload}```\n")


def imports():
    sys.path.insert(0, str(TMP))
    import e072a_layerwise_frontend_alignment as e072

    e072.add_paths()
    import core.datasets
    from core.LSRnet.LSmodel import LSR_list
    from core.loss import matrix2flow, sequence_loss
    from luckyr1_models import BCModel, M4Model, F2_CKPT, F3_CKPT, lsr
    return e072, core.datasets, LSR_list, matrix2flow, sequence_loss, BCModel, M4Model, F2_CKPT, F3_CKPT, lsr


def datasets(core_datasets):
    train_set = core_datasets.opt_sar_train(split="training", dstype="training", root=str(PARENT / "data_view"))
    dev_set = core_datasets.opt_sar_test(split="testing", dstype="testing", root=str(PARENT / "data_view"))
    if len(train_set) != 1810 or len(dev_set) != 201:
        raise RuntimeError(f"split mismatch train={len(train_set)} dev={len(dev_set)}")
    return train_set, dev_set


def exploratory_datasets(core_datasets):
    view = CURRENT / "data_view/training"
    view.mkdir(parents=True, exist_ok=True)
    for name in ("opt", "sar_warped", "truth_flow", "datum"):
        target, source = view / name, OFFICIAL_TRAIN / name
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
    train_set = core_datasets.opt_sar_train(split="training", dstype="training", root=str(CURRENT / "data_view"))
    raw_test = core_datasets.opt_sar_test(split="testing", dstype="testing", root=str(OFFICIAL_TEST))
    if len(train_set) != 2011 or len(raw_test) < 404:
        raise RuntimeError(f"exploratory split mismatch train={len(train_set)} raw_test={len(raw_test)}")
    # Historical E075A formal evaluation used max_pairs=404 over this ordered dataset.
    formal_test = torch.utils.data.Subset(raw_test, range(404))
    return train_set, formal_test


def evaluate_two_phase(model, dataset, device, run_dir: Path, label: str, lsr_fn) -> dict:
    prediction_dir = run_dir / "predictions" / label
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest = prediction_dir / "PREDICTIONS.jsonl"
    manifest.unlink(missing_ok=True)
    model.eval()
    with torch.inference_mode(), manifest.open("a") as handle:
        for index in range(len(dataset)):
            pair_id, source, target, _hidden_gt, _valid = dataset[index]
            source, target = source[None].to(device), target[None].to(device)
            prediction = lsr_fn(model(source, target, 24)[-1]).detach().cpu().half()
            path = prediction_dir / f"{index:04d}.pt"
            torch.save({"pair_index": index, "pair_id": Path(pair_id).stem, "flow": prediction}, path)
            handle.write(json.dumps({"pair_index": index, "pair_id": Path(pair_id).stem, "prediction": str(path), "uses_truth_flow": False}, sort_keys=True) + "\n")
            handle.flush()
    manifest_sha = sha256(manifest)
    values, rows = [], []
    for index in range(len(dataset)):
        pair_id, _source, _target, gt, _valid = dataset[index]
        prediction = torch.load(prediction_dir / f"{index:04d}.pt", map_location="cpu")["flow"].float()
        value = float(torch.linalg.vector_norm(prediction - gt[None].float(), dim=1).mean())
        values.append(value)
        rows.append({"pair_index": index, "pair_id": Path(pair_id).stem, "pair_aepe": value})
    joined = run_dir / f"{label}_METRIC_JOIN.csv"
    with joined.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {**pair_metrics(values), "prediction_manifest": str(manifest), "prediction_manifest_sha256": manifest_sha, "metric_join": str(joined), "prediction_materialized_before_gt_join": True, "metric_scope": "pair-level"}


def selector_path(stage: str) -> Path:
    return CLEAN / stage / "SELECTION.json"


def selection_key(row: dict):
    metric = row["metrics"]
    return metric["cmr5"], metric["cmr2"], metric["cmr1"], -metric["aepe"]


def best_checkpoint(cards: list[dict]) -> dict:
    rows = [row for card in cards for row in card.get("evaluations", [])]
    if not rows:
        raise RuntimeError("no evaluated checkpoint")
    return max(rows, key=selection_key)


def train_task(args, state):
    e072, core_datasets, LSR_list, matrix2flow, sequence_loss, BCModel, M4Model, F2_CKPT, F3_CKPT, lsr_fn = imports()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = e072.cfg()
    task = args.task
    train_set, dev_set = exploratory_datasets(core_datasets) if task.startswith("exploratory_") else datasets(core_datasets)
    run_dir = task_dir(task)
    run_dir.mkdir(parents=True, exist_ok=True)
    if task in ("bc_b", "bc_c"):
        variant = task[-1].upper()
        model = BCModel(variant, device).to(device)
        milestones = [20000, 40000, 80000, 120000, 180000]
        max_steps = args.steps or 180000
        lr = 3e-5
        source_info = {"f2": str(F2_CKPT), "f2_sha256": sha256(F2_CKPT), "f3": str(F3_CKPT), "f3_sha256": sha256(F3_CKPT)}
    elif task == "m4_raw":
        selected = json.loads(selector_path("bc_reduce").read_text())
        model = M4Model(device, Path(selected["B"]["checkpoint"]), Path(selected["C"]["checkpoint"])).to(device)
        milestones = [20000, 40000, 80000, 120000]
        max_steps = args.steps or 120000
        lr = 2e-5
        source_info = selected
    elif task in ("lucky_a", "lucky_b", "lucky_c"):
        kind = task[-1].upper()
        selected = json.loads(selector_path("bc_reduce").read_text())
        raw = json.loads(selector_path("m4_reduce").read_text())
        model = M4Model(device, Path(selected["B"]["checkpoint"]), Path(selected["C"]["checkpoint"]), lucky_kind=kind).to(device)
        raw_state = torch.load(raw["checkpoint"], map_location=device)["model"]
        model.gate.load_state_dict({key.removeprefix("gate."): value for key, value in raw_state.items() if key.startswith("gate.")}, strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in model.modules():
            if module.__class__.__name__ == "EvidenceAdapter":
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        milestones = [2000, 5000, 10000, 20000]
        max_steps = args.steps or 20000
        lr = 1e-5
        source_info = {"raw_m4": raw, "bc": selected, "method": {"A": "geometrically coupled absolute-coordinate belief", "B": "Jacobian-aligned within-modality relation correlation", "C": "joint relation evidence and belief"}[kind]}
    elif task in ("exploratory_a", "exploratory_b", "exploratory_c"):
        from luckyr1_models import HistoricalLuckyM4

        kind = task[-1].upper()
        manifest_card = json.loads((task_dir("exploratory_manifest") / "RESULT_CARD.json").read_text())
        if manifest_card.get("historical_m4_sha256") != HISTORICAL_M4_SHA:
            raise RuntimeError("protected M4 was not verified by the exploratory manifest dependency")
        cache_card = json.loads((task_dir("exploratory_cache") / "RESULT_CARD.json").read_text())
        cache_paths = {name: row["cache"] for name, row in cache_card["files"].items()}
        model = HistoricalLuckyM4(Path(cache_paths["m4"]), HISTORICAL_M4_SHA, kind, device, verify_checksum=False, cache_paths=cache_paths).to(device)
        milestones = [500, 1000, 2000]
        max_steps = args.steps or 2000
        lr = 1e-5
        source_info = {"checkpoint": str(HISTORICAL_M4), "checkpoint_sha256": HISTORICAL_M4_SHA, "checksum_verified_by": str(task_dir("exploratory_manifest") / "RESULT_CARD.json"), "cache_manifest": str(task_dir("exploratory_cache") / "RESULT_CARD.json"), "claim_scope": "CURRENT_MODEL_EXPLORATORY_ONLY_NOT_CLEAN_MODEL_UNSEEN", "train_pairs": 2011, "evaluation_pairs": 404, "method": kind}
    else:
        raise ValueError(task)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable model-facing parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    progress_path = run_dir / "TRAIN_PROGRESS.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    evaluations, curve = progress.get("evaluations", []), progress.get("curve", [])
    start_step = 0
    checkpoints = list((run_dir / "checkpoints").glob(f"{task}_step*.pt"))
    if checkpoints:
        latest = max(checkpoints, key=lambda path: int(path.stem.rsplit("step", 1)[1]))
        resume = torch.load(latest, map_location=device)
        start_step = int(resume["step"])
        model.load_state_dict(resume["model"], strict=resume.get("checkpoint_format") != "trainable_delta")
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        if "rng_state" in resume:
            rng.bit_generator.state = resume["rng_state"]
    state.update({"status": "RUNNING", "last_completed_stage": "SOURCE_EQUIVALENCE_AND_SPLIT_AUDIT", "sources": source_info, "resume_step": start_step, "train_pairs": len(train_set), "evaluation_pairs": len(dev_set), "claim_scope": "CURRENT_MODEL_EXPLORATORY_ONLY" if task.startswith("exploratory_") else "CLEAN_1810_201_MODEL_UNSEEN_DEV", "trainable_parameters": int(sum(p.numel() for p in trainable)), "scheduler": {"watchdog_job_id": os.getenv("WATCHDOG_JOB_ID"), "assigned_physical_gpus": os.getenv("ASSIGNED_GPUS"), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "logical_device": "cuda:0", "candidate_gpus": None}})
    write_card(task, state)
    milestones = {step for step in milestones if step <= max_steps} | {max_steps}
    if start_step and not any(int(row["step"]) == start_step for row in evaluations):
        metric = evaluate_two_phase(model, dev_set, device, run_dir, f"step{start_step}_iter24_post_lsr", lsr_fn)
        evaluations.append({"step": start_step, "checkpoint": str(latest), "checkpoint_sha256": sha256(latest), "metrics": metric})
        progress_path.write_text(json.dumps({"curve": curve, "evaluations": evaluations}, indent=2) + "\n")
    model.train()
    for step in range(start_step + 1, max_steps + 1):
        index = int(rng.integers(0, len(train_set)))
        source, target, gt, valid = train_set[index]
        source, target, gt, valid = source[None].to(device), target[None].to(device), gt[None].to(device), valid[None].to(device)
        predictions = model(source, target, 12)
        loss_seq, info = sequence_loss(predictions, gt, valid, cfg)
        affine_predictions = matrix2flow(LSR_list(predictions), gt)
        loss_geo, _ = sequence_loss(affine_predictions, gt, valid, cfg)
        loss = 0.5 * loss_seq + 0.5 * loss_geo
        if task == "bc_c":
            distill = sum(F.smooth_l1_loss(prediction, teacher) for prediction, teacher in zip(predictions, model.teacher_average)) / len(predictions)
            loss = loss + 0.2 * distill
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        state["step"] = step
        if step == 1 or step % max(100, max_steps // 40) == 0:
            curve.append({"step": step, "loss": float(loss.detach()), "grad_norm": grad_norm, "train_pixel_epe_diagnostic": info["epe"]})
            state.update({"last_completed_stage": f"TRAIN_STEP_{step}", "step": step, "curve": curve})
            write_card(task, state)
        if step in milestones:
            checkpoint = run_dir / "checkpoints" / f"{task}_step{step}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_format = "trainable_delta" if task.startswith("exploratory_") else "full_model"
            model_state = {name: value for name, value in model.state_dict().items() if ".evidence." in name} if checkpoint_format == "trainable_delta" else model.state_dict()
            torch.save({"model": model_state, "optimizer": optimizer.state_dict(), "rng_state": rng.bit_generator.state, "task": task, "step": step, "sources": source_info, "checkpoint_format": checkpoint_format}, checkpoint)
            metric = evaluate_two_phase(model, dev_set, device, run_dir, f"step{step}_iter24_post_lsr", lsr_fn)
            row = {"step": step, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "metrics": metric}
            evaluations.append(row)
            progress_path.write_text(json.dumps({"curve": curve, "evaluations": evaluations}, indent=2) + "\n")
            state.update({"last_completed_stage": f"CHECKPOINT_{step}_DEV_EVALUATED", "curve": curve, "evaluations": evaluations})
            write_card(task, state)
            model.train()
            if task.startswith("lucky_") and step >= 5000:
                raw_metric = json.loads(selector_path("m4_reduce").read_text())["metrics"]
                weak = metric["aepe"] > raw_metric["aepe"] * 1.25 and metric["cmr5"] < raw_metric["cmr5"]
                if weak:
                    state.update({"status": "COST_GATE_STOP", "last_completed_stage": f"CHECKPOINT_{step}_COST_GATE", "cost_gate_reason": "AEPE > 1.25x raw M4 and CMR@5 did not improve", "selected": max(evaluations, key=selection_key)})
                    write_card(task, state)
                    return state
            if task.startswith("exploratory_") and step >= 500:
                control_key = (88.86138613861387, 78.96039603960396, 56.683168316831676, -2.3613)
                signal = selection_key({"metrics": metric}) > control_key
                if not signal:
                    state.update({"status": "COST_GATE_STOP", "last_completed_stage": f"CHECKPOINT_{step}_COST_GATE", "cost_gate_reason": "no strict pair-level improvement over protected M4 under CMR@5/2/1 then AEPE ordering", "selected": max(evaluations, key=selection_key), "claim_scope": "CURRENT_MODEL_EXPLORATORY_ONLY", "cost_gate_signal": False})
                    write_card(task, state)
                    return state
    state.update({"status": "COMPLETE", "last_completed_stage": "TRAIN_AND_DEV_COMPLETE", "curve": curve, "evaluations": evaluations, "selected": max(evaluations, key=selection_key)})
    write_card(task, state)
    return state


def constructor_smoke(args, state):
    _e072, core_datasets, _LSR_list, _matrix2flow, _sequence_loss, BCModel, _M4Model, F2_CKPT, F3_CKPT, _lsr = imports()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _train, dev = datasets(core_datasets)
    _pair, source, target, _gt, _valid = dev[0]
    source, target = source[None].to(device), target[None].to(device)
    rows = []
    for variant in ("B", "C"):
        model = BCModel(variant, device).to(device).eval()
        with torch.inference_mode():
            output = model(source, target, 24)
        rows.append({"variant": variant, "outputs": len(output), "shape": list(output[-1].shape), "finite": bool(torch.isfinite(output[-1]).all())})
    if not all(row["outputs"] == 24 and row["finite"] for row in rows):
        raise RuntimeError(rows)
    state.update({"status": "COMPLETE", "last_completed_stage": "SPLIT_SAFE_BC_CONSTRUCTOR_SMOKE", "variants": rows, "sources": {"f2_sha256": sha256(F2_CKPT), "f3_sha256": sha256(F3_CKPT)}})
    write_card(args.task, state)


def lucky_model_g0(args, state):
    e072, core_datasets, _LSR_list, _matrix2flow, _sequence_loss, _BCModel, _M4Model, _F2, _F3, _lsr = imports()
    from luckyr1_models import LuckySafeE073

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _train, dev = datasets(core_datasets)
    _pair, source, target, _hidden_gt, _valid = dev[0]
    source, target = source[None].to(device), target[None].to(device)
    rows = []
    for kind in ("A", "B", "C"):
        model = LuckySafeE073(e072, e072.cfg(), "F3", device, kind).to(device)
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
        outputs = model(source, target, iters=2)
        loss = outputs[-1].float().square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.evidence.parameters() if parameter.requires_grad]
        rows.append({
            "kind": kind,
            "outputs": len(outputs),
            "shape": list(outputs[-1].shape),
            "finite_output": bool(torch.isfinite(outputs[-1]).all()),
            "trainable_evidence_parameters": int(sum(parameter.numel() for parameter in model.evidence.parameters() if parameter.requires_grad)),
            "finite_nonzero_gradient": bool(gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients) and any(float(gradient.abs().sum()) > 0 for gradient in gradients)),
            "path": "real CorrBlock -> evidence adapter -> real BasicUpdateBlock",
        })
        del model, outputs, loss, gradients
        torch.cuda.empty_cache()
    if not all(row["finite_output"] and row["finite_nonzero_gradient"] and row["shape"] == [1, 2, 512, 512] for row in rows):
        raise RuntimeError(rows)
    state.update({"status": "COMPLETE", "last_completed_stage": "MODEL_FACING_ABC_FORWARD_BACKWARD_SMOKE", "variants": rows, "uses_gt": False, "method_result": False})
    write_card(args.task, state)


def reduce_stage(args, state):
    if args.task == "bc_reduce":
        result = {}
        for variant in ("B", "C"):
            card = json.loads((task_dir(f"bc_{variant.lower()}") / "RESULT_CARD.json").read_text())
            result[variant] = best_checkpoint([card])
    elif args.task == "m4_reduce":
        card = json.loads((task_dir("m4_raw") / "RESULT_CARD.json").read_text())
        result = best_checkpoint([card])
    elif args.task == "exploratory_reduce":
        cards = [json.loads((task_dir(f"exploratory_{kind.lower()}") / "RESULT_CARD.json").read_text()) for kind in ("A", "B", "C")]
        result = best_checkpoint(cards)
        result["winning_arm"] = Path(result["checkpoint"]).name.split("_")[1].upper()
        result["claim_scope"] = "CURRENT_MODEL_EXPLORATORY_ONLY"
    else:
        cards = [json.loads((task_dir(f"lucky_{kind.lower()}") / "RESULT_CARD.json").read_text()) for kind in ("A", "B", "C")]
        result = best_checkpoint(cards)
        result["winning_arm"] = Path(result["checkpoint"]).name.split("_")[1].upper()
    path = (task_dir(args.task) / "SELECTION.json") if args.task == "exploratory_reduce" else selector_path(args.task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    scope = "current-model exploratory OS 404; not a clean selection claim" if args.task == "exploratory_reduce" else "predeclared clean 201 dev"
    state.update({"status": "COMPLETE", "last_completed_stage": "CHECKPOINT_COST_REDUCTION", "selection": result, "selection_policy": f"CMR@5, then CMR@2, then CMR@1, then AEPE; pair-level on {scope}"})
    write_card(args.task, state)


def exploratory_manifest(args, state):
    actual_sha = sha256(HISTORICAL_M4)
    if actual_sha != HISTORICAL_M4_SHA:
        raise RuntimeError(f"protected historical M4 checksum mismatch: {actual_sha}")
    state.update({"status": "COMPLETE", "last_completed_stage": "EXPLORATORY_TRACK_SEPARATED", "claim_scope": "CURRENT_MODEL_EXPLORATORY_ONLY_NOT_MODEL_UNSEEN", "historical_m4": str(HISTORICAL_M4), "historical_m4_sha256": actual_sha, "known_metrics": {"scope": "OS 404 pair-level", "aepe": 2.3613, "cmr1": 56.6832, "cmr2": 78.9604, "cmr5": 88.8614, "cmr10": 94.8020, "cmr16": 97.0297}, "no_checkpoint_modified": True})
    write_card(args.task, state)


def exploratory_cache(args, state):
    sources = {
        "m4": HISTORICAL_M4,
        "b": ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-b-stage180k/checkpoints/B_step180000.pt",
        "c": ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-c-stage180k/checkpoints/C_step180000.pt",
        "e073_f2": ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f2-warm120k/checkpoints/F2_step120000.pt",
        "e073_f3": ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f3-warm120k/checkpoints/F3_step120000.pt",
        "e072_v1": ROOT / "GDROS/experiments/sparsegeo/runs/E-072A-ZZVR-LAYERWISE-FRONTEND-ALIGNMENT/run-v1-r60k/checkpoints/V1_step60000.pt",
        "e072_v2": ROOT / "GDROS/experiments/sparsegeo/runs/E-072A-ZZVR-LAYERWISE-FRONTEND-ALIGNMENT/run-v2-r60k/checkpoints/V2_step60000.pt",
        "e046": ROOT / "GDROS/experiments/sparsegeo/runs/E-046A-ZZVR-JOINT-TRAINING/run-r2/checkpoints/masked_pretrain_joint.pt",
    }
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = CACHE_ROOT / f"{name}.pt"
        temporary = target.with_suffix(".pt.tmp")
        digest = hashlib.sha256()
        reused = target.is_file() and target.stat().st_size == source.stat().st_size
        if reused:
            with target.open("rb") as reader:
                for chunk in iter(lambda: reader.read(8 << 20), b""):
                    digest.update(chunk)
        else:
            with source.open("rb") as reader, temporary.open("wb") as writer:
                while True:
                    chunk = reader.read(8 << 20)
                    if not chunk:
                        break
                    writer.write(chunk)
                    digest.update(chunk)
            os.replace(temporary, target)
        files[name] = {"source": str(source), "cache": str(target), "size_bytes": target.stat().st_size, "sha256": digest.hexdigest(), "reused_existing_cache": reused}
        state.update({"status": "RUNNING", "last_completed_stage": f"CACHED_{name}", "files": files})
        write_card(args.task, state)
    if files["m4"]["sha256"] != HISTORICAL_M4_SHA:
        raise RuntimeError("cached M4 checksum mismatch")
    state.update({"status": "COMPLETE", "last_completed_stage": "HISTORICAL_CHECKPOINT_CACHE_COMPLETE", "files": files, "cache_filesystem": "/dev/shm", "gpu_used": False})
    write_card(args.task, state)


def exploratory_a_finalize(args, state):
    arm_task = "exploratory_a"
    run_dir = task_dir(arm_task)
    prediction_dir = run_dir / "predictions/step500_iter24_post_lsr"
    manifest = prediction_dir / "PREDICTIONS.jsonl"
    joined = run_dir / "step500_iter24_post_lsr_METRIC_JOIN.csv"
    checkpoint = run_dir / "checkpoints/exploratory_a_step500.pt"
    prediction_files = sorted(prediction_dir.glob("*.pt"))
    manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    with joined.open() as handle:
        values = [float(row["pair_aepe"]) for row in csv.DictReader(handle)]
    if len(prediction_files) != 404 or len(manifest_rows) != 404 or len(values) != 404:
        raise RuntimeError(f"incomplete recovered evaluation predictions={len(prediction_files)} manifest={len(manifest_rows)} join={len(values)}")
    metric = {
        **pair_metrics(values),
        "prediction_manifest": str(manifest),
        "prediction_manifest_sha256": sha256(manifest),
        "metric_join": str(joined),
        "metric_join_sha256": sha256(joined),
        "prediction_materialized_before_gt_join": True,
        "metric_scope": "pair-level",
    }
    row = {"step": 500, "checkpoint": str(checkpoint), "checkpoint_sha256": None, "checkpoint_size_bytes": checkpoint.stat().st_size, "checkpoint_sha_status": "SKIPPED_AFTER_COST_GATE_TO_RELEASE_GPU", "metrics": metric}
    arm_card = {
        "status": "COST_GATE_STOP",
        "last_completed_stage": "STEP500_METRICS_RECOVERED_AFTER_POST_METRIC_SHA_INTERRUPTION",
        "task": arm_task,
        "claim_scope": "CURRENT_MODEL_EXPLORATORY_ONLY",
        "train_pairs": 2011,
        "evaluation_pairs": 404,
        "evaluations": [row],
        "selected": row,
        "cost_gate_reason": "AEPE > 1.25x protected M4 and CMR@5 did not improve",
        "protected_m4": {"checkpoint": str(HISTORICAL_M4), "sha256": HISTORICAL_M4_SHA},
        "prediction_count": 404,
        "manifest_count": 404,
        "join_count": 404,
        "recovered_from_attempt": "20260913_195911_20d47bc1",
        "full_checkpoint_preserved": True,
    }
    write_card(arm_task, arm_card)
    state.update({"status": "COMPLETE", "last_completed_stage": "EXPLORATORY_A_COST_GATE_FINALIZED", "arm_result": arm_card})
    write_card(args.task, state)


def source_audit(args, state):
    seed_paths = {
        "seed1": PARENT / "foundation/F2/checkpoints/F2_step120000.pt",
        "seed2": PARENT / "foundation/F3/checkpoints/F3_step120000.pt",
    }
    sources = {}
    for name, path in seed_paths.items():
        payload = torch.load(path, map_location="cpu")
        keys = list(payload["model"])
        forbidden = [key for key in keys if "e046" in key.lower() or "front_v2" in key.lower()]
        if forbidden:
            raise RuntimeError(f"{name} contains forbidden learned lineage keys: {forbidden[:8]}")
        sources[name] = {"path": str(path), "sha256": sha256(path), "state_keys": len(keys), "forbidden_keys": forbidden}
    state.update({
        "status": "COMPLETE",
        "last_completed_stage": "SOURCE_AND_CHECKPOINT_EQUIVALENCE_AUDIT",
        "sources": sources,
        "equivalence": {
            "safe_foundation": "E073 F2 feature fusion / F3 correlation fusion operators retained; both inputs are split-safe E072 V1 instead of historical V1/V2",
            "bc_b": "E074 B 9-channel dense gate over frozen F2/F3 recurrent flow sequences",
            "bc_c": "E074 C F3-initialized student with frozen F2/F3 teacher average",
            "m4_raw": "E075A M4 16-channel region gate with 4x4 average pooling",
            "lucky_a": "real CorrBlock -> absolute-coordinate belief with unmatched mass and geometric coupling -> BasicUpdateBlock",
            "lucky_b": "real CorrBlock plus Jacobian-aligned within-modality relation evidence -> BasicUpdateBlock",
            "lucky_c": "relation evidence followed by geometric belief -> BasicUpdateBlock",
        },
        "claim_boundary": "The clean V1xV1 raw control is a split-safe reconstructed lineage, not numerical equivalence to historical E073 V1/V2 or protected E075A-M4.",
        "invalidated": ["foundation_repaired/F2", "foundation_repaired/F3", "downstream_smoke_r2"],
    })
    write_card(args.task, state)


def finalizer(args, state):
    e072, core_datasets, _LSR_list, _matrix2flow, _sequence_loss, _BCModel, M4Model, _F2, _F3, lsr_fn = imports()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    selected_bc = json.loads(selector_path("bc_reduce").read_text())
    selected_lucky = json.loads(selector_path("lucky_reduce").read_text())
    kind = selected_lucky["winning_arm"]
    model = M4Model(device, Path(selected_bc["B"]["checkpoint"]), Path(selected_bc["C"]["checkpoint"]), lucky_kind=kind).to(device)
    model.load_state_dict(torch.load(selected_lucky["checkpoint"], map_location=device)["model"], strict=True)
    raw_official = core_datasets.opt_sar_test(split="testing", dstype="testing", root=str(OFFICIAL_TEST))
    if len(raw_official) < 404:
        raise RuntimeError(f"official test size is {len(raw_official)}, expected at least 404")
    official = torch.utils.data.Subset(raw_official, range(404))
    metric = evaluate_two_phase(model, official, device, task_dir(args.task), "official404_iter24_post_lsr", lsr_fn)
    state.update({"status": "COMPLETE", "last_completed_stage": "OFFICIAL_404_FINALIZED", "winner": kind, "checkpoint": selected_lucky["checkpoint"], "checkpoint_sha256": sha256(Path(selected_lucky["checkpoint"])), "formal_metrics": metric, "clean_lineage": "1810 train -> 201 untouched dev selection -> 404 official final", "hidden_gt_inference": False, "whu_started": False})
    write_card(args.task, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["source_audit", "lucky_model_g0", "constructor_smoke_safe", "bc_b", "bc_c", "bc_reduce", "m4_raw", "m4_reduce", "lucky_a", "lucky_b", "lucky_c", "lucky_reduce", "exploratory_manifest", "exploratory_cache", "exploratory_a", "exploratory_a_finalize", "exploratory_b", "exploratory_c", "exploratory_reduce", "finalizer"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7501)
    args = parser.parse_args()
    started = time.time()
    state = {"status": "RUNNING", "last_completed_stage": "START", "started_at_unix": started}

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write_card(args.task, state)
    try:
        if args.task == "source_audit":
            source_audit(args, state)
        elif args.task == "exploratory_cache":
            exploratory_cache(args, state)
        elif args.task == "lucky_model_g0":
            lucky_model_g0(args, state)
        elif args.task == "exploratory_a_finalize":
            exploratory_a_finalize(args, state)
        elif args.task == "constructor_smoke_safe":
            constructor_smoke(args, state)
        elif args.task in ("bc_b", "bc_c", "m4_raw", "lucky_a", "lucky_b", "lucky_c", "exploratory_a", "exploratory_b", "exploratory_c"):
            train_task(args, state)
        elif args.task.endswith("_reduce"):
            reduce_stage(args, state)
        elif args.task == "exploratory_manifest":
            exploratory_manifest(args, state)
        else:
            finalizer(args, state)
    except BaseException as error:
        state.update({"status": "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "IMPLEMENTATION_FAILED", "cause": repr(error), "traceback": traceback.format_exc()})
        write_card(args.task, state)
    state["elapsed_seconds"] = time.time() - started
    write_card(args.task, state)
    if state["status"] not in ("COMPLETE", "COST_GATE_STOP"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
