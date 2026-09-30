#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.sparsegeo.e003a.data_protocol import load_split_manifest, sha256_file
from experiments.sparsegeo.e003a.losses import selector_losses
from experiments.sparsegeo.e003a.selector import LearnedGeoSelector
from experiments.sparsegeo.e003a.provenance import runtime_identity


ALLOWED_SEEDS = (20260805, 20260806, 20260807)


class SelectorCacheDataset(Dataset):
    def __init__(self, split_manifest: Path, cache_root: Path, split: str, max_pairs=None):
        self.records = load_split_manifest(split_manifest, split)
        if max_pairs is not None:
            self.records = self.records[:max_pairs]
        self.cache_root = Path(cache_root).resolve()
        self.split = split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        pair_id = record["pair_id"]
        candidate = torch.load(
            self.cache_root / "candidates" / self.split / f"{pair_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        label = torch.load(
            self.cache_root / "labels" / self.split / f"{pair_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if candidate["uses_ground_truth"] or candidate.get("uses_datum", True):
            raise RuntimeError(f"candidate cache violates GT/datum isolation: {pair_id}")
        if not label.get("uses_raw_truth_flow", False) or label.get("uses_datum", True):
            raise RuntimeError(f"label cache provenance mismatch: {pair_id}")
        return {
            "pair_id": pair_id,
            "candidate_features": candidate["candidate_features"],
            "source_xy_norm": candidate["source_xy_norm"],
            "target_xy_norm": candidate["target_xy_norm"],
            "candidate_target": label["candidate_target"],
            "grid_xy_pixel": label["grid_xy_pixel"],
            "grid_truth_flow": label["grid_truth_flow"],
        }


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _epoch(model, loader, device, use_feedback, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {
        "total": 0.0,
        "candidate": 0.0,
        "grid": 0.0,
        "confidence": 0.0,
        "grid_error_px": 0.0,
    }
    samples = 0
    gradient_norms = []
    for batch in loader:
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                batch["candidate_features"],
                batch["source_xy_norm"],
                batch["target_xy_norm"],
                use_feedback=use_feedback,
            )
            losses = selector_losses(
                output,
                batch["candidate_target"],
                batch["grid_xy_pixel"],
                batch["grid_truth_flow"],
            )
            if not torch.isfinite(losses.total):
                raise RuntimeError("selector loss became non-finite")
            if training:
                losses.total.backward()
                squared = torch.zeros((), device=device)
                nonzero = False
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("selector gradient contains NaN/Inf")
                        squared = squared + parameter.grad.square().sum()
                        nonzero = nonzero or bool((parameter.grad != 0).any())
                if not nonzero:
                    raise RuntimeError("all selector gradients are zero")
                gradient_norms.append(float(squared.sqrt().item()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        batch_size = batch["candidate_features"].shape[0]
        samples += batch_size
        for key in ("total", "candidate", "grid", "confidence"):
            totals[key] += float(getattr(losses, key).detach().item()) * batch_size
        totals["grid_error_px"] += float(losses.actual_grid_error_px.detach().sum().item())
    result = {key: value / samples for key, value in totals.items()}
    result["samples"] = samples
    result["gradient_norm_mean"] = float(np.mean(gradient_norms)) if gradient_norms else None
    return result


def _state_to_cpu(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_one_seed(args):
    if args.seed not in ALLOWED_SEEDS:
        raise ValueError(f"formal seed must be one of {ALLOWED_SEEDS}")
    if args.variant == "unary" and args.seed != 20260805:
        raise ValueError("unary-only ablation is frozen to the primary seed")
    if not args.smoke and (
        args.max_train_pairs is not None
        or args.max_dev_pairs is not None
        or args.max_epochs != 100
    ):
        raise ValueError("reduced data/epoch budgets are smoke-only")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    train_data = SelectorCacheDataset(
        args.split_manifest, args.cache_root, "train", args.max_train_pairs
    )
    dev_data = SelectorCacheDataset(
        args.split_manifest, args.cache_root, "dev", args.max_dev_pairs
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        generator=generator,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    model = LearnedGeoSelector().to(device)
    if model.trainable_parameter_count >= 100_000:
        raise RuntimeError("selector parameter budget exceeded")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_dev = math.inf
    best_epoch = None
    best_state = None
    stale = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        use_feedback = args.variant == "main"
        train_metrics = _epoch(model, train_loader, device, use_feedback, optimizer)
        with torch.no_grad():
            dev_metrics = _epoch(model, dev_loader, device, use_feedback)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "dev": dev_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if dev_metrics["grid_error_px"] < best_dev:
            best_dev = dev_metrics["grid_error_px"]
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    runtime = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    checkpoint = {
        "model_state": best_state,
        "seed": args.seed,
        "variant": args.variant,
        "best_epoch": best_epoch,
        "best_dev_grid_loss": best_dev,
        "parameter_count": model.trainable_parameter_count,
        "config": {
            "anchors": 128,
            "candidates": 8,
            "feedback_rounds": 2,
            "use_feedback": args.variant == "main",
            "hidden_width": 64,
            "batch_size": 64,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": args.max_epochs,
            "patience": 10,
            "confidence_threshold_px": 5.0,
        },
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "runtime_provenance": getattr(args, "runtime_binding", None),
        "runtime_provenance_identity": (
            runtime_identity(args.runtime_binding) if getattr(args, "runtime_binding", None) else None
        ),
    }
    checkpoint_path = args.output_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    history_path = args.output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    result = {
        "status": "PASS",
        "seed": args.seed,
        "variant": args.variant,
        "best_epoch": best_epoch,
        "best_dev_grid_loss": best_dev,
        "epochs_run": len(history),
        "parameter_count": model.trainable_parameter_count,
        "runtime_seconds": runtime,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history": str(history_path.resolve()),
    }
    result_path = args.output_dir / "training_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("main", "unary"), default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-dev-pairs", type=int)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--formal-config", type=Path)
    parser.add_argument("--cache-gate", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    from experiments.sparsegeo.e003a.spec_binding import validate_frozen_specs
    from experiments.sparsegeo.e003a.provenance import validate_runtime_bindings

    validate_frozen_specs()
    arguments = parse_args()
    if not arguments.smoke:
        required = (
            arguments.repo_root,
            arguments.run_manifest,
            arguments.formal_config,
            arguments.cache_gate,
        )
        if any(value is None for value in required):
            raise ValueError("formal training requires all runtime provenance bindings")
        arguments.runtime_binding = validate_runtime_bindings(
            arguments.repo_root,
            arguments.run_manifest,
            arguments.formal_config,
            arguments.split_manifest,
            arguments.cache_gate,
            cache_root=arguments.cache_root,
            verify_all_cache_files=True,
        )
    train_one_seed(arguments)
