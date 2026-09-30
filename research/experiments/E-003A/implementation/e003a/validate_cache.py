#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.sparsegeo.e003a.candidate_cache import (
    ANCHOR_SHA256,
    CANDIDATE_FEATURE_DIM,
    CHECKPOINT_SHA256,
)
from experiments.sparsegeo.e003a.data_protocol import load_split_manifest, sha256_file
from experiments.sparsegeo.e003a.spec_binding import validate_frozen_specs


def run(args):
    validate_frozen_specs()
    records = load_split_manifest(args.split_manifest)
    manifest_rows = []
    for record in records:
        pair_id = record["pair_id"]
        split = record["split"]
        candidate_path = args.cache_root / "candidates" / split / f"{pair_id}.pt"
        label_path = args.cache_root / "labels" / split / f"{pair_id}.pt"
        if not candidate_path.is_file() or not label_path.is_file():
            raise RuntimeError(f"cache pair is incomplete: {pair_id}")
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
        labels = torch.load(label_path, map_location="cpu", weights_only=True)
        if candidate["pair_id"] != pair_id or candidate["split"] != split:
            raise RuntimeError(f"candidate identity mismatch: {pair_id}")
        if candidate["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise RuntimeError(f"checkpoint binding mismatch: {pair_id}")
        if candidate["anchor_sha256"] != ANCHOR_SHA256:
            raise RuntimeError(f"anchor binding mismatch: {pair_id}")
        if candidate["uses_ground_truth"] or candidate["uses_datum"]:
            raise RuntimeError(f"candidate cache leaked supervision: {pair_id}")
        if candidate["optical_sha256"] != record["files"]["opt"]["sha256"]:
            raise RuntimeError(f"candidate optical SHA mismatch: {pair_id}")
        if candidate["sar_sha256"] != record["files"]["sar_warped"]["sha256"]:
            raise RuntimeError(f"candidate SAR SHA mismatch: {pair_id}")
        expected_shapes = {
            "candidate_features": (128, 8, CANDIDATE_FEATURE_DIM),
            "candidate_scores": (128, 8),
            "candidate_indices": (128, 8),
            "source_xy_norm": (128, 2),
            "target_xy_norm": (128, 8, 2),
        }
        for key, shape in expected_shapes.items():
            if tuple(candidate[key].shape) != shape:
                raise RuntimeError(f"candidate shape mismatch {pair_id} {key}")
            if candidate[key].dtype.is_floating_point and not bool(torch.isfinite(candidate[key]).all()):
                raise RuntimeError(f"candidate non-finite {pair_id} {key}")
        if labels["pair_id"] != pair_id or labels["split"] != split:
            raise RuntimeError(f"label identity mismatch: {pair_id}")
        if not labels["uses_raw_truth_flow"] or labels.get("uses_datum", True):
            raise RuntimeError(f"label provenance mismatch: {pair_id}")
        if labels["truth_flow_sha256"] != record["files"]["truth_flow"]["sha256"]:
            raise RuntimeError(f"label truth-flow SHA mismatch: {pair_id}")
        label_shapes = {
            "candidate_target": (128, 9),
            "target_token_gt": (128, 2),
            "candidate_distance_gt": (128, 8),
            "recoverable": (128,),
            "grid_xy_pixel": (256, 2),
            "grid_truth_flow": (256, 2),
        }
        for key, shape in label_shapes.items():
            if tuple(labels[key].shape) != shape:
                raise RuntimeError(f"label shape mismatch {pair_id} {key}")
            if labels[key].dtype.is_floating_point and not bool(torch.isfinite(labels[key]).all()):
                raise RuntimeError(f"label non-finite {pair_id} {key}")
        if not torch.allclose(
            labels["candidate_target"].sum(dim=-1), torch.ones(128), atol=1e-6, rtol=0
        ):
            raise RuntimeError(f"candidate target normalization mismatch: {pair_id}")
        manifest_rows.append(
            {
                "pair_id": pair_id,
                "split": split,
                "candidate_path": str(candidate_path.resolve()),
                "candidate_sha256": sha256_file(candidate_path),
                "label_path": str(label_path.resolve()),
                "label_sha256": sha256_file(label_path),
            }
        )
    aggregate = hashlib.sha256()
    for row in manifest_rows:
        aggregate.update(
            row["pair_id"].encode("ascii")
            + b"\0"
            + row["candidate_sha256"].encode("ascii")
            + b"\0"
            + row["label_sha256"].encode("ascii")
            + b"\n"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_dir / "cache_manifest.jsonl"
    with jsonl.open("w") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "status": "PASS",
        "pair_count": len(manifest_rows),
        "train_pair_count": sum(row["split"] == "train" for row in manifest_rows),
        "dev_pair_count": sum(row["split"] == "dev" for row in manifest_rows),
        "cache_aggregate_sha256": aggregate.hexdigest(),
        "cache_manifest": str(jsonl.resolve()),
        "cache_manifest_sha256": sha256_file(jsonl),
        "candidate_uses_ground_truth": False,
        "label_uses_raw_truth_flow": True,
        "datum_used_as_label": False,
    }
    path = args.output_dir / "cache_gate.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
