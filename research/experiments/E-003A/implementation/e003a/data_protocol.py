#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import zipfile
from pathlib import Path

from PIL import Image
import numpy as np
from scipy.fft import dctn


ARCHIVE_BYTES = 5_727_075_338
ARCHIVE_SHA256 = "2887e7de8cd72290df688de0ce4071f42d7bfdec8507a24716ae2ea15c3f3cf5"
EXPECTED_PAIRS = 2011
SPLIT_SALT = "M001-v0.6-E003A|"
MODALITIES = ("opt", "sar_warped", "truth_flow", "datum")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_optical_sha(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        payload = struct.pack("<II", width, height) + image.tobytes()
    return hashlib.sha256(payload).hexdigest()


def optical_phash64(path: Path) -> int:
    resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((32, 32), resampling)
        values = np.asarray(grayscale, dtype=np.float64)
    low = dctn(values, type=2, norm="ortho")[:8, :8]
    median = np.median(low.reshape(-1)[1:])
    bits = (low >= median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _pair_id(path: Path) -> str:
    return path.stem.split("_")[0]


def validate_training_root(root: Path):
    root = Path(root).resolve()
    paths = {
        "opt": sorted((root / "opt").glob("*_opt.png")),
        "sar_warped": sorted((root / "sar_warped").glob("*_sar_warped.png")),
        "truth_flow": sorted((root / "truth_flow").glob("*.flo")),
        "datum": sorted((root / "datum").glob("*.npy")),
    }
    counts = {name: len(values) for name, values in paths.items()}
    if set(counts.values()) != {EXPECTED_PAIRS}:
        raise RuntimeError(f"training modality counts must all be {EXPECTED_PAIRS}: {counts}")
    ids = {name: {_pair_id(path) for path in values} for name, values in paths.items()}
    reference = ids["opt"]
    if any(values != reference for values in ids.values()):
        raise RuntimeError("training modalities do not contain identical pair ids")
    expected_ids = {f"{index:06d}" for index in range(EXPECTED_PAIRS)}
    if reference != expected_ids:
        raise RuntimeError("training pair ids are not the exact 000000..002010 set")
    for path in paths["opt"][:3] + paths["sar_warped"][:3]:
        with Image.open(path) as image:
            if image.size != (512, 512):
                raise RuntimeError(f"E-003A requires 512x512 images: {path} -> {image.size}")
            image.load()
    return paths


def validate_archive(archive: Path):
    archive = Path(archive).resolve()
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise RuntimeError(f"archive size mismatch: {archive.stat().st_size}")
    digest = sha256_file(archive)
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(f"archive SHA mismatch: {digest}")
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        for member in members:
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe ZIP member: {member.filename}")
        bad = source.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        counts = {
            name: sum(
                1
                for member in members
                if not member.is_dir()
                and len(Path(member.filename).parts) == 3
                and Path(member.filename).parts[:2] == ("os_dataset", name)
            )
            for name in MODALITIES
        }
    if set(counts.values()) != {EXPECTED_PAIRS}:
        raise RuntimeError(f"archive modality counts must all be {EXPECTED_PAIRS}: {counts}")
    return {"path": str(archive), "size_bytes": ARCHIVE_BYTES, "sha256": digest, "counts": counts}


def publish_archive(archive: Path, publish_root: Path):
    archive_record = validate_archive(archive)
    publish_root = Path(publish_root).resolve()
    if publish_root.exists():
        validate_training_root(publish_root)
        return archive_record, False
    staging = publish_root.with_name(publish_root.name + ".staging-e003a")
    if staging.exists():
        raise RuntimeError(f"staging path already exists; inspect it manually: {staging}")
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(staging)
    extracted = staging / "os_dataset"
    validate_training_root(extracted)
    publish_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(extracted, publish_root)
    staging.rmdir()
    validate_training_root(publish_root)
    return archive_record, True


def _testing_optical_hashes(testing_root: Path):
    optical_root = Path(testing_root).resolve() / "testing" / "opt"
    files = sorted(optical_root.glob("*.png"))
    if len(files) != 424:
        raise RuntimeError(f"overlap audit requires exactly 424 testing optical images: {len(files)}")
    return {decoded_optical_sha(path): str(path) for path in files}


def build_grouped_split_manifest(training_root: Path, testing_root: Path, output_dir: Path):
    training_root = Path(training_root).resolve()
    output_dir = Path(output_dir).resolve()
    paths = validate_training_root(training_root)
    by_id = {
        modality: {_pair_id(path): path for path in modality_paths}
        for modality, modality_paths in paths.items()
    }
    records = []
    groups = {}
    for pair_id in sorted(by_id["opt"]):
        opt_path = by_id["opt"][pair_id]
        group_key = decoded_optical_sha(opt_path)
        groups.setdefault(group_key, []).append(pair_id)
        records.append(
            {
                "pair_id": pair_id,
                "group_key": group_key,
                "optical_phash64": f"{optical_phash64(opt_path):016x}",
                "files": {
                    modality: {
                        "path": str(by_id[modality][pair_id]),
                        "sha256": sha256_file(by_id[modality][pair_id]),
                    }
                    for modality in MODALITIES
                },
            }
        )
    ordered_groups = sorted(
        groups,
        key=lambda key: hashlib.sha256((SPLIT_SALT + key).encode("ascii")).hexdigest(),
    )
    train_group_count = math.floor(0.8 * len(ordered_groups))
    train_groups = set(ordered_groups[:train_group_count])
    for record in records:
        record["split"] = "train" if record["group_key"] in train_groups else "dev"
    if {record["group_key"] for record in records if record["split"] == "train"} & {
        record["group_key"] for record in records if record["split"] == "dev"
    }:
        raise RuntimeError("decoded optical group leakage across train/dev")

    testing_hashes = _testing_optical_hashes(testing_root)
    overlaps = [
        {
            "pair_id": record["pair_id"],
            "training_optical": record["files"]["opt"]["path"],
            "testing_optical": testing_hashes[record["group_key"]],
        }
        for record in records
        if record["group_key"] in testing_hashes
    ]
    train_phash = [
        (record["pair_id"], int(record["optical_phash64"], 16))
        for record in records
        if record["split"] == "train"
    ]
    dev_phash = [
        (record["pair_id"], int(record["optical_phash64"], 16))
        for record in records
        if record["split"] == "dev"
    ]
    phash_nearest = []
    for dev_id, dev_value in dev_phash:
        distances = [bin(dev_value ^ train_value).count("1") for _, train_value in train_phash]
        minimum = min(distances)
        train_id = train_phash[distances.index(minimum)][0]
        phash_nearest.append(
            {"dev_pair_id": dev_id, "train_pair_id": train_id, "hamming": minimum}
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / "split_manifest.jsonl"
    with jsonl.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "protocol": "M001-v0.6-E003A-decoded-optical-group-hash-80-20",
        "salt": SPLIT_SALT,
        "training_root": str(training_root),
        "pair_count": len(records),
        "group_count": len(groups),
        "train_group_count": train_group_count,
        "dev_group_count": len(groups) - train_group_count,
        "train_pair_count": sum(record["split"] == "train" for record in records),
        "dev_pair_count": sum(record["split"] == "dev" for record in records),
        "train_dev_group_overlap": 0,
        "testing_optical_exact_overlap_count": len(overlaps),
        "testing_optical_exact_overlaps": overlaps,
        "phash_protocol": "PIL-L-32x32-LANCZOS-DCT2-ortho-low8x8-median64",
        "cross_split_phash_nearest_min": min(row["hamming"] for row in phash_nearest),
        "cross_split_phash_nearest_p50": float(
            np.median([row["hamming"] for row in phash_nearest])
        ),
        "cross_split_phash_hamming_le4_count": sum(
            row["hamming"] <= 4 for row in phash_nearest
        ),
        "cross_split_phash_nearest": phash_nearest,
        "manifest_jsonl": str(jsonl),
        "manifest_jsonl_sha256": sha256_file(jsonl),
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = sha256_file(summary_path)
    if overlaps:
        raise RuntimeError(
            f"found {len(overlaps)} decoded optical overlaps with OS testing; see {summary_path}"
        )
    return summary


def load_split_manifest(path: Path, allowed_split=None):
    path = Path(path).resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != EXPECTED_PAIRS or len({row["pair_id"] for row in records}) != EXPECTED_PAIRS:
        raise RuntimeError("split manifest must contain 2011 unique pairs")
    if allowed_split is not None:
        if allowed_split not in ("train", "dev"):
            raise ValueError("allowed_split must be train or dev")
        records = [row for row in records if row["split"] == allowed_split]
    return records


def enforce_split_access(record, training_root: Path, allowed_split: str):
    root = Path(training_root).resolve()
    if record.get("split") != allowed_split:
        raise PermissionError(f"pair {record.get('pair_id')} is not in split {allowed_split}")
    for value in record["files"].values():
        path = Path(value["path"]).resolve()
        if root not in path.parents:
            raise PermissionError(f"manifest path escapes training root: {path}")
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--publish-root", type=Path, required=True)
    parser.add_argument("--testing-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    from experiments.sparsegeo.e003a.spec_binding import validate_frozen_specs

    validate_frozen_specs()
    args = parse_args()
    archive, extracted = publish_archive(args.archive, args.publish_root)
    split = build_grouped_split_manifest(args.publish_root, args.testing_root, args.output_dir)
    result = {"archive": archive, "extracted_now": extracted, "split": split}
    path = args.output_dir / "data_protocol_gate.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "gate": str(path), "split": split}, sort_keys=True))


if __name__ == "__main__":
    main()
