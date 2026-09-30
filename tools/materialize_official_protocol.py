#!/usr/bin/env python3
"""Rebuild official GDROS or CRFT test layouts from raw aligned OS pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PAIR_RE = re.compile(r"^(opt|sar)(.+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class Task:
    kind: str
    split: str
    variant: str | None
    official_id: str
    raw_id: str
    optical_path: str
    sar_path: str
    output_root: str
    height: int
    width: int
    coefficients: list[list[float]]
    affine_matrix: list[list[float]]
    expected_optical_hash: str
    expected_warped_hash: str
    expected_flow_crc32: str


def pixel_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes(order="C")).hexdigest()


def discover_pairs(raw_root: Path) -> dict[str, tuple[Path, Path]]:
    modalities: dict[str, dict[str, Path]] = {"opt": {}, "sar": {}}
    for path in raw_root.iterdir():
        match = PAIR_RE.fullmatch(path.name)
        if match and path.is_file():
            modalities[match.group(1).lower()][match.group(2)] = path
    if set(modalities["opt"]) != set(modalities["sar"]):
        raise RuntimeError("Raw optical and SAR IDs do not match")
    return {
        identifier: (path, modalities["sar"][identifier])
        for identifier, path in modalities["opt"].items()
    }


def hash_raw_pair(item: tuple[str, tuple[Path, Path]]) -> tuple[str, str, str]:
    identifier, (optical_path, sar_path) = item
    optical = cv2.imread(str(optical_path), cv2.IMREAD_GRAYSCALE)
    sar = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
    if optical is None or sar is None:
        raise RuntimeError(f"Cannot read raw pair {identifier}")
    return pixel_sha256(optical), pixel_sha256(sar), identifier


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        if not cv2.imwrite(str(temporary), image):
            raise RuntimeError(f"OpenCV failed to write {temporary}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_flow(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".flo", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"PIEH")
            stream.write(struct.pack("<ii", flow.shape[1], flow.shape[0]))
            stream.write(flow.astype("<f4", copy=False).tobytes(order="C"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def output_paths(task: Task) -> tuple[Path, Path, Path]:
    root = Path(task.output_root)
    if task.kind == "gdros":
        return (
            root / task.split / "opt" / f"{task.official_id}_opt.png",
            root / task.split / "sar_warped" / f"{task.official_id}_sar_warped.png",
            root / task.split / "truth_flow" / f"{task.official_id}.flo",
        )
    if not task.variant:
        raise ValueError("CRFT task is missing a variant")
    variant_root = root / task.split / "warped" / task.variant
    return (
        variant_root / "image_pair" / f"{task.official_id}_opt.tif",
        variant_root / "image_pair" / f"{task.official_id}_sar_warped.tif",
        variant_root / "truth_flow" / f"{task.official_id}.flo",
    )


def materialize(task: Task) -> dict[str, object]:
    optical = cv2.imread(task.optical_path, cv2.IMREAD_GRAYSCALE)
    sar = cv2.imread(task.sar_path, cv2.IMREAD_GRAYSCALE)
    if optical is None or sar is None:
        raise RuntimeError(f"Failed to read raw pair {task.raw_id}")
    if optical.shape != (task.height, task.width) or sar.shape != optical.shape:
        raise RuntimeError(
            f"Unexpected raw shape for {task.raw_id}: optical={optical.shape}, sar={sar.shape}"
        )

    coefficients = np.asarray(task.coefficients, dtype=np.float64)
    affine = np.asarray(task.affine_matrix, dtype=np.float64)
    warped = cv2.warpAffine(
        sar,
        affine,
        (task.width, task.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    grid_y, grid_x = np.mgrid[: task.height, : task.width]
    flow = np.empty((task.height, task.width, 2), dtype=np.float32)
    for channel in range(2):
        flow[..., channel] = (
            coefficients[channel, 0] * grid_x
            + coefficients[channel, 1] * grid_y
            + coefficients[channel, 2]
        ).astype(np.float32)

    optical_output, warped_output, flow_output = output_paths(task)
    if task.kind == "gdros":
        optical_to_write = cv2.cvtColor(optical, cv2.COLOR_GRAY2BGR)
    else:
        optical_to_write = optical
    write_image(optical_output, optical_to_write)
    write_image(warped_output, warped)
    write_flow(flow_output, flow)
    actual_optical_hash = pixel_sha256(optical)
    actual_warped_hash = pixel_sha256(warped)
    actual_flow_crc32 = f"{zlib.crc32(flow_output.read_bytes()) & 0xffffffff:08x}"
    return {
        "official_id": task.official_id,
        "raw_id": task.raw_id,
        "optical_pixel_sha256": actual_optical_hash,
        "official_optical_pixel_sha256": task.expected_optical_hash,
        "optical_pixel_exact": actual_optical_hash == task.expected_optical_hash,
        "warped_pixel_sha256": actual_warped_hash,
        "official_warped_pixel_sha256": task.expected_warped_hash,
        "warped_pixel_exact": actual_warped_hash == task.expected_warped_hash,
        "flow_crc32": actual_flow_crc32,
        "official_flow_crc32": task.expected_flow_crc32,
        "flow_bytes_exact": actual_flow_crc32 == task.expected_flow_crc32,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_pairs = discover_pairs(args.raw_root.expanduser().resolve())
    output_root = args.output_root.expanduser().resolve()
    split = manifest.get("split") or ("testing" if manifest["kind"] == "gdros" else "test")
    if manifest["kind"] == "gdros":
        target_root = output_root / split
    else:
        target_root = output_root / split / "warped" / manifest["variant"]
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"Output split is not empty: {target_root}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        hashed = list(executor.map(hash_raw_pair, raw_pairs.items()))
    hash_to_ids: dict[str, list[str]] = {}
    pair_hash_to_ids: dict[tuple[str, str], list[str]] = {}
    raw_hashes_by_id: dict[str, tuple[str, str]] = {}
    for optical_digest, sar_digest, identifier in hashed:
        hash_to_ids.setdefault(optical_digest, []).append(identifier)
        pair_hash_to_ids.setdefault((optical_digest, sar_digest), []).append(identifier)
        raw_hashes_by_id[identifier] = (optical_digest, sar_digest)

    tasks: list[Task] = []
    unmatched: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for record in manifest["records"]:
        digest = record["raw_optical_pixel_sha256"]
        sar_digest = record.get("raw_sar_pixel_sha256")
        declared_raw_id = record.get("raw_id")
        if declared_raw_id is not None:
            declared_raw_id = str(declared_raw_id)
            actual_hashes = raw_hashes_by_id.get(declared_raw_id)
            hashes_match = actual_hashes is not None and actual_hashes[0] == digest
            if sar_digest is not None:
                hashes_match = hashes_match and actual_hashes[1] == sar_digest
            candidates = [declared_raw_id] if hashes_match else []
        else:
            candidates = (
                pair_hash_to_ids.get((digest, sar_digest), [])
                if sar_digest is not None
                else hash_to_ids.get(digest, [])
            )
        if not candidates:
            unmatched.append(record["official_id"])
            continue
        if len(candidates) != 1:
            ambiguous[record["official_id"]] = candidates
            continue
        raw_id = candidates[0]
        optical_path, sar_path = raw_pairs[raw_id]
        tasks.append(
            Task(
                kind=manifest["kind"],
                split=split,
                variant=manifest.get("variant"),
                official_id=record["official_id"],
                raw_id=raw_id,
                optical_path=str(optical_path),
                sar_path=str(sar_path),
                output_root=str(output_root),
                height=int(record["height"]),
                width=int(record["width"]),
                coefficients=record["flow_coefficients"],
                affine_matrix=record["affine_matrix"],
                expected_optical_hash=record["raw_optical_pixel_sha256"],
                expected_warped_hash=record["official_warped_pixel_sha256"],
                expected_flow_crc32=record["flow_source_crc32"],
            )
        )
    if unmatched or ambiguous:
        raise RuntimeError(
            f"Cannot uniquely map official optical images: unmatched={unmatched[:20]}, "
            f"ambiguous={dict(list(ambiguous.items())[:20])}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(materialize, tasks))

    if manifest["kind"] == "crft":
        (output_root / f"{split}_list.txt").write_text(
            "".join(f"{task.official_id}\n" for task in tasks), encoding="utf-8"
        )

    audit = {
        "format": "official-os-affine-materialization-v1",
        "protocol_manifest": str(manifest_path),
        "kind": manifest["kind"],
        "split": split,
        "pair_count": len(results),
        "raw_root": str(args.raw_root.expanduser().resolve()),
        "output_root": str(output_root),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "exact_optical_pixel_matches": sum(item["optical_pixel_exact"] for item in results),
        "exact_warped_pixel_matches": sum(item["warped_pixel_exact"] for item in results),
        "exact_flow_byte_matches": sum(item["flow_bytes_exact"] for item in results),
        "records": results,
    }
    audit_path = output_root / f"materialization_audit_{split}.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
