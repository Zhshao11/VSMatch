#!/usr/bin/env python3
"""Regenerate affine optical-SAR benchmarks from aligned OSdataset pairs.

The script writes either the GDROS or CRFT public-repository layout together
with a JSONL manifest containing every sampled affine parameter. Generated data
is suitable for protocol diagnostics, but it is not byte-identical to the
authors' released processed archives unless their original random seeds and
generation implementation are known.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


FLO_MAGIC = 202021.25
PAIR_RE = re.compile(r"^(opt|sar)(.+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class Protocol:
    name: str
    translation_min: int
    translation_max: int
    scale_min: float
    scale_max: float
    scale_step: float | None
    rotation_min: float
    rotation_max: float
    rotation_step: float | None
    crop_size: int | None
    output_format: str
    variant: str | None = None


PROTOCOLS = {
    "gdros-paper": Protocol(
        name="gdros-paper",
        translation_min=-30,
        translation_max=30,
        scale_min=0.8,
        scale_max=1.2,
        scale_step=0.05,
        rotation_min=-20.0,
        rotation_max=20.0,
        rotation_step=1.0,
        crop_size=400,
        output_format="gdros",
    ),
    "crft-paper": Protocol(
        name="crft-paper",
        translation_min=-30,
        translation_max=30,
        scale_min=0.9,
        scale_max=1.1,
        scale_step=None,
        rotation_min=-45.0,
        rotation_max=45.0,
        rotation_step=None,
        crop_size=None,
        output_format="crft",
        variant="512_30Trans0.1Scale45Rot",
    ),
    "crft-release": Protocol(
        name="crft-release",
        translation_min=-15,
        translation_max=15,
        scale_min=0.9,
        scale_max=1.1,
        scale_step=None,
        rotation_min=-30.0,
        rotation_max=30.0,
        rotation_step=None,
        crop_size=None,
        output_format="crft",
        variant="512_15Trans0.1Scale30Rot",
    ),
}


@dataclass(frozen=True)
class PairTask:
    pair_id: str
    opt_path: str
    sar_path: str
    output_root: str
    output_split: str
    output_format: str
    variant: str | None
    crop_size: int | None
    tx: float
    ty: float
    scale: float
    angle_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-split", help="Defaults to the repository-specific split name")
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, help="Generate only the first N numerically sorted pairs")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing generated split",
    )
    return parser.parse_args()


def numeric_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def discover_pairs(split_root: Path) -> list[tuple[str, Path, Path]]:
    modalities: dict[str, dict[str, Path]] = {"opt": {}, "sar": {}}
    for path in split_root.iterdir():
        match = PAIR_RE.match(path.name)
        if match and path.is_file():
            modalities[match.group(1).lower()][match.group(2)] = path

    opt_ids = set(modalities["opt"])
    sar_ids = set(modalities["sar"])
    if opt_ids != sar_ids:
        raise RuntimeError(
            "Raw split is not fully paired: "
            f"opt_only={sorted(opt_ids - sar_ids)[:10]}, "
            f"sar_only={sorted(sar_ids - opt_ids)[:10]}"
        )
    if not opt_ids:
        raise RuntimeError(f"No opt*.png/sar*.png pairs found in {split_root}")

    return [
        (pair_id, modalities["opt"][pair_id], modalities["sar"][pair_id])
        for pair_id in sorted(opt_ids, key=numeric_sort_key)
    ]


def discrete_values(low: float, high: float, step: float) -> np.ndarray:
    count = int(round((high - low) / step)) + 1
    values = low + np.arange(count, dtype=np.float64) * step
    if not math.isclose(float(values[-1]), high, abs_tol=1e-8):
        raise ValueError(f"Range [{low}, {high}] is not divisible by step {step}")
    return values


def sample_parameters(protocol: Protocol, count: int, seed: int) -> list[tuple[float, float, float, float]]:
    rng = np.random.default_rng(seed)
    translations = rng.integers(
        protocol.translation_min,
        protocol.translation_max + 1,
        size=(count, 2),
    )

    if protocol.scale_step is None:
        scales = rng.uniform(protocol.scale_min, protocol.scale_max, size=count)
    else:
        scales = rng.choice(
            discrete_values(protocol.scale_min, protocol.scale_max, protocol.scale_step),
            size=count,
        )

    if protocol.rotation_step is None:
        angles = rng.uniform(protocol.rotation_min, protocol.rotation_max, size=count)
    else:
        angles = rng.choice(
            discrete_values(protocol.rotation_min, protocol.rotation_max, protocol.rotation_step),
            size=count,
        )

    return [
        (float(tx), float(ty), float(scale), float(angle))
        for (tx, ty), scale, angle in zip(translations, scales, angles)
    ]


def output_paths(task: PairTask) -> tuple[Path, Path, Path, Path]:
    root = Path(task.output_root)
    if task.output_format == "gdros":
        split_root = root / task.output_split
        return (
            split_root / "opt" / f"{task.pair_id}.png",
            split_root / "sar_warped" / f"{task.pair_id}.png",
            split_root / "truth_flow" / f"{task.pair_id}.flo",
            split_root / "datum" / f"{task.pair_id}.json",
        )

    if task.output_format == "crft":
        if not task.variant:
            raise ValueError("CRFT output requires a variant directory")
        warped_root = Path(task.output_root) / task.output_split / "warped" / task.variant
        return (
            warped_root / "image_pair" / f"{task.pair_id}_opt.tif",
            warped_root / "image_pair" / f"{task.pair_id}_sar_warped.tif",
            warped_root / "truth_flow" / f"{task.pair_id}.flo",
            warped_root / "datum" / f"{task.pair_id}.json",
        )

    raise ValueError(f"Unsupported output format: {task.output_format}")


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        if not cv2.imwrite(str(temporary), image):
            raise RuntimeError(f"OpenCV failed to write {temporary}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_flo(path: Path, flow: np.ndarray) -> None:
    if flow.dtype != np.float32 or flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected HxWx2 float32 flow, got {flow.shape} {flow.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".flo", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(struct.pack("f", FLO_MAGIC))
            stream.write(struct.pack("i", flow.shape[1]))
            stream.write(struct.pack("i", flow.shape[0]))
            flow.tofile(stream)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def center_crop(array: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = array.shape[:2]
    if size > height or size > width:
        raise ValueError(f"Cannot crop {size} from {height}x{width}")
    top = (height - size) // 2
    left = (width - size) // 2
    return array[top : top + size, left : left + size].copy(), (left, top)


def process_pair(task: PairTask) -> dict[str, object]:
    optical = cv2.imread(task.opt_path, cv2.IMREAD_UNCHANGED)
    sar = cv2.imread(task.sar_path, cv2.IMREAD_UNCHANGED)
    if optical is None or sar is None:
        raise RuntimeError(f"Failed to read raw pair {task.opt_path}, {task.sar_path}")
    if optical.shape[:2] != sar.shape[:2]:
        raise RuntimeError(
            f"Shape mismatch for {task.pair_id}: optical={optical.shape}, sar={sar.shape}"
        )

    height, width = optical.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    affine = cv2.getRotationMatrix2D(center, task.angle_deg, task.scale).astype(np.float64)
    affine[0, 2] += task.tx
    affine[1, 2] += task.ty

    warped_sar = cv2.warpAffine(
        sar,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    warped_x = affine[0, 0] * grid_x + affine[0, 1] * grid_y + affine[0, 2]
    warped_y = affine[1, 0] * grid_x + affine[1, 1] * grid_y + affine[1, 2]
    flow = np.stack((warped_x - grid_x, warped_y - grid_y), axis=-1).astype(np.float32)

    crop_origin = (0, 0)
    if task.crop_size is not None:
        optical, crop_origin = center_crop(optical, task.crop_size)
        warped_sar, warped_crop_origin = center_crop(warped_sar, task.crop_size)
        flow, flow_crop_origin = center_crop(flow, task.crop_size)
        if crop_origin != warped_crop_origin or crop_origin != flow_crop_origin:
            raise AssertionError("Inconsistent center-crop origin")

    opt_output, sar_output, flow_output, datum_output = output_paths(task)
    write_image(opt_output, optical)
    write_image(sar_output, warped_sar)
    write_flo(flow_output, flow)

    datum = {
        "pair_id": task.pair_id,
        "source_optical": str(Path(task.opt_path).resolve()),
        "source_sar": str(Path(task.sar_path).resolve()),
        "affine_source_to_warped": affine.tolist(),
        "translation_px": [task.tx, task.ty],
        "scale": task.scale,
        "rotation_deg": task.angle_deg,
        "source_shape_hw": [height, width],
        "output_shape_hw": list(flow.shape[:2]),
        "crop_origin_xy": list(crop_origin),
        "flow_convention": "target_xy_minus_source_xy",
    }
    datum_output.parent.mkdir(parents=True, exist_ok=True)
    datum_output.write_text(json.dumps(datum, sort_keys=True) + "\n", encoding="utf-8")
    return datum


def default_output_split(protocol: Protocol, raw_split: str) -> str:
    if protocol.output_format == "gdros":
        return {"train": "training", "val": "validation", "test": "testing"}[raw_split]
    return raw_split


def ensure_output_target(tasks: Sequence[PairTask], overwrite: bool) -> None:
    if not tasks:
        raise ValueError("No generation tasks")
    representative_paths = output_paths(tasks[0])
    generated_root = representative_paths[0].parent.parent
    existing = any(path.exists() for task in tasks[: min(20, len(tasks))] for path in output_paths(task)[:3])
    if existing and not overwrite:
        raise FileExistsError(
            f"Generated files already exist under {generated_root}; pass --overwrite to replace them"
        )


def run_tasks(tasks: Sequence[PairTask], workers: int) -> list[dict[str, object]]:
    if workers == 1:
        return [process_pair(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(process_pair, tasks, chunksize=max(1, len(tasks) // (workers * 8))))


def write_manifests(
    output_root: Path,
    protocol: Protocol,
    raw_split: str,
    output_split: str,
    seed: int,
    records: Iterable[dict[str, object]],
) -> None:
    metadata_root = output_root / "protocol_metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    stem = f"{output_split}_{protocol.name}_seed{seed}"
    records = list(records)
    manifest_path = metadata_root / f"{stem}.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "protocol": asdict(protocol),
        "raw_split": raw_split,
        "output_split": output_split,
        "seed": seed,
        "pair_count": len(records),
        "manifest": str(manifest_path.resolve()),
        "generator": str(Path(__file__).resolve()),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "status": "regenerated_protocol_not_author_archive",
    }
    (metadata_root / f"{stem}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_crft_list(output_root: Path, output_split: str, pair_ids: Sequence[str]) -> None:
    list_path = output_root / f"{output_split}_list.txt"
    list_path.write_text("".join(f"{pair_id}\n" for pair_id in pair_ids), encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocol = PROTOCOLS[args.protocol]
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_root = raw_root / args.raw_split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Raw split not found: {split_root}")

    pairs = discover_pairs(split_root)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        pairs = pairs[: args.limit]

    output_split = args.output_split or default_output_split(protocol, args.raw_split)
    sampled = sample_parameters(protocol, len(pairs), args.seed)
    width = max(6, max(len(pair_id) for pair_id, _, _ in pairs))
    tasks = [
        PairTask(
            pair_id=pair_id.zfill(width),
            opt_path=str(opt_path),
            sar_path=str(sar_path),
            output_root=str(output_root),
            output_split=output_split,
            output_format=protocol.output_format,
            variant=protocol.variant,
            crop_size=protocol.crop_size,
            tx=parameters[0],
            ty=parameters[1],
            scale=parameters[2],
            angle_deg=parameters[3],
        )
        for (pair_id, opt_path, sar_path), parameters in zip(pairs, sampled)
    ]

    ensure_output_target(tasks, args.overwrite)
    print(
        f"Generating {len(tasks)} {protocol.name} pairs from {split_root} "
        f"into {output_root} with seed {args.seed} and {args.workers} workers",
        flush=True,
    )
    records = run_tasks(tasks, args.workers)
    write_manifests(
        output_root,
        protocol,
        args.raw_split,
        output_split,
        args.seed,
        records,
    )
    if protocol.output_format == "crft":
        write_crft_list(output_root, output_split, [task.pair_id for task in tasks])
    print(f"Completed {len(records)} pairs", flush=True)


if __name__ == "__main__":
    main()
