#!/usr/bin/env python3
"""Build test-only archives while representing affine flow fields compactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import zipfile
from pathlib import Path

import numpy as np


GDROS_FLOW = re.compile(r"^testing/truth_flow/(\d{6})\.flo$")
CRFT_VARIANT = "512_0Trans0.1Scale45Rot"
CRFT_FLOW = re.compile(
    rf"^OSdataset/test/{re.escape(CRFT_VARIANT)}/truth_flow/(\d{{6}})\.flo$"
)


def read_flow(payload: bytes) -> np.ndarray:
    if len(payload) < 12 or payload[:4] != b"PIEH":
        raise ValueError("Invalid Middlebury .flo payload")
    width, height = struct.unpack("<ii", payload[4:12])
    expected = 12 + width * height * 2 * 4
    if len(payload) != expected:
        raise ValueError(f"Invalid .flo size: expected {expected}, got {len(payload)}")
    return np.frombuffer(payload, dtype="<f4", offset=12).reshape(height, width, 2)


def fit_affine_flow(flow: np.ndarray) -> tuple[np.ndarray, float, float]:
    height, width = flow.shape[:2]
    step = max(1, min(height, width) // 32)
    sample_y, sample_x = np.mgrid[0:height:step, 0:width:step]
    design = np.stack(
        (sample_x.ravel(), sample_y.ravel(), np.ones(sample_x.size)), axis=1
    ).astype(np.float64)
    sampled_flow = flow[::step, ::step].reshape(-1, 2).astype(np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, sampled_flow, rcond=None)
    coefficients = coefficients.T

    grid_y, grid_x = np.mgrid[:height, :width]
    reconstructed = np.empty_like(flow, dtype=np.float32)
    for channel in range(2):
        reconstructed[..., channel] = (
            coefficients[channel, 0] * grid_x
            + coefficients[channel, 1] * grid_y
            + coefficients[channel, 2]
        ).astype(np.float32)
    epe = np.linalg.norm(reconstructed - flow.astype(np.float64), axis=2)
    return coefficients, float(epe.mean()), float(epe.max())


def copy_entry(
    source: zipfile.ZipFile, destination: zipfile.ZipFile, name: str
) -> None:
    destination.writestr(name, source.read(name), compress_type=zipfile.ZIP_STORED)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive(
    source_path: Path,
    output_path: Path,
    kind: str,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    flow_pattern = GDROS_FLOW if kind == "gdros" else CRFT_FLOW

    if kind == "gdros":
        image_prefixes = ("testing/opt/", "testing/sar_warped/")
        extra_files: tuple[str, ...] = ()
        manifest_name = "testing/flow_affine_manifest.json"
    else:
        root = f"OSdataset/test/{CRFT_VARIANT}/"
        image_prefixes = (root + "image_pair/",)
        extra_files = ("OSdataset/test_list.txt",)
        manifest_name = root + "flow_affine_manifest.json"

    records: list[dict[str, object]] = []
    copied_images = 0
    image_bytes = 0
    worst_mean_epe = 0.0
    worst_max_epe = 0.0

    with zipfile.ZipFile(source_path, "r") as source, zipfile.ZipFile(
        temporary_path, "w", allowZip64=True
    ) as destination:
        names = source.namelist()
        for name in names:
            if name.endswith("/"):
                continue
            if name.startswith(image_prefixes):
                copy_entry(source, destination, name)
                copied_images += 1
                image_bytes += source.getinfo(name).file_size

        for name in extra_files:
            copy_entry(source, destination, name)

        flow_names = sorted(name for name in names if flow_pattern.fullmatch(name))
        for index, name in enumerate(flow_names, start=1):
            payload = source.read(name)
            flow = read_flow(payload)
            coefficients, mean_epe, max_epe = fit_affine_flow(flow)
            worst_mean_epe = max(worst_mean_epe, mean_epe)
            worst_max_epe = max(worst_max_epe, max_epe)
            records.append(
                {
                    "id": Path(name).stem,
                    "path": name,
                    "height": int(flow.shape[0]),
                    "width": int(flow.shape[1]),
                    "coefficients": coefficients.tolist(),
                    "source_crc32": f"{source.getinfo(name).CRC:08x}",
                    "fit_mean_epe": mean_epe,
                    "fit_max_epe": max_epe,
                }
            )
            if index % 50 == 0 or index == len(flow_names):
                print(f"{kind}: fitted {index}/{len(flow_names)} flows", flush=True)

        manifest = {
            "format": "affine-flow-manifest-v1",
            "source_archive": source_path.name,
            "kind": kind,
            "flow_count": len(records),
            "records": records,
        }
        destination.writestr(
            manifest_name,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )

    temporary_path.replace(output_path)
    digest = file_sha256(output_path)
    summary = {
        "kind": kind,
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        "sha256": digest,
        "image_count": copied_images,
        "image_bytes": image_bytes,
        "flow_count": len(records),
        "worst_fit_mean_epe": worst_mean_epe,
        "worst_fit_max_epe": worst_max_epe,
        "manifest": manifest_name,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdros-zip", type=Path, required=True)
    parser.add_argument("--crft-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = [
        build_archive(
            args.gdros_zip.expanduser().resolve(),
            args.output_dir.expanduser().resolve() / "gdros_official_test_slim.zip",
            "gdros",
        ),
        build_archive(
            args.crft_zip.expanduser().resolve(),
            args.output_dir.expanduser().resolve() / "crft_official_test_slim.zip",
            "crft",
        ),
    ]
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
