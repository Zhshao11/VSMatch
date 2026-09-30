#!/usr/bin/env python3
"""Build compact reconstruction manifests from slim official test archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


CRFT_VARIANT = "512_0Trans0.1Scale45Rot"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_grayscale(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got {pixels.shape}")
    return np.ascontiguousarray(pixels)


def pixel_sha256(pixels: np.ndarray) -> str:
    return hashlib.sha256(pixels.tobytes(order="C")).hexdigest()


def archive_layout(kind: str) -> tuple[re.Pattern[str], re.Pattern[str], str]:
    if kind == "gdros":
        return (
            re.compile(r"^testing/opt/(\d{6})_opt\.png$"),
            re.compile(r"^testing/sar_warped/(\d{6})_sar_warped\.png$"),
            "testing/flow_affine_manifest.json",
        )
    root = f"OSdataset/test/{CRFT_VARIANT}"
    return (
        re.compile(rf"^{re.escape(root)}/image_pair/(\d{{6}})_opt\.tif$"),
        re.compile(rf"^{re.escape(root)}/image_pair/(\d{{6}})_sar_warped\.tif$"),
        f"{root}/flow_affine_manifest.json",
    )


def build_manifest(
    archive_path: Path, kind: str, matrix_path: Path | None
) -> dict[str, object]:
    optical_pattern, warped_pattern, flow_manifest_name = archive_layout(kind)
    exact_matrices: dict[str, list[list[float]]] | None = None
    if matrix_path is not None:
        matrix_records = json.loads(matrix_path.read_text(encoding="utf-8"))
        exact_matrices = {record["id"]: record["matrix"] for record in matrix_records}
    with zipfile.ZipFile(archive_path, "r") as archive:
        entries = archive.namelist()
        optical_entries = {
            match.group(1): name
            for name in entries
            if (match := optical_pattern.fullmatch(name))
        }
        warped_entries = {
            match.group(1): name
            for name in entries
            if (match := warped_pattern.fullmatch(name))
        }
        flow_manifest = json.loads(archive.read(flow_manifest_name))
        flow_records = {record["id"]: record for record in flow_manifest["records"]}

        identifiers = sorted(optical_entries)
        if identifiers != sorted(warped_entries) or identifiers != sorted(flow_records):
            raise RuntimeError(
                "Official archive components do not have identical IDs: "
                f"opt={len(optical_entries)}, warped={len(warped_entries)}, "
                f"flow={len(flow_records)}"
            )
        if exact_matrices is not None and identifiers != sorted(exact_matrices):
            raise RuntimeError(
                f"Exact matrix IDs do not match archive IDs: matrices={len(exact_matrices)}, "
                f"archive={len(identifiers)}"
            )

        records: list[dict[str, object]] = []
        for index, identifier in enumerate(identifiers, start=1):
            optical_name = optical_entries[identifier]
            warped_name = warped_entries[identifier]
            optical = decode_grayscale(archive.read(optical_name))
            warped = decode_grayscale(archive.read(warped_name))
            flow_record = flow_records[identifier]
            fitted_coefficients = np.asarray(flow_record["coefficients"], dtype=np.float64)
            if exact_matrices is None:
                affine_matrix = fitted_coefficients.copy()
                affine_matrix[0, 0] += 1.0
                affine_matrix[1, 1] += 1.0
            else:
                affine_matrix = np.asarray(exact_matrices[identifier], dtype=np.float64)
                exact_coefficients = affine_matrix.copy()
                exact_coefficients[0, 0] -= 1.0
                exact_coefficients[1, 1] -= 1.0
                coefficient_delta = float(
                    np.max(np.abs(exact_coefficients - fitted_coefficients))
                )
                if coefficient_delta > 1e-5:
                    raise RuntimeError(
                        f"Exact matrix disagrees with dense flow for {identifier}: "
                        f"max coefficient delta={coefficient_delta}"
                    )
            expected_shape = (int(flow_record["height"]), int(flow_record["width"]))
            if optical.shape != expected_shape or warped.shape != expected_shape:
                raise RuntimeError(
                    f"Shape mismatch for {identifier}: opt={optical.shape}, "
                    f"warped={warped.shape}, flow={expected_shape}"
                )
            records.append(
                {
                    "official_id": identifier,
                    "height": expected_shape[0],
                    "width": expected_shape[1],
                    "raw_optical_pixel_sha256": pixel_sha256(optical),
                    "official_warped_pixel_sha256": pixel_sha256(warped),
                    "optical_source_crc32": f"{archive.getinfo(optical_name).CRC:08x}",
                    "warped_source_crc32": f"{archive.getinfo(warped_name).CRC:08x}",
                    "flow_source_crc32": flow_record["source_crc32"],
                    "affine_matrix": affine_matrix.tolist(),
                    "flow_coefficients": exact_coefficients.tolist()
                    if exact_matrices is not None
                    else fitted_coefficients.tolist(),
                    "flow_fit_mean_epe": flow_record["fit_mean_epe"],
                    "flow_fit_max_epe": flow_record["fit_max_epe"],
                }
            )
            if index % 50 == 0 or index == len(identifiers):
                print(f"{kind}: hashed {index}/{len(identifiers)} pairs", flush=True)

    return {
        "format": "official-os-affine-protocol-v1",
        "kind": kind,
        "split": "testing" if kind == "gdros" else "test",
        "variant": CRFT_VARIANT if kind == "crft" else None,
        "source_archive": archive_path.name,
        "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": sha256_file(archive_path),
        "matrix_source": matrix_path.name if matrix_path is not None else "dense-flow-fit",
        "matrix_source_sha256": sha256_file(matrix_path) if matrix_path is not None else None,
        "pair_count": len(records),
        "pixel_hash_format": "sha256 of row-major 512x512 uint8 grayscale bytes",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("gdros", "crft"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--matrix-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    archive_path = args.archive.expanduser().resolve()
    matrix_path = args.matrix_json.expanduser().resolve() if args.matrix_json else None
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(archive_path, args.kind, matrix_path)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "pairs": manifest["pair_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
