#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

EXPECTED_ARCHIVE_SIZE = 1_973_635_350
EXPECTED_ARCHIVE_SHA256 = (
    "e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218"
)
EXPECTED_DATA_MANIFEST_SHA256 = (
    "c129ae7147a7d9ad1c1a7ead35bb6da2d8c85ad8da73eef4723cf277e5ea7e96"
)
EXPECTED_PAIR_COUNT = 424
EXPECTED_WIDTH = 512
EXPECTED_HEIGHT = 512
FLOW_MAGIC = 202021.25


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the official GDROS E-000 archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--published-root", type=Path)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive-only", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_relative_files():
    files = []
    for index in range(EXPECTED_PAIR_COUNT):
        pair_id = f"{index:06d}"
        files.extend(
            [
                f"testing/datum/{pair_id}.npy",
                f"testing/opt/{pair_id}_opt.png",
                f"testing/sar_warped/{pair_id}_sar_warped.png",
                f"testing/truth_flow/{pair_id}.flo",
                f"testing/truth_flow/scene/{pair_id}.flo",
            ]
        )
    return sorted(files)


def expected_archive_entries():
    directories = [
        "testing/",
        "testing/datum/",
        "testing/opt/",
        "testing/sar_warped/",
        "testing/truth_flow/",
        "testing/truth_flow/scene/",
    ]
    return sorted(directories + expected_relative_files())


def validate_archive(archive, perform_crc_test):
    archive = archive.resolve()
    size = archive.stat().st_size
    digest = sha256_file(archive)
    if size != EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError(f"archive size mismatch: {size}")
    if digest != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"archive SHA256 mismatch: {digest}")

    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate member names")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise RuntimeError(f"unsafe archive member: {name}")
            if pure.parts[0] != "testing":
                raise RuntimeError(f"unexpected archive root: {name}")
        expected = expected_archive_entries()
        if sorted(names) != expected:
            missing = sorted(set(expected) - set(names))[:20]
            extra = sorted(set(names) - set(expected))[:20]
            raise RuntimeError(f"archive member mismatch: missing={missing}, extra={extra}")
        bad_member = zipped.testzip() if perform_crc_test else None
        if bad_member is not None:
            raise RuntimeError(f"archive CRC failure: {bad_member}")

    return {
        "path": str(archive),
        "size_bytes": size,
        "sha256": digest,
        "entry_count": len(names),
        "regular_file_count": sum(not info.is_dir() for info in infos),
        "directory_entry_count": sum(info.is_dir() for info in infos),
        "member_set_matches_expected": True,
        "crc_test": "passed" if perform_crc_test else "deferred_until_full_validation",
    }


def read_flow_header(path):
    with path.open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12:
        raise RuntimeError(f"truncated flow header: {path}")
    magic, width, height = struct.unpack("<fii", header)
    if magic != FLOW_MAGIC:
        raise RuntimeError(f"invalid flow magic in {path}: {magic}")
    expected_size = 12 + width * height * 2 * 4
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"invalid flow byte count: {path}")
    return width, height


def read_npy_header(path):
    import numpy as np

    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    stream
                )
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    stream
                )
            else:
                raise RuntimeError(f"unsupported NPY format version in {path}: {version}")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid NPY header in {path}: {error}") from error
    return version, shape, fortran_order, dtype


def evaluation_content_sha256(path):
    if path.suffix.lower() == ".flo":
        return sha256_file(path)
    if path.suffix.lower() != ".png":
        raise RuntimeError(f"unsupported evaluation-content file: {path}")

    from PIL import Image

    digest = hashlib.sha256()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest.update(struct.pack("<II", *rgb.size))
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def file_manifest(dataset_root, relative_files):
    aggregate = hashlib.sha256()
    hashes = {}
    category_aggregates = {}
    category_hashers = {}
    category_counts = Counter()
    for relative in relative_files:
        path = dataset_root / relative
        digest = sha256_file(path)
        hashes[relative] = digest
        encoded = relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n"
        aggregate.update(encoded)
        parts = PurePosixPath(relative).parts
        category = "/".join(parts[:-1])
        category_hashers.setdefault(category, hashlib.sha256()).update(encoded)
        category_counts[category] += 1
    for category, hasher in sorted(category_hashers.items()):
        category_aggregates[category] = hasher.hexdigest()
    return (
        hashes,
        aggregate.hexdigest(),
        category_aggregates,
        dict(sorted(category_counts.items())),
    )


def require_expected_data_manifest(manifest_sha):
    if manifest_sha != EXPECTED_DATA_MANIFEST_SHA256:
        raise RuntimeError(
            "extracted data manifest does not match the verified official archive: "
            f"{manifest_sha}"
        )


def validate_extracted(dataset_root, published_root, comparison_root):
    from PIL import Image

    dataset_root = dataset_root.resolve()
    relative_files = expected_relative_files()
    expected_set = set(relative_files)
    expected_directories = {
        entry.rstrip("/")
        for entry in expected_archive_entries()
        if entry.endswith("/")
    }
    entries = list(dataset_root.rglob("*"))
    symlinks = [
        path.relative_to(dataset_root).as_posix()
        for path in entries
        if path.is_symlink()
    ]
    if symlinks:
        raise RuntimeError(f"extracted dataset contains symlinks: {symlinks[:20]}")
    actual_directories = {
        path.relative_to(dataset_root).as_posix()
        for path in entries
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)[:20]
        extra = sorted(actual_directories - expected_directories)[:20]
        raise RuntimeError(
            f"extracted directory mismatch: missing={missing}, extra={extra}"
        )
    special_entries = [
        path.relative_to(dataset_root).as_posix()
        for path in entries
        if not path.is_file() and not path.is_dir()
    ]
    if special_entries:
        raise RuntimeError(
            f"extracted dataset contains special entries: {special_entries[:20]}"
        )
    actual_set = {
        path.relative_to(dataset_root).as_posix()
        for path in entries
        if path.is_file()
    }
    allowed_metadata = {"E000_PROVENANCE.json"}
    data_set = actual_set - allowed_metadata
    if data_set != expected_set or actual_set - expected_set - allowed_metadata:
        missing = sorted(expected_set - data_set)[:20]
        extra = sorted(actual_set - expected_set - allowed_metadata)[:20]
        raise RuntimeError(f"extracted file mismatch: missing={missing}, extra={extra}")

    image_shapes = Counter()
    image_modes = Counter()
    for relative in relative_files:
        if not (relative.startswith("testing/opt/") or relative.startswith("testing/sar_warped/")):
            continue
        with Image.open(dataset_root / relative) as image:
            image_shapes[str(image.size)] += 1
            image_modes[image.mode] += 1
            if image.size != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
                raise RuntimeError(f"unexpected image size in {relative}: {image.size}")

    flow_shapes = Counter()
    for relative in relative_files:
        if not relative.endswith(".flo"):
            continue
        shape = read_flow_header(dataset_root / relative)
        flow_shapes[str(shape)] += 1
        if shape != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            raise RuntimeError(f"unexpected flow size in {relative}: {shape}")

    datum_shapes = Counter()
    datum_dtypes = Counter()
    datum_versions = Counter()
    datum_fortran_orders = Counter()
    for relative in relative_files:
        if not relative.endswith(".npy"):
            continue
        version, shape, fortran_order, dtype = read_npy_header(dataset_root / relative)
        datum_versions[str(version)] += 1
        datum_shapes[str(tuple(shape))] += 1
        datum_dtypes[str(dtype)] += 1
        datum_fortran_orders[str(fortran_order)] += 1

    hashes, manifest_sha, category_sha, category_file_counts = file_manifest(
        dataset_root, relative_files
    )
    require_expected_data_manifest(manifest_sha)
    comparison = None
    if comparison_root is not None:
        comparison_root = comparison_root.resolve()
        evaluated_files = [
            relative
            for relative in relative_files
            if relative.startswith("testing/opt/")
            or relative.startswith("testing/sar_warped/")
            or (
                relative.startswith("testing/truth_flow/")
                and "/scene/" not in relative
            )
        ]
        missing = []
        mismatches = []
        category_matches = Counter()
        for relative in evaluated_files:
            candidate = comparison_root / relative
            if not candidate.is_file():
                missing.append(relative)
            elif evaluation_content_sha256(candidate) != evaluation_content_sha256(
                dataset_root / relative
            ):
                mismatches.append(relative)
            else:
                category_matches["/".join(PurePosixPath(relative).parts[:-1])] += 1
        if missing or mismatches:
            raise RuntimeError(
                f"comparison mismatch: missing={missing[:20]}, content={mismatches[:20]}"
            )
        comparison = {
            "root": str(comparison_root),
            "evaluated_file_count": len(evaluated_files),
            "missing_count": 0,
            "content_mismatch_count": 0,
            "category_content_match_counts": dict(sorted(category_matches.items())),
            "comparison_semantics": (
                "decoded RGB size and pixels for PNG; exact file bytes for FLO"
            ),
            "all_evaluation_content_identical": True,
            "all_evaluation_files_identical": True,
        }

    return {
        "dataset_root": str(published_root.resolve()),
        "validated_source_root": str(dataset_root),
        "validated_source_root_note": (
            "The extraction script validates staging before publication and validates the "
            "published dataset root again after the atomic rename."
        ),
        "provenance_file_present_during_validation": (
            "E000_PROVENANCE.json" in actual_set
        ),
        "pair_count": EXPECTED_PAIR_COUNT,
        "data_file_count": len(relative_files),
        "category_counts": {
            "datum": EXPECTED_PAIR_COUNT,
            "opt": EXPECTED_PAIR_COUNT,
            "sar_warped": EXPECTED_PAIR_COUNT,
            "truth_flow": EXPECTED_PAIR_COUNT,
            "truth_flow_scene": EXPECTED_PAIR_COUNT,
        },
        "image_shapes": dict(image_shapes),
        "image_modes": dict(image_modes),
        "flow_shapes": dict(flow_shapes),
        "datum_shapes": dict(datum_shapes),
        "datum_dtypes": dict(datum_dtypes),
        "datum_format_versions": dict(datum_versions),
        "datum_fortran_orders": dict(datum_fortran_orders),
        "manifest_sha256": manifest_sha,
        "expected_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "category_manifest_sha256": category_sha,
        "category_manifest_file_counts": category_file_counts,
        "comparison_to_equivalent_materialization": comparison,
    }


def main():
    args = parse_args()
    archive = validate_archive(args.archive, perform_crc_test=not args.archive_only)
    if args.archive_only:
        print(json.dumps({"archive": archive}, indent=2, sort_keys=True))
        return
    if args.dataset_root is None or args.published_root is None or args.output is None:
        raise ValueError("full validation requires dataset-root, published-root and output")

    extracted = validate_extracted(
        args.dataset_root, args.published_root, args.comparison_root
    )
    provenance = {
        "schema_version": 1,
        "experiment": "E-000",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "archive": archive,
        "published_dataset_root": str(args.published_root.resolve()),
        "extracted": extracted,
        "evaluation_protocol": {
            "split": "testing",
            "pair_count": EXPECTED_PAIR_COUNT,
            "image_size": [EXPECTED_HEIGHT, EXPECTED_WIDTH],
            "evaluation_mask": "none; all pixels including black borders",
            "reference_pool": "row-major [10000:210000]",
        },
        "validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version.replace("\n", " "),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
