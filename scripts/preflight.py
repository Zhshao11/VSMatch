#!/usr/bin/env python3
"""Check external assets needed by the SparseGeo research workflows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path("/mnt/sda/zhushihao/VS_Match")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_path(name: str, path: Path, kind: str = "path") -> dict:
    exists = path.is_dir() if kind == "directory" else path.is_file()
    return {"name": name, "path": str(path), "exists": exists, "kind": kind}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("os", "whu", "all"), default="all")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    workspace = args.root or Path(os.environ.get("SPARSEGEO_ROOT", DEFAULT_WORKSPACE))
    manifest = json.loads((REPO_ROOT / "research/CHECKPOINTS.json").read_text())
    scopes = {"os", "whu"} if args.scope == "all" else {args.scope}

    packages = {
        name: importlib.util.find_spec(module) is not None
        for name, module in {
            "numpy": "numpy",
            "opencv-python": "cv2",
            "Pillow": "PIL",
            "PyYAML": "yaml",
            "scipy": "scipy",
            "torch": "torch",
            "torchvision": "torchvision",
        }.items()
    }

    paths = [
        check_path(
            "upstream_gdros",
            workspace / "external_repos/E068A_GDROS_official",
            "directory",
        ),
        check_path(
            "migrated_modules",
            workspace / "image_matching_part2_migrated/.codex_tmp",
            "directory",
        ),
    ]
    if "os" in scopes:
        paths.extend([
            check_path(
                "os_training",
                workspace / "datasets/extracted/gdros_official_e003a_training",
                "directory",
            ),
            check_path(
                "os_testing",
                workspace / "datasets/extracted/gdros_official_e003a_testing",
                "directory",
            ),
        ])
    if "whu" in scopes:
        whu_train = Path(os.environ.get(
            "SPARSEGEO_WHU_TRAIN",
            "/mnt/sda/zhushihao/data/vis_sar/WHU_OPT_SAR/extracted_e077a/training/whu_opt_sar512",
        ))
        whu_test = Path(os.environ.get(
            "SPARSEGEO_WHU_TEST",
            "/mnt/sda/zhushihao/data/vis_sar/WHU_OPT_SAR/extracted_e077a/testing/testing",
        ))
        paths.extend([
            check_path("whu_training", whu_train, "directory"),
            check_path("whu_testing", whu_test, "directory"),
        ])

    checkpoints = []
    for item in manifest["checkpoints"]:
        if item["scope"] not in scopes:
            continue
        path = workspace / item["path"]
        result = {
            "id": item["id"],
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": item["sha256"],
        }
        if path.is_file() and args.verify_hashes:
            result["actual_sha256"] = sha256(path)
            result["hash_matches"] = result["actual_sha256"] == item["sha256"]
        checkpoints.append(result)

    failures = [name for name, present in packages.items() if not present]
    failures.extend(item["name"] for item in paths if not item["exists"])
    failures.extend(item["id"] for item in checkpoints if not item["exists"])
    failures.extend(
        item["id"] for item in checkpoints if item.get("hash_matches") is False
    )
    report = {
        "status": "PASS" if not failures else "INCOMPLETE",
        "scope": sorted(scopes),
        "workspace": str(workspace),
        "packages": packages,
        "paths": paths,
        "checkpoints": checkpoints,
        "failures": failures,
    }

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status: {report['status']}")
        print(f"workspace: {workspace}")
        for name, present in packages.items():
            print(f"package {name}: {'ok' if present else 'missing'}")
        for item in paths:
            print(f"{item['name']}: {'ok' if item['exists'] else 'missing'} ({item['path']})")
        for item in checkpoints:
            state = "ok" if item["exists"] else "missing"
            if item.get("hash_matches") is False:
                state = "hash-mismatch"
            print(f"checkpoint {item['id']}: {state} ({item['path']})")

    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    sys.exit(main())
