#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.sparsegeo.e003a.data_protocol import sha256_file
from experiments.sparsegeo.e003a.spec_binding import validate_frozen_specs


LOCAL_PREFIXES = ("core", "experiments")


def _module_candidates(repo_root: Path, module: str, imported_names=()):
    relative = Path(*module.split("."))
    candidates = [repo_root / relative.with_suffix(".py"), repo_root / relative / "__init__.py"]
    candidates.extend(repo_root / relative / f"{name}.py" for name in imported_names)
    return [path for path in candidates if path.is_file()]


def _local_imports(repo_root: Path, path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(LOCAL_PREFIXES):
                    found.extend(_module_candidates(repo_root, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith(LOCAL_PREFIXES):
                found.extend(
                    _module_candidates(repo_root, node.module, [alias.name for alias in node.names])
                )
    return found


def source_manifest(repo_root: Path):
    repo_root = Path(repo_root).resolve()
    e003a = repo_root / "experiments/sparsegeo/e003a"
    pending = list(sorted(path for path in e003a.glob("*.py") if path.is_file()))
    paths = set()
    while pending:
        path = pending.pop()
        if path in paths:
            continue
        paths.add(path)
        pending.extend(candidate for candidate in _local_imports(repo_root, path) if candidate not in paths)
    files = {str(path.relative_to(repo_root)): sha256_file(path) for path in paths}
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    return files, digest.hexdigest()


def validate_runtime_bindings(
    repo_root,
    run_manifest,
    formal_config,
    split_manifest,
    cache_gate,
    cache_root=None,
    verify_all_cache_files=False,
    cache_pair_ids=None,
):
    validate_frozen_specs()
    repo_root = Path(repo_root).resolve()
    run_manifest = Path(run_manifest).resolve()
    formal_config = Path(formal_config).resolve()
    split_manifest = Path(split_manifest).resolve()
    cache_gate = Path(cache_gate).resolve()
    manifest = json.loads(run_manifest.read_text())
    config = json.loads(formal_config.read_text())
    cache = json.loads(cache_gate.read_text())
    cache_manifest = Path(cache["cache_manifest"]).resolve()
    _, source_sha = source_manifest(repo_root)
    checks = {
        "source_manifest": source_sha == manifest["implementation"]["source_manifest_sha256"],
        "config": sha256_file(formal_config) == manifest["implementation"]["config_sha256"],
        "split": sha256_file(split_manifest) == manifest["dataset"]["split_manifest_sha256"],
        "cache_manifest": cache["cache_manifest_sha256"] == manifest["cache"]["manifest_sha256"],
        "cache_aggregate": cache["cache_aggregate_sha256"] == manifest["cache"]["aggregate_sha256"],
        "cache_manifest_file": sha256_file(cache_manifest) == cache["cache_manifest_sha256"],
        "config_source": config["source_manifest_sha256"] == source_sha,
        "config_split": config["input"]["split_manifest_sha256"] == sha256_file(split_manifest),
        "config_cache_manifest": config["input"]["cache_manifest_sha256"] == cache["cache_manifest_sha256"],
        "config_cache_aggregate": config["input"]["cache_aggregate_sha256"] == cache["cache_aggregate_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal E-003A runtime binding failed: {checks}")
    verified_cache_files = 0
    if verify_all_cache_files or cache_pair_ids is not None:
        if cache_root is None:
            raise ValueError("cache_root is required when verifying cache files")
        cache_root = Path(cache_root).resolve()
        selected = None if verify_all_cache_files else set(cache_pair_ids)
        rows = [json.loads(line) for line in cache_manifest.read_text().splitlines() if line]
        if selected is not None:
            rows = [row for row in rows if row["pair_id"] in selected]
            if {row["pair_id"] for row in rows} != selected:
                raise RuntimeError("cache manifest does not contain all requested pair ids")
        for row in rows:
            for kind in ("candidate", "label"):
                path = Path(row[f"{kind}_path"]).resolve()
                if cache_root not in path.parents:
                    raise RuntimeError(f"cache manifest path escapes cache root: {path}")
                if sha256_file(path) != row[f"{kind}_sha256"]:
                    raise RuntimeError(f"runtime cache file SHA mismatch: {path}")
                verified_cache_files += 1
    return {
        "status": "PASS",
        "checks": checks,
        "source_manifest_sha256": source_sha,
        "config_sha256": sha256_file(formal_config),
        "split_manifest_sha256": sha256_file(split_manifest),
        "cache_aggregate_sha256": cache["cache_aggregate_sha256"],
        "cache_manifest_sha256": cache["cache_manifest_sha256"],
        "verified_cache_files": verified_cache_files,
    }


def runtime_identity(binding):
    keys = (
        "source_manifest_sha256",
        "config_sha256",
        "split_manifest_sha256",
        "cache_aggregate_sha256",
        "cache_manifest_sha256",
    )
    return {key: binding[key] for key in keys}


def run(args):
    specs = validate_frozen_specs()
    files, aggregate = source_manifest(args.repo_root)
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=args.repo_root, text=True
    )
    result = {
        "status": "PASS",
        "frozen_specs": specs,
        "source_files": files,
        "source_manifest_sha256": aggregate,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True
        ).strip(),
        "git_status_short": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "source_manifest_sha256": aggregate}, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
