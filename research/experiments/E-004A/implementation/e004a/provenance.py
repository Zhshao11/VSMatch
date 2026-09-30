import hashlib
import json
from pathlib import Path

from experiments.sparsegeo.e003a.data_protocol import sha256_file
from experiments.sparsegeo.e004a.spec_binding import (
    BINDING_ID,
    CHECKPOINT_SHA256,
    FROZEN_SHA256,
    SPLIT_MANIFEST_SHA256,
    UNIFORM_REGION_OFFSETS,
)


def source_manifest(module_root: Path):
    module_root = Path(module_root).resolve()
    rows = []
    for path in sorted(module_root.glob("*.py")):
        rows.append({"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"files": rows, "sha256": hashlib.sha256(encoded).hexdigest()}


def runtime_provenance(spec_root: Path, checkpoint: Path, split_manifest: Path, module_root: Path):
    checkpoint_sha = sha256_file(checkpoint)
    split_sha = sha256_file(split_manifest)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    if split_sha != SPLIT_MANIFEST_SHA256:
        raise RuntimeError(f"split manifest SHA mismatch: {split_sha}")
    from experiments.sparsegeo.e004a.spec_binding import validate_frozen_specs
    return {
        "experiment": "E-004A",
        "method": "M001 v0.7",
        "stage": "P0",
        "frozen_specs": validate_frozen_specs(spec_root),
        "expected_specs": FROZEN_SHA256,
        "checkpoint_sha256": checkpoint_sha,
        "split_manifest_sha256": split_sha,
        "binding": {"id": BINDING_ID, "uniform_region_offsets": UNIFORM_REGION_OFFSETS},
        "source": source_manifest(module_root),
        "uses_os424": False,
        "uses_datum_as_label": False,
    }
