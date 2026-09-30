import hashlib
from pathlib import Path


EXPECTED = {
    "METHOD_SPEC.md": "6d259b05900e4fea404841d35584492da4cfa8ec92391bcb8fe8bced6f51b3a2",
    "METHOD_DELTA.md": "07b245e9ef803e76ced26b676cee84bc6e0f57b6c0a2fe5a3361bffcba7fe3e8",
    "INNOVATION_CONTRACT.yaml": "f764876bed16ddf57c11aad7a25039d3358fdbd9163aa0e8200a6159995963e4",
    "E-003A.md": "6d9dd55c2ecc03c8c8e0379272a8cc1c030d8270464453bb266001b154093791",
    "CURRENT_STATE.md": "2aa64e7e9ade15e8f19d31b865eb275764977f392a508abb36337078e9143e57",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_specs():
    root = Path(__file__).resolve().parent / "frozen_specs"
    actual = {}
    for name, expected in EXPECTED.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"frozen E-003A specification is missing: {path}")
        actual[name] = _sha256(path)
        if actual[name] != expected:
            raise RuntimeError(
                f"frozen E-003A specification changed: {name} expected {expected}, got {actual[name]}"
            )
    return actual
