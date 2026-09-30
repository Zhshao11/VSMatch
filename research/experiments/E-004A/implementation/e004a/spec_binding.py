from pathlib import Path

from experiments.sparsegeo.e003a.data_protocol import sha256_file


FROZEN_SHA256 = {
    "CURRENT_STATE.md": "4c8b6d1edb7791bd4f2c8de258c1888aee1fad98962419bee17e06f6c6220997",
    "METHOD_SPEC.md": "e1c6b14664214018c622f17de54baabafc7b0f3674cc03e49397ad89ab9847d7",
    "METHOD_DELTA.md": "d7410557e30378707a93731a15f0d4aa43ebfe39f4d340d3982ce9e3f746dd5a",
    "INNOVATION_CONTRACT.yaml": "c01dcfaf170778687f00fa3b6ec7d74a35ccce6f0252f847dc8ca0297298359a",
    "E-004A.md": "6110948722ff841257a938e59ac8e7682666491944b57c75eadfefb42b9ccdc9",
}
CHECKPOINT_SHA256 = "80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683"
SPLIT_MANIFEST_SHA256 = "e8c990888123896af425923970215eb1a9ec8aca89eae0421998d2bdc13a99d3"
UNIFORM_REGION_OFFSETS = ((2, 2), (6, 6))
BINDING_ID = "BIND-001"


def validate_frozen_specs(spec_root: Path):
    spec_root = Path(spec_root).resolve()
    observed = {}
    for name, expected in FROZEN_SHA256.items():
        path = spec_root / name
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen E-004A identity mismatch for {path}: {actual}")
        observed[name] = actual
    return observed
