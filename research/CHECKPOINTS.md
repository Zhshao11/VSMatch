# Checkpoint Manifest

The verified OS inference implementation is named **M4-streamlined**.
Its source checkpoint remains `os_e075a_m4_80k`. The optional extracted compact
state, full SHA256, size and legacy-constructor requirement are recorded in
[`models/M4-streamlined.json`](models/M4-streamlined.json) and
[`docs/M4_STREAMLINED.md`](../docs/M4_STREAMLINED.md). Both weight files stay
outside Git; the public Python name does not change their state_dict keys.

Model weights are intentionally excluded from Git. The machine-readable paths,
roles, and SHA256 values are in `CHECKPOINTS.json`.

| ID | Role | SHA256 |
| --- | --- | --- |
| `os_e074_b_180k` | E075A balanced B branch | `f40d80f...ea33218` |
| `os_e074_c_180k` | E075A high-precision C branch | `c3c67e6d...ecc85b9` |
| `os_e075a_m4_80k` | Current protected OS method | `55d6e6cd...6d81b` |
| `whu_e081a_m4_120k` | Current protected WHU method | `4122c27a...ca745a` |

The manifest records expected paths relative to `$SPARSEGEO_ROOT`. It is not a
download manifest because these experiment checkpoints are not published as
repository assets. Obtain them through the authorized project storage, place
them at the recorded paths, and run:

```bash
python scripts/preflight.py --scope all --verify-hashes --strict
```

Do not substitute a checkpoint with the same filename but a different hash.
