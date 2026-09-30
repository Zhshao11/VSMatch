# E-008A Specification To Code/Test Mapping

| Frozen requirement | Code | Automated evidence |
|---|---|---|
| 4x4/8x8/16x16 region matching | `regions.py::match_multiscale_regions` | region shape/top-L/tie/no-GT tests |
| dynamic source proposals, no cap | `proposals.py::generate_dynamic_source_proposals` | minima, threshold, determinism, padding-is-not-semantics tests |
| matched-region dense-lite search | `search.py::matched_region_dense_lite_search` | exact region-union membership and max-live-score tests |
| local descriptor and hard candidates | `descriptors.py`, `generator.py` | gradient, toy loss, strict checkpoint and hard-coordinate tests |
| dump before raw-flow labels | `build_observations.py`, `build_labels.py` | content identity, join order and mismatch rejection tests |
| P0 then conditional P1 | `evaluate.py` | old-baseline SHA, metric parity and gate-boundary tests |
| resource trace and no hidden N2 | build/evaluate traces | source audit, tensor-shape and synchronized timing tests |

Formal order: automatic tests -> smoke -> two targeted reviews -> observation/label watchdog -> two-seed train -> P0 -> P1 only if P0 PASS -> result card.
