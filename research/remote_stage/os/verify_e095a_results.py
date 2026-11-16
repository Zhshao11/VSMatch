#!/usr/bin/env python3
"""Independently recompute the compact E095A cards with the standard library."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def close(actual, expected):
    if abs(actual - expected) > 1e-9:
        raise AssertionError((actual, expected))


def read_rows(root, name):
    path = root / f"{name}.csv"
    if path.exists():
        with path.open() as handle:
            return list(csv.DictReader(handle))
    return json.loads((root / f"{name}_ROWS.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-predictions", action="store_true",
                        help="Rehash every dense prediction at its manifest path on the experiment host")
    args = parser.parse_args()
    root = args.directory
    card = json.loads((root / "RESULT_CARD.json").read_text())
    comparisons = json.loads((root / "COMPARISONS.json").read_text())
    timings = read_rows(root, "TIMINGS")
    metrics = read_rows(root, "METRICS")
    assert card["status"] == "COMPLETE" and card["pairs"] == 424
    assert card["training"] is False and card["iterations"] == 32
    assert card["checkpoint_sha256"] == "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b"
    assert len(timings) == 848 and len(metrics) == len(comparisons) == 424
    assert [int(row["pair_index"]) for row in metrics] == list(range(424))
    assert [row["pair_index"] for row in comparisons] == list(range(424))
    for variant, calls in (("reference5", 5), ("clean", 5)):
        selected = [row for row in timings if row["variant"] == variant]
        assert sorted(int(row["pair_index"]) for row in selected) == list(range(424))
        assert all(int(row["lsr_calls"]) == calls for row in selected)
        for row in selected:
            index = int(row["pair_index"])
            expected_first = "reference5" if index % 2 == 0 else "clean"
            assert int(row["order"]) == (0 if variant == expected_first else 1)
        values = [float(row["elapsed_ms"]) for row in selected]
        close(statistics.fmean(values), card["timing"][variant]["mean_ms"])
        close(statistics.median(values), card["timing"][variant]["median_ms"])
        close(statistics.pstdev(values), card["timing"][variant]["std_ms"])
        close(sum(values) / 1000, card["timing"][variant]["total_seconds"])
        for metric in ("public_component", "vector_epe"):
            errors = [float(row[f"{variant}__{metric}"]) for row in metrics]
            expected = card["metrics"][variant][metric]
            close(statistics.fmean(errors), expected["mean"])
            for threshold in (1, 2, 3, 4, 5, 10, 16):
                close(sum(value < threshold for value in errors) / 424 * 100,
                      expected[f"cmr@{threshold}"])
                assert sum(value >= threshold for value in errors) == expected[f"failures_ge_{threshold}"]
    for name, tolerance in (("raw", 1e-5), ("post", 1e-4)):
        maximum = max(row[name]["max_abs"] for row in comparisons)
        assert maximum <= tolerance
        close(maximum, card["equivalence"][f"{name}_max_abs"])
        assert sum(row[name]["bitwise_equal"] for row in comparisons) == card["equivalence"][f"{name}_bitwise_equal_pairs"]
    close(card["timing"]["reference5"]["mean_ms"] / card["timing"]["clean"]["mean_ms"], card["speedup_mean"])
    assert card["source_files_unchanged_during_run"]
    assert not card["leaf_parameter_coverage_uncalled"]
    audit = card["pruning_audit"]
    assert audit["clean"]["parameters"] < audit["original"]["parameters"]
    assert audit["unique_dino_backbones"] == len(set(r["group"] for r in audit["dino_groups"]))
    assert audit["unique_descriptor_frontends"] == len(set(r["adapter_group"] for r in audit["dino_groups"]))
    manifest = json.loads((root / "PREDICTION_MANIFEST.json").read_text())
    assert len(manifest) == 848
    by_key = {(r["pair_index"], r["variant"]): r for r in manifest}
    assert len(by_key) == 848
    if args.verify_predictions:
        for record in manifest:
            digest = hashlib.sha256()
            with Path(record["path"]).open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            assert digest.hexdigest() == record["sha256"]
    if card["equivalence"]["post_bitwise_equal_pairs"] == 424:
        assert all(by_key[i, "reference5"]["sha256"] == by_key[i, "clean"]["sha256"] for i in range(424))
    print(json.dumps({"status": "PASS", "pairs": 424, "timings": 848,
                      "equivalence": card["equivalence"], "speedup_mean": card["speedup_mean"]}, indent=2))


if __name__ == "__main__":
    main()
