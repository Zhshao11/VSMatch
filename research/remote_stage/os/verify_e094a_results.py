#!/usr/bin/env python3
"""Independently recompute the compact E094A cards with the standard library."""
import argparse
import csv
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
    for variant, calls in (("original129", 129), ("final5", 5)):
        selected = [row for row in timings if row["variant"] == variant]
        assert sorted(int(row["pair_index"]) for row in selected) == list(range(424))
        assert all(int(row["lsr_calls"]) == calls for row in selected)
        for row in selected:
            index = int(row["pair_index"])
            expected_first = "original129" if index % 2 == 0 else "final5"
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
    close(card["timing"]["original129"]["mean_ms"] / card["timing"]["final5"]["mean_ms"], card["speedup_mean"])
    assert card["source_files_unchanged_during_run"]
    print(json.dumps({"status": "PASS", "pairs": 424, "timings": 848,
                      "equivalence": card["equivalence"], "speedup_mean": card["speedup_mean"]}, indent=2))


if __name__ == "__main__":
    main()
