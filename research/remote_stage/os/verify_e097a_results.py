#!/usr/bin/env python3
"""Independent compact-statistics and optional dense-file audit of E097A."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def close(left, right):
    if abs(left-right) > 1e-9:
        raise AssertionError((left, right))


def rows(root, stem):
    path = root / f"{stem}.csv"
    if path.exists():
        with path.open() as handle:
            return list(csv.DictReader(handle))
    return json.loads((root / f"{stem}_ROWS.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-predictions", action="store_true")
    args = parser.parse_args()
    root = args.directory
    card = json.loads((root / "RESULT_CARD.json").read_text())
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    manifest = json.loads((root / "PREDICTION_MANIFEST.json").read_text())
    comparisons = json.loads((root / "COMPARISONS.json").read_text())
    timings, metrics = rows(root, "TIMINGS"), rows(root, "METRICS")
    policies = {"reference5": [1, 2, 3, 4, 5], "final1": [5], "last45": [4, 5]}
    assert card["status"] == "COMPLETE" and card["pairs"] == 424
    assert card["training"] is False and card["iterations"] == 32
    assert card["policies"] == protocol["policies"] == policies
    assert card["checkpoint_sha256"] == "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b"
    assert len(timings) == len(manifest) == 1272 and len(metrics) == len(comparisons) == 424
    assert [int(r["pair_index"]) for r in metrics] == list(range(424))
    assert [r["pair_index"] for r in comparisons] == list(range(424))
    assert len({(r["variant"], r["pair_index"]) for r in manifest}) == 1272
    for variant, slots in policies.items():
        selected = [r for r in timings if r["variant"] == variant]
        assert sorted(int(r["pair_index"]) for r in selected) == list(range(424))
        for record in selected:
            index = int(record["pair_index"])
            order = protocol["orders"][index % 6]
            assert int(record["order"]) == order.index(variant)
            assert int(record["lsr_calls"]) == len(slots)
        values = [float(r["elapsed_ms"]) for r in selected]
        assert min(values) > 0
        expected = card["timing"][variant]
        close(statistics.fmean(values), expected["mean_ms"])
        close(statistics.median(values), expected["median_ms"])
        close(statistics.pstdev(values), expected["std_ms"])
        close(statistics.quantiles(values, n=100, method="inclusive")[94], expected["p95_ms"])
        close(sum(values)/1000, expected["total_seconds"])
        close(card["timing"]["reference5"]["mean_ms"]/expected["mean_ms"], card["speedup_vs_reference5"][variant])
        for name in ("public_component", "vector_epe"):
            errors = [float(r[f"{variant}__{name}"]) for r in metrics]
            expected = card["metrics"][variant][name]
            close(statistics.fmean(errors), expected["mean"])
            close(statistics.pstdev(errors), expected["std"])
            for threshold in (1, 2, 3, 4, 5, 10, 16):
                close(sum(e < threshold for e in errors)/424*100, expected[f"cmr@{threshold}"])
                assert sum(e >= threshold for e in errors) == expected[f"failures_ge_{threshold}"]
    for record in manifest:
        assert record["pair_id"] == metrics[record["pair_index"]]["pair_id"]
        if args.verify_predictions:
            digest = hashlib.sha256()
            with Path(record["path"]).open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            assert digest.hexdigest() == record["sha256"]
    maximum = max(r["reference_vs_archive"]["max_abs"] for r in comparisons)
    assert maximum <= 1e-4
    close(maximum, card["reference_vs_archive"]["max_abs"])
    assert sum(r["reference_vs_archive"]["bitwise_equal"] for r in comparisons) == card["reference_vs_archive"]["bitwise_equal_pairs"]
    for variant in ("final1", "last45"):
        for name in ("raw", "post"):
            close(max(r[variant][name]["max_abs"] for r in comparisons), card["output_changes"][variant][name]["max_abs"])
            assert sum(r[variant][name]["bitwise_equal"] for r in comparisons) == card["output_changes"][variant][name]["bitwise_equal_pairs"]
    assert card["source_files_unchanged_during_run"]
    assert card["source_hashes_start"] == card["source_hashes_end"]
    print(json.dumps({"status": "PASS", "pairs": 424, "timings": 1272,
                      "dense_predictions_rehashed": 1272 if args.verify_predictions else 0,
                      "policies": policies, "reference_vs_archive": card["reference_vs_archive"],
                      "speedup_vs_reference5": card["speedup_vs_reference5"]}, indent=2))


if __name__ == "__main__":
    main()
