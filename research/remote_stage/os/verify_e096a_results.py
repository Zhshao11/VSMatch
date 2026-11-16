#!/usr/bin/env python3
"""Independently verify E096A compact statistics; optionally rehash predictions."""
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
    manifest = json.loads((root / "PREDICTION_MANIFEST.json").read_text())
    parity = json.loads((root / "M4_ARCHIVE_COMPARISONS.json").read_text())
    timings, metrics = rows(root, "TIMINGS"), rows(root, "METRICS")
    assert card["status"] == "COMPLETE" and card["pairs"] == 424
    assert card["training"] is False and card["iterations"] == 32
    assert card["checkpoint_sha256"] == {
        "gdros": "80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683",
        "m4_clean": "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b"}
    assert len(timings) == len(manifest) == 848 and len(metrics) == len(parity) == 424
    assert [int(r["pair_index"]) for r in metrics] == list(range(424))
    assert [r["pair_index"] for r in parity] == list(range(424))
    assert len({(r["variant"], r["pair_index"]) for r in manifest}) == 848
    for variant, calls in (("gdros", 1), ("m4_clean", 5)):
        selected = [r for r in timings if r["variant"] == variant]
        assert sorted(int(r["pair_index"]) for r in selected) == list(range(424))
        for record in selected:
            index = int(record["pair_index"])
            first = "gdros" if index % 2 == 0 else "m4_clean"
            assert int(record["order"]) == (0 if variant == first else 1)
            assert int(record["lsr_calls"]) == calls
        values = [float(r["elapsed_ms"]) for r in selected]
        assert min(values) > 0
        expected = card["timing"][variant]
        close(statistics.fmean(values), expected["mean_ms"])
        close(statistics.median(values), expected["median_ms"])
        close(statistics.pstdev(values), expected["std_ms"])
        close(statistics.quantiles(values, n=100, method="inclusive")[94], expected["p95_ms"])
        close(sum(values)/1000, expected["total_seconds"])
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
    maximum = max(r["max_abs"] for r in parity)
    assert maximum <= 1e-4
    close(maximum, card["m4_vs_e095a"]["max_abs"])
    assert sum(r["bitwise_equal"] for r in parity) == card["m4_vs_e095a"]["bitwise_equal_pairs"]
    close(card["timing"]["m4_clean"]["mean_ms"]/card["timing"]["gdros"]["mean_ms"],
          card["m4_over_gdros_latency_ratio"])
    assert card["source_files_unchanged_during_run"]
    assert card["source_hashes_start"] == card["source_hashes_end"]
    print(json.dumps({"status": "PASS", "pairs": 424, "timings": 848,
                      "dense_predictions_rehashed": 848 if args.verify_predictions else 0,
                      "m4_vs_e095a": card["m4_vs_e095a"],
                      "m4_over_gdros_latency_ratio": card["m4_over_gdros_latency_ratio"]}, indent=2))


if __name__ == "__main__":
    main()
