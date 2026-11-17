#!/usr/bin/env python3
"""Independently recompute streamlined C statistics and validate run evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def close(actual, expected, tolerance=1e-8):
    assert math.isfinite(actual) and abs(actual - expected) <= tolerance, (actual, expected)


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-predictions", action="store_true")
    args = parser.parse_args()
    root = args.directory

    def load(name):
        return json.loads((root / name).read_text())

    card, protocol = load("RESULT_CARD.json"), load("PROTOCOL.json")
    audit, coverage = load("MODEL_AUDIT.json"), load("COVERAGE.json")
    metrics, timings = load("METRICS_ROWS.json"), load("TIMINGS_ROWS.json")
    manifest, comparisons = load("PREDICTION_MANIFEST.json"), load("C_CACHE_COMPARISONS.json")
    assert card["status"] == "COMPLETE" and card["fresh_inference"] and not card["training"]
    assert not card["full_m4_comparison"] and card["pairs"] == 424 and card["iterations"] == 32
    assert card["checkpoint_sha256"] == "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b"
    assert protocol["prediction_before_gt_join"] and protocol["dataset_provenance"]["all_match_archive"]
    assert card["load_report"]["missing_keys"] == card["load_report"]["unexpected_keys"] == []
    for rows in (metrics, timings, manifest, comparisons):
        assert len(rows) == 424 and [r["pair_index"] for r in rows] == list(range(424))
        assert len({r["pair_id"] for r in rows}) == 424
    for i in range(424):
        assert len({r[i]["pair_id"] for r in (metrics, timings, manifest, comparisons)}) == 1
        assert timings[i]["lsr_calls"] == 1 and timings[i]["elapsed_ms"] > 0
        assert comparisons[i]["max_abs"] <= 1e-4
        assert not manifest[i]["uses_truth_flow"]
    for name in ("public_component", "vector_epe"):
        values = [r[name] for r in metrics]
        expected = card["metrics"][name]
        assert all(math.isfinite(x) and x >= 0 for x in values)
        close(statistics.fmean(values), expected["mean"])
        close(statistics.pstdev(values), expected["std"])
        close(statistics.pvariance(values), expected["variance"])
        close(quantile(values, .95), expected["p95"])
        close(quantile(values, .99), expected["p99"])
        for threshold in (1, 2, 3, 4, 5, 10, 16):
            close(sum(v < threshold for v in values) / 424 * 100, expected[f"cmr@{threshold}"])
            assert sum(v >= threshold for v in values) == expected[f"failures_ge_{threshold}"]
    elapsed = [r["elapsed_ms"] for r in timings]
    timing = card["timing"]
    close(statistics.fmean(elapsed), timing["mean_ms"])
    close(statistics.median(elapsed), timing["median_ms"])
    close(statistics.pstdev(elapsed), timing["std_ms"])
    close(quantile(elapsed, .95), timing["p95_ms"])
    close(sum(elapsed) / 1000, timing["total_seconds"])
    close(1000 / statistics.fmean(elapsed), timing["fps_from_mean"])
    assert audit["stats"] == card["model_parameters"]
    assert set(audit["top_level_children"]) == {"dino", "descriptor_adapters", "matcher"}
    for group, members, extra in ((audit["stats"], audit["groups"], 0),
                                  (audit["groups"]["matcher"], audit["matcher_groups"], audit["matcher_direct_parameters"])):
        assert sum(v["parameters"] for v in members.values()) + extra == group["parameters"]
        assert group["parameter_bytes"] == 4 * group["parameters"]
    assert coverage["instrumentation_removed"] and not coverage["uncalled_parameter_leaves"]
    assert all(coverage["leaf_calls"].values())
    assert coverage["instrumented_vs_plain"]["max_abs"] == 0
    assert coverage["counts"] == card["actual_counts_per_pair"] == {
        "gru": 32, "motion": 32, "flow_head": 32, "mask_head": 1, "context": 1,
        "frontend_v1": 1, "frontend_v2": 1, "correlation_pyramids": 2,
        "correlation_sampling": 256, "dino_forward_features": 2, "terminal_lsr": 1}
    direct = load("DIRECT_C_PARITY.json")
    assert [r["pair_index"] for r in direct] == [0, 211, 423]
    assert all(r["raw"]["max_abs"] <= 1e-5 and r["post"]["max_abs"] <= 1e-4 for r in direct)
    assert card["c_cache_parity"]["bitwise_equal_pairs"] == sum(r["bitwise_equal"] for r in comparisons)
    close(card["c_cache_parity"]["max_abs"], max(r["max_abs"] for r in comparisons))
    assert card["source_hashes_start"] == card["source_hashes_end"] == protocol["sources"]
    assert card["memory"]["reference_deleted_before_timing"]
    snapshots = root / "source_snapshot"
    if snapshots.is_dir():
        for path, digest in card["source_hashes_start"].items():
            assert sha256(snapshots / Path(path).name) == digest
    verified = 0
    if args.verify_predictions:
        for item in manifest:
            assert sha256(Path(item["path"])) == item["sha256"]
            verified += 1
        state = card["compact_state"]
        assert Path(state["path"]).stat().st_size == state["bytes"]
        assert sha256(Path(state["path"])) == state["sha256"]
    print(json.dumps({"status": "PASS", "pairs": 424, "standalone_timing_records": len(timings),
                      "dense_predictions_rehashed": verified, "parameters": audit["stats"]["parameters"],
                      "c_cache_bitwise_equal_pairs": card["c_cache_parity"]["bitwise_equal_pairs"],
                      "instrumentation_output_change": 0}, indent=2))


if __name__ == "__main__":
    main()
