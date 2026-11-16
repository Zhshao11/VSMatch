#!/usr/bin/env python3
"""Independently audit compact E092C-R artifacts, without models/data/GPU.

Example: python verify_e092cr_results.py --run-dir /path/to/run --seeds 901
Pass --aggregate only after all three seeds have completed. This checks the
exported CSV/card/manifest chain; it does not re-read remote prediction arrays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

THRESHOLDS = (1, 2, 3, 4, 5, 10, 16)
COUNT = 1625


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    if expected is None:
        assert actual is None, (actual, expected)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def audit_protocol(run):
    """Check frozen radiometry/source identities and per-pair affine metadata."""
    export = run.parent
    baseline = None
    reports = []
    for seed in (901, 902, 903):
        meta = export / "protocol_metadata" / f"seed{seed}"
        manifest_path, params_path = meta / "manifest.json", meta / "params.jsonl"
        manifest = read_json(manifest_path)
        preflight = read_json(run / f"seed{seed}/preflight/RESULT_CARD.json")
        expected = preflight["generated_protocol_audit"]
        assert preflight["status"] == expected["status"] == "PASS"
        assert digest(manifest_path) == expected["manifest_sha256"]
        assert digest(params_path) == expected["params_sha256"]
        assert manifest["seed"] == seed and manifest["pair_count"] == COUNT
        assert manifest["transform"] == {
            "translation_px": [-15, 15, 1], "scale": [.9, 1.1, .05],
            "rotation_deg": [-10, 10, 1], "crop": None, "border": "constant 0"}
        assert manifest["sar_conversion"]["lo_db"] == -28.307898330688477
        assert manifest["sar_conversion"]["hi_db"] == 10.545851440429704
        params = [json.loads(line) for line in params_path.read_text().splitlines()]
        identities = [(p["pair_id"], p["source_stem"]) for p in params]
        assert len(identities) == len(set(identities)) == COUNT
        assert [p["pair_id"] for p in params] == [f"{i:06d}" for i in range(COUNT)]
        if baseline is None:
            baseline = identities
        assert identities == baseline
        for p in params:
            assert p["shape_hw"] == [512, 512]
            tx, ty = p["translation_px"]
            angle, scale = p["rotation_deg"], p["scale"]
            assert -15 <= tx <= 15 and int(tx) == tx
            assert -15 <= ty <= 15 and int(ty) == ty
            assert -10 <= angle <= 10 and int(angle) == angle
            assert any(math.isclose(scale, v, abs_tol=1e-12) for v in (.9, .95, 1, 1.05, 1.1))
            radians = math.radians(angle)
            expected_matrix = [[scale * math.cos(radians), -scale * math.sin(radians), tx],
                               [scale * math.sin(radians), scale * math.cos(radians), ty]]
            for actual_row, expected_row in zip(p["affine_src_to_target"], expected_matrix):
                for actual, value in zip(actual_row, expected_row):
                    close(actual, value)
            assert p["sar_conversion"]["lo_db"] == manifest["sar_conversion"]["lo_db"]
            assert p["sar_conversion"]["hi_db"] == manifest["sar_conversion"]["hi_db"]
        reports.append({"seed": seed, "manifest_sha256": digest(manifest_path),
                        "params_sha256": digest(params_path)})
    audit = read_json(run / "audit/GENERATOR_AUDIT_CARD.json")
    same_version = audit["generator_environment_verification"]
    assert audit["status"] == "PASS" and same_version["warp_max_abs_difference_uint8"] == 0
    assert same_version["sampled_pairs"] == 12
    assert digest(export / "code/verify_ubcv2_testset.py") == same_version["verifier_sha256"]
    assert digest(run / "audit/GENERATOR_ENVIRONMENT_VERIFY.log") == same_version["output_log_sha256"]
    return {"status": "PASS", "seeds": reports,
            "same_source_order_and_frozen_sar_mapping": True,
            "all_affine_metadata_checked": True,
            "same_opencv_warp0_evidence_hashes_checked": True}


def audit_seed(run, seed):
    root = run / f"seed{seed}"
    card = read_json(root / "RESULT_CARD.json")
    assert card["status"] == "COMPLETE" and card["seed"] == seed
    assert card["prediction_manifests_verified_before_gt_load"] is True
    assert card["whole_image_no_mask"] is True
    assert card["public_formula_parity"]["status"] == "PASS"
    path = root / "join/PER_PAIR_METRICS.csv"
    assert digest(path) == card["per_pair_csv_sha256"]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["pair_id"] for row in rows]
    assert len(rows) == len(set(ids)) == COUNT
    assert [int(row["pair_index"]) for row in rows] == list(range(COUNT))
    preflight = read_json(root / "preflight/RESULT_CARD.json")
    assert preflight["status"] == "PASS" and preflight["seed"] == seed
    assert preflight["hash_mismatches"] == {}
    assert preflight["official_loader_input_parity"]["source_target_tensor_max_abs_difference"] == 0
    assert preflight["official_loader_input_parity"]["pairs"] == COUNT
    manifests = {}
    for model in ("gdros", "m4"):
        prediction_card = read_json(root / f"predict_{model}/RESULT_CARD.json")
        assert prediction_card["status"] == "COMPLETE"
        assert prediction_card["ground_truth_loaded"] is False
        assert prediction_card["load_report"]["missing_keys"] == []
        assert prediction_card["load_report"]["unexpected_keys"] == []
        for label, info in prediction_card["manifests"].items():
            manifest = root / "predictions" / label / "PREDICTIONS.jsonl"
            assert digest(manifest) == info["sha256"]
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            assert len(records) == info["records"] == COUNT
            assert [record["pair_id"] for record in records] == ids
            assert [record["pair_index"] for record in records] == list(range(COUNT))
            assert all(record["uses_truth_flow"] is False and record["post_lsr"] is True
                       and record["shape"] == [1, 2, 512, 512]
                       and record["dtype"] == "float32" for record in records)
            manifests[label] = info["sha256"]
    for label, kinds in card["metrics"].items():
        assert label in manifests
        for kind, expected in kinds.items():
            values = [float(row[f"{label}__{kind}"]) for row in rows]
            assert all(math.isfinite(value) and value >= 0 for value in values)
            assert expected["pairs"] == COUNT
            for field, value in {"mean": statistics.fmean(values),
                                 "variance": statistics.pvariance(values),
                                 "std": statistics.pstdev(values),
                                 "p95": quantile(values, .95),
                                 "p99": quantile(values, .99)}.items():
                close(value, expected[field])
            for threshold in THRESHOLDS:
                subset = [value for value in values if value < threshold]
                close(len(subset) / COUNT * 100, expected[f"cmr@{threshold}"])
                assert COUNT - len(subset) == expected[f"failures_ge_{threshold}"]
                if threshold <= 5:
                    result = expected[f"success_subset@{threshold}"]
                    assert result["count"] == len(subset)
                    close(statistics.fmean(subset) if subset else None, result["mean"])
                    close(statistics.pvariance(subset) if subset else None,
                          result["variance_official_rmse_field"])
                    close(statistics.pstdev(subset) if subset else None, result["std"])
    return card, {"seed": seed, "status": "PASS", "pairs": COUNT,
                  "csv_sha256": digest(path), "manifest_sha256": manifests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=(901, 902, 903), default=[901, 902, 903])
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--protocol", action="store_true", help="also check all three groups' exported metadata")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert len(set(args.seeds)) == len(args.seeds), "duplicate seeds"
    cards, results = {}, []
    for seed in args.seeds:
        card, result = audit_seed(args.run_dir, seed)
        cards[seed] = card
        results.append(result)
    if args.aggregate:
        assert set(cards) == {901, 902, 903}
        aggregate = read_json(args.run_dir / "RESULT_CARD.json")
        assert aggregate["status"] == "COMPLETE" and aggregate["seeds"] == [901, 902, 903]
        for label, kinds in aggregate["metrics"].items():
            for kind, fields in kinds.items():
                for field, expected in fields.items():
                    values = [cards[seed]["metrics"][label][kind][field] for seed in (901, 902, 903)]
                    assert expected["per_seed"] == values
                    close(statistics.fmean(values), expected["cross_seed_mean"])
                    close(statistics.pstdev(values), expected["cross_seed_std_ddof0"])
                    close(statistics.stdev(values), expected["cross_seed_std_ddof1"])
    report = {"status": "PASS", "seeds": results, "aggregate_checked": args.aggregate,
              "scope": "Independent stdlib recomputation from compact CSV/card/manifest artifacts; "
                       "full remote prediction arrays are not re-read by this audit.",
              "verifier_sha256": digest(Path(__file__))}
    if args.protocol:
        report["protocol"] = audit_protocol(args.run_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
