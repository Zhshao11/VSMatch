#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


EXPECTED_ARCHIVE_SHA256 = (
    "e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683"
)
EXPECTED_REFERENCE_SHA256 = (
    "fcafa4d9b077b2b7e88e43e7efe2bfd638f6a4af2b1007961f8f056efb6cc6db"
)
EXPECTED_PAIR_COUNT = 424
THRESHOLDS = (1, 2, 3, 4, 5)
SUMMARY_TOLERANCE = 1e-9
PERCENT_TOLERANCE = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(description="Check formal GDROS E-000 metrics")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--max-pair-delta", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def reject_nonfinite(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(path.read_bytes(), parse_constant=reject_nonfinite)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_float(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric") from error
    require(math.isfinite(result), f"{label} is non-finite")
    return result


def validate_metric_summary(label, payload, values):
    require(len(values) == EXPECTED_PAIR_COUNT, f"{label} value count mismatch")
    require(payload.get("num_pairs") == EXPECTED_PAIR_COUNT, f"{label} pair count mismatch")

    aepe = finite_float(payload.get("aepe"), f"{label} AEPE")
    pair_std = finite_float(payload.get("paper_rmse"), f"{label} pair standard deviation")
    pair_variance = finite_float(
        payload.get("legacy_variance"), f"{label} pair variance"
    )
    recomputed_aepe = statistics.fmean(values)
    recomputed_std = statistics.pstdev(values)
    recomputed_variance = statistics.pvariance(values)
    require(
        abs(aepe - recomputed_aepe) <= SUMMARY_TOLERANCE,
        f"{label} AEPE summary is inconsistent",
    )
    require(
        abs(pair_std - recomputed_std) <= SUMMARY_TOLERANCE,
        f"{label} pair standard deviation is inconsistent",
    )
    require(
        abs(pair_variance - recomputed_variance) <= SUMMARY_TOLERANCE,
        f"{label} pair variance is inconsistent",
    )

    thresholds = payload.get("thresholds")
    require(isinstance(thresholds, dict), f"{label} thresholds are missing")
    checked_thresholds = {}
    for threshold in THRESHOLDS:
        key = str(threshold)
        record = thresholds.get(key)
        require(isinstance(record, dict), f"{label} threshold {threshold} is missing")
        matched_values = [value for value in values if value < threshold]
        matched_count = len(matched_values)
        matched_pairs = record.get("matched_pairs")
        require(
            type(matched_pairs) is int and matched_pairs == matched_count,
            f"{label} matched-pair count is inconsistent at threshold {threshold}",
        )
        require(
            record.get("threshold") == threshold,
            f"{label} threshold label is inconsistent at threshold {threshold}",
        )
        cmr = finite_float(
            record.get("cmr_percent"), f"{label} CMR at threshold {threshold}"
        )
        recomputed_cmr = 100.0 * matched_count / EXPECTED_PAIR_COUNT
        require(
            abs(cmr - recomputed_cmr) <= PERCENT_TOLERANCE,
            f"{label} CMR percentage is inconsistent at threshold {threshold}",
        )

        if matched_values:
            matched_aepe = finite_float(
                record.get("aepe"), f"{label} matched AEPE at threshold {threshold}"
            )
            matched_std = finite_float(
                record.get("paper_rmse"),
                f"{label} matched standard deviation at threshold {threshold}",
            )
            matched_variance = finite_float(
                record.get("legacy_variance"),
                f"{label} matched variance at threshold {threshold}",
            )
            require(
                abs(matched_aepe - statistics.fmean(matched_values))
                <= SUMMARY_TOLERANCE,
                f"{label} matched AEPE is inconsistent at threshold {threshold}",
            )
            require(
                abs(matched_std - statistics.pstdev(matched_values))
                <= SUMMARY_TOLERANCE,
                f"{label} matched standard deviation is inconsistent at threshold {threshold}",
            )
            require(
                abs(matched_variance - statistics.pvariance(matched_values))
                <= SUMMARY_TOLERANCE,
                f"{label} matched variance is inconsistent at threshold {threshold}",
            )
        else:
            require(record.get("aepe") is None, f"{label} empty matched AEPE must be null")
            require(
                record.get("paper_rmse") is None,
                f"{label} empty matched standard deviation must be null",
            )
            require(
                record.get("legacy_variance") is None,
                f"{label} empty matched variance must be null",
            )
        checked_thresholds[key] = {
            "matched_pairs": matched_count,
            "cmr_percent": cmr,
        }

    return {
        "aepe": aepe,
        "pair_std": pair_std,
        "pair_variance": pair_variance,
        "thresholds": checked_thresholds,
    }


def main():
    args = parse_args()
    metrics_path = args.metrics.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    provenance_path = args.provenance.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(
        output
        not in {
            metrics_path,
            reference_path,
            provenance_path,
            checkpoint_path,
        },
        "output must not overwrite an input file",
    )
    require(
        math.isfinite(args.max_pair_delta)
        and 0 <= args.max_pair_delta <= 1e-4,
        "max-pair-delta must be finite and no greater than 1e-4",
    )
    require((repo_root / "experiments/sparsegeo/e001").is_dir(), "invalid repo root")
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "core"))
    from experiments.sparsegeo.e001.evaluate_sparse_lsr import (  # noqa: E402
        load_dataset_provenance,
    )

    metrics = load_json(metrics_path)
    reference = load_json(reference_path)
    provenance_record = load_dataset_provenance(
        provenance_path, dataset_root, formal_run=True
    )
    provenance = provenance_record["content"]
    checkpoint_sha = sha256_file(checkpoint_path)
    reference_sha = sha256_file(reference_path)

    require(checkpoint_sha == EXPECTED_CHECKPOINT_SHA256, "checkpoint SHA256 mismatch")
    require(reference_sha == EXPECTED_REFERENCE_SHA256, "reference metrics SHA256 mismatch")
    require(metrics.get("num_pairs") == EXPECTED_PAIR_COUNT, "formal pair count mismatch")
    require(metrics.get("dataset_split") == "testing", "formal split mismatch")
    require(metrics.get("seed") == 1234, "formal seed mismatch")
    require(
        Path(metrics.get("dataset_root", "")).expanduser().resolve() == dataset_root,
        "metrics dataset root mismatch",
    )
    require(
        Path(metrics.get("checkpoint", "")).expanduser().resolve() == checkpoint_path,
        "metrics checkpoint path mismatch",
    )
    require(provenance.get("experiment") == "E-000", "provenance experiment mismatch")
    require(provenance.get("schema_version") == 1, "provenance schema mismatch")
    require(
        provenance.get("archive", {}).get("sha256") == EXPECTED_ARCHIVE_SHA256,
        "provenance archive SHA256 mismatch",
    )
    require(
        Path(provenance.get("published_dataset_root", "")).expanduser().resolve()
        == dataset_root,
        "provenance dataset root mismatch",
    )
    extracted = provenance.get("extracted", {})
    require(extracted.get("pair_count") == EXPECTED_PAIR_COUNT, "provenance pair count mismatch")
    require(extracted.get("data_file_count") == 2120, "provenance file count mismatch")
    require(
        extracted.get("provenance_file_present_during_validation") is True,
        "provenance is not from the post-publication validation pass",
    )
    require(
        provenance_path == dataset_root / "E000_PROVENANCE.json",
        "provenance path is not bound to the dataset root",
    )
    require(
        Path(extracted.get("dataset_root", "")).expanduser().resolve()
        == dataset_root,
        "extracted dataset root mismatch",
    )
    require(
        Path(extracted.get("validated_source_root", "")).expanduser().resolve()
        == dataset_root,
        "provenance is not the post-publication record",
    )

    formal_pairs = metrics.get("per_pair")
    reference_pairs = reference.get("per_pair")
    require(isinstance(formal_pairs, list), "formal per_pair is missing")
    require(isinstance(reference_pairs, list), "reference per_pair is missing")
    require(len(formal_pairs) == EXPECTED_PAIR_COUNT, "formal per_pair length mismatch")
    require(len(reference_pairs) == EXPECTED_PAIR_COUNT, "reference per_pair length mismatch")

    pair_deltas = []
    formal_values = []
    reference_values = []
    for index, (formal_pair, reference_pair) in enumerate(
        zip(formal_pairs, reference_pairs)
    ):
        expected_id = f"{index:06d}_opt"
        require(formal_pair.get("index") == index, f"pair index mismatch at {index}")
        require(formal_pair.get("pair_id") == expected_id, f"pair id mismatch at {index}")
        require(reference_pair.get("index") == index, f"reference index mismatch at {index}")
        require(reference_pair.get("pair_id") == expected_id, f"reference id mismatch at {index}")
        expected_path = (dataset_root / f"testing/opt/{expected_id}.png").resolve()
        require(
            Path(formal_pair.get("optical_path", "")).expanduser().resolve()
            == expected_path,
            f"pair path mismatch at {index}",
        )
        formal_epe = finite_float(formal_pair.get("epe"), f"formal EPE at {index}")
        reference_epe = finite_float(
            reference_pair.get("epe"), f"reference EPE at {index}"
        )
        formal_values.append(formal_epe)
        reference_values.append(reference_epe)
        pair_deltas.append(abs(formal_epe - reference_epe))

    formal_summary = validate_metric_summary("formal", metrics, formal_values)
    reference_summary = validate_metric_summary("reference", reference, reference_values)
    aepe = formal_summary["aepe"]
    reference_aepe = reference_summary["aepe"]
    max_pair_delta = max(pair_deltas)
    require(
        max_pair_delta <= args.max_pair_delta,
        f"formal/reference pair delta {max_pair_delta} exceeds {args.max_pair_delta}",
    )
    require(
        abs(aepe - reference_aepe) <= args.max_pair_delta,
        "formal/reference AEPE delta exceeds tolerance",
    )

    threshold_summary = {}
    for threshold in THRESHOLDS:
        key = str(threshold)
        formal_threshold = formal_summary["thresholds"][key]
        reference_threshold = reference_summary["thresholds"][key]
        require(
            formal_threshold["matched_pairs"] == reference_threshold["matched_pairs"],
            f"CMR matched-pair count differs at threshold {threshold}",
        )
        threshold_summary[key] = {
            "matched_pairs": formal_threshold["matched_pairs"],
            "cmr_percent": formal_threshold["cmr_percent"],
            "reference_cmr_percent": reference_threshold["cmr_percent"],
        }

    result = {
        "passed": True,
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "reference": str(reference_path),
        "reference_sha256": reference_sha,
        "provenance": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "provenance_validation": provenance_record["validation"],
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "num_pairs": EXPECTED_PAIR_COUNT,
        "aepe": aepe,
        "reference_aepe": reference_aepe,
        "recomputed_aepe": formal_summary["aepe"],
        "recomputed_pair_std": formal_summary["pair_std"],
        "recomputed_pair_variance": formal_summary["pair_variance"],
        "recomputed_reference_aepe": reference_summary["aepe"],
        "recomputed_reference_pair_std": reference_summary["pair_std"],
        "recomputed_reference_pair_variance": reference_summary["pair_variance"],
        "aepe_delta": aepe - reference_aepe,
        "max_abs_pair_delta": max_pair_delta,
        "pair_delta_tolerance": args.max_pair_delta,
        "thresholds": threshold_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
