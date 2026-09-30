#!/usr/bin/env python3
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("check_gdros_e000_metrics.py")
SPEC = importlib.util.spec_from_file_location("check_gdros_e000_metrics", MODULE_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def make_summary(values):
    import statistics

    result = {
        "num_pairs": len(values),
        "aepe": statistics.fmean(values),
        "paper_rmse": statistics.pstdev(values),
        "legacy_variance": statistics.pvariance(values),
        "thresholds": {},
    }
    for threshold in CHECKER.THRESHOLDS:
        matched = [value for value in values if value < threshold]
        result["thresholds"][str(threshold)] = {
            "threshold": threshold,
            "matched_pairs": len(matched),
            "cmr_percent": 100.0 * len(matched) / len(values),
            "aepe": statistics.fmean(matched) if matched else None,
            "paper_rmse": statistics.pstdev(matched) if matched else None,
            "legacy_variance": statistics.pvariance(matched) if matched else None,
        }
    return result


class MetricSummaryTests(unittest.TestCase):
    def setUp(self):
        self.values = [0.5 + 5.0 * index / 423 for index in range(424)]
        self.summary = make_summary(self.values)

    def test_valid_summary_passes(self):
        checked = CHECKER.validate_metric_summary(
            "synthetic", self.summary, self.values
        )
        self.assertEqual(checked["thresholds"]["1"]["matched_pairs"], 43)

    def test_strict_threshold_comparison(self):
        values = [1.0] * 424
        summary = make_summary(values)
        checked = CHECKER.validate_metric_summary("synthetic", summary, values)
        self.assertEqual(checked["thresholds"]["1"]["matched_pairs"], 0)

    def test_rejects_inconsistent_aepe(self):
        self.summary["aepe"] += 1e-3
        with self.assertRaisesRegex(RuntimeError, "AEPE summary"):
            CHECKER.validate_metric_summary("synthetic", self.summary, self.values)

    def test_rejects_inconsistent_cmr(self):
        self.summary["thresholds"]["2"]["cmr_percent"] += 1.0
        with self.assertRaisesRegex(RuntimeError, "CMR percentage"):
            CHECKER.validate_metric_summary("synthetic", self.summary, self.values)

    def test_rejects_inconsistent_matched_statistics(self):
        self.summary["thresholds"]["3"]["paper_rmse"] += 1e-3
        with self.assertRaisesRegex(RuntimeError, "matched standard deviation"):
            CHECKER.validate_metric_summary("synthetic", self.summary, self.values)

    def test_rejects_nonfinite_numeric_value(self):
        self.summary["thresholds"]["4"]["cmr_percent"] = math.inf
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            CHECKER.validate_metric_summary("synthetic", self.summary, self.values)


class JsonTests(unittest.TestCase):
    def test_rejects_nan_anywhere_in_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text('{"unchecked": NaN}\n')
            with self.assertRaisesRegex(RuntimeError, "non-finite JSON constant"):
                CHECKER.load_json(path)

    def test_accepts_strict_json_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps({"value": 1.0}) + "\n")
            self.assertEqual(CHECKER.load_json(path), {"value": 1.0})


if __name__ == "__main__":
    unittest.main()
