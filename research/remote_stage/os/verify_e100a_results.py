#!/usr/bin/env python3
"""Rehash E100A predictions and independently score with official Torch formula.

Can audit one completed group while the other groups are still predicting.
Run --provenance once to recheck WHU archive and all UBC affine/input audits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch


GROUPS = ("whu700", "ubcv2_seed901", "ubcv2_seed902", "ubcv2_seed903")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_flow(path):
    with path.open("rb") as f:
        magic, width, height = struct.unpack("<fii", f.read(12))
        assert magic == 202021.25 and width == height == 512
        a = np.fromfile(f, dtype="<f4")
    assert a.size == width * height * 2
    return a.reshape(height, width, 2).transpose(2, 0, 1).copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--group", choices=GROUPS)
    ap.add_argument("--provenance", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Join prior complete independent audits to final metric cards, without repeating dense scoring")
    args = ap.parse_args()
    root = args.run.resolve()
    torch.set_num_threads(1)
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    assert digest(Path(protocol["compact_state"])) == protocol["compact_sha256"]
    for path, expected in protocol["sources"].items():
        assert digest(Path(path)) == expected
    groups = {}
    if args.aggregate_only:
        assert not args.group and not args.provenance
        for group in GROUPS:
            directory = root / group
            path = directory / "INDEPENDENT_VERIFICATION.json"
            report = json.loads(path.read_text())
            card = json.loads((directory / "RESULT_CARD.json").read_text())
            assert report["status"] == "PASS" and card["status"] == "COMPLETE"
            complete = json.loads((directory / "PREDICT_COMPLETE.json").read_text())
            assert digest(directory / "PREDICTIONS.jsonl") == complete["manifest_sha256"]
            assert report["official_loader_input_parity"]["status"] == "PASS"
            if group == "whu700":
                assert report["archive_parity"]["all_match_archive"]
            else:
                assert report["generated_protocol_audit"]["status"] == "PASS"
            for key, value in report["metrics"].items():
                assert np.isclose(value, card["metrics"]["public_component"][key], rtol=1e-6, atol=1e-5)
            report["result_agreement"] = True
            path.write_text(json.dumps(report, indent=2) + "\n")
            groups[group] = {key: value for key, value in report.items() if key != "gt_hashes"}
        (root / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(dict(status="PASS", groups=groups), indent=2) + "\n")
        print(json.dumps(dict(status="PASS", groups=list(groups), result_cards_agree=True)))
        return
    for group in (args.group,) if args.group else GROUPS:
        directory = root / group
        prior = json.loads((directory / "PRIOR_PREFLIGHT.json").read_text())
        assert digest(Path(prior["path"])) == prior["sha256"]
        records = [json.loads(line) for line in (directory / "PREDICTIONS.jsonl").read_text().splitlines()]
        complete = json.loads((directory / "PREDICT_COMPLETE.json").read_text())
        assert digest(directory / "PREDICTIONS.jsonl") == complete["manifest_sha256"]
        assert len(records) == (700 if group == "whu700" else 1625)
        prior_card = prior["card"]
        import e092b_whu_eval as whu
        import e092cr_ubcv2_eval as ubc
        helper = whu if group == "whu700" else ubc
        if helper is ubc:
            ubc.set_seed(int(group[-3:]))
        rows = helper.pair_rows(True)
        values, gt_hashes, maximum = [], {}, 0.0
        for index, (row, rec) in enumerate(zip(rows, records)):
            assert rec["pair_index"] == index and row["pair_id"] == rec["pair_id"]
            for key in ("source", "target"):
                assert str(row[key]) == rec[key] and digest(row[key]) == rec[key + "_sha256"]
            path = Path(rec["prediction"])
            assert digest(path) == rec["sha256"] and not rec["uses_truth_flow"]
            pred = np.load(path, allow_pickle=False)
            assert pred.shape == (1, 2, 512, 512) and pred.dtype == np.float32 and np.isfinite(pred).all()
            gt = read_flow(row["gt"])
            # Exact expression used by official test.py, including batch broadcasting.
            error = torch.sum((torch.from_numpy(pred) - torch.from_numpy(gt)) ** 2, dim=0).sqrt().mean()
            values.append(float(error))
            maximum = max(maximum, abs(float(error) - float(np.abs(pred[0] - gt).mean())))
            gt_hashes[row["pair_id"]] = digest(row["gt"])
        a = np.array(values, dtype=np.float64)
        metrics = dict(mean=float(a.mean()), variance=float(a.var()), std=float(a.std()),
                       pairs=len(a), p95=float(np.quantile(a, .95)), p99=float(np.quantile(a, .99)))
        for threshold in (1, 2, 3, 4, 5, 10, 16):
            metrics[f"cmr@{threshold}"] = float((a < threshold).mean() * 100)
            metrics[f"failures_ge_{threshold}"] = int((a >= threshold).sum())
        result_path = directory / "RESULT_CARD.json"
        agreement = None
        if result_path.exists():
            expected = json.loads(result_path.read_text())["metrics"]["public_component"]
            for key, value in metrics.items():
                assert np.isclose(value, expected[key], rtol=1e-6, atol=1e-5), (group, key, value, expected[key])
            agreement = True
        report = dict(status="PASS", group=group, metrics=metrics,
                      result_agreement=agreement, hashes_verified=len(records),
                      official_formula_vs_component_max_abs_difference=maximum,
                      gt_hashes=gt_hashes)
        if args.provenance:
            sys.path.insert(0, str(whu.OFFICIAL))
            if helper is whu:
                archive = whu.archive_parity()
                assert archive["all_match_archive"]
                assert archive["content_manifest_sha256"] == prior_card["archive_parity"]["content_manifest_sha256"]
                whu.RUN = root / "provenance_whu"
                loader = whu.official_loader_parity(rows)
                report["archive_parity"] = archive
            else:
                audit = ubc.generated_protocol_audit(rows)
                assert audit["status"] == "PASS"
                loader = ubc.official_loader_parity(rows)
                report["generated_protocol_audit"] = audit
            assert loader["status"] == "PASS"
            report["official_loader_input_parity"] = loader
        (directory / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
        groups[group] = {key: value for key, value in report.items() if key != "gt_hashes"}
        print(json.dumps(dict(group=group, status="PASS", metrics=metrics)), flush=True)
    if not args.group:
        assert all(row["result_agreement"] for row in groups.values())
        (root / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(dict(status="PASS", groups=groups), indent=2) + "\n")


if __name__ == "__main__":
    main()
