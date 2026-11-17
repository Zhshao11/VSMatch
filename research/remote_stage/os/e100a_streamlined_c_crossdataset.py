#!/usr/bin/env python3
"""Frozen OS C/student on existing WHU700 and UBCv2-1625x3 inputs.

Stage with E092B/E092CR/E093A and M4-streamlined-C-student helpers in
SPARSEGEO_ROOT/image_matching_part2_migrated/.codex_tmp; run from project root.
Weights are unchanged. This is evaluation, not a training entry point.
"""
import argparse
import csv
import gc
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import e092b_whu_eval as whu
import e092cr_ubcv2_eval as ubc
import e093a_bc_ablation_profile as branches
from m4_streamlined_c_student import M4StreamlinedCStudent


ROOT = branches.ROOT
COMPACT = ROOT / "GDROS/experiments/sparsegeo/runs/E-099A-STREAMLINED-C-STUDENT-OS424-20261116/M4-streamlined-C-student-state.pt"
COMPACT_SHA = "c2d4f79b45ce190df48f52f588e40a7654fa6cce0a32da1bc0c32020b35636ad"
GROUPS = ("whu700", "ubcv2_seed901", "ubcv2_seed902", "ubcv2_seed903")


def context(group):
    if group == "whu700":
        return whu, whu.ROOT / "GDROS/experiments/sparsegeo/runs/E-092B-WHU-OFFICIAL700-FROZEN-OS-M4-EVAL"
    ubc.set_seed(int(group[-3:]))
    return ubc, ubc.RUN


def summary(values):
    a = np.asarray(values, np.float64)
    assert a.size and np.isfinite(a).all()
    result = dict(pairs=int(a.size), mean=float(a.mean()), variance=float(a.var()),
                  std=float(a.std()), p95=float(np.quantile(a, .95)), p99=float(np.quantile(a, .99)))
    for threshold in (1, 2, 3, 4, 5, 10, 16):
        selected = a[a < threshold]
        result[f"cmr@{threshold}"] = float(100 * selected.size / a.size)
        result[f"failures_ge_{threshold}"] = int(a.size - selected.size)
        result[f"success_subset@{threshold}"] = {
            "count": int(selected.size), "mean": float(selected.mean()) if selected.size else None,
            "variance_official_rmse_field": float(selected.var()) if selected.size else None}
    return result


def sources():
    modules = (whu, ubc, branches, importlib.import_module("m4_streamlined_c_student"),
               importlib.import_module("m4_clean_deploy"))
    paths = [Path(__file__).resolve(), *[Path(m.__file__).resolve() for m in modules]]
    paths += [branches.TMP / name for name in (
        "e072a_layerwise_frontend_alignment.py", "e073a_v1_v2_fusion_longtrain.py",
        "e074a_f2_f3_twostage_fusion.py", "e075a_preserve_high_precision_recover_c2c5.py")]
    paths += [branches.OFFICIAL / name for name in (
        "test.py", "core/datasets.py", "core/corr.py", "core/update.py", "core/LSRnet/LSmodel.py")]
    return {str(p): whu.sha256(p) for p in paths}


def predict(out):
    assert not out.exists(), "Use a fresh run directory"
    out.mkdir(parents=True)
    os.chdir(branches.TMP.parent)
    torch.set_num_threads(1)
    branches.configure_backends()
    assert whu.sha256(COMPACT) == COMPACT_SHA
    assert branches.sha256(branches.M4_CKPT) == branches.EXPECTED[branches.M4_CKPT]
    source_hashes = sources()
    for source in source_hashes:
        destination = out / "source_snapshot" / Path(source).name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
    protocol = dict(method=M4StreamlinedCStudent.method_name, training=False, os_weights_frozen=True,
                    iterations=32, internal_lsr=0, terminal_lsr=1, lsr_sample_count=200000,
                    fp32=True, tf32=False, batch_size=1, full_image_no_mask=True,
                    compact_state=str(COMPACT), compact_sha256=COMPACT_SHA,
                    protected_m4_sha256=branches.EXPECTED[branches.M4_CKPT], sources=source_hashes,
                    groups=list(GROUPS), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    metric="mean over pixels and x/y components of abs(pred-gt), CMR strict per-pair mean < threshold",
                    official_rmse_field="population variance of pair means, not square-root MSE",
                    whu_scope="author-public 700 pairs; one group, not paper three-seed aggregate",
                    ubcv2_scope="official raw 1625 pairs, independent existing seeds 901/902/903; not author-screened 1447 or exact unpublished transforms",
                    timing_scope="shared GPU; runtime diagnostic only, not a speed benchmark")
    whu.write_json(out / "PROTOCOL.json", protocol)
    print(json.dumps({"stage": "LOAD_STUDENT_ONLY"}), flush=True)
    payload, state = branches.checkpoint_state()
    reference, load_report = branches.build_c_student(torch.device("cpu"), state)
    del payload, state
    model = M4StreamlinedCStudent(reference).eval()
    del reference
    gc.collect()
    compact = torch.load(COMPACT, map_location="cpu", weights_only=True)
    model.load_state_dict(compact, strict=True)
    del compact
    gc.collect()
    assert sum(p.numel() for p in model.parameters()) == 319303309
    assert list(dict(model.named_children())) == ["dino", "descriptor_adapters", "matcher"]
    whu.write_json(out / "MODEL_AUDIT.json", dict(load_report=load_report, pruning=model.audit,
                   parameters=319303309, compact_load="strict", teachers_constructed=False))
    device = torch.device("cuda:0")
    model.to(device)
    from core.LSRnet.LSmodel import LSR
    with torch.inference_mode():
        for group in GROUPS:
            helper, prior = context(group)
            rows = helper.pair_rows(False)
            preflight_path = prior / "preflight/RESULT_CARD.json"
            prior_preflight = json.loads(preflight_path.read_text())
            assert prior_preflight["status"] == "PASS"
            assert prior_preflight["official_loader_input_parity"]["source_target_tensor_max_abs_difference"] == 0
            group_dir = out / group
            group_dir.mkdir()
            (group_dir / "predictions").mkdir()
            whu.write_json(group_dir / "PRIOR_PREFLIGHT.json", dict(path=str(preflight_path),
                           sha256=whu.sha256(preflight_path), card=prior_preflight))
            if group != "whu700":
                for key in ("manifest", "params"):
                    audit = prior_preflight["generated_protocol_audit"]
                    assert whu.sha256(Path(audit[key])) == audit[key + "_sha256"]
            start = time.time()
            manifest = group_dir / "PREDICTIONS.jsonl"
            with manifest.open("x") as handle:
                for row in rows:
                    a = helper.image_tensor(row["source"]).to(device)
                    b = helper.image_tensor(row["target"]).to(device)
                    raw = model(a, b, iters=32)
                    pred = LSR(1, 512, 512, raw, 200000).detach().cpu().numpy().astype(np.float32, copy=False)
                    assert pred.shape == (1, 2, 512, 512) and np.isfinite(pred).all()
                    target = group_dir / "predictions" / f"{row['pair_index']:06d}.npy"
                    np.save(target, pred, allow_pickle=False)
                    record = dict(pair_index=row["pair_index"], pair_id=row["pair_id"],
                                  prediction=str(target), sha256=whu.sha256(target),
                                  source=str(row["source"]), source_sha256=whu.sha256(row["source"]),
                                  target=str(row["target"]), target_sha256=whu.sha256(row["target"]),
                                  uses_truth_flow=False)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    if (row["pair_index"] + 1) % 100 == 0:
                        print(json.dumps(dict(group=group, done=row["pair_index"] + 1,
                                              total=len(rows), elapsed_s=time.time() - start)), flush=True)
                    del a, b, raw, pred
            whu.write_json(group_dir / "PREDICT_COMPLETE.json", dict(pairs=len(rows),
                           manifest_sha256=whu.sha256(manifest), uses_truth_flow=False,
                           elapsed_wall_seconds=time.time() - start))
            print(json.dumps(dict(group=group, stage="PREDICT_COMPLETE", pairs=len(rows))), flush=True)
    assert source_hashes == sources(), "Source changed during evaluation"
    assert whu.sha256(COMPACT) == COMPACT_SHA
    whu.write_json(out / "PREDICT_COMPLETE.json", dict(groups=list(GROUPS), weights_unchanged=True))


def join(out):
    torch.set_num_threads(1)
    protocol = json.loads((out / "PROTOCOL.json").read_text())
    assert protocol["sources"] == sources()
    assert (out / "PREDICT_COMPLETE.json").exists()
    cards = {}
    assert ubc.formula_fixture()["status"] == "PASS"
    for group in GROUPS:
        helper, prior = context(group)
        group_dir = out / group
        records = [json.loads(line) for line in (group_dir / "PREDICTIONS.jsonl").read_text().splitlines()]
        complete = json.loads((group_dir / "PREDICT_COMPLETE.json").read_text())
        assert whu.sha256(group_dir / "PREDICTIONS.jsonl") == complete["manifest_sha256"]
        rows = helper.pair_rows(True)
        assert len(records) == len(rows) == complete["pairs"]
        pairs = []
        for row, rec in zip(rows, records):
            assert row["pair_id"] == rec["pair_id"] and row["pair_index"] == rec["pair_index"]
            assert not rec["uses_truth_flow"]
            for key in ("source", "target"):
                assert str(row[key]) == rec[key]
                assert whu.sha256(row[key]) == rec[key + "_sha256"]
            prediction = Path(rec["prediction"])
            assert whu.sha256(prediction) == rec["sha256"]
            pred = np.load(prediction, allow_pickle=False)
            assert pred.shape == (1, 2, 512, 512) and pred.dtype == np.float32
            gt = helper.read_flo(row["gt"])
            delta = pred[0] - gt
            public = float(np.mean((np.abs(delta[0]) + np.abs(delta[1])) / 2))
            official = float(torch.sum((torch.from_numpy(pred) - torch.from_numpy(gt)) ** 2, dim=0).sqrt().mean())
            assert abs(public - official) <= 2e-5 * max(1, public)
            pairs.append(dict(pair_id=row["pair_id"], pair_index=row["pair_index"],
                              public_component=public, vector_epe=float(np.linalg.norm(delta, axis=0).mean()),
                              gt=str(row["gt"]), gt_sha256=whu.sha256(row["gt"])))
        with (group_dir / "PER_PAIR_METRICS.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
        prior_result = prior / "RESULT_CARD.json"
        old_card = json.loads(prior_result.read_text())
        assert old_card["status"] == "COMPLETE"
        card = dict(status="COMPLETE", group=group,
                    metrics={key: summary([row[key] for row in pairs]) for key in ("public_component", "vector_epe")},
                    baseline_source=dict(path=str(prior_result), sha256=whu.sha256(prior_result)),
                    prior_metrics=old_card["metrics"], prediction_hashes_all_verified=True,
                    prediction_before_gt_join=True, official_formula_parity=True)
        whu.write_json(group_dir / "RESULT_CARD.json", card)
        cards[group] = card
        print(json.dumps(dict(group=group, stage="JOIN_COMPLETE", metrics=card["metrics"]["public_component"])), flush=True)
    aggregate = {}
    for field in ("mean", "variance", "cmr@1", "cmr@2", "cmr@5", "p95", "p99"):
        a = np.asarray([cards[g]["metrics"]["public_component"][field] for g in GROUPS[1:]], np.float64)
        aggregate[field] = dict(per_seed=a.tolist(), mean=float(a.mean()), std_ddof0=float(a.std()), std_ddof1=float(a.std(ddof=1)))
    whu.write_json(out / "RESULT_CARD.json", dict(status="COMPLETE", experiment="E100A",
                   protocol=protocol, groups=cards, ubcv2_three_group_aggregate=aggregate,
                   no_training=True, no_new_gdros_inference=True,
                   baseline_note="previous accepted E092B/E092CR same inputs and protocol; performance only, no timing comparison"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "join"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (predict if args.stage == "predict" else join)(args.output_dir.resolve())


if __name__ == "__main__":
    main()
