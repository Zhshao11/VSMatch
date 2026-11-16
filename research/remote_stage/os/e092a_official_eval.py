#!/usr/bin/env python3
"""E092A: fixed-weight OS-424 prediction materialization and CPU metric join."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(os.environ.get("SPARSEGEO_ROOT", "/mnt/sda/zhushihao/VS_Match"))
TMP = ROOT / "image_matching_part2_migrated/.codex_tmp"
OFFICIAL = ROOT / "external_repos/E068A_GDROS_official"
DATASET = ROOT / "datasets/extracted/gdros_official_e003a_testing/testing"
ARCHIVE = ROOT / "datasets/official_archives/gdros/testing.zip"
RUN = ROOT / "GDROS/experiments/sparsegeo/runs/E-092A-OS-OFFICIAL424-M4-GDROS-EVAL"
GDROS_CKPT = ROOT / "checkpoints/gdros/GDROS-os512.pth"
M4_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt"
B_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-b-stage180k/checkpoints/B_step180000.pt"
C_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-c-stage180k/checkpoints/C_step180000.pt"
F2_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f2-warm120k/checkpoints/F2_step120000.pt"
F3_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f3-warm120k/checkpoints/F3_step120000.pt"

EXPECTED = {
    ARCHIVE: "e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218",
    GDROS_CKPT: "80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683",
    M4_CKPT: "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b",
    B_CKPT: "f40d80f2dca62e74ec86c1823963282a490d47f336f2cfce4a28fc211ea33218",
    C_CKPT: "c3c67e6d65aca0dd55cf7af2c2a9eedccadadc0fec5ad11fe6e493e08ecc85b9",
    F2_CKPT: "4a52307fa4b7b67756ac979310fa37d34a131cc182a39e2c45d86711972dab06",
    F3_CKPT: "e5a9c612a7b9b85c568b88776ef784873d5df693cb570ad3dbbbf52fe4801ffd",
}
THRESHOLDS = (1, 2, 3, 4, 5, 10, 16)
REPLAY_TOLERANCES = {"mean": 0.005, "cmr_pp": 0.5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_card(task: str, card: dict) -> None:
    target = RUN / task
    payload = dict(card, task=task, updated_at_unix=time.time())
    write_json(target / "RESULT_CARD.json", payload)
    (target / "RESULT_CARD.md").write_text(
        f"# E092A {task}\n\nStatus: `{payload.get('status')}`\n\n"
        f"Last completed stage: `{payload.get('last_completed_stage')}`\n\n"
        f"```json\n{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n```\n"
    )
    attempt = os.environ.get("WATCHDOG_JOB_ID", f"manual-{os.getpid()}")
    attempt_dir = target / "attempts" / attempt
    write_json(attempt_dir / "RESULT_CARD.json", payload)
    (attempt_dir / "RESULT_CARD.md").write_text(
        f"# E092A {task} attempt {attempt}\n\nStatus: `{payload.get('status')}`\n"
    )


def pairs(include_gt: bool = False) -> list[dict]:
    opt = sorted((DATASET / "opt").glob("*.png"))
    sar = sorted((DATASET / "sar_warped").glob("*.png"))
    if len(opt) != 424 or len(sar) != 424:
        raise RuntimeError(f"expected 424 image pairs, got opt={len(opt)} sar={len(sar)}")
    rows = []
    for index, (source, target) in enumerate(zip(opt, sar)):
        pair_id = source.name.removesuffix("_opt.png")
        if target.name != f"{pair_id}_sar_warped.png":
            raise RuntimeError(f"pair mismatch at {index}: {source.name} / {target.name}")
        row = {"pair_index": index, "pair_id": pair_id, "source": source, "target": target}
        if include_gt:
            row["gt"] = DATASET / "truth_flow" / f"{pair_id}.flo"
        rows.append(row)
    if len({row["pair_id"] for row in rows}) != 424:
        raise RuntimeError("pair IDs are not unique")
    return rows


def image_tensor(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path), dtype=np.uint8)
    if array.ndim == 2:
        array = np.tile(array[..., None], (1, 1, 3))
    else:
        array = array[..., :3]
    if array.shape != (512, 512, 3):
        raise RuntimeError(f"unexpected image shape {array.shape} for {path}")
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()[None]


def environment() -> dict:
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "seed": 1234,
        "fp32": True,
        "batch_size": 1,
    }


def hash_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_extracted_dataset() -> dict:
    """Bind every image/GT byte consumed by the loader to the audited ZIP."""
    relative_paths = []
    for directory, pattern in (("opt", "*.png"), ("sar_warped", "*.png"), ("truth_flow", "*.flo")):
        relative_paths.extend(path.relative_to(DATASET.parent) for path in sorted((DATASET / directory).glob(pattern)))
    if len(relative_paths) != 424 * 3:
        raise RuntimeError(f"expected 1272 loader files, got {len(relative_paths)}")
    combined = hashlib.sha256()
    mismatches = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        for relative in relative_paths:
            normalized = relative.as_posix()
            candidates = [name for name in names if name == normalized or name.endswith("/" + normalized)]
            if len(candidates) != 1:
                mismatches.append({"relative": normalized, "archive_candidates": candidates})
                continue
            extracted_hash = sha256(DATASET.parent / relative)
            with archive.open(candidates[0]) as handle:
                archive_hash = hash_stream(handle)
            if extracted_hash != archive_hash:
                mismatches.append({"relative": normalized, "archive_member": candidates[0], "extracted": extracted_hash, "archive": archive_hash})
            combined.update(normalized.encode())
            combined.update(extracted_hash.encode())
    return {
        "loader_files": len(relative_paths), "all_match_archive": not mismatches,
        "mismatches": mismatches, "loader_content_manifest_sha256": combined.hexdigest(),
        "loader_semantics": "PIL uint8; grayscale tiled to 3 channels; color truncated to first 3 channels; CHW float32; no resize/crop/mask",
    }


def code_hashes() -> dict:
    paths = [
        TMP / "e092a_official_eval.py", TMP / "e075a_preserve_high_precision_recover_c2c5.py",
        TMP / "e074a_f2_f3_twostage_fusion.py", TMP / "e073a_v1_v2_fusion_longtrain.py",
        TMP / "e072a_layerwise_frontend_alignment.py", OFFICIAL / "test.py",
        OFFICIAL / "core/datasets.py", OFFICIAL / "core/LSRnet/LSmodel.py",
        OFFICIAL / "core/GDROSnet/FeatureFormer/gdros.py",
    ]
    return {str(path): sha256(path) for path in paths}


def preflight() -> None:
    actual = {str(path): sha256(path) for path in EXPECTED}
    mismatches = {str(path): {"expected": expected, "actual": actual[str(path)]}
                  for path, expected in EXPECTED.items() if actual[str(path)] != expected}
    rows = pairs(include_gt=True)
    missing_gt = [str(row["gt"]) for row in rows if not row["gt"].is_file()]
    dataset_provenance = verify_extracted_dataset()
    official_commit = os.popen(f"cd {OFFICIAL} && git rev-parse HEAD").read().strip()
    card = {
        "status": "PASS" if not mismatches and not missing_gt and dataset_provenance["all_match_archive"] else "FAIL",
        "last_completed_stage": "CPU_PREFLIGHT",
        "protocol": {
            "pairs": 424, "shape": [512, 512], "whole_image": True, "valid_mask": False,
            "lsr_npoint": 200000, "lsr_flat_slice": [10000, 210000],
            "main_table": ["gdros_iter32_post_lsr", "m4_iter32_post_lsr"],
            "preregistered_supplement": "m4_iter24_post_lsr",
            "prediction_reads_gt": False, "join_is_cpu_only": True,
        },
        "hashes": actual,
        "hash_mismatches": mismatches,
        "missing_gt": missing_gt,
        "dataset_provenance": dataset_provenance,
        "core_dependency_code_hashes": code_hashes(),
        "pair_ids_unique": len({row["pair_id"] for row in rows}) == 424,
        "official_source_commit": official_commit,
        "environment": environment(),
    }
    write_card("preflight", card)
    if card["status"] != "PASS":
        raise RuntimeError(f"preflight failed: {mismatches=} {missing_gt=} dataset_ok={dataset_provenance['all_match_archive']}")


def official_imports():
    for value in (str(OFFICIAL),):
        if value not in sys.path:
            sys.path.insert(0, value)
    from configs.default import get_cfg
    from core.GDROSnet.FeatureFormer.gdros import GDROS
    from core.LSRnet.LSmodel import LSR
    return get_cfg, GDROS, LSR


def load_gdros(device):
    get_cfg, GDROS, _ = official_imports()
    cfg = get_cfg()
    cfg.defrost()
    settings = {
        "small": False, "mixed_precision": False, "alternate_corr": False,
        "image_size": [512, 512], "corr_levels": 4, "corr_radius": 4,
        "gpus": [0], "num_classes": [8], "Npoint": 200000,
    }
    for key, value in settings.items():
        setattr(cfg, key, value)
    model = GDROS(cfg[cfg.transformer]).to(device)
    raw = torch.load(GDROS_CKPT, map_location=device, weights_only=False)
    state = {key.removeprefix("module."): value for key, value in raw.items()}
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(str(loaded))
    return model.eval(), {"missing_keys": [], "unexpected_keys": [], "state_keys": len(state)}


def load_m4(device):
    if str(TMP) not in sys.path:
        sys.path.insert(0, str(TMP))
    import e075a_preserve_high_precision_recover_c2c5 as e075
    e074, _e072, _e073 = e075.add_paths()
    model = e075.PairFusion(e074, "region", device).to(device)
    payload = torch.load(M4_CKPT, map_location=device, weights_only=False)
    loaded = model.load_state_dict(payload["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(str(loaded))
    return model.eval(), {"missing_keys": [], "unexpected_keys": [], "state_keys": len(payload["model"]), "checkpoint_step": payload.get("step")}


def prediction_path(label: str, index: int) -> Path:
    return RUN / "predictions" / label / f"{index:06d}.npy"


def predict(model_name: str) -> None:
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if model_name == "gdros":
        model, load_report = load_gdros(device)
        labels = ("gdros_iter32_post_lsr",)
    else:
        model, load_report = load_m4(device)
        labels = ("m4_iter24_post_lsr", "m4_iter32_post_lsr")
    _, _, LSR = official_imports()
    task = f"predict_{model_name}"
    manifests = {label: RUN / "predictions" / label / "PREDICTIONS.jsonl" for label in labels}
    for label, manifest in manifests.items():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.unlink(missing_ok=True)
    write_card(task, {"status": "RUNNING", "last_completed_stage": "MODEL_STRICT_LOAD", "load_report": load_report})
    handles = {label: manifest.open("a") for label, manifest in manifests.items()}
    try:
        with torch.inference_mode():
            for row in pairs(include_gt=False):
                source = image_tensor(row["source"]).to(device)
                target = image_tensor(row["target"]).to(device)
                if model_name == "gdros":
                    _low, dense = model(source, target, iters=32, test_mode=True)
                    outputs = {labels[0]: LSR(1, 512, 512, dense, 200000)}
                else:
                    sequence = model(source, target, iters=32, lsr_fn=e075_lsr_adapter())
                    outputs = {
                        "m4_iter24_post_lsr": LSR(1, 512, 512, sequence[23], 200000),
                        "m4_iter32_post_lsr": LSR(1, 512, 512, sequence[31], 200000),
                    }
                for label, tensor in outputs.items():
                    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                    if array.shape != (1, 2, 512, 512):
                        raise RuntimeError(f"bad prediction shape {array.shape}")
                    path = prediction_path(label, row["pair_index"])
                    np.save(path, array, allow_pickle=False)
                    record = {
                        "pair_index": row["pair_index"], "pair_id": row["pair_id"],
                        "prediction": str(path), "sha256": sha256(path),
                        "shape": list(array.shape), "dtype": str(array.dtype),
                        "uses_truth_flow": False, "post_lsr": True,
                    }
                    handles[label].write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    handles[label].flush()
                del source, target, outputs
                torch.cuda.empty_cache()
    finally:
        for handle in handles.values():
            handle.close()
    manifest_report = {}
    for label, manifest in manifests.items():
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        if len(records) != 424 or [record["pair_index"] for record in records] != list(range(424)):
            raise RuntimeError(f"incomplete manifest {label}: {len(records)}")
        manifest_report[label] = {"path": str(manifest), "sha256": sha256(manifest), "records": len(records)}
    write_card(task, {
        "status": "COMPLETE", "last_completed_stage": "424_PREDICTIONS_MATERIALIZED",
        "load_report": load_report, "manifests": manifest_report, "environment": environment(),
        "ground_truth_loaded": False, "logical_device": "cuda:0",
    })


def e075_lsr_adapter():
    from core.LSRnet.LSmodel import LSR
    return lambda flow: LSR(1, 512, 512, flow, 200000)


def read_flo(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = np.fromfile(handle, np.float32, count=1)
        width = np.fromfile(handle, np.int32, count=1)
        height = np.fromfile(handle, np.int32, count=1)
        if magic.size != 1 or float(magic[0]) != 202021.25:
            raise RuntimeError(f"invalid .flo magic: {path}")
        array = np.fromfile(handle, np.float32, count=int(width[0] * height[0] * 2))
    return array.reshape(int(height[0]), int(width[0]), 2).transpose(2, 0, 1)


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("metric input is empty or non-finite")
    result = {
        "pairs": int(array.size), "mean": float(array.mean()),
        "variance": float(array.var()), "std": float(array.std()),
        "p95": float(np.quantile(array, 0.95)), "p99": float(np.quantile(array, 0.99)),
    }
    for threshold in THRESHOLDS:
        result[f"cmr@{threshold}"] = float((array < threshold).mean() * 100.0)
        result[f"failures_ge_{threshold}"] = int((array >= threshold).sum())
    return result


def public_formula_fixture() -> dict:
    gt = torch.zeros((2, 2, 2), dtype=torch.float32)
    prediction = torch.stack((torch.full((2, 2), 3.0), torch.full((2, 2), 4.0)))
    difference = prediction - gt
    explicit = float(((difference[0].abs() + difference[1].abs()) / 2.0).mean())
    official = float(torch.sum((prediction.unsqueeze(0) - gt) ** 2, dim=0).sqrt().mean())
    vector = float(torch.linalg.vector_norm(difference, dim=0).mean())
    passed = abs(explicit - official) <= 1e-7 and abs(explicit - vector) > 1.0
    return {"dx": 3.0, "dy": 4.0, "explicit_public": explicit, "official_expression": official,
            "vector_epe": vector, "public_differs_from_vector": explicit != vector,
            "tolerance": 1e-7, "status": "PASS" if passed else "FAIL"}


def join() -> None:
    labels = ("gdros_iter32_post_lsr", "m4_iter32_post_lsr", "m4_iter24_post_lsr")
    manifests = {}
    for label in labels:
        path = RUN / "predictions" / label / "PREDICTIONS.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) != 424:
            raise RuntimeError(f"{label} manifest has {len(records)} records")
        for index, record in enumerate(records):
            pred = Path(record["prediction"])
            if record["pair_index"] != index or sha256(pred) != record["sha256"]:
                raise RuntimeError(f"manifest/hash failure {label} index={index}")
            prediction = np.load(pred, allow_pickle=False)
            if prediction.shape != (1, 2, 512, 512) or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
                raise RuntimeError(f"prediction integrity failure {label} index={index} shape={prediction.shape} dtype={prediction.dtype}")
        manifests[label] = {record["pair_id"]: record for record in records}
    expected_ids = [row["pair_id"] for row in pairs(include_gt=False)]
    if any(list(records) != expected_ids for records in manifests.values()):
        raise RuntimeError("prediction pair IDs do not match the image-only manifest")

    # No GT path has been constructed or read before all 1272 predictions pass this point.
    per_pair = []
    metric_values = {label: {"public_component": [], "vector_epe": []} for label in labels}
    public_parity_max_abs = 0.0
    serialized_public_max_abs = 0.0
    for row in pairs(include_gt=True):
        gt = read_flo(row["gt"]).astype(np.float32, copy=False)
        output = {"pair_index": row["pair_index"], "pair_id": row["pair_id"]}
        for label in labels:
            prediction = np.load(manifests[label][row["pair_id"]]["prediction"], allow_pickle=False)[0]
            difference = prediction - gt
            explicit = float(np.mean((np.abs(difference[0]) + np.abs(difference[1])) / 2.0))
            pred_tensor = torch.from_numpy(prediction)
            gt_tensor = torch.from_numpy(gt)
            official_expression = float(torch.sum((pred_tensor.unsqueeze(0) - gt_tensor) ** 2, dim=0).sqrt().mean())
            tensor_explicit = float((pred_tensor - gt_tensor).abs().mean())
            parity_abs = abs(tensor_explicit - official_expression)
            public_parity_max_abs = max(public_parity_max_abs, parity_abs)
            serialized_public_max_abs = max(serialized_public_max_abs, abs(explicit - tensor_explicit))
            public = explicit
            vector = float(np.mean(np.sqrt(np.square(difference[0]) + np.square(difference[1]))))
            metric_values[label]["public_component"].append(public)
            metric_values[label]["vector_epe"].append(vector)
            output[f"{label}__public_component"] = public
            output[f"{label}__vector_epe"] = vector
        per_pair.append(output)

    output_csv = RUN / "join" / "PER_PAIR_METRICS.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_pair[0]))
        writer.writeheader()
        writer.writerows(per_pair)
    metrics = {label: {metric: summarize(values) for metric, values in by_metric.items()}
               for label, by_metric in metric_values.items()}
    if not all(np.isfinite(value) for by_label in metric_values.values() for values in by_label.values() for value in values):
        raise RuntimeError("non-finite joined metric")
    gdros_observed = metrics["gdros_iter32_post_lsr"]["public_component"]
    m4_404_observed = summarize(metric_values["m4_iter24_post_lsr"]["vector_epe"][:404])
    gdros_reference = {"mean": 1.461486, "cmr@1": 33.73, "cmr@2": 81.60, "cmr@5": 99.53}
    m4_reference = {"mean": 2.361299, "cmr@1": 56.683168, "cmr@2": 78.960396, "cmr@5": 88.861386}

    def regression(observed, reference):
        deltas = {key: float(observed[key] - value) for key, value in reference.items()}
        passed = abs(deltas["mean"]) <= REPLAY_TOLERANCES["mean"] and all(
            abs(deltas[key]) <= REPLAY_TOLERANCES["cmr_pp"] for key in ("cmr@1", "cmr@2", "cmr@5")
        )
        return {"status": "PASS" if passed else "REPLAY_MISMATCH", "observed": observed, "reference": reference,
                "deltas": deltas, "predeclared_tolerances": REPLAY_TOLERANCES}

    regressions = {
        "gdros_public_424": regression(gdros_observed, gdros_reference),
        "m4_iter24_vector_first404": regression(m4_404_observed, m4_reference),
    }
    fixture = public_formula_fixture()
    replay_pass = all(value["status"] == "PASS" for value in regressions.values())
    formula_pass = public_parity_max_abs <= 1e-6
    final_status = "COMPLETE" if replay_pass and formula_pass and fixture["status"] == "PASS" else "REPLAY_MISMATCH"
    preflight_card = json.loads((RUN / "preflight/RESULT_CARD.json").read_text())
    preflight_script_sha = preflight_card["core_dependency_code_hashes"][str(TMP / "e092a_official_eval.py")]
    join_script_sha = sha256(TMP / "e092a_official_eval.py")
    card = {
        "status": final_status, "last_completed_stage": "CPU_GT_JOIN_424",
        "metrics": metrics, "regressions": regressions,
        "per_pair_csv": str(output_csv), "per_pair_csv_sha256": sha256(output_csv),
        "prediction_manifests_verified_before_gt_load": True,
        "strict_thresholds": True, "whole_image_no_mask": True,
        "public_formula_parity": {"max_abs_difference": public_parity_max_abs, "tolerance": 1e-6,
                                  "status": "PASS" if public_parity_max_abs <= 1e-6 else "FAIL",
                                  "official_expression": "mean(sqrt(sum((prediction[1,2,H,W]-gt[2,H,W])**2, dim=0)))",
                                  "explicit_expression": "mean(abs(prediction[2,H,W]-gt[2,H,W])) = mean((abs(dx)+abs(dy))/2)",
                                  "serialized_numpy_metric_max_abs_difference": serialized_public_max_abs,
                                  "serialized_numpy_metric_tolerance": 1e-5,
                                  "fixture": fixture},
        "main_table": ["gdros_iter32_post_lsr", "m4_iter32_post_lsr"],
        "supplement_only": "m4_iter24_post_lsr",
        "join_code_amendment": {
            "reason": "Correct official-formula parity tensor shape and final Markdown status; prediction implementation unchanged in the running processes",
            "preflight_script_sha256": preflight_script_sha,
            "join_script_sha256": join_script_sha,
            "changed_after_prediction_preflight": preflight_script_sha != join_script_sha,
        },
    }
    write_card("join", card)
    write_json(RUN / "RESULT_CARD.json", card)
    (RUN / "RESULT_CARD.md").write_text(
        f"# E092A final result\n\nStatus: `{final_status}`\n\n"
        f"```json\n{json.dumps(card, indent=2, sort_keys=True, allow_nan=False)}\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("preflight", "predict_gdros", "predict_m4", "join"))
    args = parser.parse_args()
    task = args.task

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if task == "preflight":
            preflight()
        elif task == "predict_gdros":
            predict("gdros")
        elif task == "predict_m4":
            predict("m4")
        else:
            join()
    except BaseException as exc:
        write_card(task, {
            "status": "FAILED", "last_completed_stage": "EXCEPTION",
            "failure_reason": repr(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
