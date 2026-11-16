#!/usr/bin/env python3
"""E092B: frozen OS-M4 and official GDROS-WHU evaluation on public WHU-700."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import struct
import subprocess
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
WHU_BASE = Path("/mnt/sda/zhushihao/data/vis_sar/WHU_OPT_SAR")
ARCHIVE = WHU_BASE / "testing.zip"
DATASET = WHU_BASE / "extracted_e077a/testing/testing"
RUN = ROOT / "GDROS/experiments/sparsegeo/runs/E-092B-WHU-OFFICIAL700-FROZEN-OS-M4-EVAL"
GDROS_CKPT = ROOT / "GDROS/official_models_reference/GDROS-whu-opt-sar512.pth"
M4_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt"
B_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-b-stage180k/checkpoints/B_step180000.pt"
C_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-c-stage180k/checkpoints/C_step180000.pt"
F2_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f2-warm120k/checkpoints/F2_step120000.pt"
F3_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f3-warm120k/checkpoints/F3_step120000.pt"
EXPECTED = {
    ARCHIVE: "7c658901f6aed4ecd988c198edc6d79fd5775b5441fb04cf8ed7a8997c21a49f",
    GDROS_CKPT: "b6ebe175e59429272ae6aecda5f8fe5c5da2d451a283ac95b4892b5732d22a16",
    M4_CKPT: "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b",
    B_CKPT: "f40d80f2dca62e74ec86c1823963282a490d47f336f2cfce4a28fc211ea33218",
    C_CKPT: "c3c67e6d65aca0dd55cf7af2c2a9eedccadadc0fec5ad11fe6e493e08ecc85b9",
    F2_CKPT: "4a52307fa4b7b67756ac979310fa37d34a131cc182a39e2c45d86711972dab06",
    F3_CKPT: "e5a9c612a7b9b85c568b88776ef784873d5df693cb570ad3dbbbf52fe4801ffd",
}
LABELS = ("gdros_whu_iter32_post_lsr", "os_m4_iter32_post_lsr", "os_m4_iter24_post_lsr")
THRESHOLDS = (1, 2, 3, 4, 5, 10, 16)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_card(task: str, card: dict) -> None:
    payload = dict(card, task=task, updated_at_unix=time.time())
    target = RUN / task
    write_json(target / "RESULT_CARD.json", payload)
    (target / "RESULT_CARD.md").write_text(
        f"# E092B {task}\n\nStatus: `{payload.get('status')}`\n\n"
        f"```json\n{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n```\n"
    )
    attempt = os.environ.get("WATCHDOG_JOB_ID", f"manual-{os.getpid()}")
    attempt_dir = target / "attempts" / attempt
    write_json(attempt_dir / "RESULT_CARD.json", payload)
    (attempt_dir / "RESULT_CARD.md").write_text(
        f"# E092B {task} attempt {attempt}\n\nStatus: `{payload.get('status')}`\n"
    )


def pair_rows(include_gt: bool = False) -> list[dict]:
    opt = sorted((DATASET / "opt").glob("*_opt.tif"))
    sar = sorted((DATASET / "sar_warped").glob("*_sar_warped.tif"))
    if len(opt) != 700 or len(sar) != 700:
        raise RuntimeError(f"expected WHU-700, got opt={len(opt)} sar={len(sar)}")
    rows = []
    for index, (source, target) in enumerate(zip(opt, sar)):
        pair_id = source.name.removesuffix("_opt.tif")
        if target.name != f"{pair_id}_sar_warped.tif":
            raise RuntimeError(f"pair mismatch {source.name} / {target.name}")
        row = {"pair_index": index, "pair_id": pair_id, "source": source, "target": target}
        if include_gt:
            row["gt"] = DATASET / "truth_flow" / f"{pair_id}.flo"
        rows.append(row)
    if len({row["pair_id"] for row in rows}) != 700:
        raise RuntimeError("non-unique WHU pair IDs")
    return rows


def image_tensor(path: Path) -> torch.Tensor:
    array = np.array(Image.open(path), dtype=np.uint8, copy=True)
    if array.ndim == 2:
        array = np.tile(array[..., None], (1, 1, 3))
    else:
        array = array[..., :3]
    if array.shape != (512, 512, 3):
        raise RuntimeError(f"unexpected image shape {array.shape}: {path}")
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()[None]


def code_hashes() -> dict:
    paths = [TMP / "e092b_whu_eval.py", TMP / "e075a_preserve_high_precision_recover_c2c5.py",
             TMP / "e074a_f2_f3_twostage_fusion.py", TMP / "e073a_v1_v2_fusion_longtrain.py",
             TMP / "e072a_layerwise_frontend_alignment.py", OFFICIAL / "test.py",
             OFFICIAL / "core/datasets.py", OFFICIAL / "core/LSRnet/LSmodel.py",
             OFFICIAL / "core/GDROSnet/FeatureFormer/gdros.py"]
    return {str(path): sha256(path) for path in paths}


def archive_parity() -> dict:
    extracted = []
    for directory in ("opt", "sar_warped", "truth_flow", "datum"):
        extracted.extend(sorted((DATASET / directory).glob("*")))
    if len(extracted) != 2800:
        raise RuntimeError(f"expected 2800 extracted files, got {len(extracted)}")
    mismatches = []
    combined = hashlib.sha256()
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = set(name for name in archive.namelist() if not name.endswith("/"))
        expected_names = {f"testing/{path.parent.name}/{path.name}" for path in extracted}
        unexpected = sorted(names - expected_names)
        missing_from_archive = sorted(expected_names - names)
        for path in extracted:
            member = f"testing/{path.parent.name}/{path.name}"
            if member not in names:
                mismatches.append({"path": str(path), "missing_member": member})
                continue
            disk_hash = sha256(path)
            with archive.open(member) as handle:
                zip_hash = hashlib.sha256(handle.read()).hexdigest()
            if disk_hash != zip_hash:
                mismatches.append({"member": member, "disk": disk_hash, "zip": zip_hash})
            combined.update(member.encode()); combined.update(disk_hash.encode())
    return {"files": len(extracted), "archive_files": len(names),
            "exact_member_set": not unexpected and not missing_from_archive,
            "unexpected_members": unexpected, "missing_members": missing_from_archive,
            "all_match_archive": not mismatches and not unexpected and not missing_from_archive,
            "mismatches": mismatches,
            "content_manifest_sha256": combined.hexdigest()}


def official_loader_parity(rows: list[dict]) -> dict:
    view = RUN / "official_loader_view/testing"
    for directory in ("opt", "sar_warped", "truth_flow"):
        (view / directory).mkdir(parents=True, exist_ok=True)
    for row in rows:
        links = {
            view / "opt" / f"{row['pair_id']}_opt.png": row["source"],
            view / "sar_warped" / f"{row['pair_id']}_sar_warped.png": row["target"],
            view / "truth_flow" / f"{row['pair_id']}.flo": row["gt"],
        }
        for link, target in links.items():
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)

    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    from core.datasets import opt_sar_test
    dataset = opt_sar_test(split="testing", dstype="testing", root=str(view.parent))
    if len(dataset) != 700:
        raise RuntimeError(f"official loader expected 700 pairs, got {len(dataset)}")
    max_abs_difference = 0.0
    pair_ids = []
    for index, row in enumerate(rows):
        file_opt, official_source, official_target, _flow, _valid = dataset[index]
        direct_source = image_tensor(row["source"])[0]
        direct_target = image_tensor(row["target"])[0]
        pair_ids.append(Path(file_opt).name.removesuffix("_opt.png"))
        max_abs_difference = max(
            max_abs_difference,
            float((official_source - direct_source).abs().max()),
            float((official_target - direct_target).abs().max()),
        )
    expected_ids = [row["pair_id"] for row in rows]
    status = "PASS" if pair_ids == expected_ids and max_abs_difference == 0.0 else "FAIL"
    return {
        "status": status, "pairs": len(dataset), "view_root": str(view.parent),
        "input_kind": "official .png glob over symlinks to the audited TIFF bytes",
        "pair_ids_exact_order": pair_ids == expected_ids,
        "source_target_tensor_max_abs_difference": max_abs_difference,
        "tensor_shape": [3, 512, 512], "tensor_dtype": "torch.float32",
    }


def preflight() -> None:
    actual = {str(path): sha256(path) for path in EXPECTED}
    hash_mismatches = {str(path): {"expected": expected, "actual": actual[str(path)]}
                       for path, expected in EXPECTED.items() if actual[str(path)] != expected}
    provenance = archive_parity()
    rows = pair_rows(include_gt=True)
    missing = [str(row["gt"]) for row in rows if not row["gt"].is_file()]
    sample = image_tensor(rows[0]["source"])
    gt_checks = []
    if not missing:
        for row in rows:
            gt = read_flo(row["gt"])
            if gt.shape != (2, 512, 512) or not np.isfinite(gt).all():
                gt_checks.append({"pair_id": row["pair_id"], "shape": list(gt.shape)})
    loader_parity = official_loader_parity(rows) if not missing and not gt_checks else {"status": "NOT_RUN"}
    try:
        official_commit = subprocess.check_output(
            ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        official_commit = None
    status = "PASS" if (not hash_mismatches and provenance["all_match_archive"] and not missing
                         and not gt_checks and loader_parity["status"] == "PASS") else "FAIL"
    card = {
        "status": status, "last_completed_stage": "CPU_PREFLIGHT",
        "dataset": "WHU-OPT-SAR author-public single test group", "pairs": 700,
        "paper_protocol_scope": "one public test group; not the paper's three-seed mean+-std",
        "archive": str(ARCHIVE), "hashes": actual, "hash_mismatches": hash_mismatches,
        "archive_parity": provenance, "missing_gt": missing, "sample_tensor_shape": list(sample.shape),
        "invalid_gt": gt_checks, "official_loader_input_parity": loader_parity,
        "gt_format": "Middlebury .flo float32 [H,W,2]", "whole_image_no_mask": True,
        "main_table": [LABELS[0], LABELS[1]], "supplement_only": LABELS[2],
        "m4_identity": "OS E075A step80K only; WHU E081A forbidden",
        "official_source_commit": official_commit,
        "core_dependency_code_hashes": code_hashes(),
        "environment": {"hostname": platform.node(), "python": sys.version, "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "numpy": np.__version__, "pillow": Image.__version__,
                        "torch_num_threads": torch.get_num_threads()},
    }
    write_card("preflight", card)
    if status != "PASS":
        raise RuntimeError("WHU preflight failed")


def official_imports():
    sys.path.insert(0, str(OFFICIAL))
    from configs.default import get_cfg
    from core.GDROSnet.FeatureFormer.gdros import GDROS
    from core.LSRnet.LSmodel import LSR
    return get_cfg, GDROS, LSR


def load_gdros(device):
    get_cfg, GDROS, _ = official_imports()
    cfg = get_cfg(); cfg.defrost()
    for key, value in {"small": False, "mixed_precision": False, "alternate_corr": False,
                       "image_size": [512, 512], "corr_levels": 4, "corr_radius": 4,
                       "gpus": [0], "num_classes": [8], "Npoint": 200000}.items():
        setattr(cfg, key, value)
    model = GDROS(cfg[cfg.transformer]).to(device)
    raw = torch.load(GDROS_CKPT, map_location=device, weights_only=False)
    state = {key.removeprefix("module."): value for key, value in raw.items()}
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(str(loaded))
    return model.eval(), {"missing_keys": [], "unexpected_keys": [], "state_keys": len(state)}


def load_m4(device):
    sys.path.insert(0, str(TMP))
    import e075a_preserve_high_precision_recover_c2c5 as e075
    e074, _e072, _e073 = e075.add_paths()
    model = e075.PairFusion(e074, "region", device).to(device)
    payload = torch.load(M4_CKPT, map_location=device, weights_only=False)
    loaded = model.load_state_dict(payload["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(str(loaded))
    return model.eval(), {"missing_keys": [], "unexpected_keys": [], "state_keys": len(payload["model"]),
                          "checkpoint_step": payload.get("step")}


def prediction_path(attempt: str, label: str, index: int) -> Path:
    return RUN / "prediction_attempts" / attempt / label / f"{index:06d}.npy"


def predict(kind: str) -> None:
    torch.manual_seed(1234); np.random.seed(1234)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    if kind == "gdros":
        model, load_report = load_gdros(device); labels = (LABELS[0],)
    else:
        model, load_report = load_m4(device); labels = (LABELS[2], LABELS[1])
    _, _, LSR = official_imports()
    task = f"predict_{kind}"
    attempt = os.environ.get("WATCHDOG_JOB_ID", f"manual-{os.getpid()}")
    manifests = {label: RUN / "prediction_attempts" / attempt / label / "PREDICTIONS.jsonl" for label in labels}
    for manifest in manifests.values():
        manifest.parent.mkdir(parents=True, exist_ok=False)
    write_card(task, {"status": "RUNNING", "last_completed_stage": "STRICT_MODEL_LOAD",
                      "load_report": load_report, "prediction_attempt": attempt})
    handles = {label: path.open("a") for label, path in manifests.items()}
    try:
        with torch.inference_mode():
            for row in pair_rows(include_gt=False):
                source = image_tensor(row["source"]).to(device); target = image_tensor(row["target"]).to(device)
                if kind == "gdros":
                    _low, dense = model(source, target, iters=32, test_mode=True)
                    outputs = {LABELS[0]: LSR(1, 512, 512, dense, 200000)}
                else:
                    adapter = lambda flow: LSR(1, 512, 512, flow, 200000)
                    sequence = model(source, target, iters=32, lsr_fn=adapter)
                    outputs = {LABELS[2]: adapter(sequence[23]), LABELS[1]: adapter(sequence[31])}
                for label, tensor in outputs.items():
                    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                    if array.shape != (1, 2, 512, 512) or not np.isfinite(array).all():
                        raise RuntimeError(f"invalid prediction {label} {row['pair_id']}")
                    path = prediction_path(attempt, label, row["pair_index"]); np.save(path, array, allow_pickle=False)
                    record = {"pair_index": row["pair_index"], "pair_id": row["pair_id"],
                              "prediction": str(path), "sha256": sha256(path), "shape": list(array.shape),
                              "dtype": str(array.dtype), "uses_truth_flow": False, "post_lsr": True}
                    handles[label].write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n"); handles[label].flush()
                del source, target, outputs; torch.cuda.empty_cache()
    finally:
        for handle in handles.values(): handle.close()
    report = {}
    for label, manifest in manifests.items():
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        if len(records) != 700 or [r["pair_index"] for r in records] != list(range(700)):
            raise RuntimeError(f"incomplete manifest {label}")
        canonical = RUN / "predictions" / label / "PREDICTIONS.jsonl"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        temporary = canonical.with_suffix(f".{attempt}.tmp")
        temporary.write_bytes(manifest.read_bytes())
        os.replace(temporary, canonical)
        report[label] = {"path": str(canonical), "sha256": sha256(canonical), "records": len(records),
                         "immutable_attempt_manifest": str(manifest), "prediction_attempt": attempt}
    write_card(task, {"status": "COMPLETE", "last_completed_stage": "700_PREDICTIONS_MATERIALIZED",
                      "load_report": load_report, "manifests": report, "ground_truth_loaded": False,
                      "logical_device": "cuda:0", "prediction_attempt": attempt})


def read_flo(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) != 12: raise RuntimeError(f"short .flo header {path}")
        magic, width, height = struct.unpack("<fii", header)
        if magic != 202021.25: raise RuntimeError(f"bad flo {path}")
        data = np.fromfile(f, dtype="<f4", count=width * height * 2)
        if data.size != width * height * 2 or f.read(1):
            raise RuntimeError(f"invalid .flo payload length {path}")
    return data.reshape(height, width, 2).transpose(2, 0, 1)


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, np.float64)
    if array.size != 700 or not np.isfinite(array).all(): raise RuntimeError("invalid metrics")
    result = {"pairs": 700, "mean": float(array.mean()), "variance": float(array.var()),
              "std": float(array.std()), "p95": float(np.quantile(array, .95)), "p99": float(np.quantile(array, .99))}
    for threshold in THRESHOLDS:
        selected = array[array < threshold]
        result[f"cmr@{threshold}"] = float(selected.size / array.size * 100)
        result[f"failures_ge_{threshold}"] = int(array.size - selected.size)
        if threshold <= 5:
            result[f"success_subset@{threshold}"] = {
                "count": int(selected.size), "mean": float(selected.mean()) if selected.size else None,
                "variance_official_rmse_field": float(selected.var()) if selected.size else None,
                "std": float(selected.std()) if selected.size else None,
            }
    return result


def formula_fixture() -> dict:
    gt = torch.zeros((2, 2, 2)); pred = torch.stack((torch.full((2, 2), 3.), torch.full((2, 2), 4.)))
    official = float(torch.sum((pred.unsqueeze(0) - gt) ** 2, dim=0).sqrt().mean())
    public = float((pred - gt).abs().mean()); vector = float(torch.linalg.vector_norm(pred - gt, dim=0).mean())
    return {"public": public, "official": official, "vector": vector,
            "status": "PASS" if public == official == 3.5 and vector == 5.0 else "FAIL"}


def join() -> None:
    manifests = {}
    for label in LABELS:
        path = RUN / "predictions" / label / "PREDICTIONS.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) != 700: raise RuntimeError(f"manifest count {label}")
        for index, record in enumerate(records):
            pred_path = Path(record["prediction"]); array = np.load(pred_path, allow_pickle=False)
            if record["pair_index"] != index or sha256(pred_path) != record["sha256"]:
                raise RuntimeError(f"hash/index failure {label} {index}")
            if array.shape != (1, 2, 512, 512) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise RuntimeError(f"prediction integrity failure {label} {index}")
        manifests[label] = {record["pair_id"]: record for record in records}
    expected_ids = [row["pair_id"] for row in pair_rows(False)]
    if any(list(rows) != expected_ids for rows in manifests.values()): raise RuntimeError("pair ID mismatch")

    values = {label: {"public_component": [], "vector_epe": []} for label in LABELS}
    per_pair, parity_max, numpy_delta_max = [], 0.0, 0.0
    for row in pair_rows(True):
        gt = read_flo(row["gt"]).astype(np.float32, copy=False)
        out = {"pair_index": row["pair_index"], "pair_id": row["pair_id"]}
        for label in LABELS:
            pred = np.load(manifests[label][row["pair_id"]]["prediction"], allow_pickle=False)[0]
            diff = pred - gt
            public_numpy = float(np.mean((np.abs(diff[0]) + np.abs(diff[1])) / 2.0))
            vector = float(np.mean(np.sqrt(np.square(diff[0]) + np.square(diff[1]))))
            pt, gt_t = torch.from_numpy(pred), torch.from_numpy(gt)
            official = float(torch.sum((pt.unsqueeze(0) - gt_t) ** 2, dim=0).sqrt().mean())
            public_tensor = float((pt - gt_t).abs().mean())
            parity_max = max(parity_max, abs(official - public_tensor))
            numpy_delta_max = max(numpy_delta_max, abs(public_numpy - public_tensor))
            values[label]["public_component"].append(public_numpy); values[label]["vector_epe"].append(vector)
            out[f"{label}__public_component"] = public_numpy; out[f"{label}__vector_epe"] = vector
        per_pair.append(out)
    csv_path = RUN / "join/PER_PAIR_METRICS.csv"; csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_pair[0])); writer.writeheader(); writer.writerows(per_pair)
    metrics = {label: {metric: summarize(vals) for metric, vals in by_metric.items()} for label, by_metric in values.items()}
    fixture = formula_fixture(); parity_status = "PASS" if parity_max <= 1e-6 and fixture["status"] == "PASS" else "FAIL"
    status = "COMPLETE" if parity_status == "PASS" else "FORMULA_PARITY_FAILED"
    card = {"status": status, "last_completed_stage": "CPU_GT_JOIN_WHU700", "metrics": metrics,
            "main_table": [LABELS[0], LABELS[1]], "supplement_only": LABELS[2],
            "prediction_manifests_verified_before_gt_load": True, "whole_image_no_mask": True,
            "paper_protocol_scope": "single author-public group; no three-seed mean+-std claim",
            "public_formula_parity": {"status": parity_status, "max_abs_difference": parity_max,
                                      "tolerance": 1e-6, "numpy_reduction_max_abs_difference": numpy_delta_max,
                                      "fixture": fixture},
            "per_pair_csv": str(csv_path), "per_pair_csv_sha256": sha256(csv_path)}
    write_card("join", card); write_json(RUN / "RESULT_CARD.json", card)
    (RUN / "RESULT_CARD.md").write_text(
        f"# E092B WHU-700 final result\n\nStatus: `{status}`\n\n"
        f"```json\n{json.dumps(card, indent=2, sort_keys=True, allow_nan=False)}\n```\n"
    )
    if status != "COMPLETE":
        raise RuntimeError("WHU public formula parity failed")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--task", required=True, choices=("preflight", "predict_gdros", "predict_m4", "join")); args = parser.parse_args()
    if args.task in ("preflight", "join"):
        torch.set_num_threads(1)
    def interrupted(signum, _frame): raise InterruptedError(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        if args.task == "preflight": preflight()
        elif args.task == "predict_gdros": predict("gdros")
        elif args.task == "predict_m4": predict("m4")
        else: join()
    except BaseException as exc:
        write_card(args.task, {"status": "FAILED", "last_completed_stage": "EXCEPTION",
                               "failure_reason": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__": main()
