#!/usr/bin/env python3
"""E093A frozen B/C ablation, parity, OS-424 evaluation, and profiling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


ROOT = Path(os.environ.get("SPARSEGEO_ROOT", "/mnt/sda/zhushihao/VS_Match"))
TMP = ROOT / "image_matching_part2_migrated/.codex_tmp"
OFFICIAL = ROOT / "external_repos/E068A_GDROS_official"
DATASET = ROOT / "datasets/extracted/gdros_official_e003a_testing/testing"
ARCHIVE = ROOT / "datasets/official_archives/gdros/testing.zip"
RUN = ROOT / "GDROS/experiments/sparsegeo/runs/E-093A-OS-BC-FROZEN-ABLATION-AND-PROFILE"
M4_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt"
B_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-b-stage180k/checkpoints/B_step180000.pt"
C_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-c-stage180k/checkpoints/C_step180000.pt"
F2_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f2-warm120k/checkpoints/F2_step120000.pt"
F3_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f3-warm120k/checkpoints/F3_step120000.pt"
OLD_E092A = ROOT / "GDROS/experiments/sparsegeo/runs/E-092A-OS-OFFICIAL424-M4-GDROS-EVAL"

EXPECTED = {
    ARCHIVE: "e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218",
    M4_CKPT: "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b",
    B_CKPT: "f40d80f2dca62e74ec86c1823963282a490d47f336f2cfce4a28fc211ea33218",
    C_CKPT: "c3c67e6d65aca0dd55cf7af2c2a9eedccadadc0fec5ad11fe6e493e08ecc85b9",
    F2_CKPT: "4a52307fa4b7b67756ac979310fa37d34a131cc182a39e2c45d86711972dab06",
    F3_CKPT: "e5a9c612a7b9b85c568b88776ef784873d5df693cb570ad3dbbbf52fe4801ffd",
}
VARIANTS = ("b", "c", "avg", "m4")
ITERS = (24, 32)
THRESHOLDS = (1, 2, 3, 4, 5, 10, 16)
PARITY_INDICES = (0, 60, 120, 180, 240, 300, 360, 423)
PROFILE_INDICES = tuple(math.floor(k * 423 / 63) for k in range(64))
RAW_TOL = 1e-5
POST_TOL = 1e-4


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
    payload = dict(card, task=task, updated_at_unix=time.time())
    target = RUN / task
    write_json(target / "RESULT_CARD.json", payload)
    (target / "RESULT_CARD.md").write_text(
        f"# E093A {task}\n\nStatus: `{payload.get('status')}`\n\n"
        f"Last completed stage: `{payload.get('last_completed_stage')}`\n\n"
        f"```json\n{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n```\n"
    )
    attempt = os.environ.get("WATCHDOG_JOB_ID", f"manual-{os.getpid()}")
    attempt_dir = target / "attempts" / attempt
    write_json(attempt_dir / "RESULT_CARD.json", payload)
    (attempt_dir / "RESULT_CARD.md").write_text(
        f"# E093A {task} attempt {attempt}\n\nStatus: `{payload.get('status')}`\n"
    )


def write_exception_card(task: str, exc: BaseException) -> None:
    target = RUN / task / "RESULT_CARD.json"
    failure = {"status": "FAILED", "last_completed_stage": "EXCEPTION",
               "failure_reason": repr(exc), "traceback": traceback.format_exc()}
    if target.exists():
        existing = json.loads(target.read_text())
        if existing.get("status") not in (None, "RUNNING", "FAILED"):
            attempt = os.environ.get("WATCHDOG_JOB_ID", f"manual-{os.getpid()}")
            write_json(RUN / task / "attempts" / attempt / "EXCEPTION.json", failure)
            return
    write_card(task, failure)


def configure_backends() -> dict:
    torch.manual_seed(1234)
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {
        "fp32": True,
        "autocast": False,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def environment() -> dict:
    return {
        "hostname": platform.node(), "python": sys.version, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "numpy": np.__version__, "pillow": Image.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "watchdog_allocated_gpus": os.environ.get("WATCHDOG_ALLOCATED_GPUS"),
        "backend": configure_backends(), "batch_size": 1,
    }


def pairs(include_gt: bool = False) -> list[dict]:
    opt = sorted((DATASET / "opt").glob("*.png"))
    sar = sorted((DATASET / "sar_warped").glob("*.png"))
    if len(opt) != 424 or len(sar) != 424:
        raise RuntimeError(f"expected 424 pairs, got opt={len(opt)} sar={len(sar)}")
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


def read_flo(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = np.fromfile(handle, np.float32, count=1)
        width = np.fromfile(handle, np.int32, count=1)
        height = np.fromfile(handle, np.int32, count=1)
        if magic.size != 1 or float(magic[0]) != 202021.25:
            raise RuntimeError(f"invalid .flo: {path}")
        array = np.fromfile(handle, np.float32, count=int(width[0] * height[0] * 2))
    return array.reshape(int(height[0]), int(width[0]), 2).transpose(2, 0, 1)


def hash_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_dataset() -> dict:
    paths = []
    for directory, pattern in (("opt", "*.png"), ("sar_warped", "*.png"), ("truth_flow", "*.flo")):
        paths.extend(path.relative_to(DATASET.parent) for path in sorted((DATASET / directory).glob(pattern)))
    mismatches = []
    combined = hashlib.sha256()
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        for relative in paths:
            normalized = relative.as_posix()
            candidates = [name for name in names if name == normalized or name.endswith("/" + normalized)]
            if len(candidates) != 1:
                mismatches.append({"relative": normalized, "candidates": candidates})
                continue
            extracted = sha256(DATASET.parent / relative)
            with archive.open(candidates[0]) as handle:
                archived = hash_stream(handle)
            if extracted != archived:
                mismatches.append({"relative": normalized, "extracted": extracted, "archive": archived})
            combined.update(normalized.encode()); combined.update(extracted.encode())
    return {"files": len(paths), "all_match_archive": not mismatches, "mismatches": mismatches,
            "loader_content_manifest_sha256": combined.hexdigest()}


def add_model_paths():
    if str(TMP) not in sys.path:
        sys.path.insert(0, str(TMP))
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    import e075a_preserve_high_precision_recover_c2c5 as e075
    e074, e072, e073 = e075.add_paths()
    return e075, e074, e072, e073


def lsr(flow: torch.Tensor) -> torch.Tensor:
    from core.LSRnet.LSmodel import LSR
    return LSR(1, 512, 512, flow, 200000)


def checkpoint_state() -> tuple[dict, dict]:
    payload = torch.load(M4_CKPT, map_location="cpu", weights_only=False)
    state = payload["model"]
    return payload, state


def extract_state(state: dict, prefix: str) -> dict:
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise RuntimeError(f"no checkpoint state for prefix {prefix}")
    return selected


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def strict_load(module: nn.Module, state: dict, name: str) -> dict:
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{name} strict load failed: {result}")
    return {"name": name, "keys": len(state), "missing_keys": [], "unexpected_keys": [],
            "state_sha256": state_digest(state)}


def freeze(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def build_b(device, state):
    _e075, e074, _e072, e073 = add_model_paths()
    model = e074.E074Model(e073, "B", device).to(device)
    report = strict_load(model, extract_state(state, "b."), "m4.b")
    return freeze(model).eval(), report


def build_c_student(device, state):
    _e075, _e074, e072, e073 = add_model_paths()
    model = e073.E073FusionMatcher(e072, e072.cfg(), "F3", device).to(device)
    report = strict_load(model, extract_state(state, "c.student."), "m4.c.student")
    return freeze(model).eval(), report


def build_gate(device, state):
    e075, _e074, _e072, _e073 = add_model_paths()
    model = e075.ScalarOrRegionGate(16, mode="region").to(device)
    report = strict_load(model, extract_state(state, "gate."), "m4.gate")
    return freeze(model).eval(), report


class DeployBundle(nn.Module):
    def __init__(self, device, state, need_b=True, need_c=True, need_gate=False):
        super().__init__()
        self.reports = []
        self.b = None
        self.c_student = None
        self.gate = None
        if need_b:
            self.b, report = build_b(device, state); self.reports.append(report)
        if need_c:
            self.c_student, report = build_c_student(device, state); self.reports.append(report)
        if need_gate:
            self.gate, report = build_gate(device, state); self.reports.append(report)

    @staticmethod
    def gate_features(pb, pc):
        lb = lsr(pb).detach().to(pb.device)
        lc = lsr(pc).detach().to(pc.device)
        return torch.cat([pb.detach(), pc.detach(), (pb - pc).detach(), torch.abs(pb - pc).detach(),
                          lb, lc, torch.abs(lb - pb).detach(), torch.abs(lc - pc).detach()], dim=1)

    def branches(self, source, target, iters=32):
        pb = self.b(source, target, iters=iters) if self.b is not None else None
        pc = self.c_student(source, target, iters=iters, return_features=False) if self.c_student is not None else None
        return pb, pc

    def fuse_m4(self, pb, pc):
        outputs = []
        for xb, xc in zip(pb, pc):
            g = self.gate(self.gate_features(xb, xc))
            outputs.append((1 - g) * xb + g * xc)
        return outputs

    def forward_variant(self, variant, source, target, iters=32):
        if variant == "b":
            return self.b(source, target, iters=iters)
        if variant == "c":
            return self.c_student(source, target, iters=iters, return_features=False)
        pb, pc = self.branches(source, target, iters)
        if variant == "avg":
            return [0.5 * (xb + xc) for xb, xc in zip(pb, pc)]
        if variant == "m4":
            return self.fuse_m4(pb, pc)
        raise ValueError(variant)


def build_deploy_variant(variant, device, state):
    if variant == "b": return DeployBundle(device, state, need_b=True, need_c=False)
    if variant == "c": return DeployBundle(device, state, need_b=False, need_c=True)
    if variant == "avg": return DeployBundle(device, state, need_b=True, need_c=True)
    if variant == "m4": return DeployBundle(device, state, need_b=True, need_c=True, need_gate=True)
    raise ValueError(variant)


def build_original_m4(device, state):
    e075, e074, _e072, _e073 = add_model_paths()
    model = e075.PairFusion(e074, "region", device).to(device)
    report = strict_load(model, state, "original_m4")
    return freeze(model).eval(), report


def preflight() -> None:
    hashes = {str(path): sha256(path) for path in EXPECTED}
    mismatches = {str(path): {"expected": expected, "actual": hashes[str(path)]}
                  for path, expected in EXPECTED.items() if hashes[str(path)] != expected}
    rows = pairs(include_gt=True)
    missing_gt = [str(row["gt"]) for row in rows if not row["gt"].is_file()]
    provenance = verify_dataset()
    script_paths = [TMP / "e093a_bc_ablation_profile.py", TMP / "e075a_preserve_high_precision_recover_c2c5.py",
                    TMP / "e074a_f2_f3_twostage_fusion.py", TMP / "e073a_v1_v2_fusion_longtrain.py",
                    TMP / "e072a_layerwise_frontend_alignment.py", OFFICIAL / "core/LSRnet/LSmodel.py"]
    payload, state = checkpoint_state()
    prefixes = {prefix: len(extract_state(state, prefix)) for prefix in ("b.", "c.student.", "gate.")}
    status = "PASS" if not mismatches and not missing_gt and provenance["all_match_archive"] else "FAIL"
    card = {
        "status": status, "last_completed_stage": "CPU_PREFLIGHT", "experiment": "E093A",
        "protocol": {"pairs": 424, "shape": [512, 512], "batch": 1, "fp32": True,
                     "main_iters": 32, "supplement_iters": 24, "lsr_npoint": 200000,
                     "lsr_flat_slice": [10000, 210000], "whole_image_no_mask": True,
                     "prediction_before_gt_join": True, "profile_indices": list(PROFILE_INDICES),
                     "parity_indices": list(PARITY_INDICES)},
        "hashes": hashes, "hash_mismatches": mismatches, "missing_gt": missing_gt,
        "dataset_provenance": provenance, "pair_ids": [row["pair_id"] for row in rows],
        "checkpoint_step": payload.get("step"), "m4_state_keys": len(state),
        "effective_substate_key_counts": prefixes,
        "code_hashes": {str(path): sha256(path) for path in script_paths},
        "environment": environment(), "training": False, "feature_ablation": False,
    }
    write_card("preflight", card)
    if status != "PASS": raise RuntimeError("preflight failed")


class DiffAccumulator:
    def __init__(self):
        self.maximum = 0.0; self.total = 0.0; self.count = 0

    def add(self, a, b):
        difference = (a.float() - b.float()).abs()
        self.maximum = max(self.maximum, float(difference.max()))
        self.total += float(difference.double().sum()); self.count += difference.numel()

    def result(self):
        return {"max_abs_difference": self.maximum, "mean_abs_difference": self.total / self.count,
                "elements": self.count}


def tensor_diff(a, b):
    difference = (a.float() - b.float()).abs()
    return {"max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.double().mean()),
            "elements": difference.numel()}


def parity() -> None:
    configure_backends()
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    _payload, state = checkpoint_state()
    selected = [pairs(False)[index] for index in PARITY_INDICES]
    original, original_report = build_original_m4(device, state)
    reference = {}
    per_pair = []
    c_direct_stats = {f"iter{it}_{stage}": DiffAccumulator() for it in ITERS for stage in ("raw", "post")}
    with torch.inference_mode():
        for row in selected:
            source = image_tensor(row["source"]).to(device); target = image_tensor(row["target"]).to(device)
            m4_sequence = original(source, target, iters=32, lsr_fn=lsr)
            b_sequence = original.last_branch_preds["B"]
            c_sequence = original.last_branch_preds["C"]
            direct_c = original.c.student(source, target, iters=32, return_features=False)
            record = {}
            for it in ITERS:
                index = it - 1
                for name, sequence in (("b", b_sequence), ("c", c_sequence), ("m4", m4_sequence)):
                    record[f"{name}_iter{it}_raw"] = sequence[index].detach().cpu()
                    record[f"{name}_iter{it}_post"] = lsr(sequence[index]).detach().cpu()
                direct_raw = direct_c[index].cpu(); direct_post = lsr(direct_c[index]).cpu()
                c_direct_stats[f"iter{it}_raw"].add(c_sequence[index].cpu(), direct_raw)
                c_direct_stats[f"iter{it}_post"].add(record[f"c_iter{it}_post"], direct_post)
                per_pair.append({"pair_index": row["pair_index"], "pair_id": row["pair_id"],
                                 "comparison": "original_c_return_vs_original_c_student",
                                 "name": f"iter{it}_raw", **tensor_diff(c_sequence[index].cpu(), direct_raw)})
                per_pair.append({"pair_index": row["pair_index"], "pair_id": row["pair_id"],
                                 "comparison": "original_c_return_vs_original_c_student",
                                 "name": f"iter{it}_post", **tensor_diff(record[f"c_iter{it}_post"], direct_post)})
            reference[row["pair_index"]] = record
            del source, target, m4_sequence, b_sequence, c_sequence, direct_c
    del original; torch.cuda.empty_cache(); torch.cuda.synchronize()

    deploy = DeployBundle(device, state, need_b=True, need_c=True, need_gate=True).eval()
    deploy_stats = {f"{name}_iter{it}_{stage}": DiffAccumulator()
                    for name in ("b", "m4") for it in ITERS for stage in ("raw", "post")}
    c_wrapper_stats = {f"iter{it}_{stage}": DiffAccumulator() for it in ITERS for stage in ("raw", "post")}
    with torch.inference_mode():
        for row in selected:
            source = image_tensor(row["source"]).to(device); target = image_tensor(row["target"]).to(device)
            pb, pc = deploy.branches(source, target, 32); fused = deploy.fuse_m4(pb, pc)
            ref = reference[row["pair_index"]]
            for it in ITERS:
                index = it - 1
                for name, sequence in (("b", pb), ("m4", fused)):
                    raw = sequence[index].cpu(); post = lsr(sequence[index]).cpu()
                    deploy_stats[f"{name}_iter{it}_raw"].add(ref[f"{name}_iter{it}_raw"], raw)
                    deploy_stats[f"{name}_iter{it}_post"].add(ref[f"{name}_iter{it}_post"], post)
                    per_pair.append({"pair_index": row["pair_index"], "pair_id": row["pair_id"],
                                     "comparison": f"original_{name}_vs_deploy_wrapper",
                                     "name": f"iter{it}_raw", **tensor_diff(ref[f"{name}_iter{it}_raw"], raw)})
                    per_pair.append({"pair_index": row["pair_index"], "pair_id": row["pair_id"],
                                     "comparison": f"original_{name}_vs_deploy_wrapper",
                                     "name": f"iter{it}_post", **tensor_diff(ref[f"{name}_iter{it}_post"], post)})
                c_raw = pc[index].cpu(); c_post = lsr(pc[index]).cpu()
                c_wrapper_stats[f"iter{it}_raw"].add(ref[f"c_iter{it}_raw"], c_raw)
                c_wrapper_stats[f"iter{it}_post"].add(ref[f"c_iter{it}_post"], c_post)
                per_pair.append({"pair_index": row["pair_index"], "pair_id": row["pair_id"],
                                 "comparison": "original_c_return_vs_deploy_c_student",
                                 "name": f"iter{it}_raw", **tensor_diff(ref[f"c_iter{it}_raw"], c_raw)})
                per_pair.append({"pair_index": row["pair_index"], "pair_id": row["pair_id"],
                                 "comparison": "original_c_return_vs_deploy_c_student",
                                 "name": f"iter{it}_post", **tensor_diff(ref[f"c_iter{it}_post"], c_post)})
    comparisons = {
        "original_c_return_vs_original_c_student": {key: value.result() for key, value in c_direct_stats.items()},
        "original_c_return_vs_deploy_c_student": {key: value.result() for key, value in c_wrapper_stats.items()},
        "original_b_and_m4_vs_deploy_wrapper": {key: value.result() for key, value in deploy_stats.items()},
    }
    failures = []
    for group, values in comparisons.items():
        for name, result in values.items():
            tolerance = POST_TOL if name.endswith("post") else RAW_TOL
            if result["max_abs_difference"] > tolerance:
                failures.append({"group": group, "name": name, "tolerance": tolerance, **result})
    card = {"status": "PASS" if not failures else "FAIL", "last_completed_stage": "EIGHT_PAIR_PARITY",
            "indices": list(PARITY_INDICES), "raw_tolerance": RAW_TOL, "post_tolerance": POST_TOL,
            "comparisons": comparisons, "failures": failures, "original_load": original_report,
            "deployment_load": deploy.reports, "per_pair_comparisons": per_pair,
            "ground_truth_loaded": False, "environment": environment()}
    write_card("parity", card)
    if failures: raise RuntimeError(f"parity failed: {failures}")


def prediction_path(label, index): return RUN / "predictions" / label / f"{index:06d}.npy"


def predict() -> None:
    configure_backends()
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    _payload, state = checkpoint_state()
    model = DeployBundle(device, state, need_b=True, need_c=True, need_gate=True).eval()
    labels = [f"{variant}_iter{it}_post_lsr" for variant in VARIANTS for it in ITERS]
    manifests = {label: RUN / "predictions" / label / "PREDICTIONS.jsonl" for label in labels}
    handles = {}
    for label, manifest in manifests.items():
        manifest.parent.mkdir(parents=True, exist_ok=True); manifest.unlink(missing_ok=True)
        handles[label] = manifest.open("a")
    write_card("predict", {"status": "RUNNING", "last_completed_stage": "DEPLOYMENT_STRICT_LOAD",
                           "load_report": model.reports, "ground_truth_loaded": False})
    try:
        with torch.inference_mode():
            for row in pairs(False):
                source = image_tensor(row["source"]).to(device); target = image_tensor(row["target"]).to(device)
                pb, pc = model.branches(source, target, 32)
                sequences = {"b": pb, "c": pc, "avg": [0.5 * (xb + xc) for xb, xc in zip(pb, pc)],
                             "m4": model.fuse_m4(pb, pc)}
                for variant, sequence in sequences.items():
                    for it in ITERS:
                        label = f"{variant}_iter{it}_post_lsr"
                        array = lsr(sequence[it - 1]).detach().cpu().numpy().astype(np.float32, copy=False)
                        if array.shape != (1, 2, 512, 512): raise RuntimeError(f"bad shape {array.shape}")
                        path = prediction_path(label, row["pair_index"]); np.save(path, array, allow_pickle=False)
                        record = {"pair_index": row["pair_index"], "pair_id": row["pair_id"],
                                  "prediction": str(path), "sha256": sha256(path), "shape": list(array.shape),
                                  "dtype": str(array.dtype), "uses_truth_flow": False, "post_lsr": True,
                                  "variant": variant, "iters": it}
                        handles[label].write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n"); handles[label].flush()
                del source, target, pb, pc, sequences
    finally:
        for handle in handles.values(): handle.close()
    report = {}
    for label, manifest in manifests.items():
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        if len(records) != 424 or [x["pair_index"] for x in records] != list(range(424)):
            raise RuntimeError(f"incomplete manifest {label}: {len(records)}")
        report[label] = {"path": str(manifest), "sha256": sha256(manifest), "records": len(records)}
    write_card("predict", {"status": "COMPLETE", "last_completed_stage": "FOUR_VARIANTS_24_32_MATERIALIZED",
                           "load_report": model.reports, "manifests": report, "ground_truth_loaded": False,
                           "environment": environment()})


def summarize(values):
    array = np.asarray(values, np.float64)
    result = {"pairs": int(array.size), "mean": float(array.mean()), "variance": float(array.var()),
              "std": float(array.std()), "p95": float(np.quantile(array, .95)),
              "p99": float(np.quantile(array, .99))}
    for threshold in THRESHOLDS:
        subset = array[array < threshold]
        result[f"cmr@{threshold}"] = float(len(subset) / len(array) * 100)
        result[f"failures_ge_{threshold}"] = int(len(array) - len(subset))
        if threshold in (1, 2, 5):
            result[f"success_subset@{threshold}"] = {
                "count": int(len(subset)), "mean": float(subset.mean()) if len(subset) else None,
                "variance": float(subset.var()) if len(subset) else None,
                "std": float(subset.std()) if len(subset) else None}
    return result


def validated_manifests():
    labels = [f"{variant}_iter{it}_post_lsr" for variant in VARIANTS for it in ITERS]
    manifests = {}
    for label in labels:
        path = RUN / "predictions" / label / "PREDICTIONS.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) != 424: raise RuntimeError(f"{label}: {len(records)} records")
        for index, record in enumerate(records):
            pred = Path(record["prediction"])
            if record["pair_index"] != index or record["pair_id"] != pairs(False)[index]["pair_id"]:
                raise RuntimeError(f"identity mismatch {label} {index}")
            if record["uses_truth_flow"] or sha256(pred) != record["sha256"]:
                raise RuntimeError(f"hash/GT mismatch {label} {index}")
            array = np.load(pred, allow_pickle=False)
            if record.get("shape") != [1, 2, 512, 512] or array.shape != (1, 2, 512, 512):
                raise RuntimeError(f"shape mismatch {label} {index}: {array.shape}")
            if record.get("dtype") != "float32" or array.dtype != np.float32:
                raise RuntimeError(f"dtype mismatch {label} {index}: {array.dtype}")
            if not np.isfinite(array).all():
                raise RuntimeError(f"non-finite prediction {label} {index}")
        manifests[label] = records
    return manifests


def old_m4_regression(manifests):
    output = {}
    for it in ITERS:
        new = manifests[f"m4_iter{it}_post_lsr"]
        old_path = OLD_E092A / "predictions" / f"m4_iter{it}_post_lsr" / "PREDICTIONS.jsonl"
        old = [json.loads(line) for line in old_path.read_text().splitlines()]
        if len(old) != 424: raise RuntimeError(f"old E092A iter{it} incomplete")
        maximum = 0.0; total = 0.0; count = 0; byte_equal = 0; per_pair = []
        for a, b in zip(new, old):
            if a["pair_id"] != b["pair_id"] or a["pair_index"] != b["pair_index"]: raise RuntimeError("old identity mismatch")
            old_prediction = Path(b["prediction"])
            if sha256(old_prediction) != b["sha256"]:
                raise RuntimeError(f"old E092A prediction hash mismatch: iter{it} pair {b['pair_index']}")
            if a["sha256"] == b["sha256"]: byte_equal += 1
            x = np.load(a["prediction"], allow_pickle=False); y = np.load(old_prediction, allow_pickle=False)
            if y.shape != (1, 2, 512, 512) or y.dtype != np.float32 or not np.isfinite(y).all():
                raise RuntimeError(f"invalid old E092A prediction: iter{it} pair {b['pair_index']}")
            diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
            pair_max = float(diff.max()); pair_mean = float(diff.mean())
            maximum = max(maximum, pair_max); total += float(diff.sum()); count += diff.size
            per_pair.append({"pair_index": a["pair_index"], "pair_id": a["pair_id"],
                             "max_abs_difference": pair_max, "mean_abs_difference": pair_mean,
                             "byte_identical": a["sha256"] == b["sha256"]})
        maximum_pair = max(per_pair, key=lambda row: row["max_abs_difference"])
        output[f"iter{it}"] = {"max_abs_difference": maximum, "mean_abs_difference": total / count,
                                "byte_identical_predictions": byte_equal, "pairs": 424,
                                "old_manifest_sha256": sha256(old_path),
                                "new_manifest_sha256": sha256(RUN / "predictions" / f"m4_iter{it}_post_lsr" / "PREDICTIONS.jsonl"),
                                "old_prediction_hashes_verified": True,
                                "maximum_difference_pair": maximum_pair, "per_pair": per_pair}
    return output


def historical_context() -> dict:
    old_preflight = json.loads((OLD_E092A / "preflight/RESULT_CARD.json").read_text())
    new_preflight = json.loads((RUN / "preflight/RESULT_CARD.json").read_text())
    old_join = json.loads((OLD_E092A / "join/RESULT_CARD.json").read_text())
    shared_paths = [TMP / "e075a_preserve_high_precision_recover_c2c5.py",
                    TMP / "e074a_f2_f3_twostage_fusion.py", TMP / "e073a_v1_v2_fusion_longtrain.py",
                    TMP / "e072a_layerwise_frontend_alignment.py", OFFICIAL / "core/LSRnet/LSmodel.py"]
    return {
        "e092a_environment": old_preflight.get("environment"),
        "e093a_environment": new_preflight.get("environment"),
        "backend_difference": "E092A did not record or explicitly set TF32/cuDNN benchmark flags; E093A explicitly disables matmul TF32, cuDNN TF32, and cuDNN benchmark",
        "checkpoint_sha256_equal": old_preflight["hashes"].get(str(M4_CKPT)) == new_preflight["hashes"].get(str(M4_CKPT)),
        "checkpoint_sha256": sha256(M4_CKPT),
        "shared_dependency_sha256_now": {str(path): sha256(path) for path in shared_paths},
        "e092a_dependency_sha256": old_preflight.get("core_dependency_code_hashes"),
        "e093a_dependency_sha256": new_preflight.get("code_hashes"),
        "e092a_join_status": old_join.get("status"),
        "invocation_difference": "E092A loaded the original full PairFusion and evaluated pairs sequentially; E093A predict loaded the clean deployment bundle and evaluated B/C once before AVG/M4 fusion",
    }


def join(task_name: str = "join", enforce_historical_gate: bool = True) -> None:
    manifests = validated_manifests()
    regression = old_m4_regression(manifests)
    # GT is first opened only after all eight prediction manifests and hashes pass.
    rows = pairs(True)
    values = {label: {"public_component": [], "vector_epe": []} for label in manifests}
    output_rows = []
    for index, row in enumerate(rows):
        gt = read_flo(row["gt"]).astype(np.float64)
        output = {"pair_index": index, "pair_id": row["pair_id"]}
        for label, records in manifests.items():
            prediction = np.load(records[index]["prediction"], allow_pickle=False)[0].astype(np.float64)
            difference = prediction - gt
            public = float(np.mean((np.abs(difference[0]) + np.abs(difference[1])) / 2.0))
            vector = float(np.mean(np.sqrt(np.square(difference[0]) + np.square(difference[1]))))
            values[label]["public_component"].append(public); values[label]["vector_epe"].append(vector)
            output[f"{label}__public_component"] = public; output[f"{label}__vector_epe"] = vector
        output_rows.append(output)
    csv_path = RUN / task_name / "PER_PAIR_METRICS.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    metrics = {label: {kind: summarize(items) for kind, items in kinds.items()} for label, kinds in values.items()}
    cross = {}
    for it in ITERS:
        for kind in ("public_component", "vector_epe"):
            b = np.asarray(values[f"b_iter{it}_post_lsr"][kind]); c = np.asarray(values[f"c_iter{it}_post_lsr"][kind])
            entry = {"wins": {"b": int((b < c).sum()), "c": int((c < b).sum()), "ties": int((b == c).sum())},
                     "delta_c_minus_b": summarize((c - b).tolist()), "success_cross_tables": {}}
            for threshold in (1, 2, 5):
                bs, cs = b < threshold, c < threshold
                entry["success_cross_tables"][str(threshold)] = {
                    "both_success": int((bs & cs).sum()), "b_only": int((bs & ~cs).sum()),
                    "c_only": int((~bs & cs).sum()), "both_fail": int((~bs & ~cs).sum())}
            cross[f"iter{it}_{kind}"] = entry
    parity_max = max(value["max_abs_difference"] for value in regression.values())
    historical_match = parity_max <= POST_TOL
    card = {"status": "COMPLETE" if historical_match else "ACCURACY_COMPLETE_HISTORICAL_REPLAY_MISMATCH",
            "last_completed_stage": "CPU_GT_JOIN_424", "metrics": metrics,
            "main_table": [f"{variant}_iter32_post_lsr" for variant in VARIANTS],
            "supplement_only": [f"{variant}_iter24_post_lsr" for variant in VARIANTS],
            "per_pair_csv": str(csv_path), "per_pair_csv_sha256": sha256(csv_path),
            "prediction_manifests_verified_before_gt_load": True, "strict_thresholds": True,
            "whole_image_no_mask": True, "b_c_diagnostics": cross,
            "e092a_original_m4_prediction_regression": regression, "regression_tolerance": POST_TOL,
            "historical_regression_matches": historical_match, "historical_context": historical_context(),
            "accuracy_acceptance": "VALID_CURRENT_RUN"}
    write_card(task_name, card)
    if enforce_historical_gate and not historical_match:
        card["status"] = "REGRESSION_MISMATCH"
        write_card(task_name, card)
        raise RuntimeError("E092A M4 regression mismatch")


def historical_replay() -> None:
    join_card = json.loads((RUN / "join_v2/RESULT_CARD.json").read_text())
    selected_indices = sorted({entry["maximum_difference_pair"]["pair_index"]
                               for entry in join_card["e092a_original_m4_prediction_regression"].values()})
    torch.manual_seed(1234); np.random.seed(1234)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(1234)
    backend = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "source": "fresh-process PyTorch defaults matching E092A, which made no explicit backend assignments",
    }
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    _payload, state = checkpoint_state()
    selected_rows = [pairs(False)[index] for index in selected_indices]
    original, original_report = build_original_m4(device, state)
    original_outputs = {}
    with torch.inference_mode():
        for row in selected_rows:
            source = image_tensor(row["source"]).to(device); target = image_tensor(row["target"]).to(device)
            sequence = original(source, target, iters=32, lsr_fn=lsr)
            original_outputs[row["pair_index"]] = {
                it: {"raw": sequence[it - 1].detach().cpu(), "post": lsr(sequence[it - 1]).detach().cpu()}
                for it in ITERS}
    del original; torch.cuda.empty_cache(); torch.cuda.synchronize()
    deploy = build_deploy_variant("m4", device, state).eval()
    manifests = validated_manifests()
    comparisons = []
    with torch.inference_mode():
        for row in selected_rows:
            source = image_tensor(row["source"]).to(device); target = image_tensor(row["target"]).to(device)
            sequence = deploy.forward_variant("m4", source, target, iters=32)
            for it in ITERS:
                raw = sequence[it - 1].detach().cpu(); post = lsr(sequence[it - 1]).detach().cpu()
                old_record = json.loads((OLD_E092A / "predictions" / f"m4_iter{it}_post_lsr" /
                                         "PREDICTIONS.jsonl").read_text().splitlines()[row["pair_index"]])
                new_record = manifests[f"m4_iter{it}_post_lsr"][row["pair_index"]]
                old_post = torch.from_numpy(np.load(old_record["prediction"], allow_pickle=False))
                new_post = torch.from_numpy(np.load(new_record["prediction"], allow_pickle=False))
                comparisons.append({
                    "pair_index": row["pair_index"], "pair_id": row["pair_id"], "iters": it,
                    "original_vs_deployment_raw": tensor_diff(original_outputs[row["pair_index"]][it]["raw"], raw),
                    "original_vs_deployment_post": tensor_diff(original_outputs[row["pair_index"]][it]["post"], post),
                    "historical_backend_original_vs_e092a_post": tensor_diff(original_outputs[row["pair_index"]][it]["post"], old_post),
                    "historical_backend_original_vs_e093a_post": tensor_diff(original_outputs[row["pair_index"]][it]["post"], new_post),
                    "historical_backend_deployment_vs_e092a_post": tensor_diff(post, old_post),
                    "historical_backend_deployment_vs_e093a_post": tensor_diff(post, new_post),
                })
    deployment_equivalent = all(row["original_vs_deployment_raw"]["max_abs_difference"] <= RAW_TOL and
                                row["original_vs_deployment_post"]["max_abs_difference"] <= POST_TOL
                                for row in comparisons)
    reproduces_e092a = all(row["historical_backend_original_vs_e092a_post"]["max_abs_difference"] <= POST_TOL
                          for row in comparisons)
    status = "ROOT_CAUSE_REPRODUCED_HISTORICAL_BACKEND" if reproduces_e092a else "DIAGNOSTIC_COMPLETE_UNRESOLVED"
    card = {"status": status, "last_completed_stage": "NO_GT_MAX_DIFFERENCE_PAIR_REPLAY",
            "selected_without_gt": True, "selection_source": "maximum E092A/E093A prediction difference from join_v2",
            "indices": selected_indices, "backend": backend, "original_load": original_report,
            "deployment_load": deploy.reports, "deployment_equivalent": deployment_equivalent,
            "reproduces_e092a_within_tolerance": reproduces_e092a, "raw_tolerance": RAW_TOL,
            "post_tolerance": POST_TOL, "comparisons": comparisons, "ground_truth_loaded": False}
    write_card("historical_replay", card)
    if not deployment_equivalent: raise RuntimeError("original/deployment mismatch under historical backend")


def record_profile_cancellations() -> None:
    for task in ("profile_b", "profile_c", "profile_avg", "profile_m4", "profile_original_m4"):
        write_card(task, {"status": "CANCELLED_BY_USER", "last_completed_stage": "NOT_STARTED",
                          "latency_measured": False, "memory_measured": False,
                          "synthetic_values_used": False, "preserve_existing_artifacts": True,
                          "reason": "User cancelled all E093A timing and GPU-memory profiling"})


def smi_snapshot():
    fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm"
    try:
        output = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True)
        process_fields = "gpu_uuid,pid,process_name,used_memory"
        process_output = subprocess.check_output(
            ["nvidia-smi", f"--query-compute-apps={process_fields}", "--format=csv,noheader,nounits"], text=True)
        processes = []
        for line in process_output.splitlines():
            if not line.strip(): continue
            gpu_uuid, pid, name, memory = [part.strip() for part in line.split(",", 3)]
            processes.append({"gpu_uuid": gpu_uuid, "pid": int(pid), "process_name": name,
                              "used_memory_mib": float(memory)})
        return {"query": fields, "rows": [line.strip() for line in output.splitlines()],
                "process_query": process_fields, "compute_processes": processes}
    except Exception as exc:
        return {"error": repr(exc)}


def selected_device_identity():
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) != 1:
        raise RuntimeError(f"expected one physical CUDA_VISIBLE_DEVICES entry, got {visible}")
    fields = "index,uuid,name,memory.total"
    output = subprocess.check_output(
        ["nvidia-smi", "--id", visible[0], f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True).strip()
    index, uuid, name, total = [part.strip() for part in output.split(",", 3)]
    return {"cuda_visible_devices": visible[0], "physical_index": int(index), "uuid": uuid,
            "name": name, "total_memory_mib": float(total), "logical_device": "cuda:0"}


def external_compute_processes(snapshot, selected):
    if "error" in snapshot:
        raise RuntimeError(f"nvidia-smi snapshot unavailable: {snapshot['error']}")
    return [process for process in snapshot["compute_processes"]
            if process["gpu_uuid"] == selected["uuid"] and process["pid"] != os.getpid()]


def memory_stats(device):
    mib = 1024 ** 2
    return {"allocated_mib": torch.cuda.memory_allocated(device) / mib,
            "reserved_mib": torch.cuda.memory_reserved(device) / mib,
            "max_allocated_mib": torch.cuda.max_memory_allocated(device) / mib,
            "max_reserved_mib": torch.cuda.max_memory_reserved(device) / mib}


def parameter_report(model):
    return {"registered_parameters": sum(p.numel() for p in model.parameters()),
            "requires_grad_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "registered_tensors": len(list(model.parameters())),
            "requires_grad_tensors": sum(1 for p in model.parameters() if p.requires_grad)}


def profile(variant: str) -> None:
    configure_backends()
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    selected_device = selected_device_identity()
    before = memory_stats(device); before_smi = smi_snapshot()
    _payload, state = checkpoint_state()
    if variant == "original_m4":
        model, report = build_original_m4(device, state); load_report = [report]
    else:
        model = build_deploy_variant(variant, device, state).eval(); load_report = model.reports
    torch.cuda.synchronize(); after_load = memory_stats(device)
    initialization_peak = {"max_allocated_mib": after_load["max_allocated_mib"],
                           "max_reserved_mib": after_load["max_reserved_mib"]}
    selected = [pairs(False)[index] for index in PROFILE_INDICES]
    inputs = [(image_tensor(row["source"]).to(device), image_tensor(row["target"]).to(device)) for row in selected]
    torch.cuda.synchronize(); after_preload = memory_stats(device)

    def execute(source, target):
        if variant == "original_m4": sequence = model(source, target, iters=32, lsr_fn=lsr)
        else: sequence = model.forward_variant(variant, source, target, iters=32)
        output = lsr(sequence[31]).detach().cpu().float()
        if tuple(output.shape) != (1, 2, 512, 512): raise RuntimeError(f"bad profile output {output.shape}")
        return output

    with torch.inference_mode():
        for index in range(10):
            output = execute(*inputs[index]); del output
    torch.cuda.synchronize(); after_warmup = memory_stats(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    middle_smi = None
    with torch.inference_mode():
        for repeat in range(3):
            for order, (row, input_pair) in enumerate(zip(selected, inputs)):
                torch.cuda.synchronize(); started = time.perf_counter()
                output = execute(*input_pair)
                torch.cuda.synchronize(); elapsed_ms = (time.perf_counter() - started) * 1000.0
                timings.append({"repeat": repeat, "order": order, "pair_index": row["pair_index"],
                                "pair_id": row["pair_id"], "elapsed_ms": elapsed_ms})
                del output
            if repeat == 1:
                middle_smi = smi_snapshot()
    torch.cuda.synchronize(); after_measurement = memory_stats(device); after_smi = smi_snapshot()
    csv_path = RUN / f"profile_{variant}/PER_SAMPLE_TIMINGS.csv"; csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0])); writer.writeheader(); writer.writerows(timings)
    ms = [row["elapsed_ms"] for row in timings]
    summary = {"samples": len(ms), "warmup_samples": 10, "repeats": 3, "pairs_per_repeat": 64,
               "median_ms_per_pair": statistics.median(ms), "mean_ms_per_pair": statistics.fmean(ms),
               "p95_ms_per_pair": float(np.quantile(np.asarray(ms), .95)), "std_ms_per_pair": statistics.pstdev(ms)}
    steady = after_warmup
    memory = {"before_model_load": before, "after_model_load": after_load, "after_input_preload": after_preload,
              "after_warmup_steady_baseline": steady, "measurement_peak_absolute": after_measurement,
              "measurement_peak_increment_over_steady_mib": {
                  "allocated": after_measurement["max_allocated_mib"] - steady["allocated_mib"],
                  "reserved": after_measurement["max_reserved_mib"] - steady["reserved_mib"]},
              "initialization_peak": initialization_peak}
    snapshots = {"start": before_smi, "middle": middle_smi, "end": after_smi}
    interference = {name: external_compute_processes(snapshot, selected_device)
                    for name, snapshot in snapshots.items()}
    interfered = {name: processes for name, processes in interference.items() if processes}
    if interfered:
        write_json(RUN / f"profile_{variant}/INTERFERENCE.json",
                   {"status": "INVALID_ATTEMPT_EXTERNAL_COMPUTE", "selected_device": selected_device,
                    "our_pid": os.getpid(), "external_processes": interfered, "snapshots": snapshots})
        raise RuntimeError(f"external compute process on profiled GPU: {interfered}")
    dependency_paths = [TMP / "e093a_bc_ablation_profile.py", TMP / "e075a_preserve_high_precision_recover_c2c5.py",
                        TMP / "e074a_f2_f3_twostage_fusion.py", TMP / "e073a_v1_v2_fusion_longtrain.py",
                        TMP / "e072a_layerwise_frontend_alignment.py", OFFICIAL / "core/LSRnet/LSmodel.py"]
    write_card(f"profile_{variant}", {"status": "COMPLETE", "last_completed_stage": "INDEPENDENT_192_SAMPLE_PROFILE",
               "variant": variant, "effective_deployment": variant != "original_m4",
               "original_teacher_redundancy": variant == "original_m4", "indices": list(PROFILE_INDICES),
               "timing_scope": "model compute including internal CPU transfers/synchronization and terminal official LSR; excludes disk decode, H2D input, output archival, and GT join",
               "summary": summary, "memory": memory, "parameters": parameter_report(model),
               "load_report": load_report, "timings_csv": str(csv_path), "timings_csv_sha256": sha256(csv_path),
               "selected_device": selected_device, "our_pid": os.getpid(), "interference_checks": interference,
               "smi_start": before_smi, "smi_middle": middle_smi, "smi_end": after_smi,
               "timing_source_sha256": sha256(TMP / "e093a_bc_ablation_profile.py"),
               "dependency_sha256": {str(path): sha256(path) for path in dependency_paths},
               "environment": environment()})


def aggregate() -> None:
    join_card = json.loads((RUN / "join_v2/RESULT_CARD.json").read_text())
    parity_card = json.loads((RUN / "parity/RESULT_CARD.json").read_text())
    historical_card = json.loads((RUN / "historical_replay/RESULT_CARD.json").read_text())
    accepted_join_statuses = {"COMPLETE", "ACCURACY_COMPLETE_HISTORICAL_REPLAY_MISMATCH"}
    if join_card["status"] not in accepted_join_statuses or parity_card["status"] != "PASS":
        raise RuntimeError("upstream accuracy or deployment parity incomplete")
    historical_resolved = historical_card["status"] == "ROOT_CAUSE_REPRODUCED_HISTORICAL_BACKEND"
    card = {"status": "COMPLETE" if historical_resolved else "RESULTS_COMPLETE_HISTORICAL_REPLAY_UNRESOLVED",
            "last_completed_stage": "E093A_AGGREGATED",
            "metrics": join_card["metrics"], "main_table": join_card["main_table"],
            "supplement_only": join_card["supplement_only"],
            "performance_evaluation": {"status": "CANCELLED_BY_USER", "latency_measured": False,
                                       "memory_measured": False, "synthetic_values_used": False,
                                       "cancelled_tasks": ["profile_b", "profile_c", "profile_avg",
                                                           "profile_m4", "profile_original_m4"]},
            "parity": {"status": parity_card["status"], "indices": parity_card["indices"],
                       "raw_tolerance": parity_card["raw_tolerance"], "post_tolerance": parity_card["post_tolerance"]},
            "per_pair_csv": join_card["per_pair_csv"], "per_pair_csv_sha256": join_card["per_pair_csv_sha256"],
            "b_c_diagnostics": join_card["b_c_diagnostics"],
            "e092a_original_m4_prediction_regression": join_card["e092a_original_m4_prediction_regression"],
            "historical_replay": historical_card,
            "current_run_accuracy_valid": True, "deployment_equivalence_verified": True,
            "cross_experiment_historical_replay_resolved": historical_resolved,
            "training": False, "feature_ablation": False}
    write_json(RUN / "RESULT_CARD.json", card)
    (RUN / "RESULT_CARD.md").write_text(
        f"# E093A final result\n\nStatus: `{card['status']}`\n\n"
        f"```json\n{json.dumps(card, indent=2, sort_keys=True, allow_nan=False)}\n```\n")
    write_card("aggregate", card)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=("preflight", "parity", "predict", "join", "join_v2", "historical_replay",
                                 "record_profile_cancellations", "profile", "aggregate"))
    parser.add_argument("--variant", choices=(*VARIANTS, "original_m4"))
    args = parser.parse_args()
    task_name = f"profile_{args.variant}" if args.task == "profile" else args.task

    def interrupted(signum, _frame): raise InterruptedError(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        if args.task == "preflight": preflight()
        elif args.task == "parity": parity()
        elif args.task == "predict": predict()
        elif args.task == "join": join()
        elif args.task == "join_v2": join("join_v2", enforce_historical_gate=False)
        elif args.task == "historical_replay": historical_replay()
        elif args.task == "record_profile_cancellations": record_profile_cancellations()
        elif args.task == "profile":
            if args.variant is None: raise ValueError("--variant required for profile")
            profile(args.variant)
        else: aggregate()
    except BaseException as exc:
        write_exception_card(task_name, exc)
        raise


if __name__ == "__main__": main()
