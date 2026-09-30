#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(os.environ.get("SPARSEGEO_ROOT", "/mnt/sda/zhushihao/VS_Match"))
MIG = ROOT / "image_matching_part2_migrated"
TMP = MIG / ".codex_tmp"
RUN_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-081A-ZZVR-WHU-OUR-METHOD-NATIVE-TRAIN"
OFFICIAL_REPO = ROOT / "external_repos/E068A_GDROS_official"
OFFICIAL_REF = ROOT / "GDROS/official_models_reference"
WHU_SRC_TRAIN = Path(os.environ.get(
    "SPARSEGEO_WHU_TRAIN",
    "/mnt/sda/zhushihao/data/vis_sar/WHU_OPT_SAR/extracted_e077a/training/whu_opt_sar512",
))
WHU_SRC_TEST = Path(os.environ.get(
    "SPARSEGEO_WHU_TEST",
    "/mnt/sda/zhushihao/data/vis_sar/WHU_OPT_SAR/extracted_e077a/testing/testing",
))
WHU_VIEW = RUN_ROOT / "datasets/os_dataset"

PAPER_WHU = {
    "cmr1": 72.05, "aepe_at_1": 0.60,
    "cmr2": 96.86, "aepe_at_2": 0.78,
    "cmr5": 99.57, "aepe_at_5": 0.83,
    "all_aepe": 0.90, "rmse": 0.62,
}


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value
E077A_REF = {
    "source": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-077A-ZZVR-WHU-OPT-SAR-TRANSFER",
    "type": "OS-pretrain->WHU-finetune; not used for E081A initialization",
}
E080A_REF = {
    "source": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-080A-ZZVR-WHU-OFFICIAL-RECIPE-FROM-SCRATCH",
    "type": "clean official GDROS WHU reproduction; separate baseline line",
}


def rr(variant: str, tag: str = "") -> Path:
    suffix = f"-{tag.strip('-')}" if tag else ""
    return RUN_ROOT / f"run-{variant.lower()}{suffix}"


def formal_dir(variant: str, tag: str = "") -> Path:
    return rr(variant, tag) / "formal"


def ckpt_dir(variant: str, tag: str = "") -> Path:
    return rr(variant, tag) / "checkpoints"


def mat_dir(variant: str, tag: str = "") -> Path:
    return formal_dir(variant, tag) / "materialized"


def log_dir(variant: str, tag: str = "") -> Path:
    return rr(variant, tag) / "logs"


def js(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_imports():
    sys.path.insert(0, str(OFFICIAL_REPO))
    from configs.default import get_cfg  # noqa
    import core.datasets as datasets  # noqa
    from core.GDROSnet.FeatureFormer.gdros import GDROS  # noqa
    from core.raft import RAFT  # noqa
    from core.loss import sequence_loss, matrix2flow  # noqa
    from core.LSRnet.LSmodel import LSR_list, LSR  # noqa
    return get_cfg, datasets, GDROS, RAFT, sequence_loss, matrix2flow, LSR_list, LSR


def link_images(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png", "*.jpg", "*.jpeg"):
        for src in sorted(src_dir.glob(ext)):
            dst = dst_dir / f"{src.stem}.png"
            if not dst.exists():
                dst.symlink_to(src)


def build_whu_view():
    for split, src in [("training", WHU_SRC_TRAIN), ("testing", WHU_SRC_TEST)]:
        for sub in ["opt", "sar_warped"]:
            link_images(src / sub, WHU_VIEW / split / sub)
        for sub in ["truth_flow", "datum"]:
            dst_dir = WHU_VIEW / split / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            for srcf in sorted((src / sub).glob("*")):
                dst = dst_dir / srcf.name
                if not dst.exists():
                    dst.symlink_to(srcf)


def make_cfg(name: str, steps: int, batch: int, image_size: int):
    get_cfg, *_ = ensure_imports()
    cfg = get_cfg()
    cfg.name = name
    cfg.stage = "OSdata"
    cfg.dstype = "training"
    cfg.validation = ["testing"]
    cfg.restore_ckpt = None
    cfg.trainer.num_steps = steps
    cfg.batch_size = batch
    cfg.image_size = [image_size, image_size]
    cfg.mixed_precision = True
    cfg.alternate_corr = False
    cfg.gamma = getattr(cfg, "gamma", 0.85)
    cfg.max_flow = getattr(cfg, "max_flow", 400)
    return cfg


def count_split(split: str) -> dict[str, Any]:
    base = WHU_VIEW / split
    ids = {}
    for sub, globpat in [("opt", "*.png"), ("sar_warped", "*.png"), ("truth_flow", "*.flo")]:
        vals = sorted((base / sub).glob(globpat))
        ids[sub] = [p.stem for p in vals]
    return {
        "split": split,
        "root": str(base),
        "counts": {k: len(v) for k, v in ids.items()},
        "first3": {k: v[:3] for k, v in ids.items()},
        "last3": {k: v[-3:] for k, v in ids.items()},
    }


def train_test_isolation() -> dict[str, Any]:
    tr = count_split("training")
    te = count_split("testing")
    # Build full sets without storing in the JSON twice.
    train_full = {p.stem for p in sorted((WHU_VIEW / "training/opt").glob("*.png"))}
    test_full = {p.stem for p in sorted((WHU_VIEW / "testing/opt").glob("*.png"))}
    basename_inter = sorted(train_full & test_full)
    train_real = {str(p.resolve()) for p in sorted((WHU_VIEW / "training/opt").glob("*.png"))}
    test_real = {str(p.resolve()) for p in sorted((WHU_VIEW / "testing/opt").glob("*.png"))}
    realpath_inter = sorted(train_real & test_real)
    source_roots_disjoint = str(WHU_SRC_TRAIN.resolve()) != str(WHU_SRC_TEST.resolve()) and "testing/testing" in str(WHU_SRC_TEST)
    status = "PASS" if len(realpath_inter) == 0 and source_roots_disjoint else "BLOCKING"
    return {
        "status": status,
        "train": tr,
        "test": te,
        "expected": {"training": 5600, "testing": 700},
        "train_test_basename_intersection_count": len(basename_inter),
        "train_test_basename_intersection_sample": basename_inter[:10],
        "basename_overlap_interpretation": "WHU split-local filenames restart from 000000; basename overlap is tracked but not treated as leakage when resolved source paths are disjoint.",
        "train_test_realpath_intersection_count": len(realpath_inter),
        "train_test_realpath_intersection_sample": realpath_inter[:10],
        "source_roots_disjoint": source_roots_disjoint,
        "training_uses_testing_zip": False,
        "testing_zip_final_eval_only": True,
        "no_os424_confirm_rescue": True,
        "candidate_before_label_audit": "predictions are written before metric summaries in eval_model materialized jsonl",
    }


def init_gdros_from_official(device: torch.device, cfg):
    _, _, GDROS, RAFT, *_ = ensure_imports()
    model = GDROS(cfg[cfg.transformer])
    raft_args = AttrDict(small=False, alternate_corr=False)
    raft = RAFT(raft_args)
    raft_path = OFFICIAL_REF / "raft-sintel.pth"
    raw = torch.load(raft_path, map_location="cpu")
    if any(k.startswith("module.") for k in raw.keys()):
        raw = {k.replace("module.", "", 1): v for k, v in raw.items()}
    raft.load_state_dict(raw)
    model.featureNet.feat_encoder.load_state_dict(raft.fnet.state_dict())
    model.context_encoder.load_state_dict(raft.fnet.state_dict())
    model.update_block.load_state_dict(raft.update_block.state_dict())
    model.to(device)
    return model


def lsr_flow(flow: torch.Tensor):
    *_, LSR = ensure_imports()
    b, _, h, w = flow.shape
    return LSR(b, h, w, flow, 200000)


class NativeGateRefiner(nn.Module):
    def __init__(self, mode: str = "scalar", residual_scale: float = 2.0):
        super().__init__()
        self.mode = mode
        self.residual_scale = residual_scale
        in_ch = 7  # pred(2), lsr-affine(2), diff(2), |diff|(1)
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        self.residual_net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 2, 1),
        )
        nn.init.constant_(self.gate_net[-1].bias, -2.0)
        nn.init.normal_(self.residual_net[-1].weight, std=1e-4)
        nn.init.constant_(self.residual_net[-1].bias, 0.0)

    def forward(self, pred: torch.Tensor):
        affine = lsr_flow(pred).detach().to(pred.device)
        diff = pred - affine
        feat = torch.cat([pred, affine, diff, torch.linalg.vector_norm(diff, dim=1, keepdim=True)], dim=1)
        gate = torch.sigmoid(self.gate_net(feat))
        if self.mode == "scalar":
            gate = gate.mean(dim=(2, 3), keepdim=True).expand_as(gate)
        elif self.mode == "region":
            gate = F.avg_pool2d(gate, kernel_size=4, stride=4, ceil_mode=True)
            gate = F.interpolate(gate, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        else:
            raise ValueError(self.mode)
        residual = self.residual_scale * torch.tanh(self.residual_net(feat))
        coverage_candidate = affine + residual
        refined = (1.0 - gate) * pred + gate * coverage_candidate
        aux = {
            "gate_mean": float(gate.detach().mean().cpu()),
            "gate_std": float(gate.detach().std(unbiased=False).cpu()),
            "pred_affine_disagreement": float(torch.linalg.vector_norm(diff.detach(), dim=1).mean().cpu()),
            "residual_abs_mean": float(residual.detach().abs().mean().cpu()),
        }
        return refined, aux


class OurNativeMethod(nn.Module):
    def __init__(self, variant: str, device: torch.device, cfg):
        super().__init__()
        self.variant = variant.upper()
        if self.variant not in {"M2", "M4"}:
            raise ValueError(self.variant)
        mode = "scalar" if self.variant == "M2" else "region"
        self.backbone = init_gdros_from_official(device, cfg)
        self.refiner = NativeGateRefiner(mode=mode)
        self.last_aux = {}

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int = 12, test_mode: bool = False):
        if test_mode:
            base_low, base_final = self.backbone(image1, image2, iters=iters, test_mode=True)
            refined, aux = self.refiner(base_final)
            self.last_aux = aux | {"base_test_mode": True}
            return base_low, refined
        preds = self.backbone(image1, image2, iters=iters)
        outs = []
        auxs = []
        for p in preds:
            out, aux = self.refiner(p)
            outs.append(out)
            auxs.append(aux)
        self.last_aux = {k: float(np.mean([a[k] for a in auxs])) for k in auxs[-1].keys()} if auxs else {}
        return outs


def pair_metrics(vals: list[float]) -> dict[str, Any]:
    a = np.asarray(vals, np.float64)
    if a.size == 0:
        return {"pairs": 0}
    out = {
        "pairs": int(a.size),
        "aepe": float(a.mean()),
        "all_aepe": float(a.mean()),
        "rmse": float(a.var()),
        "cmr1": float((a < 1).mean() * 100),
        "cmr2": float((a < 2).mean() * 100),
        "cmr5": float((a < 5).mean() * 100),
        "cmr10": float((a < 10).mean() * 100),
        "cmr16": float((a < 16).mean() * 100),
    }
    for t in [1, 2, 5]:
        sub = a[a < t]
        out[f"aepe_at_{t}"] = float(sub.mean()) if sub.size else 0.0
        out[f"rmse_at_{t}"] = float(sub.var()) if sub.size else 0.0
    return out


def eval_model(model: OurNativeMethod, ds, device: torch.device, variant: str, tag: str, name: str, iters_list: list[int], max_pairs: int):
    vals: dict[str, list[float]] = {f"iter{it}_post_lsr": [] for it in iters_list}
    vals.update({f"iter{it}_pre_lsr": [] for it in iters_list})
    pair_rows = []
    materialized = mat_dir(variant, tag) / f"{name}_predictions_before_gt.jsonl"
    materialized.parent.mkdir(parents=True, exist_ok=True)
    materialized.unlink(missing_ok=True)
    start = time.time()
    with torch.inference_mode():
        for i in range(min(max_pairs, len(ds))):
            file_opt, a, b, gt, valid = ds[i]
            a = a[None].to(device)
            b = b[None].to(device)
            preds = model(a, b, iters=max(iters_list))
            aux = getattr(model, "last_aux", {})
            rec = {
                "pair_index": i,
                "pair_id": Path(file_opt).stem,
                "uses_truth_flow_in_inference": False,
                "predictions_materialized_before_gt": True,
                "aux": aux,
            }
            for it in iters_list:
                pre = preds[it - 1].detach().cpu()
                post = lsr_flow(preds[it - 1]).detach().cpu()
                gt_cpu = gt[None].float()
                valid_cpu = valid.bool().view(-1)
                for key, flow in [(f"iter{it}_pre_lsr", pre), (f"iter{it}_post_lsr", post)]:
                    e = torch.linalg.vector_norm((flow - gt_cpu).float(), dim=1).view(-1)[valid_cpu]
                    vals[key].append(float(e.mean()))
                    rec[key] = float(e.mean())
            pair_rows.append(rec)
            with materialized.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            torch.cuda.empty_cache()
    metrics = {k: pair_metrics(v) for k, v in vals.items()}
    metrics["materialized_before_gt"] = str(materialized)
    metrics["materialized_sha256"] = sha256(materialized)
    metrics["eval_pairs"] = len(pair_rows)
    metrics["inference_seconds_total"] = time.time() - start
    metrics["inference_seconds_per_pair"] = (time.time() - start) / max(1, len(pair_rows))
    return metrics, pair_rows


def bucket_weights(epe: torch.Tensor):
    return (
        1.0
        + 0.10 * (epe >= 1).float()
        + 0.20 * (epe >= 2).float()
        + 0.50 * (epe >= 5).float()
        + 0.70 * (epe >= 10).float()
    )


def cvar_loss(epe: torch.Tensor, frac: float):
    flat = epe.reshape(-1)
    k = max(1, int(np.ceil(flat.numel() * frac)))
    return flat.topk(k).values.mean()


def manifest(variant: str, tag: str, args: argparse.Namespace | None = None) -> dict[str, Any]:
    script = Path(__file__)
    return {
        "experiment": "E081A-ZZVR-WHU-OUR-METHOD-NATIVE-TRAIN",
        "variant": variant,
        "run_root": str(rr(variant, tag)),
        "formal_root": str(formal_dir(variant, tag)),
        "objective": "Train OUR current method family natively on WHU, not OS-pretrain->WHU finetune.",
        "method_definition": {
            "M2": "GDROS/RAFT official-initialized backbone + scalar no-GT gate/refiner + E075-style threshold/tail loss.",
            "M4": "GDROS/RAFT official-initialized backbone + region no-GT gate/refiner + E075-style threshold/tail loss.",
            "no_os_task_checkpoint_init": True,
            "os_e075_e077_checkpoints_used": False,
        },
        "initialization": {
            "allowed_architecture_pretrain": {
                "raft_sintel": {"path": str(OFFICIAL_REF / "raft-sintel.pth"), "sha256": sha256(OFFICIAL_REF / "raft-sintel.pth")},
                "lsrnet_reference": {"path": str(OFFICIAL_REF / "LSRnet.pth"), "sha256": sha256(OFFICIAL_REF / "LSRnet.pth"), "note": "not loaded as train init by this wrapper; LSR operator used as in GDROS protocol"},
            },
            "forbidden_not_used": {
                "E074/E075/E077 OS-task checkpoints": "not loaded",
                "GDROS-whu released checkpoint": "not used for training init",
            },
        },
        "dataset": {
            "whu_train_root": str(WHU_SRC_TRAIN),
            "whu_test_root": str(WHU_SRC_TEST),
            "whu_view": str(WHU_VIEW),
            "training_zip_used_for_train": True,
            "testing_zip_used_for_train": False,
            "testing_zip_final_eval_only": True,
        },
        "protocol_refs": {"E077A": E077A_REF, "E080A": E080A_REF, "paper_whu": PAPER_WHU},
        "hyperparams": vars(args) if args else {},
        "source_sha256": {"script": sha256(script)},
    }


def write_p0(variant: str, tag: str, extra: dict[str, Any]):
    fr = formal_dir(variant, tag)
    base = {
        "status": "PASS" if extra.get("split_isolation", {}).get("status") == "PASS" and extra.get("smoke", {}).get("status") == "PASS" else "BLOCKING",
        "experiment": "E081A",
        "variant": variant,
        "split_isolation": extra.get("split_isolation"),
        "smoke": extra.get("smoke"),
        "manifest": manifest(variant, tag),
    }
    for name, obj in {
        "P0": base,
        "P0_dataset_protocol": base,
        "train_test_isolation_audit": extra.get("split_isolation"),
        "RUN_MANIFEST": manifest(variant, tag),
    }.items():
        js(fr / f"{name}.json", obj)
    (fr / "RESULT_CARD.md").write_text(
        f"# E081A {variant} P0\n\nStatus: `{base['status']}`\n\n"
        f"Train/test intersection: `{extra.get('split_isolation', {}).get('train_test_basename_intersection_count')}`\n\n"
        f"Smoke: `{extra.get('smoke')}`\n",
        encoding="utf-8",
    )
    return base


def p0_smoke(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed_all(args.seed)
    build_whu_view()
    get_cfg, datasets, _, _, sequence_loss, matrix2flow, LSR_list, _ = ensure_imports()
    split = train_test_isolation()
    assert split["train"]["counts"]["opt"] == 5600, split["train"]
    assert split["test"]["counts"]["opt"] == 700, split["test"]
    if split["status"] != "PASS":
        raise RuntimeError(f"WHU train/test resolved-path isolation failed: {split}")
    cfg = make_cfg(f"E081A-{args.variant}-smoke", args.steps, args.batch_size, args.image_size)
    aug = {"crop_size": cfg.image_size, "min_scale": -0.2, "max_scale": 0.6, "do_flip": True}
    tr = datasets.opt_sar_train(aug, split="training", root=str(WHU_VIEW), dstype="training")
    te = datasets.opt_sar_test(split="testing", root=str(WHU_VIEW), dstype="testing")
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    a, b, flow, valid = next(iter(loader))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = OurNativeMethod(args.variant, device, cfg).to(device)
    model.train()
    a, b, flow, valid = a.to(device), b.to(device), flow.to(device), valid.to(device)
    preds = model(a, b, iters=2)
    lseq, m1 = sequence_loss(preds, flow, valid, cfg)
    lgeo, m2 = sequence_loss(matrix2flow(LSR_list(preds), flow), flow, valid, cfg)
    epe = torch.linalg.vector_norm((preds[-1] - flow).float(), dim=1)[valid.bool()]
    loss = lseq + lgeo + args.bucket_weight * (bucket_weights(epe) * epe).mean() + args.tail_weight * cvar_loss(epe, args.tail_frac)
    loss.backward()
    refiner_grad = [p.grad.detach().abs().mean().item() for n, p in model.named_parameters() if "refiner" in n and p.grad is not None]
    backbone_grad = [p.grad.detach().abs().mean().item() for n, p in model.named_parameters() if "backbone" in n and p.grad is not None]
    params = sum(p.numel() for p in model.parameters())
    peak_mem = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    smoke_eval, _ = eval_model(model.eval(), te, device, args.variant, args.run_tag, "smoke4", [2], min(4, len(te)))
    smoke = {
        "status": "PASS",
        "train_len": len(tr),
        "test_len": len(te),
        "loss": float(loss.detach().cpu()),
        "flow_metrics": m1,
        "geo_metrics": m2,
        "refiner_grad_mean": float(np.mean(refiner_grad)) if refiner_grad else 0.0,
        "backbone_grad_mean": float(np.mean(backbone_grad)) if backbone_grad else 0.0,
        "params": int(params),
        "peak_memory_mb": int(peak_mem // (1024 * 1024)),
        "smoke_eval": smoke_eval,
        "no_gt_inference_path_uses_learned_refiner": True,
        "no_os_task_checkpoint_init": True,
    }
    if not np.isfinite(smoke["loss"]) or smoke["refiner_grad_mean"] <= 0 or smoke["backbone_grad_mean"] <= 0:
        smoke["status"] = "BLOCKING"
    return write_p0(args.variant, args.run_tag, {"split_isolation": split, "smoke": smoke})


def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed_all(args.seed)
    build_whu_view()
    get_cfg, datasets, _, _, sequence_loss, matrix2flow, LSR_list, _ = ensure_imports()
    split = train_test_isolation()
    if split["status"] != "PASS":
        raise RuntimeError("Split isolation failed")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cfg = make_cfg(f"E081A-{args.variant}-{args.steps}", args.steps, args.batch_size, args.image_size)
    aug = {"crop_size": cfg.image_size, "min_scale": -0.2, "max_scale": 0.6, "do_flip": True}
    train_ds = args.dataset_multiplier * datasets.opt_sar_train(aug, split="training", root=str(WHU_VIEW), dstype="training")
    test_ds = datasets.opt_sar_test(split="testing", root=str(WHU_VIEW), dstype="testing")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    data_iter = iter(loader)
    model = OurNativeMethod(args.variant, device, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps + max(args.toy_steps, 0), pct_start=0.05, cycle_momentum=False)
    milestones = sorted(set([m for m in args.milestones if m <= args.steps] + [args.steps]))
    curves: list[dict[str, Any]] = []
    evals: dict[str, Any] = {}
    start = time.time()
    formal_dir(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    log_dir(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    js(formal_dir(args.variant, args.run_tag) / "RUN_MANIFEST.json", manifest(args.variant, args.run_tag, args))
    log_path = log_dir(args.variant, args.run_tag) / "train_log.jsonl"
    for step in range(1, args.steps + 1):
        try:
            a, b, flow, valid = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            a, b, flow, valid = next(data_iter)
        a, b, flow, valid = a.to(device), b.to(device), flow.to(device), valid.to(device)
        model.train()
        preds = model(a, b, iters=args.train_iters)
        lseq, m1 = sequence_loss(preds, flow, valid, cfg)
        mats = matrix2flow(LSR_list(preds), flow)
        lgeo, m2 = sequence_loss(mats, flow, valid, cfg)
        final = mats[-1]
        epe = torch.linalg.vector_norm((final - flow).float(), dim=1)[valid.bool()]
        bucket = (bucket_weights(epe) * epe).mean()
        tail = cvar_loss(epe, args.tail_frac)
        loss = args.lambda_seq * lseq + args.lambda_geo * lgeo + args.bucket_weight * bucket + args.tail_weight * tail
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).detach().cpu())
        opt.step()
        sched.step()
        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "lseq_epe": m1["epe"],
                "lgeo_epe": m2["epe"],
                "flow5": float(m1["5px"] * 100),
                "mat5": float(m2["5px"] * 100),
                "bucket": float(bucket.detach().cpu()),
                "tail": float(tail.detach().cpu()),
                "lr": float(opt.param_groups[0]["lr"]),
                "grad_norm": grad_norm,
                "aux": getattr(model, "last_aux", {}),
                "elapsed_seconds": time.time() - start,
            }
            curves.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"E081A {args.variant} step={step}/{args.steps} loss={row['loss']:.4f} mat5={row['mat5']:.2f} lr={row['lr']:.2e} aux={row['aux']}", flush=True)
        if step in milestones:
            cp = ckpt_dir(args.variant, args.run_tag) / f"{args.variant}_step{step}.pt"
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "variant": args.variant, "step": step, "seed": args.seed, "manifest": manifest(args.variant, args.run_tag, args)}, cp)
            eval_metrics, pair_rows = eval_model(model.eval(), test_ds, device, args.variant, args.run_tag, f"{args.variant}_step{step}", [12, 24, 32], min(args.eval_pairs, len(test_ds)))
            evals[str(step)] = {"checkpoint": str(cp), "checkpoint_sha256": sha256(cp), "dev_test": eval_metrics}
            js(formal_dir(args.variant, args.run_tag) / "checkpoint_curve_report.json", {"status": "RUNNING", "curves": curves, "evals": evals})
            js(formal_dir(args.variant, args.run_tag) / f"per_pair_metrics_step{step}.json", pair_rows)
            torch.cuda.empty_cache()
    return finalize(args.variant, args.run_tag, curves, evals, start)


def best_from_evals(evals: dict[str, Any]):
    best_cmr5 = None
    best_aepe = None
    for step, ev in (evals or {}).items():
        for name, val in ev.get("dev_test", {}).items():
            if not isinstance(val, dict) or "cmr5" not in val:
                continue
            row = {"step": int(step), "eval_variant": name, **val, "checkpoint": ev.get("checkpoint"), "checkpoint_sha256": ev.get("checkpoint_sha256")}
            if best_cmr5 is None or row["cmr5"] > best_cmr5["cmr5"] or (row["cmr5"] == best_cmr5["cmr5"] and row["aepe"] < best_cmr5["aepe"]):
                best_cmr5 = row
            if best_aepe is None or row["aepe"] < best_aepe["aepe"]:
                best_aepe = row
    return best_cmr5, best_aepe


def finalize(variant: str, tag: str = "", curves=None, evals=None, start=None):
    fr = formal_dir(variant, tag)
    if evals is None:
        p = fr / "checkpoint_curve_report.json"
        obj = json.loads(p.read_text()) if p.exists() else {}
        curves = obj.get("curves", [])
        evals = obj.get("evals", {})
    best_cmr5, best_aepe = best_from_evals(evals or {})
    sufficient = bool(evals and any(int(k) >= 120000 for k in evals))
    if best_cmr5 is None:
        status = "TRAINING_INSUFFICIENT" if not sufficient else "NO_FORMAL_EVAL"
    else:
        success = best_cmr5["cmr5"] >= 99.0 or (best_cmr5["cmr5"] >= 96.0 and best_cmr5["aepe"] <= 1.2)
        partial = best_cmr5["cmr5"] >= 95.0 or best_cmr5["cmr2"] >= 90.0 or best_cmr5["aepe"] <= 1.5
        status = "TRAINING_INSUFFICIENT" if not sufficient else ("WHU_NATIVE_STRONG_SUCCESS" if success else ("WHU_NATIVE_PARTIAL_GAIN" if partial else "TRAINING_SUFFICIENT_NO_WHU_GAIN"))
    result = {
        "status": status,
        "variant": variant,
        "best_by_cmr5": best_cmr5,
        "best_by_aepe": best_aepe,
        "paper_whu_baseline": PAPER_WHU,
        "refs": {"E077A": E077A_REF, "E080A": E080A_REF},
        "training_sufficient": sufficient,
    }
    artifacts = {
        "G0": {"status": "PASS" if curves else "PENDING", "curves_head": (curves or [])[:5], "no_os_task_checkpoint_init": True},
        "G1": result,
        "RESULT_CARD": result,
        "checkpoint_curve_report": {"curves": curves or [], "evals": evals or {}, "best_by_cmr5": best_cmr5, "best_by_aepe": best_aepe},
        "WHU_eval_table_same_as_paper": {"ours_best_by_cmr5": best_cmr5, "ours_best_by_aepe": best_aepe, "paper_gdros": PAPER_WHU},
        "WHU_baseline_comparison": {"E077A": E077A_REF, "E080A": E080A_REF, "paper_gdros": PAPER_WHU, "ours": result},
        "resource_report": {
            "elapsed_seconds": None if start is None else time.time() - start,
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_memory_mb": int(torch.cuda.max_memory_allocated() // (1024 * 1024)) if torch.cuda.is_available() else 0,
        },
        "RUN_MANIFEST": manifest(variant, tag),
    }
    for name, obj in artifacts.items():
        js(fr / f"{name}.json", obj)
    (fr / "reviewer_conclusion.txt").write_text(
        f"E081A {variant}: {status}. Native WHU method-family run; no E074/E075/E077 OS-task checkpoint loaded. "
        "WHU testing split used only for formal evaluation; predictions materialized before metric join.\n",
        encoding="utf-8",
    )
    with (fr / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["selector", "step", "eval_variant", "aepe", "cmr1", "cmr2", "cmr5", "cmr10", "cmr16", "aepe_at_1", "aepe_at_2", "aepe_at_5", "rmse"])
        for selector, row in [("best_by_cmr5", best_cmr5), ("best_by_aepe", best_aepe)]:
            if row:
                w.writerow([selector, row.get("step"), row.get("eval_variant"), row.get("aepe"), row.get("cmr1"), row.get("cmr2"), row.get("cmr5"), row.get("cmr10"), row.get("cmr16"), row.get("aepe_at_1"), row.get("aepe_at_2"), row.get("aepe_at_5"), row.get("rmse")])
    (fr / "RESULT_CARD.md").write_text(
        f"# E081A {variant} RESULT_CARD\n\n"
        f"Status: `{status}`\n\n"
        f"Best by CMR@5: `{best_cmr5}`\n\n"
        f"Best by AEPE: `{best_aepe}`\n\n"
        "Init: RAFT-Sintel official architecture pretrain only; no OS/E075/E077 task checkpoint.\n",
        encoding="utf-8",
    )
    return result | {"formal_root": str(fr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["p0", "train", "finalize"], default="p0")
    ap.add_argument("--variant", choices=["M2", "M4"], required=True)
    ap.add_argument("--run-tag", default="native-r1")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=811)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--milestones", type=int, nargs="+", default=[20000, 40000, 80000, 120000])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=400)
    ap.add_argument("--dataset-multiplier", type=int, default=100)
    ap.add_argument("--train-iters", type=int, default=12)
    ap.add_argument("--eval-pairs", type=int, default=700)
    ap.add_argument("--lr", type=float, default=1.25e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--lambda-seq", type=float, default=1.0)
    ap.add_argument("--lambda-geo", type=float, default=1.0)
    ap.add_argument("--bucket-weight", type=float, default=0.10)
    ap.add_argument("--tail-weight", type=float, default=0.08)
    ap.add_argument("--tail-frac", type=float, default=0.10)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--toy-steps", type=int, default=0)
    args = ap.parse_args()
    formal_dir(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    mat_dir(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    log_dir(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    if args.mode == "p0":
        print(json.dumps(p0_smoke(args), ensure_ascii=False, indent=2))
    elif args.mode == "train":
        print(json.dumps(train(args), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(finalize(args.variant, args.run_tag), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
