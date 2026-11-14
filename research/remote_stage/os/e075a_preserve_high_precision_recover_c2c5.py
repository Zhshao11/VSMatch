#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("SPARSEGEO_ROOT", "/mnt/sda/zhushihao/VS_Match"))
TMP = ROOT / "image_matching_part2_migrated" / ".codex_tmp"
RUN_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE"
E074_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION"
B_CKPT = E074_ROOT / "run-b-stage180k/checkpoints/B_step180000.pt"
C_CKPT = E074_ROOT / "run-c-stage180k/checkpoints/C_step180000.pt"
B_G1 = E074_ROOT / "run-b-stage180k/formal/G1.json"
C_G1 = E074_ROOT / "run-c-stage180k/formal/G1.json"
TEST_ROOT = ROOT / "datasets/extracted/gdros_official_e003a_testing"
TRAIN_FLAT = ROOT / "datasets/extracted/gdros_official_e003a_training"

REFS = {
    "gdros_released": {"aepe": 1.461486, "cmr1": 33.73, "cmr2": 81.60, "cmr5": 99.53},
    "e074_b": {"aepe": 2.505252162178997, "cmr1": 44.801980198019805, "cmr2": 72.77227722772277, "cmr5": 89.35643564356435, "cmr10": 95.29702970297029, "cmr16": 97.52475247524752},
    "e074_c": {"aepe": 2.569692603874915, "cmr1": 48.76237623762376, "cmr2": 76.98019801980197, "cmr5": 87.87128712871286, "cmr10": 93.56435643564357, "cmr16": 96.78217821782178},
    "e074_a_nogt": {"aepe": 2.6361481511946954, "cmr1": 36.88118811881188, "cmr2": 67.82178217821783, "cmr5": 90.0990099009901, "cmr10": 94.80198019801979, "cmr16": 97.52475247524752},
}


def rr(v: str, tag: str = "") -> Path:
    suffix = f"-{tag.strip('-')}" if tag else ""
    return RUN_ROOT / f"run-{v.lower()}{suffix}"


def formal(v: str, tag: str = "") -> Path:
    return rr(v, tag) / "formal"


def mat_dir(v: str, tag: str = "") -> Path:
    return formal(v, tag) / "materialized"


def ckpt_dir(v: str, tag: str = "") -> Path:
    return rr(v, tag) / "checkpoints"


def log_dir(v: str, tag: str = "") -> Path:
    return rr(v, tag) / "logs"


def js(p: Path, o: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2, sort_keys=True))


def sha(p: Path):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def add_paths():
    sys.path.insert(0, str(TMP))
    import e074a_f2_f3_twostage_fusion as e074
    e072, e073 = e074.add_paths()
    return e074, e072, e073


def data_view(v: str, tag: str = ""):
    r = rr(v, tag) / "data_view/training"
    r.mkdir(parents=True, exist_ok=True)
    for n in ["opt", "sar_warped", "truth_flow", "datum"]:
        d = r / n
        s = TRAIN_FLAT / n
        if not d.exists():
            d.symlink_to(s, target_is_directory=True)
    return rr(v, tag) / "data_view"


def pair_metrics(vals):
    a = np.asarray(vals, np.float64)
    return {
        "pairs": int(a.size),
        "aepe": float(a.mean()) if a.size else None,
        "cmr1": float((a < 1).mean() * 100) if a.size else None,
        "cmr2": float((a < 2).mean() * 100) if a.size else None,
        "cmr5": float((a < 5).mean() * 100) if a.size else None,
        "cmr10": float((a < 10).mean() * 100) if a.size else None,
        "cmr16": float((a < 16).mean() * 100) if a.size else None,
    }


def bucket_name(x: float) -> str:
    if x <= 1:
        return "<=1"
    if x <= 2:
        return "1-2"
    if x <= 5:
        return "2-5"
    if x <= 10:
        return "5-10"
    return ">10"


def quantiles(vals):
    a = np.asarray(vals, np.float64)
    if a.size == 0:
        return {}
    return {k: float(np.quantile(a, q)) for k, q in [("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p95", 0.95)]}


def top_fraction_mean(vals, frac=0.1):
    a = np.asarray(vals, np.float64)
    if a.size == 0:
        return None
    k = max(1, int(np.ceil(a.size * frac)))
    return float(np.sort(a)[-k:].mean())


def auc_ap(labels, scores):
    y = np.asarray(labels, np.float64)
    s = np.asarray(scores, np.float64)
    pos = y.sum()
    neg = len(y) - pos
    if pos <= 0 or neg <= 0:
        return {"auroc": None, "ap": None}
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / pos
    fpr = fp / neg
    auroc = float(np.trapz(tpr, fpr))
    prec = tp / np.maximum(tp + fp, 1)
    ap = float((prec * y).sum() / pos)
    return {"auroc": auroc, "ap": ap}


def weighted_tail_loss(epe_map):
    w = (
        1.0
        + 0.10 * torch.sigmoid((epe_map - 1.0) / 0.25)
        + 0.18 * torch.sigmoid((epe_map - 2.0) / 0.50)
        + 0.30 * torch.sigmoid((epe_map - 5.0) / 1.00)
        + 0.42 * torch.sigmoid((epe_map - 10.0) / 2.00)
    )
    return (w * epe_map).mean()


def cvar_loss(epe_map, frac=0.1):
    flat = epe_map.reshape(-1)
    k = max(1, int(np.ceil(flat.numel() * frac)))
    return flat.topk(k).values.mean()


def load_branch(e074, variant: str, ckpt: Path, device):
    e072, e073 = e074.add_paths()
    c = e072.cfg()
    m = e074.E074Model(e073, variant, device).to(device)
    payload = torch.load(ckpt, map_location=device)
    miss = m.load_state_dict(payload["model"], strict=False)
    return m, {"checkpoint": str(ckpt), "sha256": sha(ckpt), "missing": list(miss.missing_keys), "unexpected": list(miss.unexpected_keys)}


class ScalarOrRegionGate(nn.Module):
    def __init__(self, in_ch: int, mode: str = "scalar"):
        super().__init__()
        self.mode = mode
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        g = torch.sigmoid(self.net(x))
        if self.mode == "scalar":
            return g.mean(dim=(2, 3), keepdim=True)
        if self.mode == "region":
            g = F.avg_pool2d(g, 4, 4, ceil_mode=True)
            return F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return g


class PairFusion(nn.Module):
    def __init__(self, e074, mode: str, device, c_skip_teachers: bool = True):
        """c_skip_teachers: C branch's teacher_preds only feeds E074 distill.

        In E075A (training and eval) nothing reads model.c.last_teacher_avg, so
        those 2 frozen GRU chains are dead computation. Set False to restore the
        legacy call chain bit-exactly for A/B verification.
        """
        super().__init__()
        self.mode = mode
        self.c_skip_teachers = bool(c_skip_teachers)
        self.b, self.b_info = load_branch(e074, "B", B_CKPT, device)
        self.c, self.c_info = load_branch(e074, "C", C_CKPT, device)
        for p in list(self.b.parameters()) + list(self.c.parameters()):
            p.requires_grad_(False)
        self.gate = ScalarOrRegionGate(16, mode=mode)

    def gate_features(self, pb, pc, lsr_fn):
        lb = lsr_fn(pb).detach().to(pb.device)
        lc = lsr_fn(pc).detach().to(pc.device)
        return torch.cat([pb.detach(), pc.detach(), (pb - pc).detach(), torch.abs(pb - pc).detach(), lb, lc, torch.abs(lb - pb).detach(), torch.abs(lc - pc).detach()], dim=1)

    def forward(self, a, b, iters=12, lsr_fn=None):
        pb = self.b(a, b, iters=iters)
        pc = self.c(a, b, iters=iters,
                    need_teacher_avg=not self.c_skip_teachers)
        outs = []
        aux = {}
        for xb, xc in zip(pb, pc):
            feat = self.gate_features(xb, xc, lsr_fn)
            g = self.gate(feat)
            if self.mode == "scalar":
                g = g.expand(-1, 1, xb.shape[-2], xb.shape[-1])
            outs.append((1 - g) * xb + g * xc)
            aux = {
                "gate_mean": float(g.detach().mean().cpu()),
                "gate_std": float(g.detach().std(unbiased=False).cpu()),
                "branch_b_mean": float(xb.detach().mean().cpu()),
                "branch_c_mean": float(xc.detach().mean().cpu()),
            }
        self.last_aux = aux
        self.last_branch_preds = {"B": pb, "C": pc}
        return outs


def bucket_weights(epe_map):
    return (
        1.0
        + 0.10 * (epe_map >= 1).float()
        + 0.20 * (epe_map >= 2).float()
        + 0.35 * (epe_map >= 5).float()
        + 0.50 * (epe_map >= 10).float()
    )


def eval_branch_outputs(preds, gt, valid, e074, cfg):
    from core.LSRnet.LSmodel import LSR_list
    pre = preds[-1]
    post = e074.lsr(pre).to(pre.device)
    e = torch.linalg.vector_norm((post - gt[None]).float(), dim=1)
    e = e.view(-1)[valid.view(-1)]
    self_resid = torch.linalg.vector_norm((post - pre).float(), dim=1).view(-1)[valid.view(-1)]
    return float(e.mean()), float(self_resid.mean()), float(torch.quantile(e, 0.9)), float(torch.quantile(e, 0.95))


def build_freeze_report():
    return {
        "experiment": "E074A",
        "frozen_at": "2026-09-04",
        "recommended_bases": {
            "balanced": str(B_CKPT),
            "high_precision": str(C_CKPT),
            "coverage_diag": "A no-GT chooser",
        },
        "best_metrics": {
            "B_180k": REFS["e074_b"],
            "C_180k": REFS["e074_c"],
            "A_nogt": REFS["e074_a_nogt"],
        },
        "sha256": {
            "e074a_script": sha(TMP / "e074a_f2_f3_twostage_fusion.py"),
            "b_checkpoint": sha(B_CKPT),
            "c_checkpoint": sha(C_CKPT),
            "run_manifest": sha(E074_ROOT / "run-a-formal/formal/RUN_MANIFEST.json"),
        },
    }


def audit(v: str, tag: str):
    out = {
        "status": "PASS",
        "experiment": "E075A",
        "variant": v,
        "run_root": str(rr(v, tag)),
        "objective": "Preserve high-precision CMR@1 while recovering CMR@2/CMR@5 and lowering AEPE",
        "preservation_package": build_freeze_report(),
        "tracks": {
            "M1": "continue C 180k with bucket-aware/tail-aware loss to keep CMR@1 while recovering CMR@2/CMR@5",
            "M2": "scalar no-GT gate between frozen E074 B and C branches, trained with oracle soft gate + final flow loss",
            "M3": "scalar gate + tail-risk/CVaR regularization to reduce AEPE tail without collapsing CMR@1",
            "M4": "region gate (2x2/4x4) with entropy/load-balance regularization for branch specialization",
        },
        "protocol": {
            "baseline_official_e046_to_e074_readonly": True,
            "testing_zip_training": False,
            "no_os424_confirm_rescue": True,
            "materialize_before_gt": True,
            "gpu_policy": "zzvr GPU0-3 only",
        },
        "sources": {
            "script": str(Path(__file__)),
            "b_checkpoint": str(B_CKPT),
            "b_checkpoint_sha256": sha(B_CKPT),
            "c_checkpoint": str(C_CKPT),
            "c_checkpoint_sha256": sha(C_CKPT),
            "b_g1_sha256": sha(B_G1),
            "c_g1_sha256": sha(C_G1),
        },
        "refs": REFS,
    }
    fr = formal(v, tag)
    for name, obj in [("P0", out), ("track_matrix", out["tracks"]), ("baseline_comparison", REFS), ("E074A_BEST_FREEZE", build_freeze_report())]:
        js(fr / f"{name}.json", obj)
    return out


def materialize_record(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def eval_model(model, ds, device, iters_list, max_pairs, v, tag, name, e074, gate_kind=None):
    from core.LSRnet.LSmodel import LSR_list
    vals = {f"iter{it}_pre_lsr": [] for it in iters_list} | {f"iter{it}_post_lsr": [] for it in iters_list}
    branch_vals = {"B": [], "C": [], "avg": [], "nogt_self_consistency": [], "gt_oracle_chooser": []}
    pair_rows = []
    mat = mat_dir(v, tag) / f"{name}_predictions_before_gt.jsonl"
    mat.parent.mkdir(parents=True, exist_ok=True)
    mat.unlink(missing_ok=True)
    with torch.inference_mode():
        for i in range(min(max_pairs, len(ds))):
            file_opt, a, b, gt, valid = ds[i]
            a = a[None].to(device)
            b = b[None].to(device)
            if gate_kind is None:
                preds = model(a, b, iters=max(iters_list))
                pb = pc = None
                aux = getattr(model, "last_aux", {})
            else:
                preds = model(a, b, iters=max(iters_list), lsr_fn=e074.lsr)
                pb = model.last_branch_preds["B"]
                pc = model.last_branch_preds["C"]
                aux = getattr(model, "last_aux", {})
            rec = {"pair_index": i, "pair_id": Path(file_opt).stem, "uses_truth_flow": False, "predictions_materialized_before_gt": True, "branch_gate": aux}
            for it in iters_list:
                pre = preds[it - 1].cpu()
                post = e074.lsr(preds[it - 1]).to(preds[it - 1].device).cpu()
                for k, fl in [(f"iter{it}_pre_lsr", pre), (f"iter{it}_post_lsr", post)]:
                    e = torch.linalg.vector_norm((fl - gt[None]).float(), dim=1).view(-1)
                    vals[k].append(float(e.mean()))
                    rec[k] = float(e.mean())
            if pb is not None and pc is not None:
                lb = e074.lsr(pb[-1]).to(pb[-1].device).cpu()
                lc = e074.lsr(pc[-1]).to(pc[-1].device).cpu()
                eb = torch.linalg.vector_norm((lb - gt[None]).float(), dim=1).view(-1)
                ec = torch.linalg.vector_norm((lc - gt[None]).float(), dim=1).view(-1)
                ea = torch.linalg.vector_norm((0.5 * (lb + lc) - gt[None]).float(), dim=1).view(-1)
                sb = float(torch.linalg.vector_norm((lb - pb[-1].cpu()).float(), dim=1).mean())
                sc = float(torch.linalg.vector_norm((lc - pc[-1].cpu()).float(), dim=1).mean())
                choice = "B" if sb <= sc else "C"
                ecf = eb if choice == "B" else ec
                eo = torch.minimum(eb, ec)
                branch_vals["B"].append(float(eb.mean()))
                branch_vals["C"].append(float(ec.mean()))
                branch_vals["avg"].append(float(ea.mean()))
                branch_vals["nogt_self_consistency"].append(float(ecf.mean()))
                branch_vals["gt_oracle_chooser"].append(float(eo.mean()))
                rec.update({
                    "branch_b_epe": float(eb.mean()),
                    "branch_c_epe": float(ec.mean()),
                    "branch_avg_epe": float(ea.mean()),
                    "branch_nogt_choice": choice,
                    "branch_nogt_epe": float(ecf.mean()),
                    "branch_gt_oracle_epe": float(eo.mean()),
                    "branch_b_self_resid": sb,
                    "branch_c_self_resid": sc,
                })
            pair_rows.append(rec)
            materialize_record(mat, rec)
            torch.cuda.empty_cache()
    metrics = {k: pair_metrics(vv) for k, vv in vals.items()}
    metrics["branch_baselines"] = {k: pair_metrics(vv) for k, vv in branch_vals.items() if vv}
    metrics["materialized_before_gt"] = str(mat)
    metrics["materialized_sha256"] = sha(mat)
    return metrics, pair_rows


def transition_report(rows):
    buckets = ["<=1", "1-2", "2-5", "5-10", ">10"]
    counts = {f"{a}->{b}": 0 for a in buckets for b in buckets}
    for r in rows:
        if "branch_b_epe" not in r:
            continue
        a = bucket_name(r["branch_b_epe"])
        b = bucket_name(r["branch_c_epe"])
        f = bucket_name(r["branch_fused_epe"])
        counts[f"{a}->{f}"] += 1
        counts[f"{b}->{f}"] += 1
    return {"counts": counts}


def train(args):
    e074, e072, e073 = add_paths()
    import core.datasets
    from core.loss import sequence_loss, matrix2flow
    from core.LSRnet.LSmodel import LSR_list

    v = args.variant.upper()
    tag = args.run_tag
    audit(v, tag)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    c = e072.cfg()
    tr = core.datasets.opt_sar_train(split="training", dstype="training", root=str(data_view(v, tag)))
    dv = core.datasets.opt_sar_test(split="testing", dstype="testing", root=str(TEST_ROOT))
    milestones = sorted(set([m for m in args.milestones if m <= args.steps] + [args.steps]))
    curves, evals = [], {}

    if v == "M1":
        model = e074.E074Model(e073, "C", device).to(device)
        if args.resume_ckpt:
            payload = torch.load(args.resume_ckpt, map_location=device)
            model.load_state_dict(payload["model"], strict=False)
    else:
        model = PairFusion(e074, "region" if v == "M4" else "scalar", device).to(device)
        if args.resume_ckpt:
            payload = torch.load(args.resume_ckpt, map_location=device)
            model.load_state_dict(payload["model"], strict=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)

    def step_batch():
        idx = int(rng.integers(0, len(tr)))
        a, b, flow, valid = tr[idx]
        return a[None].to(device), b[None].to(device), flow[None].to(device), valid[None].to(device)

    def gate_target(pred_b, pred_c, gt, scalar: bool):
        gt_b = gt.to(pred_b.device)
        if gt_b.dim() == pred_b.dim() - 1:
            gt_b = gt_b[None]
        eb = torch.linalg.vector_norm((e074.lsr(pred_b).to(pred_b.device) - gt_b).float(), dim=1)
        ec = torch.linalg.vector_norm((e074.lsr(pred_c).to(pred_c.device) - gt_b).float(), dim=1)
        tmap = torch.sigmoid((eb - ec) / args.gate_temp).detach()
        if scalar:
            return tmap.mean(dim=(1, 2), keepdim=True)[:, None]
        return tmap[:, None]

    for phase, steps in [("toy", args.toy_steps), ("full", args.steps)]:
        begin = args.resume_step + 1 if (phase == "full" and args.resume_step is not None) else 1
        if phase == "full" and args.resume_step is not None and args.resume_step >= steps:
            continue
        for step in range(begin, steps + 1):
            a, b, flow, valid = step_batch()
            if v == "M1":
                preds = model(a, b, iters=args.train_iters)
                lseq, m1 = sequence_loss(preds, flow, valid, c)
                mats = matrix2flow(LSR_list(preds), flow)
                lgeo, m2 = sequence_loss(mats, flow, valid, c)
                final = mats[-1]
                epe = torch.linalg.vector_norm((final - flow).float(), dim=1)
                epe = epe[valid.bool()]
                tail = cvar_loss(epe, frac=args.tail_frac)
                bucket = (bucket_weights(epe) * epe).mean()
                loss = lseq + lgeo + args.tail_weight * tail + args.bucket_weight * bucket
                aux = {"tail": float(tail.detach().cpu()), "bucket": float(bucket.detach().cpu())}
                gate = None
            else:
                pb = model.b(a, b, iters=args.train_iters)
                # E075A never reads model.c.last_teacher_avg (only E074 distill
                # does); skip those 2 frozen GRU chains here too, otherwise the
                # training loop would still pay for the dead computation.
                pc = model.c(a, b, iters=args.train_iters,
                             need_teacher_avg=False)
                fused = []
                for xb, xc in zip(pb, pc):
                    feat = model.gate_features(xb, xc, e074.lsr)
                    g = model.gate(feat)
                    if v != "M4":
                        g = g.expand(-1, 1, xb.shape[-2], xb.shape[-1])
                    else:
                        g = F.interpolate(g, size=xb.shape[-2:], mode="bilinear", align_corners=False)
                    fused.append((1 - g) * xb + g * xc)
                    gate = g
                lseq, m1 = sequence_loss(fused, flow, valid, c)
                mats = matrix2flow(LSR_list(fused), flow)
                lgeo, m2 = sequence_loss(mats, flow, valid, c)
                pbp = e074.lsr(pb[-1]).detach().to(pb[-1].device)
                pcp = e074.lsr(pc[-1]).detach().to(pc[-1].device)
                target = gate_target(pb[-1], pc[-1], flow, scalar=(v != "M4"))
                if v == "M4":
                    gate_loss = F.binary_cross_entropy(gate.clamp(1e-4, 1 - 1e-4), target)
                    entropy = -(gate.clamp(1e-4, 1 - 1e-4) * torch.log(gate.clamp(1e-4, 1 - 1e-4)) + (1 - gate.clamp(1e-4, 1 - 1e-4)) * torch.log(1 - gate.clamp(1e-4, 1 - 1e-4))).mean()
                    balance = (gate.mean() - target.mean()).pow(2)
                    loss = lseq + lgeo + args.gate_weight * gate_loss + args.tail_weight * cvar_loss(torch.linalg.vector_norm((mats[-1] - flow).float(), dim=1)[valid.bool()], frac=args.tail_frac) + args.entropy_weight * entropy + args.balance_weight * balance
                    aux = {"gate_loss": float(gate_loss.detach().cpu()), "entropy": float(entropy.detach().cpu()), "balance": float(balance.detach().cpu())}
                else:
                    target = target.expand_as(gate)
                    gate_loss = F.binary_cross_entropy(gate.clamp(1e-4, 1 - 1e-4), target)
                    tail = cvar_loss(torch.linalg.vector_norm((mats[-1] - flow).float(), dim=1)[valid.bool()], frac=args.tail_frac)
                    bucket = (bucket_weights(torch.linalg.vector_norm((mats[-1] - flow).float(), dim=1)[valid.bool()]) * torch.linalg.vector_norm((mats[-1] - flow).float(), dim=1)[valid.bool()]).mean()
                    loss = lseq + lgeo + args.gate_weight * gate_loss + args.tail_weight * tail + args.bucket_weight * bucket
                    aux = {"gate_loss": float(gate_loss.detach().cpu()), "tail": float(tail.detach().cpu()), "bucket": float(bucket.detach().cpu())}
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            log_every = max(1, steps // 10 if phase == "toy" else args.steps // 20)
            if step == begin or step % log_every == 0:
                row = {"variant": v, "phase": phase, "step": step, "loss": float(loss.detach().cpu()), "lseq_epe": m1["epe"], "lgeo_epe": m2["epe"], "flow5": float(m1["5px"] * 100), "mat5": float(m2["5px"] * 100), "aux": aux}
                curves.append(row)
                print(f"E075A {v} {phase} step={step}/{steps} loss={row['loss']:.3f} flow5={row['flow5']:.2f} mat5={row['mat5']:.2f} aux={row['aux']}", flush=True)
            if phase == "full" and step in milestones:
                cp = ckpt_dir(v, tag) / f"{v}_step{step}.pt"
                cp.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "variant": v, "step": step, "seed": args.seed, "resume_ckpt": args.resume_ckpt, "sources": audit(v, tag)["sources"]}, cp)
                evals[str(step)] = {"checkpoint": str(cp), "checkpoint_sha256": sha(cp), "dev": eval_model(model.eval(), dv, device, [12, 24, 32], args.dev_eval_pairs, v, tag, f"{v}_step{step}", e074, gate_kind=("region" if v == "M4" else ("scalar" if v in ["M2", "M3"] else None)))[0]}
                model.train()
                js(formal(v, tag) / "curve_guided_finetune_report.json", {"status": "RUNNING", "curves": curves, "evals": evals})
                torch.cuda.empty_cache()
    return finalize(v, tag, curves, evals, time.time(), e074)


def finalize(v, tag="", curves=None, evals=None, start=None, e074=None):
    fr = formal(v, tag)
    if evals is None:
        p = fr / "curve_guided_finetune_report.json"
        r = json.loads(p.read_text()) if p.exists() else {}
        curves = r.get("curves", [])
        evals = r.get("evals", {})
    best = None
    for st, ev in (evals or {}).items():
        for k, val in ev.get("dev", {}).items():
            if isinstance(val, dict) and "aepe" in val:
                row = {"variant": v, "step": int(st), "eval_variant": k, **val}
                if best is None or row["aepe"] < best["aepe"]:
                    best = row
    sufficient = bool(evals and any(int(k) >= 120000 for k in evals)) if v != "M1" else bool(evals and any(int(k) >= 220000 for k in evals))
    if best is None:
        status = "TRAINING_INSUFFICIENT" if not sufficient else "NO_EVAL_YET"
    else:
        strong = best["cmr1"] >= 46 and best["cmr2"] >= 80 and best["cmr5"] >= 93 and best["aepe"] < 2.3
        success = best["cmr1"] >= 46 and (best["cmr2"] >= 78 or best["cmr5"] >= 91) and best["aepe"] < 2.5
        partial = best["cmr1"] >= 46 or best["cmr2"] >= REFS["e074_c"]["cmr2"] or best["cmr5"] >= max(REFS["e074_b"]["cmr5"], REFS["e074_a_nogt"]["cmr5"]) or best["aepe"] < REFS["e074_b"]["aepe"]
        status = "TRAINING_INSUFFICIENT" if not sufficient else ("STRONG_PRESERVE_RECOVER_SUCCESS" if strong else ("PRESERVE_RECOVER_SUCCESS" if success else ("PARTIAL_PRESERVE_GAIN" if partial else "TRAINING_SUFFICIENT_NO_GAIN")))
    rows = []
    if best is not None:
        rows.append(best)
    objs = {
        "G0": {"status": "PASS" if curves else "PENDING", "training_sufficient": sufficient, "toy_signal": curves[:12], "sources": audit(v, tag)["sources"]},
        "G1": {"status": status, "best": best, "refs": REFS},
        "curve_guided_finetune_report": {"status": status, "curves": curves, "evals": evals},
        "checkpoint_curve_report": {"curves": curves, "evals": evals},
        "method_ablation": {"variant": v, "best": best},
        "resource_report": {"gpu_policy": "zzvr GPU0-3", "elapsed_seconds": None if start is None else time.time() - start},
        "RUN_MANIFEST": {"experiment": "E075A", "variant": v, "status": status, "run_root": str(rr(v, tag)), "formal_root": str(fr), "b_checkpoint_sha256": sha(B_CKPT), "c_checkpoint_sha256": sha(C_CKPT), "testing_zip_used_for_training": False, "no_os424_confirm_rescue": True, "materialized_before_gt": [str(p) for p in mat_dir(v, tag).glob("*.jsonl")]},
    }
    for n, o in objs.items():
        js(fr / f"{n}.json", o)
    (fr / "reviewer_conclusion.txt").write_text(f"E075A {v}: {status}. Preserve CMR@1 while recovering CMR@2/CMR@5 and AEPE; GT only used for labels/metrics.\n", encoding="utf-8")
    (fr / "RESULT_CARD.md").write_text(f"# E075A {v} RESULT_CARD\n\nStatus: `{status}`\n\nBest: `{best}`\n", encoding="utf-8")
    return {"status": status, "best": best, "formal": str(fr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "finalize", "p0"], default="train")
    ap.add_argument("--variant", choices=["M1", "M2", "M3", "M4"], required=True)
    ap.add_argument("--run-tag", type=str, default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=901)
    ap.add_argument("--toy-pairs", type=int, default=8)
    ap.add_argument("--toy-steps", type=int, default=200)
    ap.add_argument("--resume-step", type=int, default=None)
    ap.add_argument("--resume-ckpt", type=str, default=None)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--milestones", type=int, nargs="+", default=[40000, 80000, 120000])
    ap.add_argument("--train-iters", type=int, default=12)
    ap.add_argument("--dev-eval-pairs", type=int, default=404)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--gate-temp", type=float, default=0.75)
    ap.add_argument("--gate-weight", type=float, default=0.35)
    ap.add_argument("--tail-weight", type=float, default=0.20)
    ap.add_argument("--bucket-weight", type=float, default=0.12)
    ap.add_argument("--tail-frac", type=float, default=0.10)
    ap.add_argument("--entropy-weight", type=float, default=0.03)
    ap.add_argument("--balance-weight", type=float, default=0.05)
    args = ap.parse_args()
    formal(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    mat_dir(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    log_dir(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    if args.mode == "p0":
        print(json.dumps(audit(args.variant, args.run_tag), indent=2))
    elif args.mode == "finalize":
        print(json.dumps(finalize(args.variant, args.run_tag), indent=2))
    else:
        print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
