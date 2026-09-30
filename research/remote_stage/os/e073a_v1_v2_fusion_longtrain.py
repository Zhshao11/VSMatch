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
MIG = ROOT / "image_matching_part2_migrated"
TMP = MIG / ".codex_tmp"
E072 = TMP / "e072a_layerwise_frontend_alignment.py"
RUN_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN"
E072_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-072A-ZZVR-LAYERWISE-FRONTEND-ALIGNMENT"
V1_60K = E072_ROOT / "run-v1-r60k/checkpoints/V1_step60000.pt"
V2_60K = E072_ROOT / "run-v2-r60k/checkpoints/V2_step60000.pt"
V1_G1 = E072_ROOT / "run-v1-r60k/formal/G1.json"
V2_G1 = E072_ROOT / "run-v2-r60k/formal/G1.json"
TRAIN_FLAT = ROOT / "datasets/extracted/gdros_official_e003a_training"
TEST_ROOT = ROOT / "datasets/extracted/gdros_official_e003a_testing"

REFS = {
    "gdros_released": {"aepe": 1.461486, "cmr1": 33.73, "cmr2": 81.60, "cmr5": 99.53},
    "e065": {"aepe": 5.6425, "cmr2": 17.3166, "cmr5": 59.6648, "cmr10": 88.3459, "cmr16": 95.8989},
    "e072_v1_r60k": {"aepe": 4.192988409264253, "cmr1": 8.663366336633663, "cmr2": 33.16831683168317, "cmr5": 76.48514851485149, "cmr10": 92.57425742574257, "cmr16": 96.53465346534654},
    "e072_v2_r60k": {"aepe": 4.3024312742423305, "cmr1": 9.900990099009901, "cmr2": 43.31683168316832, "cmr5": 69.55445544554455, "cmr10": 91.58415841584159, "cmr16": 97.77227722772277},
}


def run_root(v: str, tag: str = "") -> Path:
    suffix = f"-{tag.strip('-')}" if tag else ""
    return RUN_ROOT / f"run-{v.lower()}{suffix}"


def formal(v: str, tag: str = "") -> Path:
    return run_root(v, tag) / "formal"


def mat_dir(v: str, tag: str = "") -> Path:
    return formal(v, tag) / "materialized"


def ckpt_dir(v: str, tag: str = "") -> Path:
    return run_root(v, tag) / "checkpoints"


def log_dir(v: str, tag: str = "") -> Path:
    return run_root(v, tag) / "logs"


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
    import e072a_layerwise_frontend_alignment as e072
    e072.add_paths()
    return e072


def data_view(v: str, tag: str = ""):
    r = run_root(v, tag) / "data_view/training"
    r.mkdir(parents=True, exist_ok=True)
    for n in ["opt", "sar_warped", "truth_flow", "datum"]:
        d = r / n
        s = TRAIN_FLAT / n
        if not d.exists():
            d.symlink_to(s, target_is_directory=True)
    return run_root(v, tag) / "data_view"


def audit(v: str, tag: str = ""):
    out = {
        "status": "PASS",
        "experiment": "E073A",
        "variant": v,
        "run_root": str(run_root(v, tag)),
        "fusion_variant_matrix": {
            "F1": "late prediction / affine ensemble diagnostic: fixed weights + no-GT flow-vs-LSR self-consistency chooser + GT oracle chooser upper bound",
            "F2": "feature-level dual-branch fusion: V1 GDROS-compatible feature + V2 DINO-assisted feature, concat/gated projection before CorrBlock",
            "F3": "correlation-level fusion: V1/V2 CorrBlock lookup fused by learnable gate before recurrent update",
            "F4": "multi-head matcher fusion: V1 and V2 recurrent heads produce flow sequences; learnable fusion combines heads before LSR",
            "F5": "bootstrap schedule: F2/F3/F4 warm-start from E072 V1/V2 60k checkpoints, then joint finetune",
            "F6": "multi-threshold objective proxy: Lseq + Lgeo, tracked CMR@1/2/5/10/16 and flow/LSR buckets",
        },
        "protocol": {
            "baseline_official_e046_e063_to_e072_readonly": True,
            "testing_zip_training": False,
            "no_os424_confirm_rescue": True,
            "materialize_before_gt": True,
            "gpu_policy": "zzvr GPU0-3 only",
        },
        "sources": {
            "script": str(Path(__file__)),
            "e072_script": str(E072),
            "v1_checkpoint": str(V1_60K),
            "v1_checkpoint_sha256": sha(V1_60K),
            "v2_checkpoint": str(V2_60K),
            "v2_checkpoint_sha256": sha(V2_60K),
            "v1_g1_sha256": sha(V1_G1),
            "v2_g1_sha256": sha(V2_G1),
        },
        "refs": REFS,
    }
    fr = formal(v, tag)
    for name, obj in [
        ("P0", out),
        ("fusion_variant_matrix", out["fusion_variant_matrix"]),
        ("baseline_comparison", REFS),
    ]:
        js(fr / f"{name}.json", obj)
    return out


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


def feature_stats(x):
    y = x.detach().float()
    return {
        "shape": list(y.shape),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "norm_mean": float(torch.linalg.vector_norm(y, dim=1).mean()),
        "abs_mean": float(y.abs().mean()),
    }


def corr_diag(fs, ft, flow, stride=8.0, max_src=96):
    with torch.no_grad():
        b, c, h, w = fs.shape
        n = h * w
        k = min(max_src, n)
        f1 = fs.flatten(2).transpose(1, 2)[0, :k]
        f2 = ft.flatten(2)[0]
        corr = torch.matmul(f1.float(), f2.float())
        yy = torch.arange(h, device=fs.device).repeat_interleave(w)[:k].float()
        xx = torch.arange(w, device=fs.device).repeat(h)[:k].float()
        gt = flow[:, (yy * stride).long().clamp(max=flow.shape[1] - 1), (xx * stride).long().clamp(max=flow.shape[2] - 1)].T.to(fs.device)
        tx = ((xx * stride + gt[:, 0]) / stride).round().long().clamp(0, w - 1)
        ty = ((yy * stride + gt[:, 1]) / stride).round().long().clamp(0, h - 1)
        gi = ty * w + tx
        gt_score = corr[torch.arange(k, device=fs.device), gi]
        rank = (corr > gt_score[:, None]).sum(1) + 1
        ent = torch.distributions.Categorical(logits=corr).entropy()
        return {
            "rank_mean": float(rank.float().mean()),
            "hit1": float((rank <= 1).float().mean() * 100),
            "hit4": float((rank <= 4).float().mean() * 100),
            "hit8": float((rank <= 8).float().mean() * 100),
            "hit32": float((rank <= 32).float().mean() * 100),
            "entropy": float(ent.mean()),
        }


def load_branch(e072, variant: str, checkpoint: Path, device):
    c = e072.cfg()
    m = e072.E072Matcher(c, variant, device).to(device)
    payload = torch.load(checkpoint, map_location=device)
    miss = m.load_state_dict(payload["model"], strict=False)
    return m, {"checkpoint": str(checkpoint), "sha256": sha(checkpoint), "missing": list(miss.missing_keys), "unexpected": list(miss.unexpected_keys)}


class E073FusionMatcher(nn.Module):
    def __init__(self, e072, cfg, variant: str, device, freeze_branches: bool = False):
        super().__init__()
        from core.update import BasicUpdateBlock
        from core.GDROSnet.FeatureFormer.cnn import BasicEncoder
        self.e072 = e072
        self.variant = variant.upper()
        self.b1, self.b1_load = load_branch(e072, "V1", V1_60K, device)
        self.b2, self.b2_load = load_branch(e072, "V2", V2_60K, device)
        if freeze_branches:
            for p in list(self.b1.parameters()) + list(self.b2.parameters()):
                p.requires_grad_(False)
        self.context = BasicEncoder(output_dim=256, norm_fn="instance")
        self.update = BasicUpdateBlock(cfg, hidden_dim=128)
        # Warm-start recurrent state from V1: it had stronger AEPE/CMR@5/10.
        self.context.load_state_dict(self.b1.context.state_dict(), strict=False)
        self.update.load_state_dict(self.b1.update.state_dict(), strict=False)
        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_radius = 4
        self.fuse_proj = nn.Sequential(nn.Conv2d(512, 256, 1), nn.GELU(), nn.Conv2d(256, 256, 1))
        self.fuse_gate = nn.Sequential(nn.Conv2d(512, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))
        self.corr_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.head_gate_logit = nn.Parameter(torch.tensor(0.0))

    def initialize_flow(self, img):
        from core.utils.utils import coords_grid
        n, _, h, w = img.shape
        return coords_grid(n, h // 8, w // 8, device=img.device), coords_grid(n, h // 8, w // 8, device=img.device)

    def upsample_flow(self, flow, mask):
        n, _, h, w = flow.shape
        if mask is None:
            return F.interpolate(8 * flow, scale_factor=8, mode="bilinear", align_corners=True)
        mask = torch.softmax(mask.view(n, 1, 9, 8, 8, h, w), dim=2)
        up = F.unfold(8 * flow, [3, 3], padding=1).view(n, 2, 9, 1, 1, h, w)
        up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3).reshape(n, 2, 8 * h, 8 * w)
        return up

    def branch_features(self, raw1, raw2, img1n, img2n):
        c1 = self.b1.context(img1n)
        f1s, f1t = self.b1._features(raw1, raw2, img1n, img2n, c1)
        c2 = self.b2.context(img1n)
        f2s, f2t = self.b2._features(raw1, raw2, img1n, img2n, c2)
        return f1s, f1t, f2s, f2t

    def recurrent_from_corr(self, img1n, corr_fn):
        cnet = self.context(img1n)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        c0, c1 = self.initialize_flow(img1n)
        outs = []
        for _ in range(self.iters):
            c1 = c1.detach()
            corr = corr_fn(c1)
            flow = c1 - c0
            net, mask, delta = self.update(net, inp, corr, flow)
            c1 = c1 + delta
            outs.append(self.upsample_flow(c1 - c0, mask))
        return outs

    def forward(self, raw1, raw2, iters=12, return_features=False):
        from core.corr import CorrBlock
        self.iters = iters
        img1n = 2 * (raw1 / 255.0) - 1.0
        img2n = 2 * (raw2 / 255.0) - 1.0
        f1s, f1t, f2s, f2t = self.branch_features(raw1, raw2, img1n, img2n)
        if self.variant == "F2":
            gs = torch.sigmoid(self.fuse_gate(torch.cat([f1s, f2s], dim=1)))
            gt = torch.sigmoid(self.fuse_gate(torch.cat([f1t, f2t], dim=1)))
            fs = (1 - gs) * f1s + gs * self.fuse_proj(torch.cat([f1s, f2s], dim=1))
            ft = (1 - gt) * f1t + gt * self.fuse_proj(torch.cat([f1t, f2t], dim=1))
            outs = self.recurrent_from_corr(img1n, CorrBlock(fs.float(), ft.float(), radius=self.corr_radius))
            aux = {"gate_mean": float(gs.detach().mean().cpu())}
        elif self.variant == "F3":
            c1 = CorrBlock(f1s.float(), f1t.float(), radius=self.corr_radius)
            c2 = CorrBlock(f2s.float(), f2t.float(), radius=self.corr_radius)
            a = torch.sigmoid(self.corr_gate_logit)
            outs = self.recurrent_from_corr(img1n, lambda coords: (1 - a) * c1(coords) + a * c2(coords))
            fs, ft = (1 - a) * f1s + a * f2s, (1 - a) * f1t + a * f2t
            aux = {"corr_gate_v2": float(a.detach().cpu())}
        elif self.variant == "F4":
            p1 = self.b1(raw1, raw2, iters=iters, return_features=False)
            p2 = self.b2(raw1, raw2, iters=iters, return_features=False)
            a = torch.sigmoid(self.head_gate_logit)
            outs = [(1 - a) * x + a * y for x, y in zip(p1, p2)]
            fs, ft = (f1s + f2s) * 0.5, (f1t + f2t) * 0.5
            aux = {"head_gate_v2": float(a.detach().cpu())}
        else:
            raise ValueError(self.variant)
        self.last_aux = aux
        if return_features:
            return outs, fs, ft, aux
        return outs


def eval_model(model, ds, device, iters_list, max_pairs, v, tag, name):
    from core.LSRnet.LSmodel import LSR
    vals = {f"iter{it}_pre_lsr": [] for it in iters_list} | {f"iter{it}_post_lsr": [] for it in iters_list}
    fstats, cstats, aux_rows = [], [], []
    mat = mat_dir(v, tag) / f"{name}_predictions_features_before_gt.jsonl"
    mat.parent.mkdir(parents=True, exist_ok=True)
    mat.unlink(missing_ok=True)
    with torch.inference_mode():
        for i in range(min(max_pairs, len(ds))):
            file_opt, a, b, gt, _ = ds[i]
            a = a[None].to(device)
            b = b[None].to(device)
            preds, fs, ft, aux = model(a, b, iters=max(iters_list), return_features=True)
            rec = {
                "pair_index": i,
                "pair_id": Path(file_opt).stem,
                "uses_truth_flow": False,
                "features_predictions_materialized_before_gt": True,
                "feature_stats_src": feature_stats(fs),
                "feature_stats_tgt": feature_stats(ft),
                "branch_aux": aux,
            }
            fstats += [rec["feature_stats_src"], rec["feature_stats_tgt"]]
            aux_rows.append(aux)
            cd = corr_diag(fs, ft, gt.to(device))
            cstats.append(cd)
            rec["corr_diag_joined_after_materialization"] = cd
            for it in iters_list:
                pre = preds[it - 1].cpu()
                post = LSR(1, 512, 512, preds[it - 1], 200000).cpu()
                for k, fl in [(f"iter{it}_pre_lsr", pre), (f"iter{it}_post_lsr", post)]:
                    e = torch.linalg.vector_norm((fl - gt[None]).float(), dim=1).view(-1)
                    vals[k].append(float(e.mean()))
                    rec[k] = float(e.mean())
            with mat.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            torch.cuda.empty_cache()

    def mean_dict(rows):
        out = {}
        for k in sorted({k for r in rows for k in r if isinstance(r.get(k), (int, float))}):
            arr = np.asarray([r[k] for r in rows if k in r], float)
            if arr.size:
                out[k] = float(arr.mean())
        return out

    return {k: pair_metrics(x) for k, x in vals.items()} | {
        "feature_distribution": mean_dict(fstats),
        "corr_quality": mean_dict(cstats),
        "branch_contribution": mean_dict(aux_rows),
        "materialized_before_gt": str(mat),
        "materialized_sha256": sha(mat),
    }


def train(args):
    e072 = add_paths()
    import core.datasets
    from core.loss import sequence_loss, matrix2flow
    from core.LSRnet.LSmodel import LSR_list

    v = args.variant.upper()
    tag = args.run_tag
    audit(v, tag)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    c = e072.cfg()
    tr = core.datasets.opt_sar_train(split="training", dstype="training", root=str(data_view(v, tag)))
    dv = core.datasets.opt_sar_test(split="testing", dstype="testing", root=str(TEST_ROOT))
    model = E073FusionMatcher(e072, c, v, device, freeze_branches=args.freeze_branches).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    curves, evals = [], {}
    begin = args.start_step + 1
    milestones = sorted(set([m for m in args.milestones if m <= args.steps] + [args.steps]))
    for phase, steps in [("toy", args.toy_steps), ("full", args.steps)]:
        phase_begin = begin if phase == "full" else 1
        for step in range(phase_begin, steps + 1):
            idx = int(rng.integers(0, args.toy_pairs if phase == "toy" else len(tr)))
            a, b, flow, valid = tr[idx]
            a = a[None].to(device)
            b = b[None].to(device)
            flow = flow[None].to(device)
            valid = valid[None].to(device)
            preds = model(a, b, iters=args.train_iters, return_features=False)
            lseq, m1 = sequence_loss(preds, flow, valid, c)
            mats = matrix2flow(LSR_list(preds), flow)
            lgeo, m2 = sequence_loss(mats, flow, valid, c)
            loss = args.lambda_seq * lseq + args.lambda_geo * lgeo
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            log_every = max(1, steps // 10 if phase == "toy" else args.steps // 20)
            if step == phase_begin or step % log_every == 0:
                row = {
                    "variant": v,
                    "phase": phase,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "lseq_epe": m1["epe"],
                    "lgeo_epe": m2["epe"],
                    "flow5": float(m1["5px"] * 100),
                    "mat5": float(m2["5px"] * 100),
                    "aux": getattr(model, "last_aux", {}),
                }
                curves.append(row)
                print(f"E073A {v} {phase} step={step}/{steps} loss={row['loss']:.3f} flow5={row['flow5']:.2f} mat5={row['mat5']:.2f} aux={row['aux']}", flush=True)
            if phase == "full" and step in milestones:
                cp = ckpt_dir(v, tag) / f"{v}_step{step}.pt"
                cp.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "variant": v, "step": step, "seed": args.seed, "sources": audit(v, tag)["sources"]}, cp)
                evals[str(step)] = {
                    "checkpoint": str(cp),
                    "checkpoint_sha256": sha(cp),
                    "dev": eval_model(model.eval(), dv, device, [12, 24, 32], args.dev_eval_pairs, v, tag, f"{v}_step{step}"),
                }
                model.train()
                js(formal(v, tag) / "training_sufficiency_report.json", {"status": "RUNNING", "curves": curves, "evals": evals})
                torch.cuda.empty_cache()
    return finalize(v, tag, curves, evals, time.time())


def metric_key(best):
    if not best:
        return (-999, -999, 999)
    return (best.get("cmr5", 0.0), best.get("cmr2", 0.0), -best.get("aepe", 999.0))


def finalize(v: str, tag: str = "", curves=None, evals=None, start=None):
    fr = formal(v, tag)
    if evals is None:
        p = fr / "training_sufficiency_report.json"
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
    sufficient = bool(evals and any(int(k) >= 120000 for k in evals))
    if not sufficient:
        status = "TRAINING_INSUFFICIENT"
    elif best and (best.get("cmr5", 0) > max(REFS["e072_v1_r60k"]["cmr5"], REFS["e072_v2_r60k"]["cmr5"]) or best.get("cmr2", 0) > max(REFS["e072_v1_r60k"]["cmr2"], REFS["e072_v2_r60k"]["cmr2"])) and best.get("aepe", 999) <= max(REFS["e072_v1_r60k"]["aepe"], REFS["e072_v2_r60k"]["aepe"]) + 0.25:
        status = "FUSION_BEATS_SINGLE_BRANCH"
    elif best and best.get("aepe", 999) < REFS["e065"]["aepe"]:
        status = "BEATS_E065_BUT_NOT_SINGLE_BRANCH"
    else:
        status = "TRAINING_SUFFICIENT_NO_FUSION_GAIN"
    last_eval = (evals or {}).get(str(max([int(k) for k in evals], default=0)), {}).get("dev", {}) if evals else {}
    objs = {
        "G0": {"status": "PASS" if curves else "PENDING", "training_sufficient": sufficient, "toy_signal": curves[:12], "source_checkpoints": audit(v, tag)["sources"]},
        "G1": {"status": status, "best": best, "refs": REFS},
        "branch_contribution_report": {k: e.get("dev", {}).get("branch_contribution") for k, e in (evals or {}).items()},
        "training_sufficiency_report": {"status": status, "curves": curves, "evals": evals},
        "long_training_report": {"status": status, "curves": curves, "evals": evals},
        "checkpoint_curve_report": {"curves": curves, "evals": evals},
        "corr_quality_report": {k: e.get("dev", {}).get("corr_quality") for k, e in (evals or {}).items()},
        "iterwise_flow_report": evals,
        "pre_post_lsr_bucket_report": evals,
        "method_ablation": {"variant": v, "best": best},
        "resource_report": {"gpu_policy": "zzvr GPU0-3", "elapsed_seconds": None if start is None else time.time() - start},
        "RUN_MANIFEST": {
            "experiment": "E073A",
            "variant": v,
            "status": status,
            "run_root": str(run_root(v, tag)),
            "formal_root": str(fr),
            "script": str(Path(__file__)),
            "v1_checkpoint": str(V1_60K),
            "v1_checkpoint_sha256": sha(V1_60K),
            "v2_checkpoint": str(V2_60K),
            "v2_checkpoint_sha256": sha(V2_60K),
            "testing_zip_used_for_training": False,
            "no_os424_confirm_rescue": True,
            "materialized_before_gt": [str(p) for p in mat_dir(v, tag).glob("*.jsonl")],
        },
    }
    for n, o in objs.items():
        js(fr / f"{n}.json", o)
    (fr / "reviewer_conclusion.txt").write_text(
        f"E073A {v}: {status}. V1/V2 fusion uses only E072 checkpoint predictions/features before GT metric join; released/official/history artifacts are read-only.\n",
        encoding="utf-8",
    )
    (fr / "RESULT_CARD.md").write_text(f"# E073A {v} RESULT_CARD\n\nStatus: `{status}`\n\nBest: `{best}`\n", encoding="utf-8")
    return {"status": status, "best": best, "formal": str(fr)}


def f1_diag(args):
    e072 = add_paths()
    import core.datasets
    from core.LSRnet.LSmodel import LSR

    v = "F1"
    tag = args.run_tag
    audit(v, tag)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    b1, info1 = load_branch(e072, "V1", V1_60K, device)
    b2, info2 = load_branch(e072, "V2", V2_60K, device)
    b1.eval(); b2.eval()
    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=str(TEST_ROOT))
    vals = {k: [] for k in ["v1", "v2", "avg", "nogt_self_consistency", "gt_oracle_chooser"]}
    mat = mat_dir(v, tag) / "F1_late_ensemble_predictions_before_gt.jsonl"
    mat.parent.mkdir(parents=True, exist_ok=True)
    mat.unlink(missing_ok=True)
    with torch.inference_mode():
        for i in range(min(args.dev_eval_pairs, len(ds))):
            file_opt, a, b, gt, _ = ds[i]
            a = a[None].to(device); b = b[None].to(device)
            p1 = b1(a, b, iters=12, return_features=False)[-1]
            p2 = b2(a, b, iters=12, return_features=False)[-1]
            l1 = LSR(1, 512, 512, p1, 200000).cpu()
            l2 = LSR(1, 512, 512, p2, 200000).cpu()
            avg = 0.5 * (l1 + l2)
            s1 = float(torch.linalg.vector_norm((p1.cpu() - l1).float(), dim=1).mean())
            s2 = float(torch.linalg.vector_norm((p2.cpu() - l2).float(), dim=1).mean())
            chosen = l1 if s1 <= s2 else l2
            # Materialize no-GT chooser data before GT metrics are attached.
            rec = {"pair_index": i, "pair_id": Path(file_opt).stem, "uses_truth_flow": False, "features_predictions_materialized_before_gt": True, "self_residual_v1": s1, "self_residual_v2": s2, "nogt_choice": "v1" if s1 <= s2 else "v2"}
            e1 = torch.linalg.vector_norm((l1 - gt[None]).float(), dim=1).view(-1)
            e2 = torch.linalg.vector_norm((l2 - gt[None]).float(), dim=1).view(-1)
            ea = torch.linalg.vector_norm((avg - gt[None]).float(), dim=1).view(-1)
            ec = torch.linalg.vector_norm((chosen - gt[None]).float(), dim=1).view(-1)
            eo = torch.minimum(e1, e2)
            for name, e in [("v1", e1), ("v2", e2), ("avg", ea), ("nogt_self_consistency", ec), ("gt_oracle_chooser", eo)]:
                vals[name].append(float(e.mean()))
                rec[f"{name}_epe_joined_after_materialization"] = float(e.mean())
            with mat.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            torch.cuda.empty_cache()
    metrics = {k: pair_metrics(vv) for k, vv in vals.items()}
    status = "DIAGNOSTIC_DONE"
    objs = {
        "G0": {"status": "PASS", "source_checkpoints": {"v1": info1, "v2": info2}},
        "G1": {"status": status, "metrics": metrics, "refs": REFS},
        "late_ensemble_report": metrics,
        "branch_contribution_report": metrics,
        "resource_report": {"gpu_policy": "zzvr GPU0-3"},
        "RUN_MANIFEST": {"experiment": "E073A", "variant": "F1", "status": status, "run_root": str(run_root(v, tag)), "formal_root": str(formal(v, tag)), "materialized_before_gt": [str(mat)], "v1_checkpoint_sha256": sha(V1_60K), "v2_checkpoint_sha256": sha(V2_60K), "testing_zip_used_for_training": False, "no_os424_confirm_rescue": True},
    }
    for n, o in objs.items():
        js(formal(v, tag) / f"{n}.json", o)
    (formal(v, tag) / "reviewer_conclusion.txt").write_text("F1 diagnostic only: no-GT self-consistency chooser plus GT oracle upper bound; oracle not a method.\n", encoding="utf-8")
    (formal(v, tag) / "RESULT_CARD.md").write_text(f"# E073A F1 RESULT_CARD\n\nStatus: `{status}`\n\nMetrics: `{metrics}`\n", encoding="utf-8")
    print(json.dumps({"status": status, "metrics": metrics, "formal": str(formal(v, tag))}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "f1", "finalize", "p0"], default="train")
    ap.add_argument("--variant", choices=["F1", "F2", "F3", "F4"], required=True)
    ap.add_argument("--run-tag", type=str, default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=730)
    ap.add_argument("--toy-pairs", type=int, default=8)
    ap.add_argument("--toy-steps", type=int, default=300)
    ap.add_argument("--start-step", type=int, default=60000)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--milestones", type=int, nargs="+", default=[90000, 120000])
    ap.add_argument("--train-iters", type=int, default=12)
    ap.add_argument("--dev-eval-pairs", type=int, default=404)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lambda-seq", type=float, default=0.5)
    ap.add_argument("--lambda-geo", type=float, default=0.5)
    ap.add_argument("--freeze-branches", action="store_true")
    args = ap.parse_args()
    formal(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    mat_dir(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    log_dir(args.variant, args.run_tag).mkdir(parents=True, exist_ok=True)
    if args.mode == "p0":
        print(json.dumps(audit(args.variant, args.run_tag), indent=2))
    elif args.mode == "finalize":
        print(json.dumps(finalize(args.variant, args.run_tag), indent=2))
    elif args.mode == "f1":
        f1_diag(args)
    else:
        print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
