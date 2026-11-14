#!/usr/bin/env python3
"""Unified OS-dataset evaluator for GDROS released baseline and E075A-M4.

Runs BOTH models through the SAME per-pixel metric used by e075a eval_model
(AEPE + CMR@1/2/5/10/16 on LSR-refined flow, masked by valid pixels), so the
baseline vs best-method comparison is apples-to-apples. Also reports per-image
CMR (mean-EPE-per-pair < T) to cross-check against the official gdros_released
reference numbers (33.73 / 81.60 / 99.53 for T=1/2/5).
"""
from __future__ import annotations
import sys, argparse, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/mnt/sda/zhushihao/VS_Match"
TMP = ROOT + "/image_matching_part2_migrated/.codex_tmp"
OFFICIAL = ROOT + "/external_repos/E068A_GDROS_official"
sys.path.insert(0, TMP)
sys.path.insert(0, OFFICIAL)  # for 'core' and 'configs'

import timm, sys
# Vendored GDROS encoder.py uses `from timm.layers import ...` (timm>=0.6 API),
# but the available timm (0.4.12) only exposes `timm.models.layers`. Inject a
# shim so the import resolves without editing the external repo.
if not hasattr(timm, "layers"):
    import timm.models.layers as _timm_layers
    sys.modules.setdefault("timm.layers", _timm_layers)

import core.datasets
from core.LSRnet.LSmodel import LSR
from core.GDROSnet.FeatureFormer.gdros import GDROS
from configs.default import get_cfg

import e074a_f2_f3_twostage_fusion as e074
e072, e073 = e074.add_paths()
import e075a_preserve_high_precision_recover_c2c5 as e075

B_CKPT = ROOT + "/GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-b-stage180k/checkpoints/B_step180000.pt"
C_CKPT = ROOT + "/GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-c-stage180k/checkpoints/C_step180000.pt"
TEST_ROOT = ROOT + "/datasets/extracted/gdros_official_e003a_testing"
BASE_CKPT = ROOT + "/GDROS/official_models_reference/GDROS-os512.pth"
M4_CKPT = ROOT + "/GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt"


def per_pixel_metrics(e_all):
    a = np.asarray(e_all, np.float64)
    return {
        "pairs": int(a.size),
        "aepe": float(a.mean()),
        "cmr1": float((a < 1).mean() * 100),
        "cmr2": float((a < 2).mean() * 100),
        "cmr5": float((a < 5).mean() * 100),
        "cmr10": float((a < 10).mean() * 100),
        "cmr16": float((a < 16).mean() * 100),
    }


def per_image_metrics(pair_epe):
    a = np.asarray(pair_epe, np.float64)
    return {
        "pairs": int(a.size),
        "mae": float(a.mean()),
        "cmr1": float((a < 1).mean() * 100),
        "cmr2": float((a < 2).mean() * 100),
        "cmr5": float((a < 5).mean() * 100),
        "cmr10": float((a < 10).mean() * 100),
        "cmr16": float((a < 16).mean() * 100),
    }


@torch.inference_mode()
def eval_baseline(device, max_pairs):
    cfg = get_cfg()
    cfg.transformer = "featureformerwithlbl"
    cfg.image_size = [512, 512]
    # Official GDROS-os512.pth was saved under DataParallel (module. prefix);
    # mirror test.py's construction so the keys line up.
    model = torch.nn.DataParallel(GDROS(cfg[cfg.transformer])).to(device)
    ckpt = torch.load(BASE_CKPT, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    e_all, pair_epe = [], []
    for i in range(min(max_pairs, len(ds))):
        file_opt, a, b, gt, valid = ds[i]
        a = a[None].to(device); b = b[None].to(device)
        # gt/valid stay on CPU; LSR returns a CPU flow (matches e075a eval_model).
        _, flow_pre1 = model(a, b, iters=32, test_mode=True)
        bs, _, h, w = a.shape
        flow_pre = LSR(bs, h, w, flow_pre1, 200000).cpu()
        e = torch.linalg.vector_norm((flow_pre - gt[None]).float(), dim=1).view(-1)[valid.view(-1).bool()]
        e = e.cpu().numpy()
        e_all.append(e)
        pair_epe.append(float(e.mean()))
        if i % 50 == 0:
            print(f"[baseline] pair {i}/{min(max_pairs, len(ds))} epe={e.mean():.3f}", flush=True)
    e_all = np.concatenate(e_all)
    return {
        "per_pixel": per_pixel_metrics(e_all),
        "per_image": per_image_metrics(pair_epe),
        "checkpoint": BASE_CKPT,
    }


@torch.inference_mode()
def eval_m4(device, max_pairs):
    model = e075.PairFusion(e074, "region", device).to(device)
    payload = torch.load(M4_CKPT, map_location=device)
    miss = model.load_state_dict(payload["model"], strict=False)
    print(f"[m4] missing={list(miss.missing_keys)} unexpected={list(miss.unexpected_keys)}", flush=True)
    model.eval()
    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    e_all, pair_epe = [], []
    for i in range(min(max_pairs, len(ds))):
        file_opt, a, b, gt, valid = ds[i]
        a = a[None].to(device); b = b[None].to(device)
        # gt/valid stay on CPU; e074.lsr output moved to CPU before comparing.
        preds = model(a, b, iters=32, lsr_fn=e074.lsr)
        post = e074.lsr(preds[-1]).cpu()
        e = torch.linalg.vector_norm((post - gt[None]).float(), dim=1).view(-1)[valid.view(-1).bool()]
        e = e.cpu().numpy()
        e_all.append(e)
        pair_epe.append(float(e.mean()))
        if i % 50 == 0:
            print(f"[m4] pair {i}/{min(max_pairs, len(ds))} epe={e.mean():.3f}", flush=True)
    e_all = np.concatenate(e_all)
    return {
        "per_pixel": per_pixel_metrics(e_all),
        "per_image": per_image_metrics(pair_epe),
        "checkpoint": M4_CKPT,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["baseline", "m4", "both"], default="both")
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--max-pairs", type=int, default=404)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    out = {}
    if args.which in ("baseline", "both"):
        out["baseline_gdros_os512"] = eval_baseline(device, args.max_pairs)
    if args.which in ("m4", "both"):
        out["e075a_m4_step80000"] = eval_m4(device, args.max_pairs)
    print("RESULT_JSON " + json.dumps(out, indent=2, ensure_ascii=False))
