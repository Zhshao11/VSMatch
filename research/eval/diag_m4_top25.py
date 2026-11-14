#!/usr/bin/env python3
"""Rank all 424 OS pairs by M4@80K EPE and attribute the worst 25 failures.

For EVERY pair, from ONE M4 forward pass we record:
  post  = LSR-refined fused flow           (final prediction, ranking metric)
  raw   = fused flow BEFORE LSR             (feature/gate stage, no geometry)
  B, C  = frozen branches, LSR-refined
  avg   = 0.5*(B+C) then LSR
  oracle= per-component min(error_B, error_C)   (upper bound of gate)
  gate  = region-gate mean/std
  gdros = GDROS released on the SAME pair    (is the pair intrinsically hard?)
  affine fits on GT and on post/raw: scale, rotation, translation,
          alpha = (c_pred . c_gt)/|c_gt|^2   (translation gain; 1.0 = correct)
          local_resid = non-affine residual of raw flow

Attribution logic:
  LSR geometry   -> raw is good but post is bad / GT-affine replacement fixes it
  region gate    -> oracle is much better than post (both branches OK, gate wrong)
  feature match  -> raw AND oracle are bad (both branches mis-match), often
                    with GDROS succeeding on the same pair

Metric: official test.py per-pair EPE(k) on per-component |dx|,|dy|.
Vector-EPE is recorded alongside for cross-checking.

Run on zzvr (cwd matters for E038B paths):
  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  CUDA_VISIBLE_DEVICES=2 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/diag_m4_top25.py --max-pairs 424 --gpu 0 --top 25 \
    --out .codex_tmp/m4_top25_diag_20261113.json
"""
from __future__ import annotations
import sys, argparse, json, math, time, gc
from pathlib import Path
import numpy as np
import torch

ROOT = "/mnt/sda/zhushihao/VS_Match"
TMP = ROOT + "/image_matching_part2_migrated/.codex_tmp"
OFFICIAL = ROOT + "/external_repos/E068A_GDROS_official"
sys.path.insert(0, TMP)
sys.path.insert(0, OFFICIAL)

import timm
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

TEST_ROOT = ROOT + "/datasets/extracted/gdros_official_e003a_testing"
BASE_CKPT = ROOT + "/GDROS/official_models_reference/GDROS-os512.pth"
M4_CKPT = (ROOT + "/GDROS/experiments/sparsegeo/runs/"
           "E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/"
           "run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt")
H = W = 512
ITERS = 32
NPOINT = 200000

# Centred-coordinate design matrix for a 6-param affine fit (shared by all pairs).
_yy, _xx = np.mgrid[0:H, 0:W]
P = torch.from_numpy(np.stack([_xx.ravel() - 255.5, _yy.ravel() - 255.5,
                               np.ones(H * W)], 1).astype(np.float64)).double()
PTP_INV = torch.linalg.inv(P.T @ P)


def fit_affine(flow_cpu):
    """flow [1,2,H,W] CPU -> (scale, rotation_deg, c[2], B[3,2]).

    Displacement model d(p) = A p + c on centred coords; absolute linear map M = I + A.
    """
    D = flow_cpu.reshape(1, 2, -1).squeeze(0).T.double()   # [N,2]
    B = PTP_INV @ (P.T @ D)                                # [3,2]
    A = B[:2].T                                            # [2,2]
    M = torch.eye(2, dtype=torch.float64) + A
    s = float(torch.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    theta = float(torch.rad2deg(torch.atan2(M[1, 0], M[0, 0])))
    return s, theta, B[2].double(), B


def comp_err(flow_cpu, gt_cpu):
    """Official test.py error map: [2,H,W] of |dx|,|dy| (flow is [1,2,H,W])."""
    return torch.sum((flow_cpu - gt_cpu[None]) ** 2, dim=0).sqrt()


def vec_err(flow_cpu, gt_cpu):
    return torch.linalg.vector_norm((flow_cpu - gt_cpu[None]).float(), dim=1)


def mean_comp(flow_cpu, gt_cpu):
    return float(comp_err(flow_cpu, gt_cpu).reshape(-1).mean())


def mean_vec(flow_cpu, gt_cpu):
    return float(vec_err(flow_cpu, gt_cpu).reshape(-1).mean())


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=424)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    device = torch.device("cuda:%d" % args.gpu)
    torch.cuda.set_device(device)

    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    n = min(args.max_pairs, len(ds))
    print("dataset pairs=%d evaluating=%d" % (len(ds), n), flush=True)

    # ---------- Phase A: GDROS released per-pair EPE (same pair order) ----------
    t0 = time.time()
    cfg = get_cfg()
    cfg.transformer = "featureformerwithlbl"
    cfg.image_size = [512, 512]
    base = torch.nn.DataParallel(GDROS(cfg[cfg.transformer])).to(device)
    base.load_state_dict(torch.load(BASE_CKPT, map_location=device))
    base.eval()
    gdros_epe = {}
    for i in range(n):
        _, a_img, b_img, gt, _ = ds[i]
        a = a_img[None].to(device)
        b = b_img[None].to(device)
        _, f1 = base.module(a, b, iters=ITERS, test_mode=True)
        fbase = LSR(a.shape[0], H, W, f1, NPOINT).cpu()
        gdros_epe[i] = mean_comp(fbase, gt)
        if i % 100 == 0:
            print("[gdros] %d/%d elapsed=%.1fs" % (i + 1, n, time.time() - t0), flush=True)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    print("[gdros] done %.1fs" % (time.time() - t0), flush=True)

    # ---------- Phase B: M4 full decomposition ----------
    t1 = time.time()
    m4 = e075.PairFusion(e074, "region", device).to(device)
    payload = torch.load(M4_CKPT, map_location=device)
    miss = m4.load_state_dict(payload["model"], strict=False)
    print("[m4] missing=%s unexpected=%s" %
          (list(miss.missing_keys), list(miss.unexpected_keys)), flush=True)
    m4.eval()
    del payload
    gc.collect()
    torch.cuda.empty_cache()

    records = []
    for i in range(n):
        file_opt, a_img, b_img, gt, valid = ds[i]
        a = a_img[None].to(device)
        b = b_img[None].to(device)
        preds = m4(a, b, iters=ITERS, lsr_fn=e074.lsr)     # exact eval-time path
        pb, pc = m4.last_branch_preds["B"], m4.last_branch_preds["C"]

        raw = preds[-1].cpu()                                # fused, before LSR
        post = e074.lsr(preds[-1]).cpu()                    # final prediction
        lb = e074.lsr(pb[-1]).cpu()
        lc = e074.lsr(pc[-1]).cpu()
        lavg = e074.lsr(0.5 * (pb[-1] + pc[-1])).cpu()

        e_b = comp_err(lb, gt)
        e_c = comp_err(lc, gt)
        e_oracle = float(torch.minimum(e_b, e_c).reshape(-1).mean())

        s_gt, th_gt, c_gt, _ = fit_affine(gt[None])
        s_post, th_post, c_post, _ = fit_affine(post)
        _, _, _, B_raw = fit_affine(raw)
        fa = (P @ B_raw).T.reshape(1, 2, H, W).float()
        local_resid = float(torch.linalg.vector_norm((raw.float() - fa), dim=1).mean())

        denom = float(torch.dot(c_gt, c_gt))
        alpha = float(torch.dot(c_post, c_gt) / denom) if denom > 1e-6 else None
        t_err = float(torch.linalg.vector_norm(c_post - c_gt))

        rec = {
            "index": i,
            "pair_id": Path(file_opt).stem,
            "opt": str(file_opt),
            "sar": str(Path(file_opt).parent.parent / "sar_warped" /
                       (Path(file_opt).stem.split("_")[0] + "_sar_warped.png")),
            "gt_flow": str(Path(file_opt).parent.parent / "truth_flow" /
                           (Path(file_opt).stem.split("_")[0] + ".flo")),
            "epe_post": mean_comp(post, gt),          # ranking metric (official)
            "epe_post_vec": mean_vec(post, gt),
            "epe_pre": mean_comp(raw, gt),             # LSR 前
            "epe_pre_vec": mean_vec(raw, gt),
            "epe_b": float(e_b.reshape(-1).mean()),
            "epe_c": float(e_c.reshape(-1).mean()),
            "epe_avg": mean_comp(lavg, gt),
            "epe_oracle": e_oracle,
            "epe_gdros": gdros_epe[i],
            "gate_mean": m4.last_aux.get("gate_mean"),
            "gate_std": m4.last_aux.get("gate_std"),
            "gate_gap_vs_oracle": mean_comp(post, gt) - e_oracle,
            "lsr_delta": mean_comp(post, gt) - mean_comp(raw, gt),
            "gt_scale": s_gt,
            "gt_rot_deg": th_gt,
            "pred_scale": s_post,
            "ds": s_post - s_gt,
            "dtheta_deg": th_post - th_gt,
            "gt_translation": [float(c_gt[0]), float(c_gt[1])],
            "pred_translation": [float(c_post[0]), float(c_post[1])],
            "alpha_translation_gain": alpha,
            "translation_err": t_err,
            "raw_nonaffine_resid": local_resid,
        }
        records.append(rec)
        if i % 25 == 0 or i == n - 1:
            print("[m4] %d/%d epe_post=%.3f elapsed=%.1fs" %
                  (i + 1, n, rec["epe_post"], time.time() - t1), flush=True)
        del preds, pb, pc, raw, post, lb, lc, lavg

    records.sort(key=lambda r: r["epe_post"], reverse=True)
    for rank, r in enumerate(records, 1):
        r["rank"] = rank

    # ---------- summaries ----------
    def avg(rows, k):
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return float(np.mean(vals)) if vals else None

    top = records[:args.top]
    rest = records[args.top:]
    keys = ["epe_post", "epe_pre", "epe_b", "epe_c", "epe_avg", "epe_oracle",
            "epe_gdros", "gate_mean", "gate_std", "lsr_delta",
            "gt_scale", "ds", "dtheta_deg", "alpha_translation_gain",
            "translation_err", "raw_nonaffine_resid", "gate_gap_vs_oracle"]
    summary = {k: {"top25": avg(top, k), "rest": avg(rest, k),
                   "all": avg(records, k)} for k in keys}
    vec_top = {r["index"] for r in sorted(records, key=lambda x: x["epe_post_vec"],
                                          reverse=True)[:args.top]}
    summary["top25_overlap_official_vs_vector_EPE"] = len(
        vec_top & {r["index"] for r in top})

    out = {
        "config": {"pairs": n, "top": args.top, "iters": ITERS, "lsr_npoint": NPOINT,
                  "metric": "official test.py per-pair EPE on |dx|,|dy|",
                  "elapsed_gdros_sec": round(time.time() - t0, 1),
                  "elapsed_m4_sec": round(time.time() - t1, 1)},
        "summary": summary,
        "top25": top,
        "all_records": records,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("WROTE " + args.out, flush=True)

    print("\n### Top-%d by EPE(k) (official)" % args.top, flush=True)
    hdr = ("| rank | pair | EPE_post | EPE_pre | LSR_delta | B | C | oracle | avg | "
           "gate | GDROS | gt_scale | alpha | t_err |")
    print(hdr, flush=True)
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|", flush=True)
    for r in top:
        g = r["gate_mean"]
        a = r["alpha_translation_gain"]
        print("| %d | %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %s | %.2f |" %
              (r["rank"], r["pair_id"], r["epe_post"], r["epe_pre"], r["lsr_delta"],
               r["epe_b"], r["epe_c"], r["epe_oracle"], r["epe_avg"],
               g if g is not None else float("nan"), r["epe_gdros"], r["gt_scale"],
               "%.2f" % a if a is not None else "--", r["translation_err"]), flush=True)
    print("\nSUMMARY_JSON " + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
