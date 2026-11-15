#!/usr/bin/env python3
"""Per-pair GDROS diagnostic on the generated UBCv2 test set.

Question this answers
---------------------
Our UBCv2 test set warps SAR with ``M = [[s cos, -s sin, tx], [s sin, s cos, ty]]``
-- i.e. rotation/scale about the **origin** (top-left), which is the convention
the official OS ``datum/*.npy[2]`` uses.  With +-10 deg about the origin a
512x512 image loses big triangular wedges to the ``borderValue=0`` fill.  Those
wedges carry no SAR information, so *no* method can predict them, yet
``truth_flow`` marks every pixel valid and the official metric averages over all
pixels.

This script records, per pair:

  epe_full      official per-pair EPE(k) over ALL pixels (the published number)
  epe_overlap   same, but restricted to pixels whose source really exists
                (mask = warpAffine(ones, M) == 1)  -> diagnostic only
  overlap       fraction of pixels with real source content
  black         fraction of sar_warped pixels that are exactly 0
  M_gt / M_pred predicted flow's least-squares affine vs the ground truth one
  (scale, rotation, tx, ty) read back from ``datum/<id>.json``

so we can bucket the error by transform magnitude and by lost area instead of
guessing why GDROS reports AEPE 22 on this set.

Run on zzvr (cwd matters for E038B path resolution)::

  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  CUDA_VISIBLE_DEVICES=3 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/diag_ubcv2_pairs.py --which baseline --gpu 0 --max-pairs 1625 \
    --out .codex_tmp/diag_ubcv2_pairs_20261115.json
"""
from __future__ import annotations
import argparse, glob, json, os, time
import cv2
import numpy as np
import torch

import eval_os_official_protocol as P

DEFAULT_ROOT = "/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/generated/seed901"


def overlap_mask(h: int, w: int, M: np.ndarray) -> np.ndarray:
    ones = np.ones((h, w), np.uint8)
    return cv2.warpAffine(ones, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def fit_affine(flow: torch.Tensor) -> np.ndarray:
    """flow: [2,H,W] cpu float.  Returns A (2x3) with flow = A@[x,y,1] - [x,y]."""
    _, h, w = flow.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                            torch.arange(w, dtype=torch.float64), indexing="ij")
    X = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.ones(h * w, dtype=torch.float64)], 1)
    dx = flow[0].reshape(-1).double() + xs.reshape(-1)
    dy = flow[1].reshape(-1).double() + ys.reshape(-1)
    Y = torch.stack([dx, dy], 1)
    sol = torch.linalg.lstsq(X, Y).solution           # [3,2]
    A = torch.zeros(2, 3, dtype=torch.float64)
    A[:, :] = sol.T
    return A.numpy()


def build_forward(which, device):
    if which == "baseline":
        from configs.default import get_cfg
        from core.GDROSnet.FeatureFormer.gdros import GDROS
        cfg = get_cfg()
        cfg.transformer = "featureformerwithlbl"
        cfg.image_size = [512, 512]
        model = torch.nn.DataParallel(GDROS(cfg[cfg.transformer])).to(device)
        model.load_state_dict(torch.load(P.BASE_CKPT, map_location=device))
        model.eval()
        net = model.module

        def forward(a, b):
            _, flow_pre1 = net(a, b, iters=P.ITERS, test_mode=True)
            return P.LSR(a.shape[0], a.shape[2], a.shape[3], flow_pre1, P.NPOINT)
        return forward, P.BASE_CKPT
    raise SystemExit("only --which baseline is implemented (M4 is 10x slower)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", default=DEFAULT_ROOT)
    ap.add_argument("--which", default="baseline", choices=["baseline"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=1625)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    P.TEST_ROOT = args.test_root
    device = torch.device("cuda:%d" % args.gpu)
    forward, ckpt = build_forward(args.which, device)

    import core.datasets
    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=P.TEST_ROOT)
    n = min(args.max_pairs, len(ds))
    datum = os.path.join(args.test_root, "testing", "datum")
    print("[diag] pairs=%d device=%s ckpt=%s" % (n, device, ckpt), flush=True)

    rows = []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(n):
            _, a_img, b_img, gt, _ = ds[i]
            a = a_img[None].to(device)
            b = b_img[None].to(device)
            post = forward(a, b)
            diff = post - gt[None]
            e_comp = torch.sum(diff ** 2, dim=0).sqrt()          # [2,H,W]
            _, _, h, w = post.shape

            meta_path = os.path.join(datum, "%06d.json" % i)
            meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
            M_gt = np.array(meta.get("affine_src_to_target", np.eye(2, 3).tolist()))
            mask = overlap_mask(h, w, M_gt)
            m = torch.from_numpy(mask)

            epe_full = float(e_comp.mean())
            epe_ovl = float(e_comp[:, m].mean()) if m.any() else float("nan")
            A_pred = fit_affine(post[0])
            # fit_affine regresses the ABSOLUTE coordinate field, so A_pred is
            # already the full affine M (NOT M-I).  Verified with a synthetic
            # self-test: feeding M-generated flow returns M within 1e-6.
            A_gt = M_gt
            rows.append({
                "pair": i,
                "epe_full": epe_full,
                "epe_overlap": epe_ovl,
                "overlap": float(mask.mean()),
                "scale": float(meta.get("scale", float("nan"))),
                "rot": float(meta.get("rotation_deg", float("nan"))),
                "tx": float(meta.get("translation_px", [float("nan")])[0]),
                "ty": float(meta.get("translation_px", [float("nan"), float("nan")])[1]),
                "affine_pred_err": float(np.abs(A_pred - A_gt).max()),
                "pred_scale": float(np.sqrt(abs(np.linalg.det(A_pred[:, :2])))),
                "pred_rot": float(np.degrees(np.arctan2(A_pred[1, 0], A_pred[0, 0] + 1e-12))),
            })
            if i % 50 == 0 or i == n - 1:
                print("[diag] %d/%d epe_full=%.3f epe_ovl=%.3f overlap=%.3f "
                      "pred_scale=%.3f gt_scale=%.3f t=%.1fs" %
                      (i + 1, n, epe_full, epe_ovl, mask.mean(),
                       rows[-1]["pred_scale"], rows[-1]["scale"], time.time() - t0),
                      flush=True)

    def agg(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}

    full = np.array([r["epe_full"] for r in rows])
    ovl = np.array([r["epe_overlap"] for r in rows])
    ovlf = np.array([r["overlap"] for r in rows])
    summary = {
        "pairs": n,
        "elapsed_sec": round(time.time() - t0, 1),
        "epe_full": agg("epe_full"),
        "epe_overlap": agg("epe_overlap"),
        "AEPE_full": float(full.mean()),
        "AEPE_overlap": float(ovl.mean()),
        "CMR5_full": 100.0 * float((full < 5).mean()),
        "CMR5_overlap": 100.0 * float((ovl < 5).mean()),
        "overlap_area": agg("overlap"),
        "corr_epe_vs_overlap": float(np.corrcoef(full, ovlf)[0, 1]),
        "corr_epe_vs_rot": float(np.corrcoef(
            full, np.abs([r["rot"] for r in rows]))[0, 1]),
        "corr_epe_vs_scale_dev": float(np.corrcoef(
            full, np.abs(np.array([r["scale"] for r in rows]) - 1.0))[0, 1]),
        "bucket_by_overlap": [],
        "bucket_by_absrot": [],
    }
    for lo, hi in ((0.0, 0.6), (0.6, 0.75), (0.75, 0.85), (0.85, 1.01)):
        sel = (ovlf >= lo) & (ovlf < hi)
        if sel.sum():
            summary["bucket_by_overlap"].append({
                "range": [lo, hi], "pairs": int(sel.sum()),
                "AEPE_full": float(full[sel].mean()),
                "AEPE_overlap": float(ovl[sel].mean()),
                "CMR5": 100.0 * float((full[sel] < 5).mean())})
    for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 10.01)):
        sel = (np.abs([r["rot"] for r in rows]) >= lo) & (np.abs([r["rot"] for r in rows]) < hi)
        if sel.sum():
            summary["bucket_by_absrot"].append({
                "range": [lo, hi], "pairs": int(sel.sum()),
                "AEPE_full": float(full[sel].mean()),
                "overlap": float(ovlf[sel].mean()),
                "CMR5": 100.0 * float((full[sel] < 5).mean())})

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "pairs": rows}, f, indent=2)
        print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
