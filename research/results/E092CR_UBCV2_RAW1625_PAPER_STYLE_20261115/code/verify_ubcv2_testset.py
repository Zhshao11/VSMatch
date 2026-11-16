#!/usr/bin/env python3
"""Read-only verification for a generated UBCv2 GDROS-layout test set.

Checks:
  1. opt / sar_warped / truth_flow counts match.
  2. ``truth_flow`` is exactly ``M@[x,y,1] - [x,y]`` for the stored datum M.
  3. ``warped_sar(M*p) == sar(p)`` on sampled interior points (same modality,
     so this is a strict image-level check of the warp/flow direction).
  4. ``core.datasets.opt_sar_test`` can load the generated split.

Run on zzvr (cwd matters for GDROS core imports)::

  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/verify_ubcv2_testset.py --root <generated>/seed901
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import struct
import sys

import cv2
import numpy as np

sys.path.insert(0, "/mnt/sda/zhushihao/VS_Match/external_repos/E068A_GDROS_official")
import core.datasets  # noqa: E402

FLO_MAGIC = 202021.25


def read_flo(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = struct.unpack("f", f.read(4))[0]
        w, h = struct.unpack("ii", f.read(8))
        assert abs(magic - FLO_MAGIC) < 1e-6, (path, magic)
        return np.fromfile(f, np.float32, count=2 * w * h).reshape(h, w, 2)


def bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample a BGR uint8 image at float (x,y); returns float [N,3] or [N]."""
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    dx = (xs - x0)[..., None]
    dy = (ys - y0)[..., None]
    a = img[y0, x0].astype(np.float64)
    b = img[y0, x1].astype(np.float64)
    c = img[y1, x0].astype(np.float64)
    d = img[y1, x1].astype(np.float64)
    return ((1 - dx) * (1 - dy) * a + dx * (1 - dy) * b
            + (1 - dx) * dy * c + dx * dy * d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="seed root containing testing/")
    ap.add_argument("--raw-root", default=None,
                    help="optional original rgb/sar root for the image-level check")
    ap.add_argument("--check-pairs", type=int, default=3)
    args = ap.parse_args()

    split = os.path.join(args.root, "testing")
    opt = sorted(glob.glob(os.path.join(split, "opt", "*.png")))
    sar = sorted(glob.glob(os.path.join(split, "sar_warped", "*.png")))
    flo = sorted(glob.glob(os.path.join(split, "truth_flow", "*.flo")))
    print("counts opt=%d sar=%d flo=%d" % (len(opt), len(sar), len(flo)), flush=True)
    assert len(opt) == len(sar) == len(flo) and opt, "count mismatch"

    ok = True
    rng = np.random.default_rng(0)
    for i in range(min(args.check_pairs, len(opt))):
        flow = read_flo(flo[i])
        h, w = flow.shape[:2]
        rec = json.load(open(os.path.join(
            split, "datum", os.path.basename(opt[i]).replace(".png", ".json"))))
        M = np.asarray(rec["affine_src_to_target"], dtype=np.float64)
        ys, xs = np.mgrid[0:h, 0:w]
        expected = np.stack((M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs,
                             M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys), -1)
        flow_delta = float(np.abs(flow - expected).max())

        # Fit M back from the written flow (independent of the stored datum).
        sel = rng.choice(h * w, 20000, replace=False)
        X = np.stack([xs.ravel()[sel], ys.ravel()[sel], np.ones(sel.size)], 1)
        Y = flow.reshape(-1, 2)[sel]
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        M_fit = np.array([[1 + B[0, 0], B[1, 0], B[2, 0]],
                          [B[0, 1], 1 + B[1, 1], B[2, 1]]])
        fit_delta = float(np.abs(M_fit - M).max())

        img_delta = None
        if args.raw_root:
            stem = rec["source_stem"]
            raw_sar = cv2.imread(os.path.join(args.raw_root, "sar", stem + ".tif"),
                                 cv2.IMREAD_UNCHANGED)
            if raw_sar.dtype != np.uint8:
                conv = rec["sar_conversion"]
                lo, hi = conv["lo_db"], conv["hi_db"]
                db = 20.0 * np.log10(np.maximum(raw_sar.astype(np.float32), 0.0) + 1e-3)
                u = np.clip((db - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)
                raw_sar = np.ascontiguousarray(np.repeat(u[..., None], 3, axis=2))
            elif raw_sar.ndim == 2:
                raw_sar = cv2.cvtColor(raw_sar, cv2.COLOR_GRAY2BGR)
            expected = cv2.warpAffine(np.ascontiguousarray(raw_sar[..., :3]), M, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped = cv2.imread(sar[i], cv2.IMREAD_UNCHANGED)
            if warped.ndim == 2:
                warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            img_delta = float(np.abs(expected.astype(np.int16)
                                     - warped[..., :3].astype(np.int16)).max())
            if img_delta > 0:
                ok = False

        print("pair[%d] flow_max_delta=%.6f M_fit_max_delta=%.6f "
              "warp_max_delta=%s" % (i, flow_delta, fit_delta,
                                     "%.1f" % img_delta if img_delta is not None else "-"),
              flush=True)
        if flow_delta > 1e-3 or fit_delta > 1e-3:
            ok = False

    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=args.root)
    name, a, b, gt, valid = ds[0]
    print("dataset load OK: n=%d img=%s gt=%s valid_frac=%.3f" %
          (len(ds), tuple(a.shape), tuple(gt.shape), float(valid.float().mean())),
          flush=True)
    assert tuple(a.shape) == (3, 512, 512) and tuple(gt.shape) == (2, 512, 512)
    print("VERIFY %s" % ("PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
