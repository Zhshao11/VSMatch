#!/usr/bin/env python3
"""Root-cause check: GDROS official test.py metric vs vector-EPE metric.

Only the GDROS released baseline is evaluated (no custom method involved).
Same checkpoint, same 424 pairs, same LSR, SAME predicted flow per pair.
For each pair we compute BOTH error definitions from that one prediction:

  A) official test.py:
         epe_comp = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
     flow_pre is [1,2,H,W], flow_gt is [2,H,W]; broadcasting yields [1,2,H,W],
     summing over dim=0 then sqrt gives [2,H,W] = |dx|, |dy|  (per-component
     absolute error), NOT the endpoint error sqrt(dx^2+dy^2).

  B) vector EPE (used by eval_os_unified.py / E075A eval_model):
         epe_vec = linalg.vector_norm(flow_pre - flow_gt, dim=1)  -> [1,H,W]

Both are aggregated the way official test.py does (per-pair mean -> MAE_list,
CMR@T over pairs), plus pixel-level CMR for cross-checking the printed
"1px / 3px / 5px" line.

Run on zzvr:
  cd /mnt/sda/zhushihao/VS_Match/external_repos/E068A_GDROS_official
  PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 \
    /home/zzvr/miniconda3/envs/compared/bin/python /tmp/diag_gdros_epe.py
"""
from __future__ import annotations
import sys, json, math, time
import numpy as np
import torch

ROOT = "/mnt/sda/zhushihao/VS_Match"
OFFICIAL = ROOT + "/external_repos/E068A_GDROS_official"
TEST_ROOT = ROOT + "/datasets/extracted/gdros_official_e003a_testing"
CKPT = ROOT + "/GDROS/official_models_reference/GDROS-os512.pth"
sys.path.insert(0, OFFICIAL)

import core.datasets
from core.LSRnet.LSmodel import LSR
from core.GDROSnet.FeatureFormer.gdros import GDROS
from configs.default import get_cfg

TAUS = (1, 2, 3, 4, 5)
NPOINT = 200000
ITERS = 32


@torch.inference_mode()
def main():
    cfg = get_cfg()
    cfg.transformer = "featureformerwithlbl"
    cfg.image_size = [512, 512]
    model = torch.nn.DataParallel(GDROS(cfg[cfg.transformer])).cuda()
    model.load_state_dict(torch.load(CKPT, map_location="cuda:0"))
    model.eval()
    net = model.module  # official test.py calls exp_func(model.module)

    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    n = len(ds)
    print("pairs=%d" % n, flush=True)

    pair_comp, pair_vec = [], []
    px_comp = {t: 0 for t in TAUS}
    px_vec = {t: 0 for t in TAUS}
    n_comp_el, n_vec_el = 0, 0
    s_comp, s_vec = 0.0, 0.0     # for pixel-level AEPE
    s2_comp, s2_vec = 0.0, 0.0
    t0 = time.time()
    for i in range(n):
        _, a_img, b_img, gt, valid = ds[i]
        a = a_img[None].cuda()
        b = b_img[None].cuda()
        bs, _, h, w = a.shape
        _, flow_pre1 = net(a, b, iters=ITERS, test_mode=True)
        flow_pre = LSR(bs, h, w, flow_pre1, NPOINT)   # CPU, [1,2,H,W]

        diff = flow_pre - gt[None]                    # [1,2,H,W]
        if i == 0:
            print("shape flow_pre=%s gt=%s diff=%s" %
                  (tuple(flow_pre.shape), tuple(gt.shape), tuple(diff.shape)), flush=True)

        # A) official
        e_comp = torch.sum(diff ** 2, dim=0).sqrt()    # [2,H,W] = |dx|,|dy|
        # B) vector EPE
        e_vec = torch.linalg.vector_norm(diff.float(), dim=1)  # [1,H,W]

        ec = e_comp.reshape(-1).double().numpy()
        ev = e_vec.reshape(-1).double().numpy()
        pair_comp.append(float(ec.mean()))
        pair_vec.append(float(ev.mean()))
        s_comp += float(ec.sum()); s2_comp += float(np.square(ec).sum()); n_comp_el += ec.size
        s_vec += float(ev.sum()); s2_vec += float(np.square(ev).sum()); n_vec_el += ev.size
        for t in TAUS:
            px_comp[t] += int((ec < t).sum())
            px_vec[t] += int((ev < t).sum())
        if i % 100 == 0 or i == n - 1:
            print("%d/%d elapsed=%.1fs" % (i + 1, n, time.time() - t0), flush=True)

    pc = np.asarray(pair_comp)
    pv = np.asarray(pair_vec)

    def report(pairs, px, n_el, s, s2, label):
        out = {
            "AEPE_pair_mean": float(pairs.mean()),
            "pixel_AEPE": s / n_el,
            "pixel_RMSE": math.sqrt(s2 / n_el),
            "var_of_pair_means": float(pairs.var()),
        }
        for t in TAUS:
            out["CMR@%d_pair" % t] = 100.0 * float((pairs < t).mean())
            out["CMR@%d_pixel" % t] = 100.0 * px[t] / n_el
            inl = pairs[pairs < t]
            out["MAE@%d_pair" % t] = float(inl.mean()) if inl.size else None
        print("\n=== %s ===" % label, flush=True)
        print(json.dumps(out, indent=2), flush=True)
        return out

    a = report(pc, px_comp, n_comp_el, s_comp, s2_comp, "A official test.py: per-component |d|")
    b = report(pv, px_vec, n_vec_el, s_vec, s2_vec, "B vector EPE sqrt(dx^2+dy^2)")
    print("\n=== ratio B/A (AEPE) = %.4f ; theoretical pi/2 = %.4f ===" %
          (b["AEPE_pair_mean"] / a["AEPE_pair_mean"], math.pi / 2), flush=True)
    print("RESULT_JSON " + json.dumps({"official_component": a, "vector_epe": b},
                                      indent=2), flush=True)


if __name__ == "__main__":
    main()
