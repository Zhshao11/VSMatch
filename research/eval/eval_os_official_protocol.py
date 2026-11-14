#!/usr/bin/env python3
"""Official GDROS test.py protocol for GDROS released AND E075A-M4@80K.

Same inference as eval_os_full_metrics.py, but the error map and aggregation
are exactly the official test.py semantics:

    epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()   # [2,H,W] = |dx|,|dy|
    MAE_list[k] = mean(epe) over the pair                  # per-pair EPE(k)
    AEPE = mean(MAE_list)
    "RMSE" = np.var(MAE_list)      <-- official code prints variance here
    CMR@T   = 100 * |{k: EPE(k) < T}| / N
    MAE@T   = mean(EPE(k) for EPE(k) < T)   (= AEPE@tau in the paper)
    "RMSE"@T = np.var(inliers)     <-- same mislabel as above

For reference we also accumulate the true vector EPE sqrt(dx^2+dy^2) from the
SAME predicted flow, so the two protocols can be compared without re-running.

Run on zzvr (cwd matters for E038B path resolution):
  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  CUDA_VISIBLE_DEVICES=2 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/eval_os_official_protocol.py --which both --max-pairs 424 --gpu 0 \
    --out .codex_tmp/os_official_protocol_20261113.json
"""
from __future__ import annotations
import sys, argparse, json, math, time, gc
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
TAUS = (1, 2, 3, 4, 5)
NPOINT = 200000
ITERS = 32


class Collector:
    """Per-pair official metric accumulation + optional pixel-level CMR."""

    def __init__(self):
        self.pairs = []
        self.px_cnt = {t: 0 for t in TAUS}
        self.px_n = 0
        self.px_s2 = 0.0
        self.px_vec_cnt = {t: 0 for t in TAUS}
        self.px_vec_n = 0
        self.vec_pairs = []

    def add_pair(self, diff):
        # diff: [1,2,H,W], CPU, exactly the tensor official test.py subtracts
        e_comp = torch.sum(diff ** 2, dim=0).sqrt().reshape(-1).double().numpy()
        e_vec = torch.linalg.vector_norm(diff.float(), dim=1).reshape(-1).double().numpy()
        self.pairs.append(float(e_comp.mean()))
        self.vec_pairs.append(float(e_vec.mean()))
        self.px_n += int(e_comp.size)
        self.px_s2 += float(np.square(e_comp).sum())
        self.px_vec_n += int(e_vec.size)
        for t in TAUS:
            self.px_cnt[t] += int((e_comp < t).sum())
            self.px_vec_cnt[t] += int((e_vec < t).sum())

    def report(self):
        p = np.asarray(self.pairs, dtype=np.float64)
        v = np.asarray(self.vec_pairs, dtype=np.float64)
        out = {
            "pairs": int(p.size),
            "AEPE": float(p.mean()),                      # official: mean(MAE_list)
            "RMSE_official_var": float(p.var()),          # official prints np.var as "RMSE"
            "RMSE_paper_formula_std": float(p.std()),      # paper eq.(15) sqrt(var)
            "pixel_RMSE_true": math.sqrt(self.px_s2 / self.px_n),
            "AEPE_vector_EPE_reference": float(v.mean()),
            "CMR@1_vector_EPE_reference": 100.0 * float((v < 1).mean()),
        }
        for t in TAUS:
            inl = p[p < t]
            out["@%dpx" % t] = {
                "CMR": 100.0 * inl.size / p.size,
                "AEPE": float(inl.mean()) if inl.size else None,
                "RMSE_official_var": float(inl.var()) if inl.size else None,
            }
        # official test.py also prints pixel-level 1px/3px/5px rates
        out["pixel_CMR"] = {"%dpx" % t: 100.0 * self.px_cnt[t] / self.px_n
                            for t in TAUS}
        out["pixel_CMR_vector_EPE"] = {"%dpx" % t: 100.0 * self.px_vec_cnt[t] / self.px_vec_n
                                       for t in TAUS}
        return out


@torch.inference_mode()
def run_one(which, device, max_pairs):
    t0 = time.time()
    if which == "baseline":
        cfg = get_cfg()
        cfg.transformer = "featureformerwithlbl"
        cfg.image_size = [512, 512]
        model = torch.nn.DataParallel(GDROS(cfg[cfg.transformer])).to(device)
        model.load_state_dict(torch.load(BASE_CKPT, map_location=device))
        model.eval()
        net = model.module          # official test.py calls exp_func(model.module)
        ckpt = BASE_CKPT

        def forward(a, b):
            _, flow_pre1 = net(a, b, iters=ITERS, test_mode=True)
            return LSR(a.shape[0], a.shape[2], a.shape[3], flow_pre1, NPOINT)
    else:
        model = e075.PairFusion(e074, "region", device).to(device)
        payload = torch.load(M4_CKPT, map_location=device)
        miss = model.load_state_dict(payload["model"], strict=False)
        print("[m4] missing=%s unexpected=%s" %
              (list(miss.missing_keys), list(miss.unexpected_keys)), flush=True)
        model.eval()
        ckpt = M4_CKPT
        del payload

        def forward(a, b):
            preds = model(a, b, iters=ITERS, lsr_fn=e074.lsr)
            return e074.lsr(preds[-1])       # == LSR(1,512,512,flow,200000)

    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    n = min(max_pairs, len(ds))
    print("[%s] dataset=%d evaluating=%d device=%s" % (which, len(ds), n, device), flush=True)
    col = Collector()
    for i in range(n):
        _, a_img, b_img, gt, valid = ds[i]
        a = a_img[None].to(device)
        b = b_img[None].to(device)
        post = forward(a, b)                  # CPU, [1,2,H,W]
        diff = post - gt[None]
        if i == 0:
            print("[%s] post=%s gt=%s diff=%s" %
                  (which, tuple(post.shape), tuple(gt.shape), tuple(diff.shape)), flush=True)
        col.add_pair(diff)
        if i % 50 == 0 or i == n - 1:
            print("[%s] %d/%d pair_epe=%.4f elapsed=%.1fs" %
                  (which, i + 1, n, col.pairs[-1], time.time() - t0), flush=True)
    result = {
        "which": which,
        "checkpoint": ckpt,
        "pairs_evaluated": n,
        "iters": ITERS,
        "lsr_npoint": NPOINT,
        "elapsed_sec": round(time.time() - t0, 1),
        "official": col.report(),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def print_table(results):
    print("\n### 官方 test.py 口径（逐分量 |Δ| + 配对级）", flush=True)
    print("| 模型 | AEPE↓ | RMSE↓(官方栏位/方差) | CMR@1↑ | AEPE@1↓ | CMR@2↑ | AEPE@2↓ | CMR@5↑ | AEPE@5↓ |", flush=True)
    print("|---|---|---|---|---|---|---|---|---|", flush=True)
    for name, r in results:
        m = r["official"]
        cells = [name, "%.4f" % m["AEPE"], "%.4f" % m["RMSE_official_var"]]
        for t in (1, 2, 5):
            q = m["@%dpx" % t]
            cells += ["%.2f" % q["CMR"], "%.4f" % q["AEPE"]]
        print("| " + " | ".join(cells) + " |", flush=True)

    print("\n### 官方栏位补充（T=1..5 的 RMSE 亦为方差）与逐像素 1/3/5px", flush=True)
    for name, r in results:
        m = r["official"]
        print("- %s: RMSE(论文公式 std)=%.4f, pixel_RMSE_true=%.4f; "
              "pixel_CMR 1/3/5px = %.2f / %.2f / %.2f" %
              (name, m["RMSE_paper_formula_std"], m["pixel_RMSE_true"],
               m["pixel_CMR"]["1px"], m["pixel_CMR"]["3px"], m["pixel_CMR"]["5px"]),
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["baseline", "m4", "both"], default="both")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=424)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    device = torch.device("cuda:%d" % args.gpu)
    which_list = ["baseline", "m4"] if args.which == "both" else [args.which]
    out = {}
    for w in which_list:
        out[w] = run_one(w, device, args.max_pairs)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("WROTE " + args.out, flush=True)
    print_table([(w, out[w]) for w in which_list])
    print("RESULT_JSON " + json.dumps(out, indent=2, ensure_ascii=False), flush=True)
