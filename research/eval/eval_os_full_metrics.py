#!/usr/bin/env python3
"""Full OS-dataset metrics for GDROS released weights and E075A-M4@80K.

For BOTH models, under TWO protocols, report:
  - AEPE(all), RMSE(all)
  - CMR@tau, AEPE@tau, RMSE@tau for tau = 1, 2, 5 (10/16 kept as sanity check)

Protocol A "official_pair_level" reproduces GDROS official test.py semantics:
    the unit is the per-pair mean EPE (MAE_list);
    CMR@T = fraction of pairs whose mean EPE < T;
    MAE@T = mean of the inlier pairs' mean EPE.
    NOTE: official test.py prints np.var(...) under the label "RMSE".
    We therefore report the true RMSE = sqrt(mean(e^2)) AND "var" (the value
    official test.py would have printed), so numbers can be matched either way.

Protocol B "per_pixel" applies the same definitions to every valid pixel,
    i.e. the protocol of eval_os_unified.py / E075A eval_model.

Both the historical 404-pair subset and the full test set are accumulated.

Run from image_matching_part2_migrated/ (E038B path resolution depends on cwd):
  CUDA_VISIBLE_DEVICES=2 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/eval_os_full_metrics.py --which baseline --max-pairs 424 --gpu 0
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
# Vendored GDROS encoder.py imports timm.layers (timm>=0.6); shim for old timm.
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
TAUS = (1, 2, 5, 10, 16)
HIST_PAIRS = 404
NPOINT = 200000
ITERS = 32


class Acc:
    """Streaming accumulator of count / sum / sum-of-squares, globally and per tau."""

    def __init__(self, taus=TAUS):
        self.taus = taus
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0
        self.tn = {t: 0 for t in taus}
        self.ts = {t: 0.0 for t in taus}
        self.ts2 = {t: 0.0 for t in taus}

    def add(self, arr):
        a = np.asarray(arr, dtype=np.float64)
        self.n += int(a.size)
        self.s += float(a.sum())
        self.s2 += float(np.square(a).sum())
        for t in self.taus:
            m = a[a < t]
            self.tn[t] += int(m.size)
            self.ts[t] += float(m.sum())
            self.ts2[t] += float(np.square(m).sum())

    def report(self):
        mean = self.s / self.n
        var = self.s2 / self.n - mean * mean
        out = {
            "n": self.n,
            "AEPE": mean,
            "RMSE": math.sqrt(self.s2 / self.n),
            "var_official_RMSE_label": var,
        }
        for t in self.taus:
            n, s, s2 = self.tn[t], self.ts[t], self.ts2[t]
            if n:
                mu = s / n
                out["@%dpx" % t] = {
                    "CMR": 100.0 * n / self.n,
                    "AEPE": mu,
                    "RMSE": math.sqrt(s2 / n),
                    "var_official_RMSE_label": s2 / n - mu * mu,
                }
            else:
                out["@%dpx" % t] = {
                    "CMR": 0.0, "AEPE": None, "RMSE": None,
                    "var_official_RMSE_label": None,
                }
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
        ckpt = BASE_CKPT

        def forward(a, b):
            _, flow_pre1 = model(a, b, iters=ITERS, test_mode=True)
            return LSR(a.shape[0], a.shape[2], a.shape[3], flow_pre1, NPOINT).cpu()
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
            return e074.lsr(preds[-1]).cpu()

    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    n_pairs = min(max_pairs, len(ds))
    print("[%s] dataset pairs=%d, evaluating=%d, device=%s" %
          (which, len(ds), n_pairs, device), flush=True)

    px_all, px_hist = Acc(), Acc()
    pr_all, pr_hist = Acc(), Acc()
    valid_min = 1.0
    for i in range(n_pairs):
        file_opt, a_img, b_img, gt, valid = ds[i]
        a = a_img[None].to(device)
        b = b_img[None].to(device)
        post = forward(a, b)
        e = torch.linalg.vector_norm((post - gt[None]).float(), dim=1).view(-1)
        vmask = valid.view(-1).bool()
        valid_min = min(valid_min, float(vmask.float().mean().item()))
        e_m = e[vmask].cpu().numpy()
        pair = np.array([e_m.mean()], dtype=np.float64)
        px_all.add(e_m)
        pr_all.add(pair)
        if i < HIST_PAIRS:
            px_hist.add(e_m)
            pr_hist.add(pair)
        if i % 20 == 0 or i == n_pairs - 1:
            print("[%s] %d/%d pair_epe=%.4f elapsed=%.1fs" %
                  (which, i + 1, n_pairs, float(pair[0]), time.time() - t0), flush=True)

    result = {
        "which": which,
        "checkpoint": ckpt,
        "pairs_evaluated": n_pairs,
        "iters": ITERS,
        "lsr_npoint": NPOINT,
        "valid_mask_min_fraction": valid_min,
        "elapsed_sec": round(time.time() - t0, 1),
        "official_pair_level": {
            "full_%d" % n_pairs: pr_all.report(),
            "first_%d" % HIST_PAIRS: pr_hist.report(),
        },
        "per_pixel": {
            "full_%d" % n_pairs: px_all.report(),
            "first_%d" % HIST_PAIRS: px_hist.report(),
        },
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def fmt_table(rows, protocol, subset):
    """Print a compact markdown table for the requested metrics."""
    print("\n### %s / %s" % (protocol, subset), flush=True)
    hdr = ("| model | AEPE↓ | RMSE↓ | CMR@1↑ | AEPE@1↓ | CMR@2↑ | AEPE@2↓ | "
           "CMR@5↑ | AEPE@5↓ |")
    print(hdr, flush=True)
    print("|---|---|---|---|---|---|---|---|---|", flush=True)
    def f(v, pattern):
        return "--" if v is None else pattern % v

    for name, r in rows:
        m = r[protocol][subset]
        cells = [name]
        cells += [f(m["AEPE"], "%.4f"), f(m["RMSE"], "%.4f")]
        for t in (1, 2, 5):
            key = "@%dpx" % t
            cells += [f(m[key]["CMR"], "%.2f"), f(m[key]["AEPE"], "%.4f")]
        print("| " + " | ".join(cells) + " |", flush=True)


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
    rows = [(w, out[w]) for w in which_list]
    n = out[which_list[0]]["pairs_evaluated"]
    fmt_table(rows, "official_pair_level", "full_%d" % n)
    fmt_table(rows, "official_pair_level", "first_%d" % HIST_PAIRS)
    fmt_table(rows, "per_pixel", "full_%d" % n)
    fmt_table(rows, "per_pixel", "first_%d" % HIST_PAIRS)
    print("RESULT_JSON " + json.dumps(out, indent=2, ensure_ascii=False), flush=True)
