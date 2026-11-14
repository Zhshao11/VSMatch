#!/usr/bin/env python3
"""Verify the C-branch teacher pruning in E075A-M4 is bit-exact.

Context
-------
E074Model(variant="C").teacher_preds() produces `last_teacher_avg`, which is
read ONLY by the E074 distill loss. E075A PairFusion (training and evaluation)
never reads it, so those 2 frozen GRU chains are dead computation. The pruned
call chain is:

    legacy : 5 GRU chains x iters = 160 updates, LSR = 4k+1 = 129
    pruned : 3 GRU chains x iters =  96 updates, LSR = 4k+1 = 129

This script does two things in the same process / with the same weights:

  1. A/B on the first `--ab-pairs` pairs: compare all `iters` prediction steps
     and the final LSR output between c_skip_teachers=False and True.
  2. Full official-protocol evaluation of the PRUNED model on `--max-pairs`
     (424) pairs, using eval_os_official_protocol.Collector, and compare with
     the archived 2026-11-13 legacy numbers.

Run on zzvr:
  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  CUDA_VISIBLE_DEVICES=5 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/verify_c_teacher_pruning.py --max-pairs 424 --gpu 0 \
    --out .codex_tmp/verify_c_teacher_pruning.json
"""
from __future__ import annotations
import argparse
import gc
import json
import time

import torch

# P inserts the official-repo / .codex_tmp paths into sys.path on import.
import eval_os_official_protocol as P
import core.datasets
import e074a_f2_f3_twostage_fusion as e074
import e075a_preserve_high_precision_recover_c2c5 as e075

# Archived legacy result, 424 pairs / iters=32 / LSR NPOINT=200000.
# Source: research/results/OS_20261113/os_official_protocol_20261113.json
HIST_REF = {
    "AEPE": 1.473827834446231,
    "CMR@1": 72.40566037735849,
    "CMR@2": 84.90566037735849,
    "CMR@5": 94.10377358490567,
    "RMSE_official_var": 8.335009898837217,
    "elapsed_sec": 422.4,          # idle-GPU 2026-11-14 ablation run
}


def build(skip: bool, device):
    model = e075.PairFusion(e074, "region", device,
                            c_skip_teachers=skip).to(device)
    payload = torch.load(P.M4_CKPT, map_location=device)
    miss = model.load_state_dict(payload["model"], strict=False)
    info = {
        "c_skip_teachers": bool(skip),
        "missing": list(miss.missing_keys),
        "unexpected": list(miss.unexpected_keys),
    }
    print("[build] %s" % json.dumps(info), flush=True)
    if info["missing"] or info["unexpected"]:
        raise RuntimeError("checkpoint keys changed; aborting verification")
    model.eval()
    del payload
    gc.collect()
    torch.cuda.empty_cache()
    return model


@torch.inference_mode()
def predict_all(model, a, b):
    """Return all `iters` fused predictions (CPU) and the final LSR output."""
    preds = model(a, b, iters=P.ITERS, lsr_fn=e074.lsr)
    post = e074.lsr(preds[-1])
    return [p.detach().cpu() for p in preds], post.detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=424)
    ap.add_argument("--ab-pairs", type=int, default=8)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    device = torch.device("cuda:%d" % args.gpu)

    ds = core.datasets.opt_sar_test(split="testing", dstype="testing",
                                    root=P.TEST_ROOT)
    n = min(args.max_pairs, len(ds))
    ab_n = min(args.ab_pairs, n)
    print("[data] dataset=%d evaluating=%d ab_pairs=%d device=%s iters=%d npoint=%d"
          % (len(ds), n, ab_n, device, P.ITERS, P.NPOINT), flush=True)

    samples = []
    for i in range(ab_n):
        _, a_img, b_img, gt, _ = ds[i]
        samples.append((a_img[None].to(device), b_img[None].to(device)))

    # --- A/B: legacy vs pruned, all 32 iterations, same weights ---
    t_ab = time.time()
    legacy = build(False, device)
    legacy_out = [predict_all(legacy, a, b) for a, b in samples]
    del legacy
    gc.collect()
    torch.cuda.empty_cache()

    pruned = build(True, device)
    pruned_out = [predict_all(pruned, a, b) for a, b in samples]

    max_iter_delta = 0.0
    max_post_delta = 0.0
    for (lp, lpost), (pp, ppost) in zip(legacy_out, pruned_out):
        for x, y in zip(lp, pp):
            max_iter_delta = max(max_iter_delta, float((x - y).abs().max()))
        max_post_delta = max(max_post_delta, float((lpost - ppost).abs().max()))
    print("[ab] pairs=%d max|delta| over all %d iters = %.9g ; post-LSR = %.9g ; %.1fs"
          % (ab_n, P.ITERS, max_iter_delta, max_post_delta, time.time() - t_ab),
          flush=True)
    ab_elapsed = round(time.time() - t_ab, 1)   # includes both model builds
    del legacy_out, pruned_out
    gc.collect()
    torch.cuda.empty_cache()

    # --- Full official-protocol evaluation of the pruned model ---
    col = P.Collector()
    t0 = time.time()
    with torch.inference_mode():
        for i in range(n):
            _, a_img, b_img, gt, valid = ds[i]
            a = a_img[None].to(device)
            b = b_img[None].to(device)
            preds = pruned(a, b, iters=P.ITERS, lsr_fn=e074.lsr)
            post = e074.lsr(preds[-1]).detach().cpu()
            col.add_pair(post - gt[None])
            if i % 50 == 0 or i == n - 1:
                print("[pruned] %d/%d pair_epe=%.4f elapsed=%.1fs"
                      % (i + 1, n, col.pairs[-1], time.time() - t0), flush=True)
    elapsed = round(time.time() - t0, 1)
    rep = col.report()

    delta = {
        "AEPE": abs(rep["AEPE"] - HIST_REF["AEPE"]),
        "CMR@1": abs(rep["@1px"]["CMR"] - HIST_REF["CMR@1"]),
        "CMR@2": abs(rep["@2px"]["CMR"] - HIST_REF["CMR@2"]),
        "CMR@5": abs(rep["@5px"]["CMR"] - HIST_REF["CMR@5"]),
        "RMSE_official_var": abs(rep["RMSE_official_var"] - HIST_REF["RMSE_official_var"]),
    }
    bit_exact = (max_iter_delta == 0.0 and max_post_delta == 0.0)
    metrics_ok = (delta["AEPE"] < 1e-6 and delta["RMSE_official_var"] < 1e-3
                  and all(delta[k] < 1e-6 for k in ("CMR@1", "CMR@2", "CMR@5")))
    ok = bit_exact and metrics_ok

    print("\n### pruned official-protocol result", flush=True)
    print("AEPE=%.7f  CMR@1/2/5=%.2f/%.2f/%.2f  RMSE_var=%.4f  elapsed=%.1fs"
          % (rep["AEPE"], rep["@1px"]["CMR"], rep["@2px"]["CMR"],
             rep["@5px"]["CMR"], rep["RMSE_official_var"], elapsed), flush=True)
    print("delta vs archived legacy: %s" % json.dumps(delta), flush=True)
    print("speedup vs idle-GPU legacy 422.4s: %.3fx"
          % (HIST_REF["elapsed_sec"] / elapsed), flush=True)
    print("SANITY %s" % ("PASS" if ok else "FAIL"), flush=True)

    out = {
        "protocol": "GDROS official test.py (component-wise |dx|,|dy| + pair-level)",
        "checkpoint": P.M4_CKPT,
        "pairs_evaluated": n,
        "iters": P.ITERS,
        "lsr_npoint": P.NPOINT,
        "ab": {
            "pairs": ab_n,
            "compared_iterations": P.ITERS,
            "max_abs_delta_all_iters": max_iter_delta,
            "max_abs_delta_post_lsr": max_post_delta,
            "bit_exact": bit_exact,
            "elapsed_sec": ab_elapsed,
            "elapsed_includes_two_model_builds": True,
        },
        "pruned": {
            "official": rep,
            "elapsed_sec": elapsed,
            "pair_EPE": col.pairs,
        },
        "archived_legacy": HIST_REF,
        "abs_delta_vs_archived": delta,
        "speedup_vs_idle_gpu_legacy": round(HIST_REF["elapsed_sec"] / elapsed, 4),
        "call_counts": {
            "gru_updates_legacy": 5 * P.ITERS,
            "gru_updates_pruned": 3 * P.ITERS,
            "lsr_calls_both": 4 * P.ITERS + 1,
            "note": "B.gate_input 2k + PairFusion.gate_features 2k + final 1; "
                    "pruning removes only C.teacher_preds (2 GRU chains), not LSR",
        },
        "sanity_pass": bool(ok),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print("[out] %s" % args.out, flush=True)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
