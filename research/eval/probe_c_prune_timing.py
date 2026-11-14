#!/usr/bin/env python3
"""Same-process A/B timing for the C-branch teacher pruning.

The 424-pair verification compares wall time ACROSS runs (317.2s pruned vs
422.4s archived legacy), which mixes in GPU/process state. This probe instead
builds legacy and pruned models in ONE process and times both on the SAME
pairs, for two call paths:

  eval path  : PairFusion.forward, iters=32 (official evaluation)
  train path : E075A's manual `model.b` + `model.c` + gate_features loop,
               iters=12 (the actual M4 training forward)

Only the forward pass is timed. A full training step also includes
LSR_list(fused), losses, backward through the L4 gate, data loading and
periodic dev eval, so the measured ratio must NOT be applied directly to the
0.741 s/step / 16.5 h training estimate.

Run on zzvr:
  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  CUDA_VISIBLE_DEVICES=5 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/probe_c_prune_timing.py --pairs 30 --gpu 0 \
    --out .codex_tmp/probe_c_prune_timing.json
"""
from __future__ import annotations
import argparse
import gc
import json
import time

import torch

import eval_os_official_protocol as P
import core.datasets
import e074a_f2_f3_twostage_fusion as e074
import e075a_preserve_high_precision_recover_c2c5 as e075


def build(skip: bool, device):
    model = e075.PairFusion(e074, "region", device,
                            c_skip_teachers=skip).to(device)
    payload = torch.load(P.M4_CKPT, map_location=device)
    miss = model.load_state_dict(payload["model"], strict=False)
    if miss.missing_keys or miss.unexpected_keys:
        raise RuntimeError("checkpoint keys changed: %s / %s"
                           % (list(miss.missing_keys), list(miss.unexpected_keys)))
    model.eval()
    del payload
    gc.collect()
    torch.cuda.empty_cache()
    return model


@torch.inference_mode()
def run_eval_path(model, a, b, iters):
    preds = model(a, b, iters=iters, lsr_fn=e074.lsr)
    return e074.lsr(preds[-1])


@torch.inference_mode()
def run_train_path(model, a, b, iters, skip_c_teachers):
    """Mirror e075a training loop: manual B/C + gate_features, no fused loss."""
    pb = model.b(a, b, iters=iters)
    pc = model.c(a, b, iters=iters,
                 need_teacher_avg=not skip_c_teachers)
    out = None
    for xb, xc in zip(pb, pc):
        feat = model.gate_features(xb, xc, e074.lsr)
        g = model.gate(feat)
        g = torch.nn.functional.interpolate(
            g, size=xb.shape[-2:], mode="bilinear", align_corners=False)
        out = (1 - g) * xb + g * xc
    return out


def timed(model, samples, iters, path, skip, warmup=3):
    fn = (lambda a, b: run_eval_path(model, a, b, iters)) if path == "eval" else (
        lambda a, b: run_train_path(model, a, b, iters, skip))
    for a, b in samples[:warmup]:
        fn(a, b)
    torch.cuda.synchronize()
    t0 = time.time()
    for a, b in samples[warmup:]:
        fn(a, b)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = max(1, len(samples) - warmup)
    return {"pairs_timed": n, "iters": iters, "total_sec": round(dt, 3),
            "per_pair_sec": round(dt / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    device = torch.device("cuda:%d" % args.gpu)

    ds = core.datasets.opt_sar_test(split="testing", dstype="testing",
                                    root=P.TEST_ROOT)
    n = min(args.pairs + args.warmup, len(ds))
    samples = []
    for i in range(n):
        _, a_img, b_img, gt, _ = ds[i]
        samples.append((a_img[None].to(device), b_img[None].to(device)))
    print("[data] pairs=%d warmup=%d device=%s" % (n, args.warmup, device),
          flush=True)

    out = {"checkpoint": P.M4_CKPT, "same_process_ab": True, "paths": {}}
    for skip, tag in ((False, "legacy"), (True, "pruned")):
        model = build(skip, device)
        row = {}
        for path, iters in (("eval", 32), ("train", 12)):
            t = timed(model, samples, iters, path, skip, args.warmup)
            row[path] = t
            print("[%s] %-6s iters=%d per_pair=%.4fs total=%.1fs"
                  % (tag, path, iters, t["per_pair_sec"], t["total_sec"]),
                  flush=True)
        out["paths"][tag] = row
        del model
        gc.collect()
        torch.cuda.empty_cache()

    out["speedup"] = {
        p: round(out["paths"]["legacy"][p]["per_pair_sec"]
                 / out["paths"]["pruned"][p]["per_pair_sec"], 4)
        for p in ("eval", "train")
    }
    out["call_counts"] = {
        "gru_updates": {"legacy": {"eval": 5 * 32, "train": 5 * 12},
                        "pruned": {"eval": 3 * 32, "train": 3 * 12}},
        "lsr_calls_eval": 4 * 32 + 1,
        "note": "pruning removes only C.teacher_preds (2 GRU chains); all LSR "
                "calls remain. The timed train path reproduces only the two "
                "per-iteration LSR sites (B.gate_input + gate_features = 4k=48); "
                "the real training step additionally runs LSR_list(fused), "
                "gate_target and pbp/pcp LSR, which are unchanged by the pruning.",
    }
    out["caveat"] = ("Forward-only timing. A full training step also includes "
                     "LSR_list(fused), losses, backward through the L4 gate, "
                     "data loading and dev eval; do NOT scale 0.741 s/step by "
                     "this ratio.")
    print("[speedup] eval=%.3fx train_forward=%.3fx"
          % (out["speedup"]["eval"], out["speedup"]["train"]), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print("[out] %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
