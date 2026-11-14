#!/usr/bin/env python3
"""Symmetric single-branch diagnostic bypass on the PRUNED E075A-M4.

One process = one config, so two configs can be launched on two different GPUs
in parallel. Both use the GDROS official test.py protocol (424 pairs, iters=32,
LSR NPOINT=200000) via eval_os_official_protocol.Collector.

What the two configs mean
-------------------------
E073 fuses two descriptor streams with gate values:

  F3: a  = sigmoid(corr_gate_logit); corr = (1-a)*Corr1 + a*Corr2
  F2: gs = sigmoid(fuse_gate(cat([f1s,f2s])));
      fs = (1-gs)*f1s + gs*fuse_proj(cat([f1s,f2s]))

  v1only (only GDROS-compatible stream): corr_gate_logit = -80 (a≈0), gs≈0
  v2only (DINO-assisted stream):         corr_gate_logit = +80 (a≈1), gs≈1
  both: original trained gates (a≈0.5)

STRUCTURAL CAVEAT — "only DINO" is NOT a pure backbone removal.
In E072Matcher._features the V2 stream is built as

    fs, ft = front_v1(...)           # GDROS-compatible frontend, always runs
    if variant == "V2": fs, ft = e046(raw1, raw2, fs, ft)   # DINO FiLM injection

so f2 already contains the V1 feature; the frozen DINO descriptors only
modulate it. Moreover F2 with gs=1 yields fuse_proj(cat([f1s,f2s])), which
still explicitly contains f1s. Therefore v2only means "push both fusion gates
to the V2 side", not "delete the GDROS backbone".

This is a frozen-weight sensitivity diagnostic, NOT a valid module ablation:
it cannot quantify either branch's marginal contribution, because the large
degradation mixes lost information with downstream out-of-distribution inputs.

Run on zzvr (two GPUs in parallel):
  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  CUDA_VISIBLE_DEVICES=5 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/ablate_backbone_single.py --config v1only --gpu 0 \
    --out .codex_tmp/ablate_v1only_pruned.json
  CUDA_VISIBLE_DEVICES=1 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/ablate_backbone_single.py --config v2only --gpu 0 \
    --out .codex_tmp/ablate_v2only_pruned.json
"""
from __future__ import annotations
import sys
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

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
import e074a_f2_f3_twostage_fusion as e074
e072, e073 = e074.add_paths()
import e075a_preserve_high_precision_recover_c2c5 as e075
import eval_os_official_protocol as P

M4_CKPT = P.M4_CKPT
TEST_ROOT = P.TEST_ROOT
NPOINT = P.NPOINT
ITERS = P.ITERS
NEG = -80.0
POS = 80.0

# Archived official-protocol values (424 pairs, iters=32), exact floats.
# both / gdros: research/results/OS_20261113/os_official_protocol_20261113.json
# v1only == older "v2off": research/results/OS_20261114/ablate_v2_off_result_official.json
HIST_REF = {
    "gdros": {"AEPE": 1.4618458366096103, "CMR@1": 33.72641509433962,
              "CMR@2": 81.60377358490567, "CMR@5": 99.52830188679245,
              "RMSE_official_var": 0.7463472918495302},
    "both": {"AEPE": 1.473827834446231, "CMR@1": 72.40566037735849,
             "CMR@2": 84.90566037735849, "CMR@5": 94.10377358490567,
             "RMSE_official_var": 8.335009898837217},
    "v1only": {"AEPE": 7.745715116406876, "CMR@1": 0.9433962264150944,
               "CMR@2": 4.245283018867925, "CMR@5": 24.764150943396228,
               "RMSE_official_var": 16.018962391754382},
}
TOL = 1e-6

CONFIG_DEF = {
    "both": "original trained gates (a≈0.5, learned fuse_gate)",
    "v1only": "corr_gate_logit=-80 (a≈0), fuse_gate=-80 (gs≈0); keep GDROS stream",
    "v2only": "corr_gate_logit=+80 (a≈1), fuse_gate=+80 (gs≈1); push gates to DINO-assisted stream",
}
STRUCTURAL_CAVEAT = (
    "E072 V2 is not an independent backbone: front_v1 always runs and E046FiLM "
    "injects frozen DINO descriptors into that V1 feature. f2 therefore contains "
    "V1 information, and F2 with gs=1 returns fuse_proj(cat([f1s,f2s])) which "
    "still contains f1s. v2only is a gate-biased bypass, not a V1 removal."
)


class ConstGate(nn.Module):
    """Constant logit for F2 fuse_gate; sigmoid(±80) is 0 or 1 in float32."""
    def __init__(self, logit: float):
        super().__init__()
        self.logit = float(logit)

    def forward(self, x):
        return x.new_full((x.shape[0], 1, x.shape[2], x.shape[3]), self.logit)


def make_state(model):
    mods = [m for m in model.modules() if isinstance(m, e073.E073FusionMatcher)]
    unknown = [m.variant for m in mods if m.variant not in ("F2", "F3")]
    if unknown:
        raise RuntimeError("unhandled E073 variants: %s" % unknown)
    return {
        "mods": mods,
        "variants": [m.variant for m in mods],
        "orig_logit": [float(m.corr_gate_logit.detach().cpu().item()) for m in mods],
        "orig_fuse": [m.fuse_gate for m in mods],
    }


def set_config(state, cfg):
    for i, m in enumerate(state["mods"]):
        if m.variant == "F3":
            if cfg == "both":
                v = state["orig_logit"][i]
            else:
                v = NEG if cfg == "v1only" else POS
            m.corr_gate_logit.data.fill_(v)
        else:
            if cfg == "both":
                m.fuse_gate = state["orig_fuse"][i]
            else:
                m.fuse_gate = ConstGate(NEG if cfg == "v1only" else POS)
    return gate_snapshot(state)


def gate_snapshot(state):
    out = []
    for m in state["mods"]:
        if m.variant == "F3":
            out.append({"variant": "F3",
                        "a_v2": round(torch.sigmoid(m.corr_gate_logit).item(), 8),
                        "corr_gate_logit": round(float(m.corr_gate_logit.item()), 3)})
        else:
            g = m.fuse_gate
            out.append({"variant": "F2", "fuse_gate": type(g).__name__,
                        "const_logit": getattr(g, "logit", None)})
    return out


@torch.inference_mode()
def forward_post(model, a, b, iters):
    preds = model(a, b, iters=iters, lsr_fn=e074.lsr)
    return e074.lsr(preds[-1]).cpu()


def evaluate(model, state, device, cfg, max_pairs, iters):
    t0 = time.time()
    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    n = min(max_pairs, len(ds))
    print("[%s] dataset=%d evaluating=%d iters=%d gates=%s"
          % (cfg, len(ds), n, iters, gate_snapshot(state)), flush=True)
    col = P.Collector()
    for i in range(n):
        _, a_img, b_img, gt, _ = ds[i]
        a = a_img[None].to(device)
        b = b_img[None].to(device)
        post = forward_post(model, a, b, iters)
        col.add_pair(post - gt[None])
        if i % 50 == 0 or i == n - 1:
            print("[%s] %d/%d pair_epe=%.4f elapsed=%.1fs"
                  % (cfg, i + 1, n, col.pairs[-1], time.time() - t0), flush=True)
    return {
        "protocol": "GDROS official test.py (component-wise |dx|,|dy| + pair-level)",
        "pairs_evaluated": n,
        "iters": iters,
        "lsr_npoint": NPOINT,
        "gates": gate_snapshot(state),
        "elapsed_sec": round(time.time() - t0, 1),
        "official": col.report(),
        "pair_EPE": [float(x) for x in col.pairs],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["both", "v1only", "v2only"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=424)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cuda:%d" % args.gpu)
    full_run = (args.iters == ITERS and args.max_pairs >= 424)

    t_load = time.time()
    model = e075.PairFusion(e074, "region", device,
                            c_skip_teachers=True).to(device)
    payload = torch.load(M4_CKPT, map_location=device)
    miss = model.load_state_dict(payload["model"], strict=False)
    print("[m4] missing=%s unexpected=%s c_skip_teachers=%s"
          % (list(miss.missing_keys), list(miss.unexpected_keys),
             model.c_skip_teachers), flush=True)
    if miss.missing_keys or miss.unexpected_keys or not model.c_skip_teachers:
        raise RuntimeError("checkpoint or pruning flag changed; aborting")
    del payload
    model.eval()
    state = make_state(model)
    print("[structure] E073FusionMatcher=%d variants=%s orig_logit=%s load_sec=%.1f"
          % (len(state["mods"]), state["variants"], state["orig_logit"],
             time.time() - t_load), flush=True)

    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)

    # 1-pair probe against `both` proves the gate switch actually changes output.
    _, a_img, b_img, _, _ = ds[0]
    a, b = a_img[None].to(device), b_img[None].to(device)
    set_config(state, "both")
    p_both = forward_post(model, a, b, args.iters).float()
    set_config(state, args.config)
    p_cfg = forward_post(model, a, b, args.iters).float()
    d = (p_cfg - p_both).abs()
    probe = {"pairs": 1, "max_abs_delta_vs_both": float(d.max()),
             "mean_abs_delta_vs_both": float(d.mean())}
    print("[probe] %s max|Δpost|=%.6f mean|Δpost|=%.6f"
          % (args.config, probe["max_abs_delta_vs_both"],
             probe["mean_abs_delta_vs_both"]), flush=True)

    result = evaluate(model, state, device, args.config,
                      args.max_pairs, args.iters)
    m = result["official"]

    print("\n### %s official-protocol result" % args.config, flush=True)
    print("AEPE=%.7f  CMR@1/2/5=%.2f/%.2f/%.2f  RMSE_var=%.4f  elapsed=%.1fs"
          % (m["AEPE"], m["@1px"]["CMR"], m["@2px"]["CMR"],
             m["@5px"]["CMR"], m["RMSE_official_var"],
             result["elapsed_sec"]), flush=True)

    checks = []
    if args.config in HIST_REF:
        ref = HIST_REF[args.config]
        cur = {"AEPE": m["AEPE"], "CMR@1": m["@1px"]["CMR"],
               "CMR@2": m["@2px"]["CMR"], "CMR@5": m["@5px"]["CMR"],
               "RMSE_official_var": m["RMSE_official_var"]}
        for k, v in ref.items():
            delta = abs(cur[k] - v)
            checks.append({"metric": k, "current": cur[k], "archived": v,
                           "abs_delta": delta, "tol": TOL, "pass": delta <= TOL})
    else:
        delta_aepe = abs(m["AEPE"] - HIST_REF["both"]["AEPE"])
        checks.append({"metric": "gate_switch_changes_output",
                       "current": probe["max_abs_delta_vs_both"],
                       "threshold": 0.1,
                       "pass": probe["max_abs_delta_vs_both"] > 0.1})
        checks.append({"metric": "AEPE_differs_from_both",
                       "current": delta_aepe, "threshold": 0.05,
                       "pass": delta_aepe > 0.05})
    sanity = all(c["pass"] for c in checks) and full_run
    for c in checks:
        print("[sanity] %-28s %s" % (c["metric"], "PASS" if c["pass"] else "FAIL"),
              flush=True)
    print("SANITY %s" % ("PASS" if sanity else "FAIL"), flush=True)

    out = {
        "protocol": "GDROS official test.py",
        "ablation_class": "diagnostic_bypass_not_module_ablation",
        "config": args.config,
        "config_definition": CONFIG_DEF[args.config],
        "structural_caveat": STRUCTURAL_CAVEAT,
        "checkpoint": M4_CKPT,
        "pruned_model": {
            "c_skip_teachers": bool(model.c_skip_teachers),
            "gru_chains": 3, "gru_updates_per_pair": 3 * args.iters,
            "lsr_calls_per_pair": 4 * args.iters + 1,
            "note": "2026-11-15 C-branch teacher pruning; metrics bit-exact vs legacy",
        },
        "structure": {"n_matchers": len(state["mods"]),
                      "variants": state["variants"],
                      "orig_corr_gate_logit": state["orig_logit"]},
        "device": str(device),
        "full_run": bool(full_run),
        "probe": probe,
        "result": result,
        "gdros_released_reference": HIST_REF["gdros"],
        "both_reference": HIST_REF["both"],
        "sanity_checks": checks,
        "sanity_pass": bool(sanity),
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("[out] %s" % args.out, flush=True)
    if not sanity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
