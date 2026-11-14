#!/usr/bin/env python3
"""零训练消融：把 E073 的 V2(DINO 辅助) 那一路从相关体融合里旁路掉，
在**同一进程 / 同一份权重 / 同一评测口径**下比较 E075A-M4@80K 的最终指标。

唯一改动是融合门控的值，不改任何层维度、不重新训练：

  F3: a = sigmoid(corr_gate_logit); corr = (1-a)*Corr1 + a*Corr2
      单路 -> corr_gate_logit = -80  =>  a ≈ 0  =>  只用 V1(GDROS 兼容) 的 CorrBlock
  F2: gs = sigmoid(fuse_gate(cat([f1s,f2s]))); fs = (1-gs)*f1s + gs*proj(...)
      单路 -> fuse_gate 换成常数 -80  =>  gs ≈ 0  =>  fs = f1s

M4 = PairFusion，B 分支含 F2/F3，C 分支含 F2/F3/student，共 5 个
E073FusionMatcher 实例，全部一起切换。V2 前端的前向仍会执行，但其输出
被乘以 ≈0，数学上等价于「只有 GDROS 那一路」。

评测口径与 eval_os_full_metrics.py 完全一致：404 对、iters=32、
LSR NPOINT=200000、valid mask、per_pixel 与 official_pair_level 两套统计。

用法（在 image_matching_part2_migrated/ 目录下）：
  CUDA_VISIBLE_DEVICES=2 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/ablate_v2_off.py --configs both,v2off --max-pairs 404 \
    --out .codex_tmp/ablate_v2_off_result.json
CPU 自检（不产出指标，只验证链路/门控/消融生效）：
  /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/ablate_v2_off.py --device cpu --max-pairs 1 --iters 1 --probe
"""
from __future__ import annotations
import sys, argparse, json, time
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
import eval_os_full_metrics as E

M4_CKPT = E.M4_CKPT
TEST_ROOT = E.TEST_ROOT
HIST_PAIRS = E.HIST_PAIRS
NPOINT = E.NPOINT
ITERS = E.ITERS
OFF_LOGIT = -80.0  # sigmoid(-80) ~ 1.8e-35，乘任何特征/corr 都严格可忽略


class ZeroGate(nn.Module):
    """替代 F2 的 fuse_gate：输出恒定大负数，使 sigmoid 后门控 ≈ 0。"""
    def forward(self, x):
        return x.new_full((x.shape[0], 1, x.shape[2], x.shape[3]), OFF_LOGIT)


def collect_fusion_modules(model):
    mods = [m for m in model.modules() if isinstance(m, e073.E073FusionMatcher)]
    return mods


def make_state(model):
    mods = collect_fusion_modules(model)
    state = {
        "mods": mods,
        "variants": [m.variant for m in mods],
        "orig_logit": [float(m.corr_gate_logit.detach().cpu().item()) for m in mods],
        "orig_fuse": [m.fuse_gate for m in mods],
    }
    unknown = [v for v in state["variants"] if v not in ("F2", "F3")]
    if unknown:
        raise RuntimeError("未处理的 E073 variant: %s" % unknown)
    return state


def set_v2(state, enabled):
    """enabled=True -> 原始两路；False -> 旁路 V2(DINO)，只留 GDROS 那一路。"""
    for i, m in enumerate(state["mods"]):
        if m.variant == "F3":
            m.corr_gate_logit.data.fill_(state["orig_logit"][i] if enabled else OFF_LOGIT)
        elif m.variant == "F2":
            m.fuse_gate = state["orig_fuse"][i] if enabled else ZeroGate()
    return gate_snapshot(state)


def gate_snapshot(state):
    out = []
    for m in state["mods"]:
        if m.variant == "F3":
            out.append({"variant": "F3", "a_v2": round(torch.sigmoid(m.corr_gate_logit).item(), 8)})
        else:
            out.append({"variant": "F2", "fuse_gate": type(m.fuse_gate).__name__})
    return out


@torch.inference_mode()
def forward_post(model, a, b, iters):
    preds = model(a, b, iters=iters, lsr_fn=e074.lsr)
    return e074.lsr(preds[-1])


@torch.inference_mode()
def evaluate(model, state, device, max_pairs, iters, tag):
    t0 = time.time()
    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    n = min(max_pairs, len(ds))
    print("[%s] pairs=%d iters=%d gates=%s" % (tag, n, iters, gate_snapshot(state)), flush=True)
    px_all, px_hist = E.Acc(), E.Acc()
    pr_all, pr_hist = E.Acc(), E.Acc()
    for i in range(n):
        _, a_img, b_img, gt, valid = ds[i]
        a = a_img[None].to(device)
        b = b_img[None].to(device)
        post = forward_post(model, a, b, iters).cpu()
        e = torch.linalg.vector_norm((post - gt[None]).float(), dim=1).view(-1)
        e_m = e[valid.view(-1).bool()].cpu().numpy()
        pair = np.array([e_m.mean()], dtype=np.float64)
        px_all.add(e_m); pr_all.add(pair)
        if i < HIST_PAIRS:
            px_hist.add(e_m); pr_hist.add(pair)
        if i % 20 == 0 or i == n - 1:
            print("[%s] %d/%d pair_epe=%.4f elapsed=%.1fs" %
                  (tag, i + 1, n, float(pair[0]), time.time() - t0), flush=True)
    return {
        "pairs_evaluated": n, "iters": iters, "lsr_npoint": NPOINT,
        "elapsed_sec": round(time.time() - t0, 1),
        "official_pair_level": {"full": pr_all.report(), "first_%d" % HIST_PAIRS: pr_hist.report()},
        "per_pixel": {"full": px_all.report(), "first_%d" % HIST_PAIRS: px_hist.report()},
    }


def fmt(out, protocol, subset):
    keys = ["AEPE", "RMSE", "@1px", "@2px", "@5px", "@10px", "@16px"]
    print("\n### %s / %s" % (protocol, subset), flush=True)
    print("| config | AEPE↓ | RMSE↓ | CMR@1↑ | CMR@2↑ | CMR@5↑ | CMR@10↑ | CMR@16↑ |", flush=True)
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for tag, r in out.items():
        m = r[protocol][subset]
        cells = [tag, "%.4f" % m["AEPE"], "%.4f" % m["RMSE"]]
        cells += ["%.2f" % m[k]["CMR"] for k in ("@1px", "@2px", "@5px", "@10px", "@16px")]
        print("| " + " | ".join(cells) + " |", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="both,v2off",
                    help="both=原始两路, v2off=旁路 DINO 那一路")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=404)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--probe", action="store_true", help="首对上比较两配置输出差异")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else "cuda:%d" % args.gpu)
    if args.device == "cpu":
        # LSR 内部硬编码 .cuda()（LSmodel.py:138），CPU 只能做链路/门控自检。
        # 用恒等映射替代，仅验证「切换 V2 会改变输出」，此模式下的指标无效。
        e074.lsr = lambda flow: flow
        print("[cpu-smoke] e074.lsr -> identity; 该模式仅自检，指标无效", flush=True)
    print("device=%s ckpt=%s" % (device, M4_CKPT), flush=True)
    t_load = time.time()
    model = e075.PairFusion(e074, "region", device).to(device)
    payload = torch.load(M4_CKPT, map_location=device)
    miss = model.load_state_dict(payload["model"], strict=False)
    print("[m4] missing=%s unexpected=%s" %
          (list(miss.missing_keys), list(miss.unexpected_keys)), flush=True)
    del payload
    model.eval()
    state = make_state(model)
    print("[structure] E073FusionMatcher=%d variants=%s orig_logit=%s load_sec=%.1f" %
          (len(state["mods"]), state["variants"], state["orig_logit"], time.time() - t_load), flush=True)

    if args.probe:
        ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
        _, a_img, b_img, _, _ = ds[0]
        a, b = a_img[None].to(device), b_img[None].to(device)
        set_v2(state, True); p_two = forward_post(model, a, b, args.iters).cpu().float()
        set_v2(state, False); p_one = forward_post(model, a, b, args.iters).cpu().float()
        d = (p_two - p_one).abs()
        print("[probe] max|Δpost|=%.6f mean|Δpost|=%.6f  (应显著>0，证明消融生效)" %
              (float(d.max()), float(d.mean())), flush=True)

    out = {}
    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        if cfg not in ("both", "v2off"):
            raise ValueError("unknown config: %s" % cfg)
        set_v2(state, enabled=(cfg == "both"))
        out[cfg] = evaluate(model, state, device, args.max_pairs, args.iters, cfg)

    for proto in ("official_pair_level", "per_pixel"):
        fmt(out, proto, "full")
    if args.out:
        payload_out = {"configs": out, "structure": {"variants": state["variants"],
                       "orig_corr_gate_logit": state["orig_logit"]},
                       "checkpoint": M4_CKPT, "device": str(device)}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload_out, f, indent=2, ensure_ascii=False)
        print("WROTE " + args.out, flush=True)
    print("RESULT_JSON " + json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
