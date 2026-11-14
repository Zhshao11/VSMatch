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

评测口径 = GDROS 官方 test.py，与 eval_os_official_protocol.py 完全一致：
  epe = sum((flow_pre - flow_gt)**2, dim=0).sqrt()   # [2,H,W] = |dx|,|dy|
  EPE(k) = mean(epe) 配对级；AEPE = mean(EPE(k))；"RMSE" = np.var(EPE(k))
  CMR@T = 100 * |{k : EPE(k) < T}| / N；AEPE@T = mean(inliers)
  424 对全量、iters=32、LSR NPOINT=200000、不额外乘 valid mask（实测全 1）

同一次运行另跑 GDROS 官方权重（baseline），用于 sanity check 与横向对比；
M4 的 both / v2off 与 GDROS 三者都在官方口径下报数，可直接并列。

用法（在 image_matching_part2_migrated/ 目录下）：
  CUDA_VISIBLE_DEVICES=2 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/ablate_v2_off.py --configs both,v2off --max-pairs 424 \
    --out .codex_tmp/ablate_v2_off_result_official.json
CPU 自检（不产出有效指标，只验证链路/门控/消融生效，会自动跳过 baseline）：
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
import eval_os_official_protocol as P

M4_CKPT = P.M4_CKPT
BASE_CKPT = P.BASE_CKPT
TEST_ROOT = P.TEST_ROOT
NPOINT = P.NPOINT
ITERS = P.ITERS
TAUS = P.TAUS
OFF_LOGIT = -80.0  # sigmoid(-80) ~ 1.8e-35，乘任何特征/corr 都严格可忽略

# 2026-11-13 官方 test.py 口径、424 对全量的历史基准，用于 sanity check
HIST_REF = {
    "gdros": {"AEPE": 1.4618, "CMR@1": 33.73, "CMR@2": 81.60,
              "CMR@5": 99.53, "RMSE_official_var": 0.7463},
    "both": {"AEPE": 1.4738, "CMR@1": 72.41, "CMR@2": 84.91,
             "CMR@5": 94.10, "RMSE_official_var": 8.3350},
}
TOL = {"AEPE": 0.02, "CMR@1": 0.10, "CMR@2": 0.10,
       "CMR@5": 0.10, "RMSE_official_var": 0.10}


class ZeroGate(nn.Module):
    """替代 F2 的 fuse_gate：输出恒定大负数，使 sigmoid 后门控 ≈ 0。"""
    def forward(self, x):
        return x.new_full((x.shape[0], 1, x.shape[2], x.shape[3]), OFF_LOGIT)


def make_state(model):
    mods = [m for m in model.modules() if isinstance(m, e073.E073FusionMatcher)]
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
    return e074.lsr(preds[-1]).cpu()


@torch.inference_mode()
def evaluate(model, state, device, max_pairs, iters, tag):
    t0 = time.time()
    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    n = min(max_pairs, len(ds))
    print("[%s] dataset=%d evaluating=%d iters=%d gates=%s" %
          (tag, len(ds), n, iters, gate_snapshot(state)), flush=True)
    col = P.Collector()
    for i in range(n):
        _, a_img, b_img, gt, _ = ds[i]
        a = a_img[None].to(device)
        b = b_img[None].to(device)
        post = forward_post(model, a, b, iters)
        diff = post - gt[None]           # 官方 test.py 的原始 diff，CPU [1,2,H,W]
        col.add_pair(diff)
        if i % 50 == 0 or i == n - 1:
            print("[%s] %d/%d pair_epe=%.4f elapsed=%.1fs" %
                  (tag, i + 1, n, col.pairs[-1], time.time() - t0), flush=True)
    return {
        "protocol": "GDROS official test.py (component-wise |dx|,|dy| + pair-level)",
        "checkpoint": M4_CKPT,
        "pairs_evaluated": n,
        "iters": iters,
        "lsr_npoint": NPOINT,
        "gates": gate_snapshot(state),
        "elapsed_sec": round(time.time() - t0, 1),
        "official": col.report(),
        # 配对级原始数列：用于 both / v2off 的逐对差值与灾难失败归因
        "pair_EPE": [float(x) for x in col.pairs],
        "pair_EPE_vector_reference": [float(x) for x in col.vec_pairs],
    }


def print_official_table(out):
    """官方口径主表；某个阈值无内点时 AEPE 为 None，打成 - 而不是崩掉。"""
    def f(x, n=4):
        return "-" if x is None else ("%.*f" % (n, x))
    print("\n### 官方 test.py 口径（逐分量 |Δ| + 配对级）", flush=True)
    print("| 配置 | AEPE↓ | RMSE↓(官方栏位/方差) | CMR@1↑ | AEPE@1↓ | CMR@2↑ | AEPE@2↓ | CMR@5↑ | AEPE@5↓ |", flush=True)
    print("|---|---|---|---|---|---|---|---|---|", flush=True)
    for name, r in out.items():
        m = r["official"]
        cells = [name, f(m["AEPE"]), f(m["RMSE_official_var"])]
        for t in (1, 2, 5):
            q = m["@%dpx" % t]
            cells += [f(q["CMR"], 2), f(q["AEPE"])]
        print("| " + " | ".join(cells) + " |", flush=True)
    print("\n### 官方栏位补充（T=1..5 的 RMSE 亦为方差）与逐像素 1/3/5px", flush=True)
    for name, r in out.items():
        m = r["official"]
        print("- %s: pairs=%s, RMSE(论文公式 std)=%s, pixel_RMSE_true=%s, "
              "AEPE(向量 EPE 参考)=%s; pixel_CMR 1/3/5px = %s / %s / %s" %
              (name, m["pairs"], f(m["RMSE_paper_formula_std"]), f(m["pixel_RMSE_true"]),
               f(m["AEPE_vector_EPE_reference"]),
               f(m["pixel_CMR"]["1px"], 2), f(m["pixel_CMR"]["3px"], 2),
               f(m["pixel_CMR"]["5px"], 2)), flush=True)


def print_sanity(out):
    """确认 GDROS 与 both 复现 2026-11-13 官方口径数值；不通过则 v2off 结论不可用。"""
    print("\n### sanity check（对照 2026-11-13 官方口径 424 对）", flush=True)
    print("| 配置 | 指标 | 本次 | 历史 | 容差 | 结论 |", flush=True)
    print("|---|---|---|---|---|---|", flush=True)
    ok = True
    for name, ref in HIST_REF.items():
        if name not in out:
            print("| %s | - | 缺失 | - | - | FAIL（跳过基线时无法核验口径） |" % name, flush=True)
            ok = False
            continue
        m = out[name]["official"]
        cur = {
            "AEPE": m["AEPE"],
            "CMR@1": m["@1px"]["CMR"],
            "CMR@2": m["@2px"]["CMR"],
            "CMR@5": m["@5px"]["CMR"],
            "RMSE_official_var": m["RMSE_official_var"],
        }
        for k, v in ref.items():
            d = abs(cur[k] - v)
            passed = d <= TOL[k]
            ok = ok and passed
            print("| %s | %s | %.4f | %.4f | ±%.4f | %s |" %
                  (name, k, cur[k], v, TOL[k], "PASS" if passed else "FAIL"), flush=True)
    print("SANITY %s" % ("PASS" if ok else "FAIL —— 不要采用 v2off 结论，先排查口径/权重"),
          flush=True)
    return ok


def print_delta(out):
    """逐对差值：v2off 相对 both 的恶化分布，比只看均值更能说明问题。"""
    if "both" not in out or "v2off" not in out:
        return
    a = np.asarray(out["both"]["pair_EPE"], dtype=np.float64)
    b = np.asarray(out["v2off"]["pair_EPE"], dtype=np.float64)
    if a.size != b.size or a.size == 0:
        return
    d = b - a
    print("\n### 逐对 ΔEPE = v2off - both（官方口径，正数=去掉 DINO 后变差）", flush=True)
    print("- mean Δ=%+.4f  median Δ=%+.4f  improved=%d/%d (%.1f%%)" %
          (d.mean(), np.median(d), int((d < 0).sum()), d.size,
           100.0 * (d < 0).mean()), flush=True)
    for t in (1, 2, 5):
        print("- Δ>+%dpx: %d 对; Δ<-%.0fpx: %d 对" %
              (t, int((d > t).sum()), t, int((d < -t).sum())), flush=True)
    order = np.argsort(-d)[:10]
    print("- 恶化最大的 10 对 (0-based idx, both, v2off, Δ): %s" %
          [(int(i), round(float(a[i]), 3), round(float(b[i]), 3), round(float(d[i]), 3))
           for i in order], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="both,v2off",
                    help="both=原始两路, v2off=旁路 DINO 那一路")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=424)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--probe", action="store_true", help="首对上比较两配置输出差异")
    ap.add_argument("--no-baseline", action="store_true", help="跳过 GDROS 官方权重参照行")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else "cuda:%d" % args.gpu)
    cpu_smoke = args.device == "cpu"
    if cpu_smoke:
        # LSR 内部硬编码 .cuda()（LSmodel.py:138），CPU 只能做链路/门控自检。
        e074.lsr = lambda flow: flow
        print("[cpu-smoke] e074.lsr -> identity; 该模式仅自检，指标无效", flush=True)
    if args.iters != ITERS or args.max_pairs < 424:
        print("[warn] iters=%d pairs=%d 不是官方设置(32/424)，本次指标不可与论文/历史值并列" %
              (args.iters, args.max_pairs), flush=True)

    # sanity check 只在完整设置下有意义
    full_run = (not cpu_smoke and args.iters == ITERS and args.max_pairs >= 424
                and not args.probe)
    run_baseline = (not args.no_baseline and not cpu_smoke and not args.probe
                    and args.iters == ITERS and args.max_pairs >= 424)
    print("device=%s ckpt=%s baseline=%s full_run=%s" %
          (device, M4_CKPT, run_baseline, full_run), flush=True)

    out = {}
    if run_baseline:
        # 先跑 GDROS 官方权重：独立模型，run_one 内部会释放显存
        out["gdros"] = P.run_one("baseline", device, args.max_pairs)

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
          (len(state["mods"]), state["variants"], state["orig_logit"], time.time() - t_load),
          flush=True)

    if args.probe:
        ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
        _, a_img, b_img, _, _ = ds[0]
        a, b = a_img[None].to(device), b_img[None].to(device)
        set_v2(state, True); p_two = forward_post(model, a, b, args.iters).float()
        set_v2(state, False); p_one = forward_post(model, a, b, args.iters).float()
        d = (p_two - p_one).abs()
        print("[probe] max|Δpost|=%.6f mean|Δpost|=%.6f  (应显著>0，证明消融生效)" %
              (float(d.max()), float(d.mean())), flush=True)

    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        if cfg not in ("both", "v2off"):
            raise ValueError("unknown config: %s" % cfg)
        set_v2(state, enabled=(cfg == "both"))
        out[cfg] = evaluate(model, state, device, args.max_pairs, args.iters, cfg)

    print_official_table(out)
    sanity = print_sanity(out) if full_run else None
    print_delta(out)

    if args.out:
        payload_out = {
            "protocol": "GDROS official test.py",
            "configs": out,
            "structure": {"variants": state["variants"],
                          "orig_corr_gate_logit": state["orig_logit"]},
            "checkpoint": M4_CKPT,
            "baseline_checkpoint": BASE_CKPT if run_baseline else None,
            "device": str(device),
            "full_run": full_run,
            "sanity_pass": sanity,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload_out, f, indent=2, ensure_ascii=False)
        print("WROTE " + args.out, flush=True)

    # 精简 RESULT_JSON：不重复打印 424 长度原始数列
    brief = {k: {kk: vv for kk, vv in v.items()
                 if kk not in ("pair_EPE", "pair_EPE_vector_reference")}
             for k, v in out.items()}
    print("RESULT_JSON " + json.dumps(brief, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
