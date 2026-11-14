#!/usr/bin/env python3
"""零训练 GRU 迭代次数 sweep：一次 32-iteration 前向，拿到所有迭代点的官方口径指标。

为什么这个干预比「旁路 V2」干净：

  * GRU 是**同一份权重循环展开**（`e073a...py:248` `for _ in range(self.iters)`），
    `iters` 是调用次数，不是结构层、不是可训练模块。改 iters 不改参数、
    不换特征、不改输入分布。
  * 训练 `--train-iters=12`，正式评测 `iters=32` 属于**外推**；把 iters 降到
    12/8/6 反而更贴近训练分布，不制造下游 OOD。
  * `PairFusion.forward` 在**每个迭代点**都调 `gate_features`，后者对 B/C 各做一次
    LSR ⇒ 一次 iters=K 的前向含 `2K` 次 LSR，再加评测对 `preds[-1]` 的 1 次 ⇒ `2K+1`。
    因此 iters 与训练/评测计算量近似线性相关，这直接对应「省训练时间」。

本脚本只做一次 `iters=32` 前向（内部 64 次 LSR），再对候选迭代点的
`preds[k-1]` 各补 1 次 LSR + 官方 `Collector`，即可同时得到全部曲线，
无需为每个 iters 重跑模型。

口径 = GDROS 官方 `test.py`（逐分量 |dx|,|dy| + 配对级），OS 424 对全量。
`k=32` 必须与 2026-11-13 官方值 `AEPE 1.4738 / CMR@1 72.41` 吻合才算有效。

用法（在 image_matching_part2_migrated/ 目录下）：
  CUDA_VISIBLE_DEVICES=2 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/sweep_gru_iters.py --max-pairs 424 --gpu 0 \
    --out .codex_tmp/sweep_gru_iters_official.json
"""
from __future__ import annotations
import sys, argparse, json, time
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
import e074a_f2_f3_twostage_fusion as e074
e072, e073 = e074.add_paths()
import e075a_preserve_high_precision_recover_c2c5 as e075
import eval_os_official_protocol as P

M4_CKPT = P.M4_CKPT
TEST_ROOT = P.TEST_ROOT
ITERS_MAX = P.ITERS          # 32
DEFAULT_SWEEP = (4, 6, 8, 12, 16, 24, 32)

# 官方口径 424 对、iters=32 的历史基准（eval_os_official_protocol.py，2026-11-13）
HIST_M4_32 = {"AEPE": 1.4738, "CMR@1": 72.41, "CMR@2": 84.91,
              "CMR@5": 94.10, "RMSE_official_var": 8.3350}
TOL = {"AEPE": 0.02, "CMR@1": 0.10, "CMR@2": 0.10,
       "CMR@5": 0.10, "RMSE_official_var": 0.10}
# 静态参照：GDROS 官方权重（官方口径 424 对），仅用于判断掉点后是否仍优于 baseline
GDROS_REF = {"AEPE": 1.4618, "CMR@1": 33.73, "CMR@2": 81.60,
             "CMR@5": 99.53, "RMSE_official_var": 0.7463}


def fmt(x, n=4):
    return "-" if x is None else ("%.*f" % (n, x))


@torch.inference_mode()
def run(model, device, max_pairs, sweep):
    ds = core.datasets.opt_sar_test(split="testing", dstype="testing", root=TEST_ROOT)
    n = min(max_pairs, len(ds))
    cols = {k: P.Collector() for k in sweep}
    t0 = time.time()
    print("[sweep] dataset=%d evaluating=%d iters_max=%d sweep=%s" %
          (len(ds), n, ITERS_MAX, list(sweep)), flush=True)
    for i in range(n):
        _, a_img, b_img, gt, _ = ds[i]
        a = a_img[None].to(device)
        b = b_img[None].to(device)
        # 一次前向；gate_features 内部对每个迭代点各做 2 次 LSR
        preds = model(a, b, iters=ITERS_MAX, lsr_fn=e074.lsr)
        if len(preds) != ITERS_MAX:
            raise RuntimeError("preds len %d != iters %d" % (len(preds), ITERS_MAX))
        for k in sweep:
            post = e074.lsr(preds[k - 1]).cpu()
            cols[k].add_pair(post - gt[None])
        if i % 50 == 0 or i == n - 1:
            anchor = max(sweep)
            print("[sweep] %d/%d epe@%d=%.4f elapsed=%.1fs" %
                  (i + 1, n, anchor, cols[anchor].pairs[-1], time.time() - t0), flush=True)
    return {str(k): {"iters": k,
                     "official": cols[k].report(),
                     "pair_EPE": [float(x) for x in cols[k].pairs],
                     # LSR 调用次数：每个迭代点 gate 用 2 次，评测再用 1 次
                     "lsr_calls_per_pair": 2 * k + 1,
                     "relative_inference_cost": round((2 * k + 1) / (2 * ITERS_MAX + 1), 4)}
            for k in sweep}, round(time.time() - t0, 1)


def print_table(res, sanity_ok):
    print("\n### 官方 test.py 口径 / GRU 迭代次数 sweep（零训练，同一份权重）", flush=True)
    print("| iters | AEPE↓ | CMR@1↑ | CMR@2↑ | CMR@5↑ | RMSE↓(方差) | 相对推理成本 | LSR/对 |", flush=True)
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for k in sorted(res, key=lambda x: int(x)):
        r = res[k]; m = r["official"]
        print("| %s | %s | %s | %s | %s | %s | %.3f | %d |" %
              (k, fmt(m["AEPE"]), fmt(m["@1px"]["CMR"], 2), fmt(m["@2px"]["CMR"], 2),
               fmt(m["@5px"]["CMR"], 2), fmt(m["RMSE_official_var"]),
               r["relative_inference_cost"], r["lsr_calls_per_pair"]), flush=True)
    print("| GDROS 官方权重(静态参照) | %s | %s | %s | %s | %s | - | - |" %
          (fmt(GDROS_REF["AEPE"]), fmt(GDROS_REF["CMR@1"], 2),
           fmt(GDROS_REF["CMR@2"], 2), fmt(GDROS_REF["CMR@5"], 2),
           fmt(GDROS_REF["RMSE_official_var"])), flush=True)
    print("\n相对推理成本 = (2·k+1)/(2·32+1)，即 LSR 调用次数之比；"
          "训练时同样按 2·k 次 LSR/step 计。", flush=True)
    print("SANITY %s" % ("PASS" if sanity_ok else "FAIL —— 不要采用本表结论"), flush=True)


def check_sanity(res):
    if str(ITERS_MAX) not in res:
        return False
    m = res[str(ITERS_MAX)]["official"]
    cur = {"AEPE": m["AEPE"], "CMR@1": m["@1px"]["CMR"], "CMR@2": m["@2px"]["CMR"],
           "CMR@5": m["@5px"]["CMR"], "RMSE_official_var": m["RMSE_official_var"]}
    print("\n### sanity check（k=32 对照 2026-11-13 官方口径 424 对）", flush=True)
    ok = True
    for key, ref in HIST_M4_32.items():
        passed = abs(cur[key] - ref) <= TOL[key]
        ok = ok and passed
        print("  %-20s cur=%.4f hist=%.4f  %s" %
              (key, cur[key], ref, "PASS" if passed else "FAIL"), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=424)
    ap.add_argument("--sweep", default=",".join(str(x) for x in DEFAULT_SWEEP))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sweep = tuple(sorted({int(x) for x in args.sweep.split(",")}))
    if min(sweep) < 1 or max(sweep) > ITERS_MAX:
        raise ValueError("sweep must be within [1, %d]" % ITERS_MAX)
    device = torch.device("cpu" if args.device == "cpu" else "cuda:%d" % args.gpu)
    if args.device == "cpu":
        e074.lsr = lambda flow: flow
        print("[cpu-smoke] e074.lsr -> identity; 仅链路自检，指标无效", flush=True)
    print("device=%s ckpt=%s" % (device, M4_CKPT), flush=True)

    t_load = time.time()
    model = e075.PairFusion(e074, "region", device).to(device)
    payload = torch.load(M4_CKPT, map_location=device)
    miss = model.load_state_dict(payload["model"], strict=False)
    print("[m4] missing=%s unexpected=%s load_sec=%.1f" %
          (list(miss.missing_keys), list(miss.unexpected_keys), time.time() - t_load),
          flush=True)
    del payload
    model.eval()

    res, elapsed = run(model, device, args.max_pairs, sweep)
    full_run = (args.device != "cpu" and args.max_pairs >= 424
                and ITERS_MAX in sweep)
    sanity = check_sanity(res) if full_run else False
    print_table(res, sanity)

    if args.out:
        out = {
            "protocol": "GDROS official test.py",
            "experiment": "zero-training GRU iteration sweep (single 32-iter forward)",
            "note": "推理 iters 的敏感性曲线；训练 iters 是否可同步减少仍需短训验证",
            "checkpoint": M4_CKPT,
            "pairs_evaluated": min(args.max_pairs, 424),
            "iters_max": ITERS_MAX,
            "train_iters_used_by_checkpoint": 12,
            "elapsed_sec": elapsed,
            "device": str(device),
            "full_run": full_run,
            "sanity_pass": sanity,
            "gdros_static_reference": GDROS_REF,
            "results": res,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("WROTE " + args.out, flush=True)

    brief = {k: {kk: vv for kk, vv in v.items() if kk != "pair_EPE"}
             for k, v in res.items()}
    print("RESULT_JSON " + json.dumps(brief, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
