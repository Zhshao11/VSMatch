#!/usr/bin/env python3
"""Evaluate GDROS checkpoints **trained on UBCv2** (in-domain) on the UBCv2 test set.

This closes the "GDROS: UBCv2-trained -> UBCv2 test" cell that pairs with the
already-published "M4: OS-trained -> UBCv2 test" row.  Everything about the
metric is unchanged: it reuses ``eval_os_official_protocol.run_one`` verbatim
(official ``test.py`` semantics, per-component ``|dx|,|dy|``, per-pair EPE(k),
``iters=32``, ``LSR NPOINT=200000``), and only overrides two module globals:

  ``P.TEST_ROOT``  -> the generated UBCv2 test set
  ``P.BASE_CKPT``  -> each UBCv2-trained checkpoint in turn

``run_one`` builds ``nn.DataParallel(GDROS(...))`` and does
``load_state_dict(torch.load(BASE_CKPT))``; ``train.py`` also saves from a
``DataParallel`` model, so the ``module.`` prefixes match and no key surgery is
needed.

Also recomputes the zero-flow identity baseline from ``datum/*.json`` so the
output is self-contained (expected 29.7910 on the seed901 set).

Run on zzvr (cwd matters for E038B path resolution)::

  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  CUDA_VISIBLE_DEVICES=5 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/eval_ubcv2_indomain.py --gpu 0 --max-pairs 1625 \
    --ckpt-glob '/mnt/sda/zhushihao/VS_Match/external_repos/E068A_GDROS_official/checkpoints/*_gdros-ubcv2-512.pth' \
    --ckpts /mnt/sda/zhushihao/VS_Match/external_repos/E068A_GDROS_official/checkpoints/gdros-ubcv2-512.pth \
    --out .codex_tmp/ubcv2_indomain_curve.json
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

import eval_os_official_protocol as P

DEFAULT_ROOT = "/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/generated/seed901"

# Anchors from research/results/UBCV2_20261115/ubcv2_official_20261115.json
# (both OS-trained, same 1625 pairs, same protocol).
INFERENCE_ANCHORS = {
    "GDROS-os512.pth (OS-trained, cross-dataset)": 22.009050187680995,
    "E075A-M4@80K (OS-trained, cross-dataset)": 3.8941,
}


def step_of(path: str) -> int:
    m = re.match(r"(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else 10 ** 9


def identity_baseline(root: str) -> float | None:
    datum = os.path.join(root, "testing", "datum")
    files = sorted(glob.glob(os.path.join(datum, "*.json")))
    if not files:
        return None
    ys, xs = np.mgrid[0:512, 0:512]
    per = []
    for f in files:
        meta = json.load(open(f, encoding="utf-8"))
        M = np.array(meta["affine_src_to_target"])
        dx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs
        dy = M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys
        per.append(float((np.abs(dx).mean() + np.abs(dy).mean()) / 2))
    return float(np.mean(per))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", default=DEFAULT_ROOT)
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--ckpt-glob", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=1625)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpts = [c for c in args.ckpts if os.path.exists(c)]
    if args.ckpt_glob:
        ckpts += [c for c in glob.glob(args.ckpt_glob) if c not in ckpts]
    ckpts = sorted(set(ckpts), key=step_of)
    if not ckpts:
        raise SystemExit("no checkpoint found; pass --ckpts and/or --ckpt-glob")
    print("[indomain] %d checkpoints:" % len(ckpts), flush=True)
    for c in ckpts:
        print("  step=%s %s" % (step_of(c), c), flush=True)

    P.TEST_ROOT = args.test_root
    device = torch.device("cuda:%d" % args.gpu)

    ident = identity_baseline(args.test_root)
    print("[indomain] zero-flow identity baseline AEPE = %s" % ident, flush=True)

    out = {
        "dataset": "UBCv2",
        "setting": "GDROS trained on UBCv2 (in-domain) -> UBCv2 test",
        "protocol": ("GDROS official test.py: per-component |dx|,|dy|, per-pair "
                     "EPE(k), iters=32, LSR NPOINT=200000"),
        "test_root": args.test_root,
        "pairs": args.max_pairs,
        "identity_baseline_AEPE": ident,
        "anchors_cross_dataset": INFERENCE_ANCHORS,
        "curve": [],
    }
    t0 = time.time()
    for c in ckpts:
        P.BASE_CKPT = c
        print("\n[indomain] ==== %s (step %s) ====" % (os.path.basename(c), step_of(c)),
              flush=True)
        r = P.run_one("baseline", device, args.max_pairs)
        r["step"] = step_of(c)
        r["ckpt_path"] = c
        out["curve"].append(r)
        m = r["official"]
        print("[indomain] step=%s AEPE=%.4f CMR@1=%.2f CMR@2=%.2f CMR@5=%.2f "
              "elapsed=%.1fs" % (step_of(c), m["AEPE"], m["@1px"]["CMR"],
                                 m["@2px"]["CMR"], m["@5px"]["CMR"],
                                 r["elapsed_sec"]), flush=True)

    print("\n### GDROS UBCv2-trained -> UBCv2 test (in-domain)\n", flush=True)
    print("| step | AEPE | CMR@1 | CMR@2 | CMR@3 | CMR@5 | 相对零流基线增益 |", flush=True)
    print("|---|---|---|---|---|---|---|", flush=True)
    for r in out["curve"]:
        m = r["official"]
        gain = ("%.1f%%" % (100 * (1 - m["AEPE"] / ident))) if ident else "-"
        print("| %s | %.4f | %.2f | %.2f | %.2f | %.2f | %s |" %
              (r["step"], m["AEPE"], m["@1px"]["CMR"], m["@2px"]["CMR"],
               m["@3px"]["CMR"], m["@5px"]["CMR"], gain), flush=True)
    print("\n参照（同一 1625 对、同一口径、OS 训练）:", flush=True)
    for k, v in INFERENCE_ANCHORS.items():
        print("- %s: AEPE %.4f%s" %
              (k, v, ("  (相对零流增益 %.1f%%)" % (100 * (1 - v / ident))) if ident else ""),
              flush=True)

    out["elapsed_sec_total"] = round(time.time() - t0, 1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nWROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
