#!/usr/bin/env python3
"""GDROS official test.py protocol on UBCv2 (or any OS-layout dataset root).

Reuses ``eval_os_official_protocol.Collector`` / ``run_one`` verbatim; the only
difference is the dataset root. The dataset must follow the GDROS OS layout:

    <root>/testing/opt/*.png
    <root>/testing/sar_warped/*.png
    <root>/testing/truth_flow/*.flo
    <root>/testing/datum/*.npy          (optional for evaluation)

Run on zzvr (cwd matters for E038B path resolution)::

  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  CUDA_VISIBLE_DEVICES=5 /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/eval_ubcv2_official_protocol.py \
    --which both --gpu 0 --max-pairs 1447 \
    --out .codex_tmp/ubcv2_official_20261115.json

Boundary: ``M4_step80000.pt`` and ``GDROS-os512.pth`` are both trained on OS.
Running them on UBCv2 is a **cross-dataset generalisation** test, NOT
comparable to the in-domain GDROS row of the paper's Table III.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import torch

import eval_os_official_protocol as P

DEFAULT_ROOT = "/mnt/sda/zhushihao/VS_Match/datasets/ubcv2"


def check_layout(root: str, dstype: str = "testing") -> dict:
    base = os.path.join(root, dstype)
    counts = {}
    for sub, pat in (("opt", "*.png"), ("sar_warped", "*.png"),
                     ("truth_flow", "*.flo"), ("datum", "*.npy")):
        counts[sub] = len(glob.glob(os.path.join(base, sub, pat)))
    problems = []
    if counts["opt"] == 0:
        problems.append("no opt/*.png")
    if counts["sar_warped"] != counts["opt"]:
        problems.append("opt/sar_warped count mismatch (%d vs %d)" %
                        (counts["opt"], counts["sar_warped"]))
    if counts["truth_flow"] != counts["opt"]:
        problems.append("opt/truth_flow count mismatch (%d vs %d)" %
                        (counts["opt"], counts["truth_flow"]))
    return {"root": root, "dstype": dstype, "counts": counts,
            "ok": not problems, "problems": problems}


def print_table_safe(results) -> None:
    """P.print_table crashes when a threshold has no inliers (AEPE=None).

    UBCv2 is hard enough that CMR@1 can legitimately be 0, so format those
    cells as '-' instead.
    """
    print("\n### GDROS official test.py protocol (UBCv2)", flush=True)
    print("| model | AEPE | RMSE_var | CMR@1 | AEPE@1 | CMR@2 | AEPE@2 | CMR@5 | AEPE@5 |",
          flush=True)
    print("|---|---|---|---|---|---|---|---|---|", flush=True)
    for name, r in results:
        m = r["official"]
        cells = [name, "%.4f" % m["AEPE"], "%.4f" % m["RMSE_official_var"]]
        for t in (1, 2, 5):
            q = m["@%dpx" % t]
            cells += ["%.2f" % q["CMR"],
                      ("%.4f" % q["AEPE"]) if q["AEPE"] is not None else "-"]
        print("| " + " | ".join(cells) + " |", flush=True)
    for name, r in results:
        m = r["official"]
        print("- %s: pairs=%d elapsed=%.1fs pixel_CMR 1/3/5px=%.2f/%.2f/%.2f "
              "vector_EPE_reference=%.4f" %
              (name, r["pairs_evaluated"], r["elapsed_sec"],
               m["pixel_CMR"]["1px"], m["pixel_CMR"]["3px"], m["pixel_CMR"]["5px"],
               m["AEPE_vector_EPE_reference"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", default=DEFAULT_ROOT,
                    help="dataset root containing <dstype>/opt|sar_warped|truth_flow")
    ap.add_argument("--dstype", default="testing",
                    help="only 'testing' is supported by P.run_one (hardcoded there)")
    ap.add_argument("--which", choices=["baseline", "m4", "both"], default="both")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--baseline-ckpt", default="",
                    help="override the GDROS checkpoint (default: released OS weights). "
                         "Point this at a UBCv2-trained checkpoint for the in-domain run.")
    ap.add_argument("--tag", default="",
                    help="free-form label written into the JSON, e.g. 'gdros_ubcv2_120k'")
    ap.add_argument("--max-pairs", type=int, default=1447,
                    help="paper UBCv2 test split is 1447 pairs after pre-screening")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    layout = check_layout(args.test_root, args.dstype)
    print("LAYOUT " + json.dumps(layout, ensure_ascii=False), flush=True)
    if not layout["ok"]:
        raise SystemExit("dataset layout invalid: " + "; ".join(layout["problems"]))
    if args.dstype != "testing":
        raise SystemExit("P.run_one hardcodes dstype='testing'; use --dstype testing")

    # P.run_one reads the module-level TEST_ROOT / BASE_CKPT at call time.
    P.TEST_ROOT = args.test_root
    if args.baseline_ckpt:
        if not os.path.exists(args.baseline_ckpt):
            raise SystemExit("baseline checkpoint not found: " + args.baseline_ckpt)
        P.BASE_CKPT = args.baseline_ckpt
        print("BASELINE_CKPT " + args.baseline_ckpt, flush=True)

    which_list = ["baseline", "m4"] if args.which == "both" else [args.which]
    device = torch.device("cuda:%d" % args.gpu)
    in_domain = bool(args.baseline_ckpt)
    out = {
        "dataset": "UBCv2",
        "tag": args.tag,
        "protocol": ("GDROS official test.py: per-component |dx|,|dy| error, "
                     "per-pair EPE(k), iters=32, LSR NPOINT=200000"),
        "test_root": args.test_root,
        "layout": layout,
        "iters": P.ITERS,
        "lsr_npoint": P.NPOINT,
        "requested_pairs": args.max_pairs,
        "weight_note": (
            "GDROS checkpoint is UBCv2-trained (in-domain); M4 is still OS-trained "
            "(cross-dataset). Not a like-for-like comparison."
            if in_domain else
            "Both checkpoints are OS-trained. This is a cross-dataset generalisation "
            "run, not the paper's in-domain Table III setting."),
        "results": {},
    }
    for w in which_list:
        out["results"][w] = P.run_one(w, device, args.max_pairs)
    print_table_safe([(w, out["results"][w]) for w in which_list])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
