#!/usr/bin/env python3
"""Summarise a UBCv2 official-protocol result JSON into a markdown table.

Pure stdlib, no GPU. Usage::

  python research/eval/summarize_ubcv2.py \
    research/results/UBCV2_20261115/ubcv2_official_20261115.json
"""
from __future__ import annotations
import argparse
import json

# OS official-protocol reference values (2026-11-13, 424 pairs, same protocol).
OS_AEPE = {"baseline": 1.4618458366096103, "m4": 1.473827834446231}


def fmt(x, n=4):
    return "-" if x is None else ("%.*f" % (n, x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    args = ap.parse_args()
    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)
    res = data["results"]
    print("| 模型 | AEPE↓ | CMR@1↑ | CMR@2↑ | CMR@3↑ | CMR@4↑ | CMR@5↑ | RMSE_var | >5px 对 | 耗时(s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in res.items():
        m = r["official"]
        n = r["pairs_evaluated"]
        gt5 = n * (1.0 - m["@5px"]["CMR"] / 100.0)
        cells = [name, fmt(m["AEPE"])]
        cells += [fmt(m["@%dpx" % t]["CMR"], 2) for t in (1, 2, 3, 4, 5)]
        cells += [fmt(m["RMSE_official_var"]), "%.0f" % gt5,
                  fmt(r["elapsed_sec"], 1)]
        print("| " + " | ".join(cells) + " |")
    print()
    for name, r in res.items():
        a = r["official"]["AEPE"]
        ref = OS_AEPE.get(name)
        if ref:
            print("- %s: UBCv2 AEPE %.4f vs OS %.4f = %.2fx (泛化落差)" %
                  (name, a, ref, a / ref))
        print("- %s: pixel_CMR 1/3/5px = %.2f/%.2f/%.2f; "
              "AEPE@1=%s, AEPE@5=%s, vector_EPE_reference=%.4f" %
              (name, r["official"]["pixel_CMR"]["1px"],
               r["official"]["pixel_CMR"]["3px"],
               r["official"]["pixel_CMR"]["5px"],
               fmt(r["official"]["@1px"]["AEPE"]),
               fmt(r["official"]["@5px"]["AEPE"]),
               r["official"]["AEPE_vector_EPE_reference"]))
    print("\ncheckpoints: " + ", ".join(
        "%s=%s" % (k, v["checkpoint"]) for k, v in res.items()))


if __name__ == "__main__":
    main()
