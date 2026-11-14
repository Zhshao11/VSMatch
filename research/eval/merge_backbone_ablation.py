#!/usr/bin/env python3
"""Merge the two single-branch bypass results into one comparison.

Pure standard library, no GPU, no numpy. Run from anywhere:

  python research/eval/merge_backbone_ablation.py \
    --v1 research/results/OS_20261115/ablate_v1only_pruned.json \
    --v2 research/results/OS_20261115/ablate_v2only_pruned.json \
    --out research/results/OS_20261115/backbone_ablation_merged.json

The `both` reference (424 pair-level EPE values) is taken from the archived
2026-11-14 ablation JSON. The 2026-11-15 C-teacher pruning is bit-exact, so
those values are still valid for the pruned model.

These are frozen-weight sensitivity diagnostics, NOT module ablations. A large
degradation mixes lost information with downstream out-of-distribution inputs
and cannot be read as either branch's marginal contribution.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HIST_BOTH = ROOT / "research/results/OS_20261114/ablate_v2_off_result_official.json"
DEF_V1 = ROOT / "research/results/OS_20261115/ablate_v1only_pruned.json"
DEF_V2 = ROOT / "research/results/OS_20261115/ablate_v2only_pruned.json"
DEF_OUT = ROOT / "research/results/OS_20261115/backbone_ablation_merged.json"


def official_of(obj, path):
    """Accept both the new single-config schema and the archived configs schema."""
    if "result" in obj:
        return obj["result"]["official"]
    if "configs" in obj:
        return obj["configs"][path]["official"]
    if "official" in obj:
        return obj["official"]
    raise KeyError("cannot find official metrics in %s" % path)


def pair_epe_of(obj, path):
    if "result" in obj:
        return obj["result"]["pair_EPE"]
    if "configs" in obj:
        return obj["configs"][path]["pair_EPE"]
    raise KeyError("cannot find pair_EPE in %s" % path)


def summary(official):
    return {
        "AEPE": official["AEPE"],
        "CMR@1": official["@1px"]["CMR"],
        "CMR@2": official["@2px"]["CMR"],
        "CMR@5": official["@5px"]["CMR"],
        "RMSE_official_var": official["RMSE_official_var"],
        "pairs": official["pairs"],
    }


def delta_stats(a, b):
    """a - b, positive means a is worse (larger per-pair EPE)."""
    if len(a) != len(b):
        raise ValueError("pair_EPE length mismatch: %d vs %d" % (len(a), len(b)))
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    return {
        "n": n,
        "mean_delta": statistics.fmean(d),
        "median_delta": statistics.median(d),
        "a_better_pairs": sum(1 for x in d if x < 0),
        "a_better_pct": 100.0 * sum(1 for x in d if x < 0) / n,
        "delta_gt_5px": sum(1 for x in d if x > 5),
        "delta_lt_minus_5px": sum(1 for x in d if x < -5),
        "a_catastrophic_gt_10px": sum(1 for x in a if x > 10),
        "b_catastrophic_gt_10px": sum(1 for x in b if x > 10),
        "worst_10": sorted(
            [{"idx": i, "a": round(a[i], 4), "b": round(b[i], 4),
              "delta": round(d[i], 4)} for i in range(n)],
            key=lambda r: -r["delta"])[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, default=DEF_V1)
    ap.add_argument("--v2", type=Path, default=DEF_V2)
    ap.add_argument("--both", type=Path, default=HIST_BOTH)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    args = ap.parse_args()

    o1 = json.loads(args.v1.read_text())
    o2 = json.loads(args.v2.read_text())
    ob = json.loads(args.both.read_text())

    m1, m2, mb = (official_of(o1, "v2off"), official_of(o2, "v2off"),
                  official_of(ob, "both"))
    p1, p2, pb = (pair_epe_of(o1, "v2off"), pair_epe_of(o2, "v2off"),
                  pair_epe_of(ob, "both"))

    table = {
        "gdros_released": o1.get("gdros_released_reference"),
        "both": summary(mb),
        "v1only": summary(m1),
        "v2only": summary(m2),
    }
    pairwise = {
        "v1only_minus_both": delta_stats(p1, pb),
        "v2only_minus_both": delta_stats(p2, pb),
        "v1only_minus_v2only": delta_stats(p1, p2),
    }
    out = {
        "protocol": "GDROS official test.py, 424 pairs, iters=32, LSR NPOINT=200000",
        "ablation_class": "diagnostic_bypass_not_module_ablation",
        "structural_caveat": o1.get("structural_caveat"),
        "pruned_model": o1.get("pruned_model"),
        "table": table,
        "pairwise": pairwise,
        "both_pair_EPE_source": str(args.both.relative_to(ROOT)),
        "source_files": [str(args.both.relative_to(ROOT)),
                         str(args.v1.relative_to(ROOT)),
                         str(args.v2.relative_to(ROOT))],
        "sanity_pass": {"v1only": o1.get("sanity_pass"),
                        "v2only": o2.get("sanity_pass")},
    }

    print("\n### 对称单路旁路（剪枝后 M4，官方口径 424 对）")
    print("| 配置 | AEPE↓ | CMR@1↑ | CMR@2↑ | CMR@5↑ | RMSE_var |")
    print("|---|---|---|---|---|---|")
    for k in ("gdros_released", "both", "v1only", "v2only"):
        r = table[k]
        if r is None:
            continue
        ae = r.get("AEPE")
        print("| %s | %s | %s | %s | %s | %s |" % (
            k, "%.4f" % ae if ae is not None else "-",
            "%.2f" % r["CMR@1"] if isinstance(r.get("CMR@1"), (int, float)) else "-",
            "%.2f" % r["CMR@2"] if isinstance(r.get("CMR@2"), (int, float)) else "-",
            "%.2f" % r["CMR@5"] if isinstance(r.get("CMR@5"), (int, float)) else "-",
            "%.4f" % r["RMSE_official_var"]
            if isinstance(r.get("RMSE_official_var"), (int, float)) else "-"))
    for k, v in pairwise.items():
        print("\n%s: mean=%+.4f median=%+.4f, a 变好 %d/%d (%.1f%%), "
              "Δ>+5px %d 对, Δ<-5px %d 对, a/b >10px 灾难失败 %d/%d"
              % (k, v["mean_delta"], v["median_delta"], v["a_better_pairs"],
                 v["n"], v["a_better_pct"], v["delta_gt_5px"],
                 v["delta_lt_minus_5px"], v["a_catastrophic_gt_10px"],
                 v["b_catastrophic_gt_10px"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n[out] %s" % args.out)


if __name__ == "__main__":
    main()
