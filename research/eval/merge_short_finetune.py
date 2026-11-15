#!/usr/bin/env python3
"""Compare the two adaptation short-finetunes against their zero-shot bypass.

Pure standard library, no GPU. Default paths are relative to the repository
root (two levels above this file).

  python research/eval/merge_short_finetune.py
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R15 = ROOT / "research/results/OS_20261115"
R14 = ROOT / "research/results/OS_20261114"

FILES = {
    "v1only": {
        "ft": R15 / "short_ft_v1only.json",
        "zero": R15 / "ablate_v1only_pruned.json",
    },
    "v2only": {
        "ft": R15 / "short_ft_v2only.json",
        "zero": R15 / "ablate_v2only_pruned.json",
    },
}
BOTH_JSON = R14 / "ablate_v2_off_result_official.json"
OUT = R15 / "short_finetune_merged.json"


def summarize(official):
    return {
        "AEPE": official["AEPE"],
        "CMR@1": official["@1px"]["CMR"],
        "CMR@2": official["@2px"]["CMR"],
        "CMR@5": official["@5px"]["CMR"],
        "RMSE_official_var": official["RMSE_official_var"],
    }


def pair_delta(a, b):
    """a - b, negative means a improved."""
    if len(a) != len(b):
        raise ValueError("pair_EPE length mismatch")
    d = [x - y for x, y in zip(a, b)]
    return {
        "n": len(d),
        "mean_delta": statistics.fmean(d),
        "median_delta": statistics.median(d),
        "improved_pairs": sum(1 for x in d if x < 0),
        "improved_pct": 100.0 * sum(1 for x in d if x < 0) / len(d),
        "catastrophic_gt_10px_before": sum(1 for x in b if x > 10),
        "catastrophic_gt_10px_after": sum(1 for x in a if x > 10),
    }


def main():
    both = json.loads(BOTH_JSON.read_text())["configs"]["both"]
    both_m = summarize(both["official"])
    both_pairs = both["pair_EPE"]

    out = {
        "protocol": "GDROS official test.py, 424 pairs, iters=32, LSR NPOINT=200000",
        "experiment_class": "adaptation_short_finetune_scheme_A",
        "reference": {
            "gdros_released_AEPE": 1.4618458366096103,
            "both": both_m,
            "both_pair_EPE_source": str(BOTH_JSON.relative_to(ROOT)),
        },
        "configs": {},
    }
    print("\n### 适配性短训 vs 零训练旁路（剪枝后 M4，官方口径 424 对）")

    rows = [("both（原始两路，未训练）", both_m, None)]
    for cfg, paths in FILES.items():
        ft = json.loads(paths["ft"].read_text())
        zero = json.loads(paths["zero"].read_text())
        ev = ft.get("official_after")
        if not ev:
            raise RuntimeError("%s 缺少 official_after（评测未完成）" % cfg)
        after = summarize(ev["official"])
        zerom = summarize(zero["result"]["official"])
        delta = pair_delta(ev["pair_EPE"], zero["result"]["pair_EPE"])
        out["configs"][cfg] = {
            "zero_shot": zerom,
            "after_short_finetune": after,
            "AEPE_improvement": zerom["AEPE"] - after["AEPE"],
            "AEPE_gap_vs_both": after["AEPE"] - both_m["AEPE"],
            "recovery_fraction_of_zeroshot_gap": (
                (zerom["AEPE"] - after["AEPE"])
                / (zerom["AEPE"] - both_m["AEPE"])
                if zerom["AEPE"] != both_m["AEPE"] else None),
            "pair_delta_after_minus_zero": delta,
            "train": {k: ft["train"].get(k) for k in
                      ("steps", "train_iters", "lr", "seed",
                       "trainable_params", "elapsed_sec", "sec_per_step")},
            "final_loss": ft["train"]["curve"][-1]["loss"],
        }
        rows.append((cfg + " 零训练旁路", zerom, None))
        rows.append((cfg + " 短训后", after, after["AEPE"] - zerom["AEPE"]))
        rec = out["configs"][cfg]["recovery_fraction_of_zeroshot_gap"]
        print("\n%s: 零训练 AEPE %.4f → 短训后 %.4f（改善 %+.4f，"
              "仍比 both 差 %+.4f，恢复零训练落差的 %s）"
              % (cfg, zerom["AEPE"], after["AEPE"],
                 zerom["AEPE"] - after["AEPE"],
                 after["AEPE"] - both_m["AEPE"],
                 "%.1f%%" % (100.0 * rec) if rec is not None else "-"))
        d = delta
        print("  逐对：mean Δ%+.4f median Δ%+.4f，改善 %d/%d (%.1f%%)，"
              ">10px 灾难失败 %d → %d"
              % (d["mean_delta"], d["median_delta"], d["improved_pairs"],
                 d["n"], d["improved_pct"],
                 d["catastrophic_gt_10px_before"],
                 d["catastrophic_gt_10px_after"]))

    print("\n| 配置 | AEPE↓ | CMR@1↑ | CMR@2↑ | CMR@5↑ | 相对零训练 ΔAEPE |")
    print("|---|---|---|---|---|---|")
    for name, m, delta in rows:
        print("| %s | %.4f | %.2f | %.2f | %.2f | %s |"
              % (name, m["AEPE"], m["CMR@1"], m["CMR@2"], m["CMR@5"],
                 "-" if delta is None else "%+.4f" % delta))

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n[out] %s" % OUT)


if __name__ == "__main__":
    main()
