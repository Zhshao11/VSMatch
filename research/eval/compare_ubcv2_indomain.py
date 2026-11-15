#!/usr/bin/env python3
"""Assemble the UBCv2 in-domain vs cross-dataset table **in the paper's layout**.

The paper's UBCv2 row reports, for each threshold, a (CMR, AEPE) pair::

    2px   14.89(+-0.38)%   1.56(+-0.02)
    3px   37.08(+-0.04)%   2.13(+-0.00)
    5px   72.50(+-0.42)%   2.97(+-0.00)
    AEPE  4.49(+-0.00)     RMSE 11.97(+-0.31)

``eval_os_official_protocol.Collector.report()`` already produces exactly this
structure (``@<t>px -> {CMR, AEPE}``), so this script only has to lay it out and
add the rows we computed ourselves.

Two things must NOT be conflated:

1. The paper row is **in-domain** (GDROS trained on UBCv2) on the authors'
   screened 1447-pair split with an unpublished transform realisation. Absolute
   CMR at a fixed 2 px threshold is not transferable: rotating about the image
   centre instead of the origin alone moves our zero-flow baseline AEPE from
   30.15 to 16.30 (1.83x). So the paper row is a *target regime*, not a number
   to hit bit-for-bit.
2. What IS comparable is the column-wise ordering **within one generated test
   split**, because every row there shares the same 1625 pairs, the same LSR and
   the same ``iters=32``, and differs only in where the weights were trained.

Usage on zzvr::

  $PY compare_ubcv2_indomain.py \\
    --test-root .../generated/seed901 \\
    --row "GDROS UBCv2-trained|UBCv2 in-domain|baseline|<in_domain>.json" \\
    --row "GDROS OS-trained|OS cross-dataset|baseline|ubcv2_official_20261115.json" \\
    --row "M4@80K|OS cross-dataset|m4|ubcv2_official_20261115.json"

Repeating a label aggregates mean +- std across those runs (one per seed).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics

import numpy as np

# Paper reference, UBCv2, GDROS (in-domain), w/ LSR.  Order: (value, +-std).
PAPER = {
    "CMR@2": (14.89, 0.38),
    "AEPE@2": (1.56, 0.02),
    "CMR@3": (37.08, 0.04),
    "AEPE@3": (2.13, 0.00),
    "CMR@5": (72.50, 0.42),
    "AEPE@5": (2.97, 0.00),
    "AEPE": (4.49, 0.00),
    "RMSE": (11.97, 0.31),
}
COLS = ["CMR@2", "AEPE@2", "CMR@3", "AEPE@3", "CMR@5", "AEPE@5", "AEPE", "RMSE"]
# The paper's single "RMSE" column is most consistent with a standard deviation
# (11.97 / 4.49 = 2.67), while the official test.py prints np.var in that slot.
# The main table therefore shows std and says so in the header; the variance and
# the true pixel RMSE are both listed in the secondary table below.
DISPLAY = {"RMSE": "RMSE(std)"}


def zero_flow_baseline(test_root: str) -> float:
    """Official-protocol AEPE of predicting zero flow, from the analytic M."""
    files = sorted(glob.glob(os.path.join(os.path.abspath(test_root), "testing",
                                          "datum", "*.json")))
    if not files:
        return float("nan")
    per = []
    for f in files:
        m = json.load(open(f, encoding="utf-8"))
        M = np.array(m["affine_src_to_target"], dtype=np.float64)
        h, w = m.get("shape_hw", [512, 512])
        ys, xs = np.mgrid[0:h, 0:w]
        dx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs
        dy = M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys
        per.append(float((np.abs(dx).mean() + np.abs(dy).mean()) / 2.0))
    return float(np.mean(per))


def extract(path: str, key: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    if key not in d["results"]:
        raise SystemExit("%s has no results[%r] (has %s)" %
                         (path, key, list(d["results"])))
    off = d["results"][key]["official"]
    row = {
        "AEPE": off["AEPE"],
        "RMSE_var": off["RMSE_official_var"],
        "RMSE_std": off["RMSE_paper_formula_std"],
        "RMSE": off["RMSE_paper_formula_std"],
        "pixel_RMSE": off["pixel_RMSE_true"],
        "pairs": off["pairs"],
        "src": "%s[%s]" % (os.path.basename(path), key),
    }
    for t in (1, 2, 3, 4, 5):
        q = off["@%dpx" % t]
        row["CMR@%d" % t] = q["CMR"]
        row["AEPE@%d" % t] = q["AEPE"] if q["AEPE"] is not None else float("nan")
    return row


def fmt(mean, std, digits=2) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "-"
    if std is None:
        return ("%%.%df" % digits) % mean
    return ("%%.%df(±%%.%df)" % (digits, digits)) % (mean, std)


def agg(rows, c):
    vals = [r[c] for r in rows
            if not (isinstance(r[c], float) and math.isnan(r[c]))]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", required=True,
                    help="one seed root; used only for the zero-flow baseline")
    ap.add_argument("--row", action="append", required=True,
                    metavar="LABEL|DOMAIN|KEY|PATH")
    ap.add_argument("--baseline-from", default="",
                    help="extra seed roots (comma separated) so the zero-flow "
                         "baseline is averaged over the same seeds as the rows")
    args = ap.parse_args()

    groups: dict[str, dict] = {}
    for spec in args.row:
        parts = spec.split("|")
        if len(parts) != 4:
            raise SystemExit("bad --row %r, want LABEL|DOMAIN|KEY|PATH" % spec)
        label, domain, key, path = (p.strip() for p in parts)
        g = groups.setdefault(label, {"domain": domain, "rows": []})
        if g["domain"] != domain:
            raise SystemExit("label %r used with two domains" % label)
        g["rows"].append(extract(path, key))

    roots = [args.test_root] + [r for r in args.baseline_from.split(",") if r.strip()]
    bases = [b for b in (zero_flow_baseline(r) for r in roots) if not math.isnan(b)]
    base = float(np.mean(bases)) if bases else float("nan")

    print("")
    print("### UBCv2（论文列序）：同一测试集，只变「权重训练在哪」")
    print("")
    print("| 对象 | 训练域 | " + " | ".join(DISPLAY.get(c, c) for c in COLS) + " |")
    print("|---" * (len(COLS) + 2) + "|")

    for label, g in groups.items():
        rows = g["rows"]
        cells = [label, g["domain"]]
        for c in COLS:
            mean, std = agg(rows, c)
            cells.append(fmt(mean, std, 4 if c in ("AEPE", "RMSE") else 2))
        print("| " + " | ".join(cells) + " |")

    paper_cells = ["论文 GDROS（UBCv2 in-domain）", "UBCv2"]
    for c in COLS:
        m, s = PAPER[c]
        paper_cells.append(fmt(m, s, 2))
    print("| " + " | ".join(paper_cells) + " |")

    print("")
    print("- 零流基线 AEPE（由 `datum/*.json` 解析算出，%d 个 seed 平均）= **%.4f**"
          % (len(bases), base))
    print("- 每个对象的 seed 数：%s；多 seed 时给 mean(±std)" %
          ", ".join("%s:%d" % (k, len(v["rows"])) for k, v in groups.items()))
    print("")
    print("**RMSE 口径待对齐**：论文写 `11.97(±0.31)`。官方 `test.py` 打印在 RMSE "
          "栏位的是 `np.var(MAE_list)`（方差），但 `11.97 / 4.49 = 2.67` 的量级更像"
          "标准差。下表把三种候选都列出，**不要直接拿方差去对 11.97**：")
    print("")
    print("| 对象 | var（官方栏位） | std = sqrt(var) | 逐像素 RMSE | std / AEPE |")
    print("|---|---|---|---|---|")
    for label, g in groups.items():
        v, _ = agg(g["rows"], "RMSE_var")
        s, _ = agg(g["rows"], "RMSE_std")
        p, _ = agg(g["rows"], "pixel_RMSE")
        a, _ = agg(g["rows"], "AEPE")
        print("| %s | %.4f | %.4f | %.4f | %.2f |" %
              (label, v, s, p, (s / a) if a else float("nan")))
    print("| 论文（若 11.97 就是 std） | 143.2800 | 11.9700 | - | 2.67 |")
    print("")
    print("- 表内各行的比较才有效：共用同一个 `--test-root`、同一个 LSR / `iters=32`，"
          "只差权重训练在哪。")
    print("- 论文行是作者筛后 1447 对 × 3 seed、变换实现未公开 ⇒ 固定 2px 阈值的绝对 "
          "CMR 不可逐位对齐，只能看是否同档。")


if __name__ == "__main__":
    main()
