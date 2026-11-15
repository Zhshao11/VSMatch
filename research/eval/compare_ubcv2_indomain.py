#!/usr/bin/env python3
"""Assemble the UBCv2 in-domain vs cross-dataset comparison table.

Motivation
----------
The paper's Table III / V UBCv2 row is **in-domain** (GDROS trained on UBCv2) on
the authors' screened 1447-pair test split. Our published cross-dataset numbers
(`GDROS 22.0091`, `M4 3.8941`) are on a locally generated 1625-pair split. Those
two are not comparable (see ``docs/UBCV2_DIAGNOSIS.md``).

This script takes the *same* script's output for several checkpoints on the
*same* generated test split, so the rows differ only in what the weights were
trained on — which is exactly the question "does my method's generalisation beat
an in-domain-trained GDROS?".

Because the metric scale depends on the transform realisation (rotation about the
origin vs about the image centre is a 1.83x difference in the zero-flow
baseline), every row is also reported as a **gain over the zero-flow baseline**
computed from ``datum/*.json``. That is the only quantity that has a chance of
being stable across implementations.

Usage on zzvr::

  PY=/home/zzvr/miniconda3/envs/compared/bin/python
  $PY compare_ubcv2_indomain.py \\
    --test-root /mnt/sda/zhushihao/VS_Match/datasets/ubcv2/generated/seed901 \\
    --row "GDROS UBCv2-trained 120K|UBCv2 in-domain|baseline|.codex_tmp/in_domain.json" \\
    --row "GDROS OS-trained|OS cross-dataset|baseline|.codex_tmp/ubcv2_official_20261115.json" \\
    --row "M4@80K|OS cross-dataset|m4|.codex_tmp/ubcv2_official_20261115.json"

Repeating the same label aggregates mean +- std across those runs (one per seed).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics

import numpy as np

# Paper reference (UBCv2, GDROS, w/ LSR).  Hardcoded because it is the target we
# are trying to reproduce; it is NOT computed by this repo.
PAPER = {
    "AEPE": (4.49, 0.00),
    "RMSE_var": (11.97, 0.31),
    "CMR@1": (0.73, 0.18),
    "CMR@2": (14.89, 0.38),
    "CMR@3": (37.08, None),   # Table III
    "CMR@5": (72.50, None),   # Table III
}
TAUS = (1, 2, 3, 4, 5)


def zero_flow_baseline(test_root: str) -> float:
    """Official-protocol AEPE of predicting zero flow, from the analytic M."""
    files = sorted(glob.glob(os.path.join(test_root, "testing", "datum", "*.json")))
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
    row = {"AEPE": off["AEPE"], "RMSE_var": off["RMSE_official_var"],
           "pairs": off["pairs"]}
    for t in TAUS:
        row["CMR@%d" % t] = off["@%dpx" % t]["CMR"]
    row["src"] = "%s[%s]" % (os.path.basename(path), key)
    return row


def fmt(mean: float, std, digits: int = 2) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "-"
    if std is None:
        return ("%%.%df" % digits) % mean
    return ("%%.%df(±%%.%df)" % (digits, digits)) % (mean, std)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", required=True,
                    help="one seed root; used only for the zero-flow baseline")
    ap.add_argument("--row", action="append", required=True,
                    metavar="LABEL|DOMAIN|KEY|PATH")
    ap.add_argument("--baseline-from", default="",
                    help="extra seed roots (comma separated) to average the "
                         "zero-flow baseline over")
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
    bases = [zero_flow_baseline(r) for r in roots]
    base = float(np.mean([b for b in bases if not math.isnan(b)])) if bases else float("nan")

    cols = ["AEPE", "RMSE_var", "CMR@1", "CMR@2", "CMR@3", "CMR@5"]
    print("")
    print("### UBCv2：同一套测试集，只变「权重训练在哪」（官方 test.py 口径）")
    print("")
    print("| 对象 | 训练域 | " + " | ".join(cols) + " | AEPE 相对零流增益 |")
    print("|---" * (len(cols) + 3) + "|")

    for label, g in groups.items():
        rows = g["rows"]
        cells = [label, g["domain"]]
        for c in cols:
            vals = [r[c] for r in rows]
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else None
            cells.append(fmt(mean, std, 4 if c in ("AEPE", "RMSE_var") else 2))
        mean_aepe = statistics.fmean([r["AEPE"] for r in rows])
        cells.append("%.1f%%" % (100.0 * (1.0 - mean_aepe / base)) if base else "-")
        print("| " + " | ".join(cells) + " |")

    paper_cells = ["论文 GDROS (Table III/V)", "UBCv2 (in-domain)"]
    for c in cols:
        m, s = PAPER.get(c, (None, None))
        paper_cells.append(fmt(m, s, 2))
    # The paper's zero-flow baseline is unknown (its transform realisation is not
    # published), so a gain column for that row would be meaningless.
    paper_cells.append("—（基线未知）")
    print("| " + " | ".join(paper_cells) + " |")
    print("")
    print("- 零流基线 AEPE（本测试集，由 `datum/*.json` 解析算出）= **%.4f**" % base)
    print("- 每个对象跑了 %s 个 seed；多 seed 时给的是 mean(±std)" %
          ", ".join("%s:%d" % (k, len(v["rows"])) for k, v in groups.items()))
    print("- 论文行是作者筛后 1447 对 × 3 seed，**与本表的 1625 对不是同一集合**，"
          "且其变换实现（旋转基准点）未公开 ⇒ 论文行只作量级参照。")
    print("- 表内各行的比较才是有效的：它们共用同一个 `--test-root`，"
          "只差权重训练在哪。")


if __name__ == "__main__":
    main()
