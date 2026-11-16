#!/usr/bin/env python3
"""Build a GDROS-layout UBCv2 test set from the official multi-modal test images.

Input is the extracted UBCv2 multi-modal ``test_images.zip``::

    <raw-root>/rgb/<stem>.tif
    <raw-root>/sar/<stem>.tif        (finely aligned RGB / SAR pair)

Output (one root per seed, ``testing/`` is required by opt_sar_test)::

    <output-root>/seed<seed>/testing/opt/<id>.png
    <output-root>/seed<seed>/testing/sar_warped/<id>.png
    <output-root>/seed<seed>/testing/truth_flow/<id>.flo
    <output-root>/seed<seed>/testing/datum/<id>.json
    <output-root>/seed<seed>/protocol_metadata/manifest.json

Affine convention (verified against the official OS test set on 2026-11-15):
``datum/000000.npy[2]`` stores ``M = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]``
and the official ``truth_flow`` is exactly ``M @ [x,y,1] - [x,y]``
(least-squares residual ~1e-6 on three checked pairs). We use the same
origin-centred rotation + translation form, and warp SAR with the default
``cv2.warpAffine`` (inverse mapping), so ``warped_sar(M*p) == sar(p)``.

UBCv2 paper parameters: translation [-15,15] px (1 px step),
scale [0.9,1.1] (0.05 step), rotation [-10,10] deg (1 deg step),
512x512 (no 400 crop, because official test.py evaluates at 512).

The official UBCv2 SAR TIFFs are float32 amplitude (roughly 0.05..13, sparse
outliers to 130), whereas GDROS/M4 were trained on 8-bit SAR. A fixed global
log-amplitude LUT (p2/p98 dB over 300 sampled tiles) converts every SAR tile
to uint8 identically; without it cv2.imwrite silently truncates to 0..13.

Run on zzvr::

  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  /home/zzvr/miniconda3/envs/compared/bin/python \
    .codex_tmp/build_ubcv2_testset.py \
    --raw-root /mnt/sda/zhushihao/VS_Match/datasets/ubcv2/raw/raw_extracted \
    --output-root /mnt/sda/zhushihao/VS_Match/datasets/ubcv2/generated \
    --seeds 901 --workers 8
"""
from __future__ import annotations
import argparse
import json
import math
import os
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

FLO_MAGIC = 202021.25


@dataclass(frozen=True)
class Task:
    pair_id: str
    rgb_path: str
    sar_path: str
    source_stem: str
    out_root: str
    tx: float
    ty: float
    scale: float
    angle_deg: float
    sar_lo_db: float
    sar_hi_db: float


def sar_to_u8(x: np.ndarray, lo_db: float, hi_db: float) -> np.ndarray:
    """Official UBCv2 SAR TIFFs are float32 backscatter amplitudes.

    Convert once with a FIXED global log-amplitude LUT (percentiles computed
    from the test set, stored in the manifest) so every pair shares the same
    radiometry. GDROS/M4 were trained on 8-bit SAR; writing the float array
    straight to PNG silently truncates it to 0..13 (almost black).
    """
    db = 20.0 * np.log10(np.maximum(x.astype(np.float32), 0.0) + 1e-3)
    u = np.clip((db - lo_db) / (hi_db - lo_db) * 255.0, 0.0, 255.0)
    return u.astype(np.uint8)


def compute_sar_lut(sar_paths, n_samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(sar_paths), min(n_samples, len(sar_paths)), replace=False)
    vals = []
    for j in picks:
        x = cv2.imread(sar_paths[j], cv2.IMREAD_UNCHANGED)
        db = 20.0 * np.log10(np.maximum(x.astype(np.float32), 0.0) + 1e-3)
        vals.append(db.ravel()[::31])
    lo, hi = np.percentile(np.concatenate(vals), [2.0, 98.0])
    return float(lo), float(hi)


def discrete(low: float, high: float, step: float) -> np.ndarray:
    count = int(round((high - low) / step)) + 1
    vals = low + np.arange(count, dtype=np.float64) * step
    if not math.isclose(float(vals[-1]), high, abs_tol=1e-8):
        raise ValueError("range not divisible by step: %s %s %s" % (low, high, step))
    return vals


def discover(raw_root: Path):
    rgb = {p.stem: p for p in (raw_root / "rgb").glob("*.tif")}
    sar = {p.stem: p for p in (raw_root / "sar").glob("*.tif")}
    common = sorted(set(rgb) & set(sar))
    if not common:
        raise RuntimeError("no paired rgb/sar tif under %s" % raw_root)
    missing = len(set(rgb) ^ set(sar))
    print("paired=%d unpaired=%d" % (len(common), missing), flush=True)
    return [(k, str(rgb[k]), str(sar[k])) for k in common]


def sample(count: int, seed: int):
    rng = np.random.default_rng(seed)
    t = rng.integers(-15, 16, size=(count, 2))
    s = rng.choice(discrete(0.9, 1.1, 0.05), size=count)
    a = rng.choice(discrete(-10.0, 10.0, 1.0), size=count)
    return [(float(t[i, 0]), float(t[i, 1]), float(s[i]), float(a[i]))
            for i in range(count)]


def affine_matrix(angle_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    th = math.radians(angle_deg)
    c, s = scale * math.cos(th), scale * math.sin(th)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


def write_flo(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".flo.tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("f", FLO_MAGIC))
        f.write(struct.pack("i", flow.shape[1]))
        f.write(struct.pack("i", flow.shape[0]))
        flow.astype(np.float32).tofile(f)
    tmp.replace(path)


def process(task: Task) -> dict:
    opt = cv2.imread(task.rgb_path, cv2.IMREAD_UNCHANGED)
    sar = cv2.imread(task.sar_path, cv2.IMREAD_UNCHANGED)
    if opt is None or sar is None:
        raise RuntimeError("read failed: %s / %s" % (task.rgb_path, task.sar_path))
    if opt.shape[:2] != sar.shape[:2]:
        raise RuntimeError("shape mismatch %s: %s vs %s" %
                           (task.pair_id, opt.shape, sar.shape))
    if opt.ndim == 2:
        opt = cv2.cvtColor(opt, cv2.COLOR_GRAY2BGR)
    opt = np.ascontiguousarray(opt[..., :3])
    if sar.dtype != np.uint8:
        sar_u8 = sar_to_u8(sar, task.sar_lo_db, task.sar_hi_db)
    else:
        sar_u8 = sar[..., 0] if sar.ndim == 3 else sar
    if sar_u8.ndim == 3:
        sar_u8 = sar_u8[..., 0]
    sar = np.ascontiguousarray(np.repeat(sar_u8[..., None], 3, axis=2))
    h, w = opt.shape[:2]

    M = affine_matrix(task.angle_deg, task.scale, task.tx, task.ty)
    warped = cv2.warpAffine(sar, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.mgrid[0:h, 0:w]
    flow = np.stack((M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs,
                     M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys),
                    axis=-1).astype(np.float32)

    split = Path(task.out_root) / "testing"
    for sub in ("opt", "sar_warped", "truth_flow", "datum"):
        (split / sub).mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(split / "opt" / (task.pair_id + ".png")), opt):
        raise RuntimeError("png write failed for %s" % task.pair_id)
    if not cv2.imwrite(str(split / "sar_warped" / (task.pair_id + ".png")), warped):
        raise RuntimeError("png write failed for %s" % task.pair_id)
    write_flo(split / "truth_flow" / (task.pair_id + ".flo"), flow)
    rec = {
        "pair_id": task.pair_id,
        "source_stem": task.source_stem,
        "translation_px": [task.tx, task.ty],
        "scale": task.scale,
        "rotation_deg": task.angle_deg,
        "affine_src_to_target": M.tolist(),
        "flow_convention": "M@[x,y,1] - [x,y] (matches GDROS OS official datum[2])",
        "sar_conversion": {
            "type": "fixed_global_log_amplitude_p2_p98_to_uint8",
            "lo_db": task.sar_lo_db,
            "hi_db": task.sar_hi_db,
        },
        "shape_hw": [h, w],
    }
    (split / "datum" / (task.pair_id + ".json")).write_text(
        json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def build_one(raw_root: Path, output_root: Path, seed: int, limit, workers: int,
              fixed_lut_manifest: Path | None):
    all_pairs = discover(raw_root)
    if fixed_lut_manifest is not None:
        reference = json.loads(fixed_lut_manifest.read_text(encoding="utf-8"))
        lo_db = float(reference["sar_conversion"]["lo_db"])
        hi_db = float(reference["sar_conversion"]["hi_db"])
        lut_source = str(fixed_lut_manifest.resolve())
    else:
        # Seed901 is the preregistered radiometric reference. Additional
        # geometry seeds must pass --fixed-lut-manifest seed901/manifest.json.
        lo_db, hi_db = compute_sar_lut([p[2] for p in all_pairs], 300, seed)
        lut_source = "300 raw SAR tiles sampled with seed %d" % seed
    print("sar_lut: db p2=%.4f p98=%.4f (300 sampled tiles)" %
          (lo_db, hi_db), flush=True)
    pairs = all_pairs[:limit] if limit else all_pairs
    params = sample(len(pairs), seed)
    out_root = output_root / ("seed%d" % seed)
    width = max(6, len(str(len(pairs) - 1)))
    tasks = [Task(str(i).zfill(width), rgb, sar, stem, str(out_root),
                  p[0], p[1], p[2], p[3], lo_db, hi_db)
             for i, ((stem, rgb, sar), p) in enumerate(zip(pairs, params))]
    print("seed=%d pairs=%d out=%s workers=%d" %
          (seed, len(tasks), out_root, workers), flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(process, tasks, chunksize=16))
    else:
        records = [process(t) for t in tasks]
    meta = out_root / "protocol_metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "manifest.json").write_text(json.dumps({
        "dataset": "UBCv2 multi-modal test images (official)",
        "raw_root": str(raw_root),
        "seed": seed,
        "pair_count": len(records),
        "sar_conversion": {
            "source": "official UBCv2 SAR TIFF is float32 backscatter amplitude",
            "type": "fixed_global_log_amplitude_p2_p98_to_uint8",
            "lo_db": lo_db,
            "hi_db": hi_db,
            "fixed_across_geometry_seeds": fixed_lut_manifest is not None,
            "lut_source": lut_source,
            "note": "20*log10(x+1e-3), clip((db-lo)/(hi-lo)*255,0,255), "
                    "then replicated to 3 channels; identical LUT for every pair",
        },
        "prescreen": "none (all official test pairs kept; paper pre-screened to 1447)",
        "transform": {
            "translation_px": [-15, 15, 1],
            "scale": [0.9, 1.1, 0.05],
            "rotation_deg": [-10, 10, 1],
            "crop": None,
            "border": "constant 0",
        },
        "flow_convention": "M@[x,y,1] - [x,y]",
        "generator": str(Path(__file__).resolve()),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (meta / "params.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8")
    print("DONE seed=%d pairs=%d" % (seed, len(records)), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--seeds", default="901", help="comma separated seeds")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fixed-lut-manifest", type=Path, default=None,
                    help="Reuse lo_db/hi_db from seed901 manifest so only geometry changes")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        build_one(args.raw_root, args.output_root, seed, args.limit, args.workers,
                  args.fixed_lut_manifest)


if __name__ == "__main__":
    main()
