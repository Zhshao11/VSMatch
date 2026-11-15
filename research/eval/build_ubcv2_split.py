#!/usr/bin/env python3
"""Build a GDROS-layout UBCv2 ``training`` / ``validation`` split.

This is a **separate file on purpose**.  ``build_ubcv2_testset.py`` generated the
published 1625-pair test set and its sha256 is recorded in
``research/results/UBCV2_20261115/manifest.json`` and in the E-092C audit card.
Editing that file would silently break the provenance chain, so the training /
validation generator lives here instead and reuses the *same* geometry and
radiometry conventions.

Differences from the test generator
-----------------------------------
1. ``--split`` selects the output directory name (``training`` / ``validation``).
2. ``--sar-lut`` lets the caller **reuse the test-set LUT** so that train, val
   and test share one identical radiometric mapping.  Without it the LUT would
   be recomputed per split and the model would see a slightly different SAR
   histogram in training than at test time.
3. Nothing else changes: same origin-centred affine form, same paper ranges
   (translation +-15 px, scale 0.9..1.1 step 0.05, rotation +-10 deg step 1),
   same ``M @ [x,y,1] - [x,y]`` flow convention, same ``cv2.warpAffine``
   inverse mapping.

Why a baked-in warp at all, when ``opt_sar_train`` already augments?
-------------------------------------------------------------------
``core.datasets.opt_sar_train`` is constructed with
``{'crop_size': image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}``
so it *adds* a zoom + crop(400) + flip on top of whatever pair it is given.  We
still bake the paper affine into ``sar_warped`` so that the residual 512x512
pair sits in the same transform family the model is evaluated on; an identity
base pair would push the whole distribution into the augmentor instead.

Usage on zzvr::

  PY=/home/zzvr/miniconda3/envs/compared/bin/python
  RAW=/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/raw/train_extracted
  OUT=/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/generated
  # identical LUT as the published test set (manifest.json of seed901)
  $PY build_ubcv2_split.py --raw-root $RAW/train --output-root $OUT/train_seed701 \
    --split training --sar-lut=-28.3079,10.5459 --seed 701 --workers 8
  $PY build_ubcv2_split.py --raw-root $RAW/val   --output-root $OUT/val_seed702 \
    --split validation --sar-lut=-28.3079,10.5459 --seed 702 --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FLO_MAGIC = 202021.25


@dataclass(frozen=True)
class Task:
    pair_id: str
    rgb_path: str
    sar_path: str
    source_stem: str
    out_root: str
    split: str
    tx: float
    ty: float
    scale: float
    angle_deg: float
    sar_lo_db: float
    sar_hi_db: float


def sar_to_u8(x: np.ndarray, lo_db: float, hi_db: float) -> np.ndarray:
    """Official UBCv2 SAR TIFFs are float32 backscatter amplitudes.

    A fixed global log-amplitude LUT is mandatory: ``cv2.imwrite`` silently
    falls back to ``CV_8U`` on float input and truncates the 0.05..13 amplitude
    range to 0..13, producing an almost black tile.
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
    print("paired=%d unpaired=%d" % (len(common), len(set(rgb) ^ set(sar))), flush=True)
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

    split = Path(task.out_root) / task.split
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
        "split": task.split,
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


def build_one(raw_root: Path, output_root: Path, split: str, seed: int,
              count, workers: int, lut, lut_source: str):
    all_pairs = discover(raw_root)
    if not lut:
        lo_db, hi_db = compute_sar_lut([p[2] for p in all_pairs], 300, seed)
        lut_source = "computed_from_this_split"
    else:
        lo_db, hi_db = lut
    print("sar_lut: db p2=%.4f p98=%.4f source=%s" % (lo_db, hi_db, lut_source),
          flush=True)
    pairs = all_pairs[:count] if count else all_pairs
    params = sample(len(pairs), seed)
    width = max(6, len(str(len(pairs) - 1)))
    tasks = [Task(str(i).zfill(width), rgb, sar, stem, str(output_root), split,
                  p[0], p[1], p[2], p[3], lo_db, hi_db)
             for i, ((stem, rgb, sar), p) in enumerate(zip(pairs, params))]
    print("split=%s seed=%d pairs=%d out=%s workers=%d" %
          (split, seed, len(tasks), output_root, workers), flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(process, tasks, chunksize=32))
    else:
        records = [process(t) for t in tasks]
    meta = Path(output_root) / "protocol_metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / ("manifest_%s.json" % split)).write_text(json.dumps({
        "dataset": "UBCv2 multi-modal %s images (official)" % split,
        "raw_root": str(raw_root),
        "split": split,
        "seed": seed,
        "pair_count": len(records),
        "sar_conversion": {
            "source": "official UBCv2 SAR TIFF is float32 backscatter amplitude",
            "type": "fixed_global_log_amplitude_p2_p98_to_uint8",
            "lo_db": lo_db,
            "hi_db": hi_db,
            "lut_source": lut_source,
            "note": "20*log10(x+1e-3), clip((db-lo)/(hi-lo)*255,0,255), "
                    "then replicated to 3 channels; identical LUT for every pair",
        },
        "transform": {
            "translation_px": [-15, 15, 1],
            "scale": [0.9, 1.1, 0.05],
            "rotation_deg": [-10, 10, 1],
            "crop": None,
            "border": "constant 0",
        },
        "flow_convention": "M@[x,y,1] - [x,y]",
        "train_time_augmentation": {
            "applied_by": "core.datasets.opt_sar_train",
            "crop_size": "cfg.image_size",
            "min_scale": -0.2,
            "max_scale": 0.6,
            "do_flip": True,
            "note": "base pair already carries the paper affine; the loader adds "
                    "zoom+crop+flip on top",
        },
        "generator": str(Path(__file__).resolve()),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (meta / ("params_%s.jsonl" % split)).write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8")
    print("DONE split=%s pairs=%d" % (split, len(records)), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--split", default="training",
                    choices=["training", "validation", "testing"])
    ap.add_argument("--sar-lut", default=None,
                    help="LO,HI in dB. Omit to compute from this split.")
    ap.add_argument("--seed", type=int, default=701)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    lut, source = None, ""
    if args.sar_lut:
        lo, hi = [float(v) for v in args.sar_lut.split(",")]
        lut, source = (lo, hi), "reused_from_reference_split"
    build_one(args.raw_root, args.output_root, args.split, args.seed,
              args.count, args.workers, lut, source)


if __name__ == "__main__":
    main()
