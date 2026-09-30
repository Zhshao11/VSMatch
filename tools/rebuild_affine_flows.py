#!/usr/bin/env python3
"""Rebuild Middlebury .flo files from an affine-flow manifest."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def write_flow(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = flow.shape[:2]
    with path.open("wb") as output:
        output.write(b"PIEH")
        output.write(struct.pack("<ii", width, height))
        output.write(flow.astype("<f4", copy=False).tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text(encoding="utf-8"))
    records = manifest["records"]
    for index, record in enumerate(records, start=1):
        height = int(record["height"])
        width = int(record["width"])
        coefficients = np.asarray(record["coefficients"], dtype=np.float64)
        grid_y, grid_x = np.mgrid[:height, :width]
        flow = np.empty((height, width, 2), dtype=np.float32)
        for channel in range(2):
            flow[..., channel] = (
                coefficients[channel, 0] * grid_x
                + coefficients[channel, 1] * grid_y
                + coefficients[channel, 2]
            ).astype(np.float32)
        write_flow(root / record["path"], flow)
        if index % 50 == 0 or index == len(records):
            print(f"rebuilt {index}/{len(records)} flows", flush=True)


if __name__ == "__main__":
    main()
