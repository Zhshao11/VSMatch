#!/usr/bin/env python3
"""Coarse small-sample CUDA-stream span audit of GDROS/clean M4.

CUDA events measure elapsed stream spans, including scheduling/CPU gaps;
these are not sums of active kernel time. Plain synchronized wall times are
recorded separately. Instrumentation never changes tensor arithmetic.
"""
import argparse
from collections import defaultdict
import functools
import gc
import importlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import e092a_official_eval as base
from m4_clean_deploy import CleanM4
from quick_timing_10pairs_20261115 import snapshot


class Spans:
    def __init__(self):
        self.enabled = False
        self.pool = defaultdict(list)
        self.reset()

    def reset(self):
        self.used = defaultdict(int)
        self.records = []

    def begin(self, category):
        if not self.enabled:
            return None
        index = self.used[category]
        self.used[category] += 1
        if index == len(self.pool[category]):
            self.pool[category].append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
        begin, end = self.pool[category][index]
        begin.record()
        return category, begin, end

    def end(self, token):
        if token is not None:
            token[2].record()
            self.records.append(token)

    def totals(self):
        result = defaultdict(float)
        for category, begin, end in self.records:
            result[category] += begin.elapsed_time(end)
        return dict(result)

    def wrap(self, function, category):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            token = self.begin(category)
            try:
                return function(*args, **kwargs)
            finally:
                self.end(token)
        return wrapped

    def hook(self, module, category):
        stack = []
        def before(_module, _args):
            stack.append(self.begin(category))
        def after(_module, _args, _output):
            self.end(stack.pop())
        return [module.register_forward_pre_hook(before), module.register_forward_hook(after)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError("Use an empty output directory")
    os.chdir(base.TMP.parent)
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    sources = [Path(__file__).resolve(), Path(importlib.import_module("m4_clean_deploy").__file__).resolve(), Path(base.__file__).resolve(),
               base.OFFICIAL / "core/corr.py", base.OFFICIAL / "core/LSRnet/LSmodel.py"]
    source_hashes = {str(p): base.sha256(p) for p in sources}
    (out / "source_snapshot").mkdir()
    for path in sources:
        (out / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    for path in (base.M4_CKPT, base.GDROS_CKPT):
        if base.sha256(path) != base.EXPECTED[path]:
            raise RuntimeError("Checkpoint hash mismatch")
    print(json.dumps({"stage": "LOADING"}), flush=True)
    legacy, _ = base.load_m4(torch.device("cpu"))
    m4 = CleanM4(legacy).eval()
    del legacy
    gc.collect()
    gdros, _ = base.load_gdros(torch.device("cpu"))
    m4, gdros = m4.to(device), gdros.to(device)
    lsr_module = importlib.import_module("core.LSRnet.LSmodel")
    corr = importlib.import_module("core.corr")
    original_lsr = lsr_module.LSR
    spans = Spans()
    calls = 0

    def lsr(flow):
        nonlocal calls
        calls += 1
        token = spans.begin("lsr")
        try:
            return original_lsr(1, 512, 512, flow, 200000)
        finally:
            spans.end(token)

    def run(variant, source, target):
        nonlocal calls
        calls = 0
        spans.reset()
        torch.cuda.synchronize(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if spans.enabled:
            begin.record()
        start = time.perf_counter()
        if variant == "m4_clean":
            raw = m4(source, target, iters=32, lsr_fn=lsr)
        else:
            _, raw = gdros(source, target, iters=32, test_mode=True)
        post = lsr(raw)
        if spans.enabled:
            end.record()
        torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter()-start)*1000
        assert calls == (5 if variant == "m4_clean" else 1)
        result = {"wall_ms": wall_ms, "lsr_calls": calls}
        if spans.enabled:
            buckets = spans.totals()
            total = begin.elapsed_time(end)
            matcher_total = buckets.pop("matcher_total", 0.0)
            nested = sum(buckets.get(key,0.0) for key in
                         ("feature_frontends", "dino", "descriptor_adapters_film", "context", "correlation_build"))
            if matcher_total < nested-0.1:
                raise RuntimeError("Nested spans exceed matcher total")
            buckets["iterative_matching_and_related"] = max(0.0, matcher_total-nested)
            residual = total-sum(buckets.values())
            if residual < -0.1:
                raise RuntimeError(f"Overlapping spans: {total}, {buckets}")
            buckets["other_coordinates_fusion_launch_gaps"] = max(0.0, residual)
            result.update(stream_total_ms=total, categories_ms=buckets, category_calls=dict(spans.used))
        return post.detach().cpu().numpy().copy(), result

    indices = np.linspace(0, 423, 10, dtype=int).tolist()
    rows = base.pairs(include_gt=False)
    inputs = [(base.image_tensor(rows[i]["source"]).to(device), base.image_tensor(rows[i]["target"]).to(device)) for i in indices]
    prior = base.ROOT / "GDROS/experiments/sparsegeo/runs/E-096A-GDROS-CLEAN-M4-OS424-20261116"
    baseline_outputs = {}
    records, parity = [], []
    smi = [{"stage": "before_warmup", "snapshot": snapshot()}]
    with torch.inference_mode():
        for i in range(3):
            for variant in ("m4_clean", "gdros"):
                run(variant, *inputs[i])
        for repeat in range(3):
            for j, (index, pair) in enumerate(zip(indices, inputs)):
                order = ("m4_clean", "gdros") if (repeat+j)%2 == 0 else ("gdros", "m4_clean")
                for variant in order:
                    array, result = run(variant, *pair)
                    old = np.load(prior / "predictions" / variant / f"{index:06d}.npy", allow_pickle=False)
                    maximum = float(np.abs(array-old).max())
                    assert maximum <= 1e-4
                    parity.append({"phase": "plain", "variant": variant, "pair_index": index, "repeat": repeat, "max_abs": maximum})
                    baseline_outputs[variant, index] = array
                    records.append({"phase": "plain", "variant": variant, "pair_index": index, "repeat": repeat, **result})
        print(json.dumps({"stage": "PLAIN_COMPLETE"}), flush=True)
        handles = []
        for adapter in m4.descriptor_adapters:
            handles += spans.hook(adapter, "descriptor_adapters_film")
        for matcher in (m4.f2, m4.f3, m4.student):
            handles += spans.hook(matcher, "matcher_total")
            for module, category in [(matcher.v1, "feature_frontends"), (matcher.v2, "feature_frontends"),
                                     (matcher.film_projection, "descriptor_adapters_film"), (matcher.context, "context")]:
                handles += spans.hook(module, category)
        for gate in (m4.b_gate, m4.m_gate):
            handles += spans.hook(gate, "fusion_gates")
        for module, category in [(gdros, "matcher_total"), (gdros.featureNet, "feature_frontends"),
                                 (gdros.context_encoder, "context")]:
            handles += spans.hook(module, category)
        patches = [(corr.CorrBlock, "__init__", corr.CorrBlock.__init__)]
        patches += [(dino, "forward_features", dino.forward_features) for dino in m4.dinos]
        for owner, name, function in patches:
            category = {"__init__": "correlation_build", "forward_features": "dino"}[name]
            setattr(owner, name, spans.wrap(function, category))
        spans.enabled = True
        # Warm event pools so measurement reuses CUDA events without allocation.
        for variant in ("m4_clean", "gdros"):
            run(variant, *inputs[0])
        for repeat in range(3):
            for j, (index, pair) in enumerate(zip(indices, inputs)):
                order = ("m4_clean", "gdros") if (repeat+j)%2 == 0 else ("gdros", "m4_clean")
                for variant in order:
                    array, result = run(variant, *pair)
                    maximum = float(np.abs(array-baseline_outputs[variant, index]).max())
                    assert maximum == 0
                    parity.append({"phase": "instrumented", "variant": variant, "pair_index": index, "repeat": repeat, "max_abs": maximum})
                    records.append({"phase": "instrumented", "variant": variant, "pair_index": index, "repeat": repeat, **result})
            print(json.dumps({"stage": "INSTRUMENTED_REPEAT_COMPLETE", "repeat": repeat}), flush=True)
        spans.enabled = False
        for handle in handles:
            handle.remove()
        for owner, name, function in patches:
            setattr(owner, name, function)
    summaries = {}
    for variant in ("m4_clean", "gdros"):
        plain = [r for r in records if r["variant"] == variant and r["phase"] == "plain"]
        measured = [r for r in records if r["variant"] == variant and r["phase"] == "instrumented"]
        keys = sorted({key for r in measured for key in r["categories_ms"]})
        summaries[variant] = {"plain_mean_ms": statistics.fmean(r["wall_ms"] for r in plain),
                              "instrumented_wall_mean_ms": statistics.fmean(r["wall_ms"] for r in measured),
                              "instrumented_stream_mean_ms": statistics.fmean(r["stream_total_ms"] for r in measured),
                              "categories_mean_ms": {key: statistics.fmean(r["categories_ms"].get(key,0) for r in measured) for key in keys},
                              "category_calls_per_pair": measured[0]["category_calls"]}
    sources_after = {str(p): base.sha256(p) for p in sources}
    assert source_hashes == sources_after
    smi.append({"stage": "after_measurement", "snapshot": snapshot()})
    card = {"status": "COMPLETE", "indices": indices, "pairs": 10, "repeats": 3, "iterations": 32,
            "summary": summaries, "parity_max_abs": max(r["max_abs"] for r in parity),
            "instrumentation_parity_max_abs": max(r["max_abs"] for r in parity if r["phase"] == "instrumented"),
            "environment": base.environment(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "shared_gpu": True, "source_hashes_start": source_hashes, "source_hashes_end": sources_after,
            "instrumentation_restored": True, "snapshots": smi,
            "method": "Coarse CUDA-event stream spans; includes scheduling/CPU gaps, not active-kernel-time sums. Iterative matching and related is inclusive matcher span minus nested frontend/DINO/adapter/context/correlation-build spans, including recurrent lookup, updates, mask/upsampling and other matcher arithmetic. Flat categories are nonoverlapping. Percentages refer to instrumented stream totals; plain wall timing is separate."}
    base.write_json(out / "RESULT_CARD.json", card)
    base.write_json(out / "ROWS.json", records)
    base.write_json(out / "PARITY.json", parity)
    print(json.dumps({"stage": "COMPLETE", "summary": summaries}), flush=True)


if __name__ == "__main__":
    main()
