#!/usr/bin/env python3
"""Full clean-lineage M4 graph gate before LuckyR1 A/B/C training."""
from __future__ import annotations

import json
import os
import signal
import time
import traceback
from pathlib import Path

import torch

import luckyr1_models_v3 as models
import luckyr1_pipeline_v3 as versioned

pipeline = versioned.pipeline
TASK = "lucky_clean_v3_g0"


def main():
    started = time.time()
    state = {
        "status": "RUNNING",
        "last_completed_stage": "START",
        "implementation_version": models.IMPLEMENTATION_VERSION,
        "watchdog_job_id": os.getenv("WATCHDOG_JOB_ID"),
        "candidate_gpus": None,
        "logical_device": "cuda:0",
        "uses_gt": False,
        "method_result": False,
    }

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    pipeline.write_card(TASK, state)
    try:
        _e072, core_datasets, *_rest = pipeline.imports()
        _train, dev = pipeline.datasets(core_datasets)
        _pair, source, target, _hidden_gt, _valid = dev[0]
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        source, target = source[None].to(device), target[None].to(device)
        bc = json.loads(pipeline.selector_path("bc_reduce").read_text())
        raw = json.loads(pipeline.selector_path("m4_reduce").read_text())
        raw_state = torch.load(raw["checkpoint"], map_location=device)["model"]
        rows = []
        for kind in ("A", "B", "C"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = models.M4Model(
                device, Path(bc["B"]["checkpoint"]), Path(bc["C"]["checkpoint"]),
                lucky_kind=kind,
            ).to(device)
            model.gate.load_state_dict(
                {key.removeprefix("gate."): value for key, value in raw_state.items() if key.startswith("gate.")},
                strict=True,
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for module in model.modules():
                if isinstance(module, models.EvidenceAdapter):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
            outputs = model(source, target, iters=2)
            outputs[-1].float().square().mean().backward()
            gradients = []
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    gradient = parameter.grad
                    gradients.append({
                        "name": name,
                        "parameter_finite": bool(torch.isfinite(parameter).all()),
                        "gradient_present": gradient is not None,
                        "gradient_finite": bool(gradient is not None and torch.isfinite(gradient).all()),
                        "gradient_abs_sum": float(gradient.abs().sum()) if gradient is not None else None,
                    })
            row = {
                "kind": kind,
                "shape": list(outputs[-1].shape),
                "finite_output": bool(torch.isfinite(outputs[-1]).all()),
                "gradients": gradients,
                "all_trainable_finite_nonzero": bool(gradients) and all(
                    g["parameter_finite"] and g["gradient_present"] and g["gradient_finite"]
                    and g["gradient_abs_sum"] > 0 for g in gradients
                ),
                "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 2**20),
                "path": "clean selected B/C -> clean M4 raw gate -> real CorrBlock -> v3 evidence -> real BasicUpdateBlock",
            }
            rows.append(row)
            if not (row["shape"] == [1, 2, 512, 512] and row["finite_output"] and row["all_trainable_finite_nonzero"]):
                raise RuntimeError(row)
            del model, outputs
        state.update({
            "status": "COMPLETE",
            "last_completed_stage": "CLEAN_M4_V3_ABC_GRAPH_MEMORY_G0",
            "variants": rows,
            "bc_selection": bc,
            "raw_m4_selection": raw,
        })
    except BaseException as error:
        state.update({
            "status": "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "IMPLEMENTATION_FAILED",
            "cause": repr(error),
            "traceback": traceback.format_exc(),
        })
    state["elapsed_seconds"] = time.time() - started
    pipeline.write_card(TASK, state)
    if state["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
