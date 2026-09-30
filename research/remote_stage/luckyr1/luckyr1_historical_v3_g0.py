#!/usr/bin/env python3
"""Per-parameter historical-M4 graph and memory gate for LuckyR1 v3."""
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
TASK = "exploratory_v3_g0"


def parameter_role(name: str) -> str:
    if ".model.b.f2.evidence." in f".{name}":
        return "E074-B F2 matcher evidence; expected active after v3 graph repair"
    if ".model.b.f3.evidence." in f".{name}":
        return "E074-B F3 matcher evidence; expected active after v3 graph repair"
    if ".model.c.student.evidence." in f".{name}":
        return "E074-C student matcher evidence; expected active"
    return "trainable evidence parameter; expected active"


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
        "started_at_unix": started,
        "v2_diagnosis": {
            "inactive_parameters": "model.b.f2/f3 evidence parameters",
            "cause": "historical E074-B teacher_preds used torch.no_grad around both matchers",
            "classification": "intentional for frozen E074 teachers, broken for newly injected trainable LuckyR1 evidence",
            "repair": "remove no_grad only from E074-B matcher calls; all base parameters remain requires_grad=False",
        },
    }

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    pipeline.write_card(TASK, state)
    try:
        _e072, core_datasets, *_rest = pipeline.imports()
        _train, evaluation = pipeline.exploratory_datasets(core_datasets)
        _pair_id, source, target, _hidden_gt, _valid = evaluation[0]
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        source, target = source[None].to(device), target[None].to(device)
        cache_card = json.loads(
            (pipeline.task_dir("exploratory_cache") / "RESULT_CARD.json").read_text()
        )
        cache_paths = {name: row["cache"] for name, row in cache_card["files"].items()}
        rows = []
        for kind in ("B", "C"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = models.HistoricalLuckyM4(
                Path(cache_paths["m4"]), pipeline.HISTORICAL_M4_SHA, kind, device,
                verify_checksum=False, cache_paths=cache_paths,
            ).to(device)
            outputs = model(source, target, iters=2)
            loss = outputs[-1].float().square().mean()
            loss.backward()
            gradients = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                gradient = parameter.grad
                gradients.append({
                    "name": name,
                    "role": parameter_role(name),
                    "parameter_finite": bool(torch.isfinite(parameter).all()),
                    "gradient_present": gradient is not None,
                    "gradient_finite": bool(gradient is not None and torch.isfinite(gradient).all()),
                    "gradient_abs_sum": float(gradient.abs().sum()) if gradient is not None else None,
                    "expected_nonzero": True,
                })
            row = {
                "kind": kind,
                "outputs": len(outputs),
                "shape": list(outputs[-1].shape),
                "finite_output": bool(torch.isfinite(outputs[-1]).all()),
                "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "trainable_tensors": len(gradients),
                "gradients": gradients,
                "all_parameters_finite": all(g["parameter_finite"] for g in gradients),
                "all_expected_gradients_present_finite_nonzero": all(
                    g["gradient_present"] and g["gradient_finite"] and g["gradient_abs_sum"] > 0
                    for g in gradients if g["expected_nonzero"]
                ),
                "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 2**20),
                "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 2**20),
                "path": "protected historical M4 -> real CorrBlock -> v3 evidence -> real BasicUpdateBlock -> region gate",
            }
            rows.append(row)
            if not (row["outputs"] == 2 and row["shape"] == [1, 2, 512, 512]
                    and row["finite_output"] and row["all_parameters_finite"]
                    and row["all_expected_gradients_present_finite_nonzero"]):
                raise RuntimeError(row)
            del model, outputs, loss
        state.update({
            "status": "COMPLETE",
            "last_completed_stage": "HISTORICAL_M4_V3_PER_PARAMETER_GRAPH_MEMORY_G0",
            "variants": rows,
            "protected_m4_sha256": pipeline.HISTORICAL_M4_SHA,
            "protected_m4_modified": False,
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
