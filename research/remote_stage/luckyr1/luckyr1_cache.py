#!/usr/bin/env python3
"""Build and verify the historical checkpoint cache without importing Torch."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import time
import traceback
from pathlib import Path

ROOT = Path("/mnt/sda/zhushihao/VS_Match")
PARENT = ROOT / "GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN"
RUN = PARENT / "current_model_exploratory/exploratory_cache"
CACHE = Path("/dev/shm/zhushihao_luckyr1_ckpts")
EXPECTED_M4 = "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b"
SOURCES = {
    "m4": ROOT / "GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt",
    "b": ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-b-stage180k/checkpoints/B_step180000.pt",
    "c": ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION/run-c-stage180k/checkpoints/C_step180000.pt",
    "e073_f2": ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f2-warm120k/checkpoints/F2_step120000.pt",
    "e073_f3": ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN/run-f3-warm120k/checkpoints/F3_step120000.pt",
    "e072_v1": ROOT / "GDROS/experiments/sparsegeo/runs/E-072A-ZZVR-LAYERWISE-FRONTEND-ALIGNMENT/run-v1-r60k/checkpoints/V1_step60000.pt",
    "e072_v2": ROOT / "GDROS/experiments/sparsegeo/runs/E-072A-ZZVR-LAYERWISE-FRONTEND-ALIGNMENT/run-v2-r60k/checkpoints/V2_step60000.pt",
    "e046": ROOT / "GDROS/experiments/sparsegeo/runs/E-046A-ZZVR-JOINT-TRAINING/run-r2/checkpoints/masked_pretrain_joint.pt",
}


def write_card(card: dict) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(card, indent=2, sort_keys=True) + "\n"
    (RUN / "RESULT_CARD.json").write_text(payload)
    (RUN / "RESULT_CARD.md").write_text(f"# LuckyR1 historical checkpoint cache\n\nStatus: `{card.get('status')}`\n\n```json\n{payload}```\n")
    attempt = os.getenv("WATCHDOG_JOB_ID", f"manual-{os.getpid()}")
    target = RUN / "attempts" / attempt
    target.mkdir(parents=True, exist_ok=True)
    (target / "RESULT_CARD.json").write_text(payload)


def main() -> None:
    started = time.time()
    state = {"status": "RUNNING", "last_completed_stage": "START", "files": {}, "gpu_used": False}

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write_card(state)
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        for name, source in SOURCES.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            target = CACHE / f"{name}.pt"
            temporary = target.with_suffix(".pt.tmp")
            digest = hashlib.sha256()
            reused = target.is_file() and target.stat().st_size == source.stat().st_size
            if reused:
                with target.open("rb") as reader:
                    for chunk in iter(lambda: reader.read(8 << 20), b""):
                        digest.update(chunk)
            else:
                with source.open("rb") as reader, temporary.open("wb") as writer:
                    while True:
                        chunk = reader.read(8 << 20)
                        if not chunk:
                            break
                        writer.write(chunk)
                        digest.update(chunk)
                os.replace(temporary, target)
            state["files"][name] = {"source": str(source), "cache": str(target), "size_bytes": target.stat().st_size, "sha256": digest.hexdigest(), "reused_existing_cache": reused}
            state["last_completed_stage"] = f"VERIFIED_{name}"
            write_card(state)
        if state["files"]["m4"]["sha256"] != EXPECTED_M4:
            raise RuntimeError("cached M4 checksum mismatch")
        state.update({"status": "COMPLETE", "last_completed_stage": "HISTORICAL_CHECKPOINT_CACHE_COMPLETE", "cache_filesystem": "/dev/shm", "elapsed_seconds": time.time() - started})
    except BaseException as error:
        state.update({"status": "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "IMPLEMENTATION_FAILED", "cause": repr(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started})
    finally:
        write_card(state)
    if state["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
