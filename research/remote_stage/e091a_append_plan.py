#!/usr/bin/env python3
"""Append the deterministic E091A DAG to the shared watchdog plan."""
import json
import os
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
plan = json.loads(plan_path.read_text())
tasks = plan["tasks"]
existing = {task["id"] for task in tasks}
root = Path(os.environ.get("SPARSEGEO_ROOT", "/mnt/sda/zhushihao/VS_Match"))
python = os.environ.get(
    "SPARSEGEO_PYTHON",
    "/mnt/sda/zhushihao/multiModel_matching/xoftr_sar/envs/xoftr5090/bin/python",
)
cwd = os.environ.get("SPARSEGEO_MIGRATED_ROOT", str(root / "image_matching_part2_migrated"))
script = ".codex_tmp/e091a_heldout_generalization.py"


def add(task_id, name, command, dependencies, gpus=0, memory=0, attempts=4):
    if task_id in existing:
        return
    tasks.append({
        "id": task_id,
        "name": name,
        "cwd": cwd,
        "gpu_count": gpus,
        "min_free_memory_mb": memory,
        "priority": 20,
        "max_attempts": attempts,
        "depends_on": dependencies,
        "command": [python, script, *command],
    })
    existing.add(task_id)


add("e091a_prepare", "E091A-PREPARE-GROUP-SPLIT", ["--mode", "prepare"], ["e090d_d1_reduce"], attempts=2)
previous = "e091a_prepare"
for phase, epochs in (("candidate", range(1, 5)), ("recurrent", range(1, 4))):
    for epoch in epochs:
        prefix = f"e091a_{phase}_e{epoch}"
        add(f"{prefix}_train", f"E091A-{phase.upper()}-EPOCH{epoch}-TRAIN", ["--mode", "train", "--phase", phase, "--epoch", str(epoch)], [previous], gpus=1, memory=18000, attempts=6)
        inference_tasks = []
        for shard in range(4):
            task_id = f"{prefix}_infer_{shard}"
            inference_tasks.append(task_id)
            add(task_id, f"E091A-{phase.upper()}-EPOCH{epoch}-DEV-SHARD{shard}", ["--mode", "infer", "--phase", phase, "--epoch", str(epoch), "--shard", str(shard)], [f"{prefix}_train"], gpus=1, memory=18000, attempts=6)
        add(f"{prefix}_join", f"E091A-{phase.upper()}-EPOCH{epoch}-DEV-JOIN", ["--mode", "join", "--phase", phase, "--epoch", str(epoch)], inference_tasks, attempts=3)
        previous = f"{prefix}_join"
add("e091a_reduce", "E091A-DEV-SELECTION-AND-GATE", ["--mode", "select"], [previous], attempts=3)
plan_path.write_text(json.dumps(plan, indent=2) + "\n")
