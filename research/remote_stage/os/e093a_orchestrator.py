#!/usr/bin/env python3
"""Isolated E093A DAG runner using the audited watchdog integration."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/sda/zhushihao/experiment_watchdog_luckyr1")
import luckyr1_orchestrator as runner

runner.ROOT = Path("/mnt/sda/zhushihao/experiment_watchdog_e093a")
runner.PLAN = runner.ROOT / "e093a_plan.json"
runner.STATE = runner.ROOT / "e093a_orchestrator_state.json"
runner.LOG = runner.ROOT / "e093a_orchestrator.log"
runner.ATTEMPT_CARDS = runner.ROOT / "attempt_cards"
runner.COLLISIONS = runner.ROOT / "collisions"


def submit_with_profile_affinity(task: dict) -> str:
    command = [
        "/usr/bin/python3", str(runner.WATCHDOG), "--root", str(runner.ROOT), "submit",
        "--name", task["name"], "--cwd", task["cwd"], "--gpus", str(task.get("gpu_count", 1)),
    ]
    if task["id"].startswith("profile_") and task["id"] != "profile_b":
        state = runner.read_json(runner.STATE, {"tasks": {}})
        allocated = state.get("tasks", {}).get("profile_b", {}).get("allocated_gpus", [])
        if len(allocated) != 1:
            raise RuntimeError(f"profile_b physical GPU unavailable: {allocated}")
        command += ["--candidate-gpus", str(allocated[0])]
    command += [
        "--min-free-memory-mb", str(task.get("min_free_memory_mb", 12000)),
        "--priority", str(task.get("priority", 0)), "--no-email", "--", *task["command"],
    ]
    return json.loads(subprocess.check_output(command, text=True))["id"]


runner.submit = submit_with_profile_affinity

if __name__ == "__main__":
    runner.main()
