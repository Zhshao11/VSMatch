#!/usr/bin/env python3
"""Persistent dependency runner for LuckyR1 watchdog jobs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/sda/zhushihao/experiment_watchdog_luckyr1")
PLAN = ROOT / "lucky_plan.json"
STATE = ROOT / "lucky_orchestrator_state.json"
LOG = ROOT / "lucky_orchestrator.log"
ATTEMPT_CARDS = ROOT / "attempt_cards"
COLLISIONS = ROOT / "collisions"
WATCHDOG = Path("/mnt/sda/zhushihao/experiment_watchdog/experiment_watchdog.py")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def log(message: str) -> None:
    with LOG.open("a") as handle:
        handle.write(f"{now()} {message}\n")


def ensure_attempt_card(task: dict, current: dict) -> None:
    """Guarantee a scheduler-level card even when the model process cannot finalize."""
    job_id = current["id"]
    target = ATTEMPT_CARDS / task["id"] / job_id
    target.mkdir(parents=True, exist_ok=True)
    card = {
        "task": task["id"],
        "job_id": job_id,
        "status": current.get("status"),
        "last_completed_stage": "WATCHDOG_TERMINAL_STATE",
        "failure_reason": current.get("failure_reason"),
        "return_code": current.get("return_code"),
        "created_at": current.get("created_at"),
        "started_at": current.get("started_at"),
        "finished_at": current.get("finished_at"),
        "allocated_gpus": current.get("allocated_gpus", []),
        "candidate_gpus": current.get("candidate_gpus"),
        "log_path": current.get("log_path"),
        "watchdog_report_json": current.get("report_json"),
        "evidence_scope": "scheduler-level; method card in the run directory is authoritative when present",
    }
    write_json(target / "RESULT_CARD.json", card)
    (target / "RESULT_CARD.md").write_text(
        f"# {task['id']} attempt {job_id}\n\n"
        f"Status: `{card['status']}`\n\n"
        f"Last stage: `{card['last_completed_stage']}`\n\n"
        f"Reason: `{card['failure_reason']}`\n"
    )


def job(job_id: str):
    return read_json(ROOT / "state" / "jobs" / f"{job_id}.json", None)


def process_group(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat[stat.rfind(")") + 2 :].split()[2])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def process_stat(pid: int) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        return {"pid": pid, "state": fields[0], "process_group": int(fields[2]), "cpu_ticks": int(fields[11]) + int(fields[12])}
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def descendants(root_pid: int) -> list[int]:
    found, pending = [], [root_pid]
    while pending:
        parent = pending.pop()
        try:
            children = [int(value) for value in Path(f"/proc/{parent}/task/{parent}/children").read_text().split()]
        except (FileNotFoundError, PermissionError, ValueError):
            children = []
        for child in children:
            if child not in found:
                found.append(child)
                pending.append(child)
    return found


def liveness_snapshot(current: dict, gpu_processes: dict[int, list[int]] | None, previous: dict | None) -> dict:
    pids = [int(current.get("pid") or -1)] + descendants(int(current.get("pid") or -1))
    if gpu_processes is not None:
        for gpu in current.get("allocated_gpus", []):
            pids.extend(gpu_processes.get(int(gpu), []))
    processes = [row for pid in sorted(set(pids)) if pid > 0 and (row := process_stat(pid)) is not None]
    total_ticks = sum(row["cpu_ticks"] for row in processes)
    previous_ticks = int((previous or {}).get("total_cpu_ticks", total_ticks))
    return {
        "sampled_at": now(),
        "processes": processes,
        "total_cpu_ticks": total_ticks,
        "cpu_ticks_since_previous_sample": max(0, total_ticks - previous_ticks),
        "allocated_gpus": current.get("allocated_gpus", []),
    }


def gpu_compute_processes() -> dict[int, list[int]] | None:
    try:
        gpu_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        process_output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        uuid_to_index = {}
        for line in gpu_output.splitlines():
            index, uuid = [part.strip() for part in line.split(",", 1)]
            uuid_to_index[uuid] = int(index)
        result = {index: [] for index in uuid_to_index.values()}
        for line in process_output.splitlines():
            if not line.strip():
                continue
            uuid, pid = [part.strip() for part in line.split(",", 1)]
            if uuid in uuid_to_index:
                result[uuid_to_index[uuid]].append(int(pid))
        return result
    except (subprocess.SubprocessError, ValueError) as exc:
        log(f"collision monitor unavailable error={exc!r}")
        return None


def cancel_on_collision(task: dict, current: dict, gpu_processes: dict[int, list[int]] | None) -> dict | None:
    if current.get("status") != "running" or gpu_processes is None:
        return None
    expected_group = int(current.get("process_group") or -1)
    for gpu in current.get("allocated_gpus", []):
        external = [pid for pid in gpu_processes.get(int(gpu), []) if process_group(pid) not in {None, expected_group}]
        if not external:
            continue
        evidence = {
            "task": task["id"],
            "job_id": current["id"],
            "status": "OUR_JOB_CANCELLED_FOR_EXTERNAL_GPU_COLLISION",
            "gpu": int(gpu),
            "our_process_group": expected_group,
            "external_pids": external,
            "detected_at": now(),
            "candidate_gpus": current.get("candidate_gpus"),
        }
        write_json(COLLISIONS / f"{current['id']}.json", evidence)
        try:
            subprocess.run(
                ["/usr/bin/python3", str(WATCHDOG), "--root", str(ROOT), "cancel", current["id"]],
                check=False,
                text=True,
                timeout=30,
            )
            log(f"collision cancelled task={task['id']} job={current['id']} gpu={gpu} external={external}")
        except subprocess.SubprocessError as exc:
            evidence["cancel_error"] = repr(exc)
            write_json(COLLISIONS / f"{current['id']}.json", evidence)
            log(f"collision cancel error task={task['id']} job={current['id']} error={exc!r}")
        return evidence
    return None


def submit(task: dict) -> str:
    command = [
        "/usr/bin/python3",
        str(WATCHDOG),
        "--root",
        str(ROOT),
        "submit",
        "--name",
        task["name"],
        "--cwd",
        task["cwd"],
        "--gpus",
        str(task.get("gpu_count", 1)),
        "--min-free-memory-mb",
        str(task.get("min_free_memory_mb", 12000)),
        "--priority",
        str(task.get("priority", 0)),
        "--no-email",
        "--",
        *task["command"],
    ]
    output = subprocess.check_output(command, text=True)
    return json.loads(output)["id"]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    ATTEMPT_CARDS.mkdir(parents=True, exist_ok=True)
    COLLISIONS.mkdir(parents=True, exist_ok=True)
    log(f"orchestrator started pid={os.getpid()}")
    while True:
        plan = read_json(PLAN, {"tasks": []})
        state = read_json(STATE, {"pid": os.getpid(), "tasks": {}})
        observed_gpu_processes = gpu_compute_processes()
        state["pid"] = os.getpid()
        state["heartbeat"] = now()
        tasks = state.setdefault("tasks", {})
        planned_ids = {task["id"] for task in plan.get("tasks", [])}
        for retired_name, retired_entry in tasks.items():
            if retired_name in planned_ids:
                retired_entry.pop("retired_from_plan", None)
                continue
            retired_job = job(retired_entry.get("job_id", "")) if retired_entry.get("job_id") else None
            if retired_job:
                retired_entry["status"] = retired_job["status"]
                retired_entry["allocated_gpus"] = retired_job.get("allocated_gpus", [])
                retired_entry["updated_at"] = retired_job.get("updated_at")
                if retired_job["status"] in {"succeeded", "failed", "cancelled"}:
                    ensure_attempt_card({"id": retired_name}, retired_job)
            retired_entry["retired_from_plan"] = True
        for task in plan.get("tasks", []):
            name = task["id"]
            entry = tasks.setdefault(name, {"status": "waiting", "attempts": 0})
            dependencies = task.get("depends_on", [])
            if not all(tasks.get(dep, {}).get("status") == "succeeded" for dep in dependencies):
                continue
            current = job(entry.get("job_id", "")) if entry.get("job_id") else None
            if current:
                collision = cancel_on_collision(task, current, observed_gpu_processes)
                if collision:
                    entry["collision"] = collision
                    entry["status"] = "collision_cancel_pending"
                    continue
                entry["status"] = current["status"]
                entry["allocated_gpus"] = current.get("allocated_gpus", [])
                entry["updated_at"] = current.get("updated_at")
                if current["status"] == "running":
                    entry["liveness"] = liveness_snapshot(current, observed_gpu_processes, entry.get("liveness"))
                if current["status"] in {"succeeded", "failed", "cancelled"}:
                    ensure_attempt_card(task, current)
                if current["status"] in {"failed", "cancelled"} and entry["attempts"] < task.get("max_attempts", 1):
                    entry.pop("job_id", None)
                    entry["status"] = "retry_pending"
                continue
            if entry["attempts"] >= task.get("max_attempts", 1):
                entry["status"] = "failed_exhausted"
                continue
            try:
                entry["job_id"] = submit(task)
                entry["attempts"] += 1
                entry["status"] = "queued"
                log(f"submitted task={name} job={entry['job_id']}")
            except Exception as exc:
                entry["status"] = "submit_error"
                entry["last_error"] = repr(exc)
                log(f"submit error task={name} error={exc!r}")
        write_json(STATE, state)
        time.sleep(15)


if __name__ == "__main__":
    main()
