#!/usr/bin/env python3
"""Isolated E092A DAG runner using the audited LuckyR1 watchdog integration."""
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/sda/zhushihao/experiment_watchdog_luckyr1")
import luckyr1_orchestrator as runner


runner.ROOT = Path("/mnt/sda/zhushihao/experiment_watchdog_e092a")
runner.PLAN = runner.ROOT / "e092a_plan.json"
runner.STATE = runner.ROOT / "e092a_orchestrator_state.json"
runner.LOG = runner.ROOT / "e092a_orchestrator.log"
runner.ATTEMPT_CARDS = runner.ROOT / "attempt_cards"
runner.COLLISIONS = runner.ROOT / "collisions"


if __name__ == "__main__":
    runner.main()
