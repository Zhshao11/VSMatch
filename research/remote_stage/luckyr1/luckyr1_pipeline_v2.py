#!/usr/bin/env python3
"""Version-pinned LuckyR1 B/C pipeline entry point."""
from __future__ import annotations

import os
import sys

import luckyr1_models_v2 as models_v2

# luckyr1_pipeline resolves the model module lazily. Pin it before importing the
# shared, already-audited data/training/evaluation implementation.
sys.modules["luckyr1_models"] = models_v2
import luckyr1_pipeline as pipeline

os.environ["LUCKYR1_IMPLEMENTATION_VERSION"] = models_v2.IMPLEMENTATION_VERSION
_write_card = pipeline.write_card


def write_versioned_card(task, card):
    payload = dict(card)
    payload["implementation_version"] = models_v2.IMPLEMENTATION_VERSION
    _write_card(task, payload)


pipeline.write_card = write_versioned_card

if __name__ == "__main__":
    pipeline.main()
