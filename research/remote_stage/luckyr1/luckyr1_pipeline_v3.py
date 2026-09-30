#!/usr/bin/env python3
"""Version-pinned LuckyR1 v3 training entry point."""
from __future__ import annotations

import os
import sys

import luckyr1_models_v3 as models_v3

sys.modules["luckyr1_models"] = models_v3
import luckyr1_pipeline as pipeline

os.environ["LUCKYR1_IMPLEMENTATION_VERSION"] = models_v3.IMPLEMENTATION_VERSION
_write_card = pipeline.write_card


def write_versioned_card(task, card):
    payload = dict(card)
    payload["implementation_version"] = models_v3.IMPLEMENTATION_VERSION
    _write_card(task, payload)


pipeline.write_card = write_versioned_card

if __name__ == "__main__":
    pipeline.main()
