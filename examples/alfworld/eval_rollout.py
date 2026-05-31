from __future__ import annotations

from argparse import Namespace
from typing import Any

from slime.rollout.fully_async_rollout import _stop_global_worker
from slime.rollout.sglang_rollout import generate_rollout as sglang_generate_rollout


def generate_rollout(args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False):
    if evaluation:
        _stop_global_worker()
    return sglang_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
