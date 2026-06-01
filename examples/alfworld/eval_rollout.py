from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

from slime.rollout.sglang_rollout import generate_rollout as sglang_generate_rollout

logger = logging.getLogger(__name__)


def _prepare_eval_barrier(args: Namespace) -> None:
    import slime.rollout.fully_async_rollout as fully_async_rollout
    import slime.utils.http_utils as http_utils
    from slime.rollout.sglang_rollout import GenerateState

    timeout_s = float(getattr(args, "alfworld_eval_stop_worker_timeout_s", 60.0))
    with fully_async_rollout._worker_lock:
        worker = fully_async_rollout._global_worker
        fully_async_rollout._global_worker = None

    if worker is not None:
        worker.running = False
        thread = worker.worker_thread
        if thread is not None and thread.is_alive():
            logger.info("ALFWorld eval: waiting for fully-async train rollout worker to stop")
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                raise RuntimeError(
                    f"fully-async train rollout worker did not stop within {timeout_s:.1f}s before eval"
                )

    GenerateState.clear_instances()
    http_utils._http_client = None
    http_utils.init_http_client(args)


def generate_rollout(args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False):
    if evaluation:
        _prepare_eval_barrier(args)
    return sglang_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
