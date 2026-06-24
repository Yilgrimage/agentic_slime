from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.fully_async_rollout import _get_global_worker
from slime.utils.async_utils import run
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger("examples.agent_env.fully_async_rollout")


async def _generate_rollout_async(args: Any, rollout_id: int, data_buffer: Any) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset

    worker = _get_global_worker(args, data_buffer)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    target = int(args.rollout_batch_size)
    logger.info(
        "agent-env fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    all_groups: list[list[Sample]] = []
    started = time.time()
    last_log = started
    log_every = 30.0

    while len(collected) < target:
        drained = 0
        for gid, group in worker.get_completed_groups():
            drained += 1
            all_groups.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue
            collected[gid] = group

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > log_every:
            logger.info(
                "agent-env fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    def _key(group: list[Sample]) -> int:
        for sample in group:
            idx = getattr(sample, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    data = sorted(collected.values(), key=_key)[:target]
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, sorted(all_groups, key=_key), data_buffer)

    logger.info(
        "agent-env fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect())


def generate_rollout_fully_async(args: Any, rollout_id: int, data_buffer: Any, evaluation: bool = False):
    if evaluation:
        raise ValueError("agent-env fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
