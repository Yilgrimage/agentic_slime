from __future__ import annotations

import atexit
import asyncio
import logging
import math
import os
import threading
import time
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.fully_async_rollout import AsyncRolloutWorker
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.misc import load_function
from slime.utils.types import Sample

from examples.agent_env.metrics import generated_train_scope_metrics

logger = logging.getLogger("examples.agent_env.fully_async_rollout")

_worker_lock = threading.Lock()
_worker: AsyncRolloutWorker | None = None
_worker_key: tuple[int, int] | None = None
_GLM_PADDING_FILTER_PATH = "examples.agent_env.rollout.glm_style_pad_groups_filter"
_GLM_PADDING_PREFILTER_PATH = "examples.agent_env.rollout.glm_style_group_padding_filter_keep"


def _iter_samples(node: Any):
    if isinstance(node, Sample):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _iter_samples(item)


class _AnnotatedDataBuffer:
    """Attach dump-only generation buckets before the async worker runs RM.

    In fully-async mode ROPD artifacts are written inside the background
    generation worker, before a later training step drains the completed group.
    The real consumer rollout_id is therefore not known yet. This wrapper gives
    diagnostics a stable generated-step bucket without changing Slime's
    training rollout_id semantics.
    """

    def __init__(self, data_buffer: Any, *, groups_per_dump_step: int):
        self._data_buffer = data_buffer
        self._groups_per_dump_step = max(1, int(groups_per_dump_step))
        self._next_submission_id = 0
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data_buffer, name)

    def add_samples(self, groups: Any) -> Any:
        return self._data_buffer.add_samples(groups)

    def get_samples(self, num_samples: int):
        groups = self._data_buffer.get_samples(num_samples)
        for group in groups or []:
            with self._lock:
                submission_id = self._next_submission_id
                self._next_submission_id += 1
            dump_step = submission_id // self._groups_per_dump_step
            index_in_dump_step = submission_id % self._groups_per_dump_step
            for sample in _iter_samples(group):
                sample_metadata = sample.metadata or {}
                sample.metadata = sample_metadata
                sample_metadata.setdefault("agent_env_async_submission_id", submission_id)
                sample_metadata.setdefault("agent_env_async_dump_step", dump_step)
                sample_metadata.setdefault("agent_env_async_index_in_dump_step", index_in_dump_step)
        return groups


def _runtime_env(args: Any, name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    train_env_vars = getattr(args, "train_env_vars", None) or {}
    if isinstance(train_env_vars, dict):
        value = train_env_vars.get(name)
        if value:
            return str(value)
    return default


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _default_max_inflight_groups(args: Any) -> int:
    sample_capacity = max(1, int(args.sglang_server_concurrency) * get_rollout_num_engines(args))
    group_size = max(1, int(args.n_samples_per_prompt))
    return max(1, math.ceil(sample_capacity / group_size))


def _max_inflight_groups(args: Any) -> int:
    configured = _positive_int(_runtime_env(args, "AGENT_ENV_ASYNC_MAX_INFLIGHT_GROUPS", ""))
    if configured is not None:
        return configured
    return _default_max_inflight_groups(args)


def _add_metrics(metrics: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        metrics[key] = metrics.get(key, 0.0) + float(value)


def _load_padding_prefilter(args: Any):
    if getattr(args, "rollout_sample_filter_path", None) != _GLM_PADDING_FILTER_PATH:
        return None
    return load_function(_GLM_PADDING_PREFILTER_PATH)


def _get_worker(args: Any, data_buffer: Any) -> AsyncRolloutWorker:
    global _worker, _worker_key
    max_groups = _max_inflight_groups(args)
    key = (id(data_buffer), max_groups)
    with _worker_lock:
        if _worker is None or not _worker.worker_thread or not _worker.worker_thread.is_alive() or _worker_key != key:
            if _worker is not None:
                _worker.stop()
            logger.info(
                "starting agent-env fully-async worker: max_inflight_groups=%d, "
                "sample_capacity=%d, group_size=%d",
                max_groups,
                int(args.sglang_server_concurrency) * get_rollout_num_engines(args),
                int(args.n_samples_per_prompt),
            )
            annotated_data_buffer = _AnnotatedDataBuffer(
                data_buffer,
                groups_per_dump_step=int(args.rollout_batch_size),
            )
            _worker = AsyncRolloutWorker(args, annotated_data_buffer, concurrency=max_groups)
            _worker.start()
            _worker_key = key
        return _worker


def _stop_worker() -> None:
    global _worker, _worker_key
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
            _worker = None
            _worker_key = None


atexit.register(_stop_worker)


async def _generate_rollout_async(args: Any, rollout_id: int, data_buffer: Any) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset

    worker = _get_worker(args, data_buffer)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    padding_prefilter = _load_padding_prefilter(args)
    metric_gatherer = MetricGatherer()
    prefilter_metrics: dict[str, float] = {}

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
            if padding_prefilter is not None:
                padding_keep, padding_metrics = padding_prefilter(args, group)
                _add_metrics(prefilter_metrics, padding_metrics)
                if not padding_keep:
                    logger.info(
                        "agent-env fully-async rollout %d: skipped group %s before collection: too few valid samples",
                        rollout_id,
                        gid,
                    )
                    continue
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
    metrics = metric_gatherer.collect()
    metrics.update(prefilter_metrics)
    metrics.update(generated_train_scope_metrics(generated_groups=all_groups, train_groups=data))
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def generate_rollout_fully_async(args: Any, rollout_id: int, data_buffer: Any, evaluation: bool = False):
    if evaluation:
        raise ValueError("agent-env fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
