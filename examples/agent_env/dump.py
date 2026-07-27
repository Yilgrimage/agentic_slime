from __future__ import annotations

import os
import re
from typing import Any

_DUMP_COUNTS: dict[str, int] = {}


def runtime_env(args: Any, name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    train_env_vars = getattr(args, "train_env_vars", None) or {}
    if isinstance(train_env_vars, dict):
        value = train_env_vars.get(name)
        if value:
            return str(value)
    return default


def int_runtime_env(args: Any, name: str, default: str = "0") -> int:
    try:
        return int(runtime_env(args, name, default) or 0)
    except (TypeError, ValueError):
        return 0


def safe_filename_part(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe[:80] if safe else "unknown"


def sample_dump_step_label(sample: Any) -> str:
    sample_metadata = getattr(sample, "metadata", None) or {}
    for value in (
        sample_metadata.get("agent_env_async_dump_step"),
        getattr(sample, "rollout_id", None),
        sample_metadata.get("rollout_id"),
        sample_metadata.get("rollout_step"),
    ):
        if value not in (None, ""):
            return str(value)
    return "unknown"


def record_dump_step_label(record: dict[str, Any]) -> str:
    for key in ("dump_step", "agent_env_async_dump_step", "rollout_id", "rollout_step"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def reserve_dump_slot(
    *,
    namespace: str,
    stage: str,
    dump_step: str,
    per_step_limit: int,
    total_limit: int = 0,
) -> tuple[int, int] | None:
    """Reserve one dump slot for a process-local stage.

    Limits are intentionally per process. Rollout and RM run inside multiple
    worker processes, so a global cross-process counter would require shared
    coordination and would make dump writes part of the hot path.
    """

    if per_step_limit <= 0:
        return None
    pid = os.getpid()
    total_key = f"{namespace}:{stage}:pid:{pid}:total"
    total_count = _DUMP_COUNTS.get(total_key, 0)
    if total_limit > 0 and total_count >= total_limit:
        return None
    step_key = f"{namespace}:{stage}:pid:{pid}:step:{dump_step}"
    step_count = _DUMP_COUNTS.get(step_key, 0)
    if step_count >= per_step_limit:
        return None
    _DUMP_COUNTS[step_key] = step_count + 1
    _DUMP_COUNTS[total_key] = total_count + 1
    return step_count, total_count
