from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.rewards.config import resolve_path, reward_cfg_path
from examples.agent_env.rewards.extractors import metadata, runtime_env

DEFAULT_TEACHER_TRACE_FIELDS = (
    "teacher_tool_trace",
    "teacher_trace",
    "teacher_trajectory",
    "teacher_action_observation_trace",
    "teacher_full_trace_text",
    "reference_trace",
    "teacher_trace_tool_io",
    "teacher_no_tool_response_text",
    "teacher_trace_no_tool_response",
    "teacher_no_tool_trace",
    "reference_no_tool_trace",
)

DEFAULT_TEACHER_INDEX_KEYS = (
    "teacher_trace_key",
    "teacher_index_key",
    "teacher_rollout_key",
    "rollout_key",
    "source_sample_id",
    "global_index",
    "prompt_index",
    "row_index",
    "sample_id",
    "task_id",
    "id",
    "task_index",
)

_TEACHER_INDEX_CACHE: dict[tuple[str, str], tuple[int, int, dict[str, dict[str, Any]]]] = {}


def list_value(value: Any, default: tuple[str, ...] = ()) -> list[str]:
    if value in (None, "", []):
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def metadata_field_values(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def mapping_text(mapping: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return str(value).strip()
    return ""


def metadata_value(sample: Sample, keys: list[str]) -> Any:
    sample_metadata = metadata(sample)
    for key in keys:
        value = sample_metadata.get(key)
        if value not in (None, "", []):
            return value
    return None


def config_value(args: Any, config_prefix: str, name: str, default: Any = None) -> Any:
    return reward_cfg_path(args, f"{config_prefix}.{name}", default)


def teacher_index_path(args: Any, *, config_prefix: str, env_var: str) -> Path | None:
    raw = str(runtime_env(args, env_var, "") or config_value(args, config_prefix, "teacher_index_path", "")).strip()
    return resolve_path(args, raw) if raw else None


def candidate_values_from_mapping(mapping: dict[str, Any], keys: list[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = mapping.get(key)
        if value in (None, "", []):
            continue
        for item in metadata_field_values(value):
            values.append(item)
    return values


def teacher_index_key_candidates(
    args: Any,
    sample: Sample,
    *,
    config_prefix: str,
    join_value: str | None = None,
) -> list[str]:
    sample_metadata = metadata(sample)
    configured_keys = config_value(args, config_prefix, "teacher_index_keys", None)
    keys = list_value(configured_keys, DEFAULT_TEACHER_INDEX_KEYS)
    candidates = candidate_values_from_mapping(sample_metadata, keys)
    source_row = sample_metadata.get("source_row")
    if isinstance(source_row, dict):
        candidates.extend(candidate_values_from_mapping(source_row, keys))
    if configured_keys in (None, "", []):
        sample_index = getattr(sample, "index", None)
        if sample_index is not None:
            candidates.append(str(sample_index))
    if join_value:
        candidates.append(join_value)
    return list(dict.fromkeys(value for value in candidates if value))


def read_teacher_index(args: Any, path: Path, *, config_prefix: str) -> dict[str, dict[str, Any]]:
    stat = path.stat()
    cache_key = (config_prefix, str(path.resolve()))
    cached = _TEACHER_INDEX_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            if all(isinstance(item, dict) for item in value.values()):
                rows = [dict(item, _index_key=str(key)) for key, item in value.items()]
            elif isinstance(value.get("items"), list):
                rows = [item for item in value["items"] if isinstance(item, dict)]
            else:
                rows = [value]
        elif isinstance(value, list):
            rows = [item for item in value if isinstance(item, dict)]
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    if not line.endswith("\n"):
                        break
                    raise ValueError(f"Invalid JSON in teacher index {path}:{line_no}: {exc}") from exc
                if isinstance(value, dict):
                    rows.append(value)

    key_fields = list_value(config_value(args, config_prefix, "teacher_index_keys", None), DEFAULT_TEACHER_INDEX_KEYS)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidates = candidate_values_from_mapping(row, key_fields)
        source_row = row.get("source_row")
        if isinstance(source_row, dict):
            candidates.extend(candidate_values_from_mapping(source_row, key_fields))
        if row.get("_index_key") not in (None, ""):
            candidates.append(str(row["_index_key"]))
        for key in dict.fromkeys(candidates):
            index.setdefault(key, row)

    _TEACHER_INDEX_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, index)
    return index


def teacher_index_row_status(row: dict[str, Any], *, has_text: bool) -> str:
    for source in (row, row.get("evidence"), row.get("adapter_result")):
        if isinstance(source, dict):
            for success_key in ("teacher_success", "success", "completed"):
                success = source.get(success_key)
                if success is True:
                    return "completed"
            value = source.get("status") or source.get("teacher_status") or source.get("episode_status")
            if value:
                status = str(value).strip().lower()
                if status in {"ok", "success", "succeeded", "done"}:
                    return "completed"
                return status
    return "completed" if has_text else "missing"


def teacher_index_row_text(
    args: Any,
    row: dict[str, Any],
    *,
    config_prefix: str,
    key_config_name: str,
    default_fields: tuple[str, ...] = DEFAULT_TEACHER_TRACE_FIELDS,
) -> str:
    keys = list_value(config_value(args, config_prefix, key_config_name, None), default_fields)
    for source in (row, row.get("evidence"), row.get("adapter_result"), row.get("source_row")):
        if isinstance(source, dict):
            text = mapping_text(source, keys)
            if text:
                return text
    return ""


def teacher_index_row_for_sample(
    args: Any,
    sample: Sample,
    *,
    config_prefix: str,
    env_var: str,
    join_value: str | None = None,
) -> dict[str, Any] | None:
    index_path = teacher_index_path(args, config_prefix=config_prefix, env_var=env_var)
    if index_path is None:
        return None
    index = read_teacher_index(args, index_path, config_prefix=config_prefix)
    for key in teacher_index_key_candidates(args, sample, config_prefix=config_prefix, join_value=join_value):
        row = index.get(key)
        if row is not None:
            return row
    return None


def teacher_trace_text(
    args: Any,
    sample: Sample,
    *,
    config_prefix: str,
    env_var: str,
    key_config_name: str,
    default_fields: tuple[str, ...] = DEFAULT_TEACHER_TRACE_FIELDS,
    join_value: str | None = None,
) -> str:
    keys = list_value(config_value(args, config_prefix, key_config_name, None), default_fields)
    value = metadata_value(sample, keys)
    if value not in (None, "", []):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return str(value).strip()

    row = teacher_index_row_for_sample(
        args,
        sample,
        config_prefix=config_prefix,
        env_var=env_var,
        join_value=join_value,
    )
    if row is None:
        return ""
    text = teacher_index_row_text(
        args,
        row,
        config_prefix=config_prefix,
        key_config_name=key_config_name,
        default_fields=default_fields,
    )
    if not text:
        return ""
    status = teacher_index_row_status(row, has_text=True)
    if status != "completed":
        return ""
    return text


def teacher_trace_value(
    args: Any,
    sample: Sample,
    *,
    config_prefix: str,
    env_var: str,
    key_config_name: str,
    default_fields: tuple[str, ...] = DEFAULT_TEACHER_TRACE_FIELDS,
    join_value: str | None = None,
) -> Any:
    """Return a structured teacher field without serializing it through text."""

    keys = list_value(config_value(args, config_prefix, key_config_name, None), default_fields)
    sample_metadata = metadata(sample)
    for key in keys:
        value = sample_metadata.get(key)
        if value not in (None, "", []):
            return value

    row = teacher_index_row_for_sample(
        args,
        sample,
        config_prefix=config_prefix,
        env_var=env_var,
        join_value=join_value,
    )
    if row is None:
        return None
    for source in (row, row.get("evidence"), row.get("adapter_result"), row.get("source_row")):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if value not in (None, "", []):
                if teacher_index_row_status(row, has_text=True) != "completed":
                    return None
                return value
    return None
