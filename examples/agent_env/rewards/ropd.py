from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from .config import resolve_path, reward_cfg_path
from . import naive
from .extractors import (
    float_value,
    metadata,
    prediction_text,
    reference_values,
    runtime_env,
    task_prompt,
    truncate,
)
from .llm_client import call_json_judge_with_metadata, judge_mode, parse_single_score
from .types import RewardResult

RUBRIC_SYSTEM_PROMPT = (
    "You write compact grading rubrics for agent trajectories. "
    "Use the teacher/student comparison only to infer evaluation criteria. "
    "Do not mention which answer came from teacher or student. Output JSON only."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict rubric-based judge. Grade the candidate answer with the given rubric. "
    "Output JSON only."
)

_TEACHER_CACHE: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}


def _cfg(args: Any, name: str, default: Any = None) -> Any:
    return reward_cfg_path(args, f"ropd.{name}", default)


def _list_value(value: Any, default: tuple[str, ...] = ()) -> list[str]:
    if value in (None, "", []):
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _metadata_value(sample: Sample, keys: list[str]) -> Any:
    sample_metadata = metadata(sample)
    for key in keys:
        value = sample_metadata.get(key)
        if value not in (None, "", []):
            return value
    return None


def _cache_path(args: Any) -> Path | None:
    path = runtime_env(args, "AGENT_ENV_ROPD_RUBRIC_CACHE_PATH", "").strip()
    if not path:
        path = str(_cfg(args, "rubric_cache_path", "") or "").strip()
    return resolve_path(args, path) if path else None


def _role_api_key_path(args: Any, role: str) -> str | None:
    upper = role.upper()
    path = runtime_env(args, f"AUX_{upper}_API_KEY_PATH", "").strip()
    return str(resolve_path(args, path)) if path else None


def _role_endpoint(args: Any, role: str) -> dict[str, str]:
    upper = role.upper()
    values: dict[str, str] = {}
    for key in ("provider", "base_url", "model"):
        key_upper = key.upper()
        value = runtime_env(args, f"AUX_{upper}_{key_upper}", "").strip()
        if value:
            values[key] = value
    return values


def _cache_key(sample: Sample) -> str:
    sample_metadata = metadata(sample)
    raw = {
        "task_id": sample_metadata.get("task_id"),
        "prompt": task_prompt(sample),
        "references": reference_values(sample),
    }
    text = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    value = json.loads(line, strict=False)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        return [item for item in value["items"] if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("tasks"), list):
        return [item for item in value["tasks"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported ROPD teacher file format: {path}")


def _join_key(args: Any) -> str:
    return str(_cfg(args, "join_key", "task_id") or "task_id")


def _join_value(args: Any, sample: Sample) -> str:
    sample_metadata = metadata(sample)
    key = _join_key(args)
    for candidate in (key, "task_id", "id", "task_index"):
        value = sample_metadata.get(candidate)
        if value not in (None, "", []):
            return str(value)
    return _cache_key(sample)


def _teacher_files(args: Any) -> list[Path]:
    override = runtime_env(args, "AGENT_ENV_ROPD_TEACHER_FILE", "").strip()
    raw = override or _cfg(args, "teacher_file", None) or _cfg(args, "teacher_files", None)
    return [resolve_path(args, item) for item in _list_value(raw)]


def _teacher_index(args: Any) -> dict[str, dict[str, Any]]:
    files = _teacher_files(args)
    if not files:
        return {}
    key = _join_key(args)
    cache_key = ("|".join(str(path) for path in files), key)
    if cache_key in _TEACHER_CACHE:
        return _TEACHER_CACHE[cache_key]

    output: dict[str, dict[str, Any]] = {}
    for path in files:
        for row in _load_json_or_jsonl(path):
            row_key = row.get(key)
            if row_key in (None, "", []):
                row_key = row.get("task_id") or row.get("id") or row.get("task_index")
            if row_key in (None, "", []):
                continue
            output[str(row_key)] = row
    _TEACHER_CACHE[cache_key] = output
    return output


def _teacher_data(args: Any, sample: Sample) -> dict[str, Any]:
    return _teacher_index(args).get(_join_value(args, sample), {})


def _teacher_or_metadata_value(args: Any, sample: Sample, teacher_data: dict[str, Any], cfg_name: str, default_keys: tuple[str, ...]) -> Any:
    keys = _list_value(_cfg(args, cfg_name, None), default_keys)
    value = _metadata_value(sample, keys)
    if value not in (None, "", []):
        return value
    for key in keys:
        value = teacher_data.get(key)
        if value not in (None, "", []):
            return value
    return None


def _existing_rubric(args: Any, sample: Sample, teacher_data: dict[str, Any]) -> Any:
    return _teacher_or_metadata_value(
        args,
        sample,
        teacher_data,
        "rubric_keys",
        ("rubric", "reward_rubric", "ropd_rubric"),
    )


def _rubric_prompt(args: Any, sample: Sample, teacher_data: dict[str, Any]) -> str:
    payload = {
        "task": truncate(task_prompt(sample), 5000),
        "reference_answers": [truncate(ref, 2000) for ref in reference_values(sample)],
        "teacher_answer": truncate(
            _teacher_or_metadata_value(
                args,
                sample,
                teacher_data,
                "teacher_answer_keys",
                ("teacher_response", "teacher_answer", "teacher_final_answer"),
            ),
            4000,
        ),
        "student_answer": truncate(
            _teacher_or_metadata_value(
                args,
                sample,
                teacher_data,
                "student_answer_keys",
                ("student_response", "student_answer", "student_final_answer"),
            ),
            4000,
        ),
    }
    return (
        "Create a grading rubric for this task. Output exactly one minified JSON object "
        'like {"rubric":["criterion"],"reason":"short reason"}.\n\n'
        + json.dumps(payload, ensure_ascii=False)
    )


def _judge_prompt(sample: Sample, rubric: Any) -> str:
    payload = {
        "task": truncate(task_prompt(sample), 5000),
        "rubric": rubric,
        "reference_answers": [truncate(ref, 2000) for ref in reference_values(sample)],
        "candidate_answer": truncate(prediction_text(sample), 5000),
    }
    return (
        "Grade the candidate answer using the rubric. Output exactly one minified JSON object "
        'like {"score":0.0,"reason":"short reason"} with score in [0,1].\n\n'
        + json.dumps(payload, ensure_ascii=False)
    )


def _weight(args: Any) -> float:
    raw = runtime_env(args, "AGENT_ENV_ROPD_TASK_SUCCESS_WEIGHT", "")
    if raw == "":
        raw = _cfg(args, "task_success_weight", reward_cfg_path(args, "outcome", 10.0))
    return float_value(raw, 10.0)


async def _rubric_for_sample(args: Any, sample: Sample, cache: dict[str, Any]) -> tuple[Any, str, dict[str, Any] | None]:
    teacher_data = _teacher_data(args, sample)
    existing = _existing_rubric(args, sample, teacher_data)
    if existing:
        return existing, "teacher", None
    key = _cache_key(sample)
    if key in cache:
        return cache[key], "cache", None
    allow_online_raw = runtime_env(args, "AGENT_ENV_ROPD_ALLOW_ONLINE_RUBRIC", "")
    if allow_online_raw == "":
        allow_online_raw = _cfg(args, "allow_online_rubric", False)
    allow_online = str(allow_online_raw).strip().lower() in {"1", "true", "yes", "on"}
    if not allow_online or judge_mode(args) != "aux":
        return None, "missing", None
    payload, call_metadata = await call_json_judge_with_metadata(
        args,
        _rubric_prompt(args, sample, teacher_data),
        system_prompt=RUBRIC_SYSTEM_PROMPT,
        api_key_path=_role_api_key_path(args, "rubric"),
        **_role_endpoint(args, "rubric"),
    )
    rubric = payload.get("rubric") if isinstance(payload, dict) else payload
    cache[key] = rubric
    return rubric, "online", call_metadata


async def score(args: Any, samples: list[Sample], *, single: bool = False) -> list[RewardResult]:
    if judge_mode(args) != "aux":
        fallback_results = await naive.score(args, samples, single=single)
        for result in fallback_results:
            result.reward_version = "ropd_v1_fallback_naive"
        return fallback_results

    cache_path = _cache_path(args)
    cache = _load_cache(cache_path)
    changed = False
    results: list[RewardResult] = []
    for sample in samples:
        rubric, rubric_source, rubric_call = await _rubric_for_sample(args, sample, cache)
        if rubric is None:
            results.append(
                RewardResult(
                    score=0.0,
                    components={"rubric_task_success": 0.0},
                    raw={
                        "fallback": "missing_rubric",
                        "rubric_source": rubric_source,
                        "join_key": _join_key(args),
                        "join_value": _join_value(args, sample),
                    },
                    reason="missing_rubric",
                    returns_total=True,
                    reward_version="ropd_v1_missing_rubric",
                )
            )
            continue
        changed = changed or rubric_source == "online"
        payload, judge_call = await call_json_judge_with_metadata(
            args,
            _judge_prompt(sample, rubric),
            system_prompt=JUDGE_SYSTEM_PROMPT,
            api_key_path=_role_api_key_path(args, "judge"),
            **_role_endpoint(args, "judge"),
        )
        value, item = parse_single_score(payload)
        bounded = max(0.0, min(1.0, float(value)))
        weighted = bounded * _weight(args)
        results.append(
            RewardResult(
                score=weighted,
                components={"rubric_task_success": weighted},
                raw={
                    "rubric": rubric,
                    "rubric_source": rubric_source,
                    "rubric_call": rubric_call,
                    "judge": item,
                    "judge_call": judge_call,
                },
                reason=str(item.get("reason") or "") if isinstance(item, dict) else "",
                returns_total=True,
                reward_version="ropd_v1",
            )
        )
    if changed:
        _save_cache(cache_path, cache)
    return results
