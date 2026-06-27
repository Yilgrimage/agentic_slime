from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

MAX_PROMPT_CHARS = 800
MAX_RESPONSE_CHARS = 1200
MAX_TURNS = 8
MAX_TURN_TEXT_CHARS = 400
MAX_OBSERVATION_CHARS = 400
MAX_ACTION_CHARS = 160
MAX_ACTIONS = 40


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


def read_secret(path: str) -> str:
    secret_path = str(path or "").strip()
    if not secret_path:
        return ""
    try:
        return Path(secret_path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def metadata(sample: Sample) -> dict[str, Any]:
    if sample.metadata is None:
        sample.metadata = {}
    return sample.metadata


def truncate(value: Any, max_chars: int, *, tail: bool = False) -> str:
    if value is None:
        return ""
    text = str(value)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 16:
        return text[-max_chars:] if tail else text[:max_chars]
    if tail:
        return "... " + text[-(max_chars - 4) :]
    head = max(1, (max_chars - 5) // 2)
    tail_len = max(1, max_chars - 5 - head)
    return text[:head] + " ... " + text[-tail_len:]


def compact_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return truncate(value, max_chars)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return truncate(text, max_chars)


def compact_actions(actions: Any) -> list[Any]:
    if not isinstance(actions, list):
        return []
    selected = actions
    if len(actions) > MAX_ACTIONS:
        selected = actions[:1] + [{"omitted_middle_actions": len(actions) - MAX_ACTIONS}] + actions[-(MAX_ACTIONS - 2) :]
    return [compact_value(action, MAX_ACTION_CHARS) for action in selected]


def select_turns(turns: list[Any]) -> list[Any]:
    if len(turns) <= MAX_TURNS:
        return turns
    return turns[:1] + [{"omitted_middle_turns": len(turns) - MAX_TURNS}] + turns[-(MAX_TURNS - 2) :]


def compact_turn(turn: Any) -> dict[str, Any]:
    if not isinstance(turn, dict):
        return {"turn": compact_value(turn, MAX_TURN_TEXT_CHARS)}
    parser_text = turn.get("parser_text")
    response_text = parser_text if parser_text not in (None, "") else turn.get("response_text")
    compact = {
        "turn": turn.get("turn"),
        "finish_type": turn.get("finish_type"),
        "response_token_count": turn.get("response_token_count"),
        "parse_mode": turn.get("parse_mode"),
        "format_valid": turn.get("format_valid"),
        "action": compact_value(turn.get("action"), MAX_ACTION_CHARS),
        "response": truncate(response_text, MAX_TURN_TEXT_CHARS),
        "observation": truncate(turn.get("observation"), MAX_OBSERVATION_CHARS, tail=True),
        "done": turn.get("done"),
        "score": turn.get("score"),
        "success": turn.get("success"),
        "discard_reason": turn.get("discard_reason"),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [])}


def action_name(action: Any) -> str:
    if isinstance(action, dict):
        if action.get("type") == "tool_call":
            return str(action.get("name") or "").strip()
        return str(action.get("name") or action.get("tool") or action.get("tool_name") or "").strip()
    return str(action or "").strip()


def action_arguments(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    args = action.get("arguments") or action.get("args") or {}
    return args if isinstance(args, dict) else {}


def normalize_names(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        parts = re.split(r"[,;|\n]+", values)
        return {part.strip().lower() for part in parts if part.strip()}
    if isinstance(values, dict):
        values = values.keys()
    if isinstance(values, (list, tuple, set)):
        return {str(item).strip().lower() for item in values if str(item).strip()}
    return {str(values).strip().lower()} if str(values).strip() else set()


def metadata_values(sample: Sample, keys: tuple[str, ...]) -> Any:
    sample_metadata = metadata(sample)
    for key in keys:
        if key in sample_metadata and sample_metadata[key] not in (None, "", []):
            return sample_metadata[key]
    for nested_key in ("task", "task_ref", "label", "reference", "mcp_server", "env_metadata"):
        nested = sample_metadata.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in keys:
            if key in nested and nested[key] not in (None, "", []):
                return nested[key]
    return None


def reference_values(sample: Sample) -> list[str]:
    keys = (
        "reference",
        "references",
        "ground_truth",
        "gt",
        "target",
        "expected_answer",
        "gold",
        "label",
    )
    value = metadata_values(sample, keys)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _info_dict_from_turn(turn: Any) -> dict[str, Any]:
    if not isinstance(turn, dict):
        return {}
    env_step = turn.get("env_step")
    if isinstance(env_step, dict) and isinstance(env_step.get("info"), dict):
        return env_step["info"]
    return {}


def prediction_text(sample: Sample) -> str:
    sample_metadata = metadata(sample)
    for key in ("prediction", "model_prediction", "final_answer", "answer"):
        value = sample_metadata.get(key)
        if value not in (None, "", []):
            return str(value)
    env_eval = sample_metadata.get("env_evaluate")
    if isinstance(env_eval, dict):
        info = env_eval.get("info")
        if isinstance(info, dict) and info.get("final_answer") not in (None, ""):
            return str(info["final_answer"])
    turns = sample_metadata.get("turns")
    if isinstance(turns, list):
        for turn in reversed(turns):
            info = _info_dict_from_turn(turn)
            if info.get("final_answer") not in (None, ""):
                return str(info["final_answer"])
    return str(getattr(sample, "response", "") or "")


def task_prompt(sample: Sample) -> str:
    sample_metadata = metadata(sample)
    value = metadata_values(sample, ("query", "task_prompt", "instruction", "question", "prompt"))
    if value not in (None, "", []):
        return str(value)
    return str(getattr(sample, "prompt", "") or "")


def expected_skills(sample: Sample) -> set[str]:
    return normalize_names(
        metadata_values(
            sample,
            (
                "expected_skills",
                "required_skills",
                "target_skills",
                "gold_skills",
            ),
        )
    )


def expected_tools(sample: Sample) -> set[str]:
    return normalize_names(
        metadata_values(
            sample,
            (
                "expected_tools",
                "required_tools",
                "target_tools",
                "gold_tools",
            ),
        )
    )


def actions(sample: Sample) -> list[Any]:
    raw = metadata(sample).get("actions") or []
    return raw if isinstance(raw, list) else []


def used_tools(sample: Sample) -> set[str]:
    explicit = normalize_names(metadata_values(sample, ("used_tools", "tools_used", "hit_tools")))
    names = {action_name(action).lower() for action in actions(sample) if action_name(action)}
    return explicit | names


def used_skills(sample: Sample) -> set[str]:
    explicit = normalize_names(metadata_values(sample, ("used_skills", "skills_used", "hit_skills")))
    found = set(explicit)
    for action in actions(sample):
        args = action_arguments(action)
        for key in ("skill", "skill_name", "skill_id", "skills"):
            found |= normalize_names(args.get(key))
    response = str(getattr(sample, "response", "") or "")
    for match in re.finditer(r"skills?\s+used\s*:\s*([^\n]+)", response, flags=re.IGNORECASE):
        found |= normalize_names(match.group(1))
    return found


def ratio_score(used: set[str], expected: set[str]) -> float:
    if not expected:
        return 0.0
    return len(used & expected) / max(1, len(expected))


def env_score(sample: Sample) -> float:
    sample_metadata = metadata(sample)
    if "env_score" in sample_metadata:
        return float_value(sample_metadata.get("env_score"))
    if "env_reward" in sample_metadata:
        return float_value(sample_metadata.get("env_reward"))
    return 0.0


def status_name(sample: Sample) -> str:
    return str(getattr(getattr(sample, "status", None), "value", getattr(sample, "status", "")))


def sample_payload(sample: Sample) -> dict[str, Any]:
    sample_metadata = metadata(sample)
    turns = sample_metadata.get("turns") if isinstance(sample_metadata.get("turns"), list) else []
    refs = reference_values(sample)
    return {
        "index": sample.index,
        "group_index": sample.group_index,
        "status": status_name(sample),
        "remove_sample": bool(getattr(sample, "remove_sample", False)),
        "prompt": truncate(task_prompt(sample), MAX_PROMPT_CHARS),
        "response": truncate(getattr(sample, "response", ""), MAX_RESPONSE_CHARS, tail=True),
        "prediction": truncate(prediction_text(sample), MAX_RESPONSE_CHARS, tail=True),
        "references": [truncate(ref, MAX_RESPONSE_CHARS, tail=True) for ref in refs],
        "actions": compact_actions(sample_metadata.get("actions") or []),
        "turns": [
            turn if isinstance(turn, dict) and "omitted_middle_turns" in turn else compact_turn(turn)
            for turn in select_turns(turns)
        ],
        "expected_skills": sorted(expected_skills(sample)),
        "used_skills": sorted(used_skills(sample)),
        "expected_tools": sorted(expected_tools(sample)),
        "used_tools": sorted(used_tools(sample)),
        "env_score": sample_metadata.get("env_score"),
        "env_success": sample_metadata.get("env_success"),
        "env_reward": sample_metadata.get("env_reward"),
        "turn_count": sample_metadata.get("turn_count", len(sample_metadata.get("actions") or [])),
        "format_errors": sample_metadata.get("format_errors", 0),
        "max_response_tokens_hits": sample_metadata.get("max_response_tokens_hits", 0),
        "truncated_reason": sample_metadata.get("truncated_reason"),
        "discard_reason": sample_metadata.get("discard_reason"),
    }


def text_for_error_scan(sample: Sample) -> str:
    sample_metadata = metadata(sample)
    chunks = [str(getattr(sample, "response", "") or ""), str(sample_metadata.get("error") or "")]
    turns = sample_metadata.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                chunks.append(str(turn.get("response_text") or turn.get("parser_text") or ""))
                chunks.append(str(turn.get("observation") or ""))
    return "\n".join(chunk for chunk in chunks if chunk)


def record_reward_result(sample: Sample, impl_name: str, result: Any) -> None:
    sample_metadata = metadata(sample)
    record = result.record()
    sample_metadata["rm_impl"] = impl_name
    sample_metadata["rm_returns_total"] = bool(record.get("returns_total", True))
    sample_metadata["rm_reward"] = record
    sample_metadata["reward_components"] = record.get("components", {})
    sample_metadata["judge_score"] = float(record.get("score", 0.0))
    if record.get("raw") is not None:
        sample_metadata["judge_raw"] = record["raw"]
    if record.get("reason"):
        sample_metadata["judge_reason"] = record["reason"]
