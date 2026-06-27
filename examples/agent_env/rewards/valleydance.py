from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from examples.agent_env.rollout import cfg_path

from . import naive
from .config import reward_cfg_path
from .extractors import (
    bool_value,
    expected_skills,
    expected_tools,
    float_value,
    int_value,
    metadata,
    ratio_score,
    reference_values,
    runtime_env,
    text_for_error_scan,
    used_skills,
    used_tools,
)
from .llm_client import judge_mode
from .types import RewardResult


def _weight(args: Any, env_name: str, cfg_names: str | tuple[str, ...], default: float) -> float:
    value = runtime_env(args, env_name, "")
    if value != "":
        return float_value(value, default)
    names = (cfg_names,) if isinstance(cfg_names, str) else cfg_names
    for cfg_name in names:
        cfg_value = reward_cfg_path(args, cfg_name, None)
        if cfg_value is not None:
            return float_value(cfg_value, default)
    return default


def task_success_weight(args: Any) -> float:
    value = reward_cfg_path(args, "valleydance.task_success_weight", None)
    if value is None:
        value = reward_cfg_path(args, "outcome", 10.0)
    return float_value(value, 10.0)


def skill_weight(args: Any) -> float:
    return _weight(args, "VALLEYDANCE_SKILL_WEIGHT", ("valleydance.skill_weight", "skill_usage.weight"), 1.0)


def tool_weight(args: Any) -> float:
    return _weight(args, "VALLEYDANCE_TOOL_WEIGHT", ("valleydance.tool_weight", "tool_usage.weight"), 0.0)


def error_penalty_weight(args: Any) -> float:
    return abs(
        _weight(
            args,
            "VALLEYDANCE_ERROR_PATTERN_PENALTY",
            ("valleydance.error_pattern_penalty", "error_pattern.penalty"),
            0.0,
        )
    )


def turn_excess_penalty_weight(args: Any) -> float:
    return abs(
        _weight(
            args,
            "VALLEYDANCE_TURN_EXCESS_PENALTY",
            ("valleydance.turn_excess_penalty", "turn.excess_penalty"),
            0.0,
        )
    )


def max_turns(args: Any) -> int:
    cfg_value = reward_cfg_path(args, "valleydance.max_turns", None)
    if cfg_value is None:
        cfg_value = reward_cfg_path(args, "turn.max_turns", None)
    if cfg_value is None:
        cfg_value = cfg_path(args, "max_turns", 20)
    return int_value(runtime_env(args, "VALLEYDANCE_MAX_TURNS", str(cfg_value)), 20)


def configured_error_patterns(args: Any) -> list[str]:
    raw = runtime_env(args, "VALLEYDANCE_ERROR_PATTERNS_JSON", "").strip()
    if raw:
        try:
            import json

            value = json.loads(raw)
            if isinstance(value, list):
                return [str(item) for item in value if str(item)]
        except Exception:
            pass
    cfg_value = reward_cfg_path(args, "valleydance.error_patterns", None)
    if cfg_value is None:
        cfg_value = reward_cfg_path(args, "error_patterns", None)
    if isinstance(cfg_value, list):
        return [str(item) for item in cfg_value if str(item)]
    raw = runtime_env(args, "VALLEYDANCE_ERROR_PATTERNS", "").strip()
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return []


def error_pattern_count(args: Any, sample: Sample) -> int:
    import re

    text = text_for_error_scan(sample)
    count = 0
    for pattern in configured_error_patterns(args):
        try:
            count += len(re.findall(pattern, text, flags=re.IGNORECASE))
        except re.error:
            count += text.lower().count(pattern.lower())
    return count


def process_components(args: Any, sample: Sample) -> dict[str, float]:
    skills_expected = expected_skills(sample)
    tools_expected = expected_tools(sample)
    skill_ratio = ratio_score(used_skills(sample), skills_expected)
    tool_ratio = ratio_score(used_tools(sample), tools_expected)

    turn_count = int_value(metadata(sample).get("turn_count"), 0)
    excess_turns = max(0, turn_count - max_turns(args))
    turn_penalty = -turn_excess_penalty_weight(args) * excess_turns

    err_count = error_pattern_count(args, sample)
    return {
        "skill_usage": skill_ratio * skill_weight(args),
        "tool_usage": tool_ratio * tool_weight(args),
        "turn_penalty": turn_penalty,
        "error_penalty": -error_penalty_weight(args) * err_count,
    }


async def score(args: Any, samples: list[Sample], *, single: bool = False) -> list[RewardResult]:
    use_llm_raw = runtime_env(args, "VALLEYDANCE_USE_LLM_JUDGE", "")
    if use_llm_raw == "":
        use_llm_raw = reward_cfg_path(args, "valleydance.use_llm_judge", True)
    use_llm = bool_value(use_llm_raw, True)
    can_call_llm = use_llm and judge_mode(args) == "aux"
    task_results: list[RewardResult]
    if can_call_llm and any(reference_values(sample) for sample in samples):
        task_results = await naive.score(args, samples, single=single)
    else:
        task_results = [naive._fallback_result(args, sample, "llm_judge_disabled_or_missing_reference") for sample in samples]

    output = []
    for sample, task_result in zip(samples, task_results, strict=True):
        components = dict(task_result.components)
        components.update(process_components(args, sample))
        total = sum(components.values())
        output.append(
            RewardResult(
                score=total,
                components=components,
                raw=task_result.raw,
                reason=task_result.reason,
                returns_total=True,
                reward_version="valleydance_v2",
            )
        )
    return output
