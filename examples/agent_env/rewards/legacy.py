from __future__ import annotations

import json
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.rollout import format_reward_adjustment, truncated_reward_adjustment

from .extractors import float_value, metadata, sample_payload
from .llm_client import call_json_judge, judge_mode, parse_scores, parse_single_score
from .types import RewardResult

SYSTEM_PROMPT = (
    "You are a reward judge for text-action agent trajectories. "
    "Score each sample independently from the trajectory summary and environment name. "
    "Reward valid actions that make progress toward the task. Penalize invalid actions, "
    "repeated loops, format errors, and trajectories that do not solve or approach the task. "
    "Follow the requested JSON schema exactly and use scores in [-1, 1]. "
    "Do not include reasoning, markdown, code fences, or any text outside the JSON object."
)


def _prompt(args: Any, samples: list[Sample], *, single: bool) -> str:
    if single:
        payload = {
            "env_name": getattr(args, "env_name", None),
            "sample": sample_payload(samples[0]),
        }
        return (
            'Score this trajectory. Output exactly one minified JSON object: '
            '{"score":0.0,"reason":"short reason"}\n\n'
            + json.dumps(payload, ensure_ascii=False)
        )
    payload = {
        "env_name": getattr(args, "env_name", None),
        "samples": [sample_payload(sample) for sample in samples],
    }
    return (
        "Score this group of trajectories. Output exactly one minified JSON object: "
        '{"scores":[{"score":0.0,"reason":"short reason"}]} with one score object per sample '
        "in the same order.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _env_reward(sample: Sample) -> float:
    return float_value(metadata(sample).get("env_reward"), 0.0)


def _combined_result(args: Any, sample: Sample, judge_score: float, raw: Any, reason: str = "") -> RewardResult:
    components = {
        "env_reward": _env_reward(sample),
        "judge": float(judge_score),
        "format": format_reward_adjustment(args, sample),
        "truncated": truncated_reward_adjustment(args, sample),
    }
    return RewardResult(
        score=sum(components.values()),
        components=components,
        raw=raw,
        reason=reason,
        returns_total=True,
        reward_version="legacy_combined_v1",
    )


async def score(args: Any, samples: list[Sample], *, single: bool = False) -> list[RewardResult]:
    if judge_mode(args) == "none":
        return [_combined_result(args, sample, 0.0, {"skipped": "judge_disabled"}) for sample in samples]
    payload = await call_json_judge(args, _prompt(args, samples, single=single), system_prompt=SYSTEM_PROMPT)
    if single:
        value, item = parse_single_score(payload)
        reason = str(item.get("reason") or "") if isinstance(item, dict) else ""
        return [_combined_result(args, samples[0], value, item, reason)]
    values, items = parse_scores(payload, len(samples))
    return [
        _combined_result(
            args,
            sample,
            value,
            item,
            str(item.get("reason") or "") if isinstance(item, dict) else "",
        )
        for sample, value, item in zip(samples, values, items, strict=True)
    ]
