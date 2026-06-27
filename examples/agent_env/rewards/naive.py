from __future__ import annotations

import json
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.rollout import cfg_path

from .config import reward_cfg_path
from .extractors import float_value, reference_values, sample_payload
from .llm_client import call_json_judge, judge_mode, parse_scores, parse_single_score
from .types import RewardResult

SYSTEM_PROMPT = (
    "You are a strict answer correctness judge. Compare the model prediction to the references. "
    "Use score 1 when the prediction is correct or semantically equivalent, 0 when incorrect, "
    "and partial credit only when the answer is materially incomplete but useful. "
    "Output JSON only."
)


def task_success_weight(args: Any) -> float:
    value = reward_cfg_path(args, "naive.task_success_weight", None)
    if value is None:
        value = reward_cfg_path(args, "outcome", 10.0)
    return float_value(value, 10.0)


def _fallback_result(args: Any, sample: Sample, reason: str) -> RewardResult:
    return RewardResult(
        score=0.0,
        components={"judge_task_success": 0.0},
        raw={"fallback": reason},
        reason=reason,
        returns_total=True,
        reward_version="naive_v1",
    )


def _prompt(args: Any, samples: list[Sample], *, single: bool) -> str:
    if single:
        payload = {"sample": sample_payload(samples[0])}
        return (
            'Judge answer correctness. Output exactly {"score":0.0,"reason":"short reason"} '
            "with score in [0,1].\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
    payload = {"samples": [sample_payload(sample) for sample in samples]}
    return (
        "Judge answer correctness for each sample. Output exactly one minified JSON object: "
        '{"scores":[{"score":0.0,"reason":"short reason"}]} with one score object per sample '
        "in the same order. Scores must be in [0,1].\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _result(args: Any, score: float, raw: Any) -> RewardResult:
    bounded = max(0.0, min(1.0, float(score)))
    weighted = bounded * task_success_weight(args)
    return RewardResult(
        score=weighted,
        components={"judge_task_success": weighted},
        raw=raw,
        reason=str(raw.get("reason") or "") if isinstance(raw, dict) else "",
        returns_total=True,
        reward_version="naive_v1",
    )


async def score(args: Any, samples: list[Sample], *, single: bool = False) -> list[RewardResult]:
    if judge_mode(args) == "none":
        return [_fallback_result(args, sample, "judge_disabled") for sample in samples]
    judged_indices = [idx for idx, sample in enumerate(samples) if reference_values(sample)]
    results: list[RewardResult | None] = [None] * len(samples)
    for idx, sample in enumerate(samples):
        if idx not in judged_indices:
            results[idx] = _fallback_result(args, sample, "missing_reference")
    if not judged_indices:
        return [item for item in results if item is not None]

    judged_samples = [samples[idx] for idx in judged_indices]
    payload = await call_json_judge(args, _prompt(args, judged_samples, single=single and len(judged_samples) == 1), system_prompt=SYSTEM_PROMPT)
    if single and len(judged_samples) == 1:
        value, item = parse_single_score(payload)
        results[judged_indices[0]] = _result(args, value, item)
    else:
        values, items = parse_scores(payload, len(judged_samples))
        for idx, value, item in zip(judged_indices, values, items, strict=True):
            results[idx] = _result(args, value, item)
    return [item if item is not None else _fallback_result(args, sample, "missing_result") for item, sample in zip(results, samples, strict=True)]
