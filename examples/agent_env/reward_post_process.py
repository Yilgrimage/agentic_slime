from __future__ import annotations

import math
from typing import Any

from slime.utils.types import Sample
from slime.rollout.filter_hub.base_types import DynamicFilterOutput

from examples.agent_env.rollout import (
    arg,
    format_reward_adjustment,
    metadata,
    truncated_reward_adjustment,
)


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_reward(args: Any, sample: Sample) -> float:
    sample_metadata = metadata(sample)
    if "env_reward" in sample_metadata:
        return _float_value(sample_metadata.get("env_reward"))
    return 0.0


def _judge_score(args: Any, sample: Sample) -> float:
    if sample.reward is None:
        return 0.0
    if isinstance(sample.reward, dict):
        reward_key = arg(args, "reward_key", None)
        if reward_key and reward_key in sample.reward:
            return _float_value(sample.reward.get(reward_key))
        for key in ("judge_score", "score", "reward"):
            if key in sample.reward:
                return _float_value(sample.reward.get(key))
        return 0.0
    return _float_value(sample.reward)


def post_process_rewards(args: Any, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Agent-env reward combiner.

    Rollout owns environment interaction and stores env score/reward metadata.
    Group RM, when enabled, returns a judge score through sample.reward. This
    function is the only place that combines those sources into train rewards.
    """
    raw_rewards = []
    for sample in samples:
        sample_metadata = metadata(sample)
        if sample.remove_sample:
            env_reward = 0.0
            judge_score = 0.0
            format_adjustment = 0.0
            truncated_adjustment = 0.0
        else:
            env_reward = _env_reward(args, sample)
            judge_score = _judge_score(args, sample)
            format_adjustment = format_reward_adjustment(args, sample)
            truncated_adjustment = truncated_reward_adjustment(args, sample)
        raw_reward = env_reward + judge_score + format_adjustment + truncated_adjustment
        sample_metadata["env_reward_for_train"] = env_reward
        sample_metadata["judge_score_for_train"] = judge_score
        sample_metadata["format_reward"] = format_adjustment
        sample_metadata["truncated_reward"] = truncated_adjustment
        sample_metadata["raw_reward"] = raw_reward
        raw_rewards.append(raw_reward)

    if not (
        arg(args, "advantage_estimator", None) in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and bool(arg(args, "rewards_normalization", False))
    ):
        return raw_rewards, list(raw_rewards)

    rewards = [0.0] * len(samples)
    n_samples = max(1, int(arg(args, "n_samples_per_prompt", 1) or 1))
    grouped: dict[int, list[int]] = {}
    for idx, sample in enumerate(samples):
        group_key = int(sample.group_index) if sample.group_index is not None else idx // n_samples
        grouped.setdefault(group_key, []).append(idx)

    use_std = arg(args, "advantage_estimator", None) in ["grpo", "gspo"] and bool(
        arg(args, "grpo_std_normalization", False)
    )
    for indices in grouped.values():
        active = [idx for idx in indices if not samples[idx].remove_sample]
        if not active:
            continue
        values = [raw_rewards[idx] for idx in active]
        mean = sum(values) / len(values)
        centered = [value - mean for value in values]
        if use_std:
            if len(values) > 1:
                variance = sum(value * value for value in centered) / (len(values) - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0
            centered = [value / (std + 1e-6) for value in centered]
        for idx, value in zip(active, centered, strict=True):
            rewards[idx] = value
    return raw_rewards, rewards


def raw_reward_for_filter(args: Any, sample: Sample) -> float:
    if sample.remove_sample:
        return 0.0
    return (
        _env_reward(args, sample)
        + _judge_score(args, sample)
        + format_reward_adjustment(args, sample)
        + truncated_reward_adjustment(args, sample)
    )


def check_reward_nonzero_std(args: Any, samples: list[Sample], **_: Any) -> DynamicFilterOutput:
    active = [sample for sample in samples if not sample.remove_sample]
    if not active:
        return DynamicFilterOutput(keep=False, reason="no_active_samples")
    rewards = [raw_reward_for_filter(args, sample) for sample in active]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / max(1, len(rewards) - 1)
    keep = math.sqrt(variance) > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
