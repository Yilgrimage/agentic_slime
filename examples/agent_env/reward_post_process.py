from __future__ import annotations

import math
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.rollout import arg, metadata


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reward_value(args: Any, sample: Sample) -> float:
    reward = sample.reward
    if isinstance(reward, dict):
        reward_key = arg(args, "reward_key", None)
        if reward_key and reward_key in reward:
            return _float_value(reward.get(reward_key))
        for key in ("score", "reward", "judge_score", "rm_score"):
            if key in reward:
                return _float_value(reward.get(key))
        return 0.0
    if reward is not None:
        return _float_value(reward)

    sample_metadata = metadata(sample)
    rm_reward = sample_metadata.get("rm_reward")
    if isinstance(rm_reward, dict) and "score" in rm_reward:
        return _float_value(rm_reward.get("score"))
    for key in ("rm_score", "judge_score"):
        if key in sample_metadata:
            return _float_value(sample_metadata.get(key))
    return 0.0


def post_process_rewards(args: Any, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Adapt RM-produced rewards to Slime's reward post-process contract.

    Environment scores, format rewards, truncation penalties, and judge rewards
    are composed by the selected RM implementation. This hook only handles
    removed samples and optional grouped reward normalization.
    """

    raw_rewards = []
    for sample in samples:
        raw_reward = 0.0 if sample.remove_sample else _reward_value(args, sample)
        sample_metadata = metadata(sample)
        sample_metadata["raw_reward"] = raw_reward
        sample_metadata["rm_reward_for_train"] = raw_reward
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
