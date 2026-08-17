from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

from examples.agent_env import credit_assignment, dump
from examples.agent_env.rollout import _is_hard_discard_sample, arg, metadata


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _runtime_bool(args: Any, name: str, default: bool) -> bool:
    value = _runtime_env(args, name, "1" if default else "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
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


def check_reward_nonzero_std(args: Any, samples: list[Sample], **_: Any) -> DynamicFilterOutput:
    """Drop groups that cannot produce useful GRPO signal or safe GLM padding."""

    active = [sample for sample in samples if not _is_hard_discard_sample(sample)]
    if not active:
        return DynamicFilterOutput(keep=False, reason="no_active_samples")

    group_size = int(arg(args, "n_samples_per_prompt", len(samples)) or len(samples))
    min_valid_fraction = _float_value(_runtime_env(args, "AGENT_ENV_GLM_PADDING_MIN_VALID_FRACTION", "0.5"), 0.5)
    if len(active) <= group_size * min_valid_fraction:
        return DynamicFilterOutput(keep=False, reason="too_few_active_samples")

    if not _runtime_bool(args, "AGENT_ENV_DYNAMIC_DROP_ZERO_STD", True):
        return DynamicFilterOutput(keep=True, reason=None)

    rewards = [_reward_value(args, sample) for sample in active]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / max(1, len(rewards) - 1)
    keep = math.sqrt(variance) > 1e-6
    if not keep and credit_assignment.group_has_process_credit_signal(args, active):
        return DynamicFilterOutput(keep=True, reason=None)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else "zero_std",
    )


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
        credit_assignment.attach_process_advantages(args, samples, scalar_rewards=raw_rewards)
        rewards = list(raw_rewards)
        _maybe_dump_reward_groups(args, samples, raw_rewards=raw_rewards, normalized_rewards=rewards)
        return raw_rewards, rewards

    rewards = [0.0] * len(samples)
    n_samples = max(1, int(arg(args, "n_samples_per_prompt", 1) or 1))
    grouped = _group_indices(samples, n_samples)

    use_std = arg(args, "advantage_estimator", None) in ["grpo", "gspo"] and bool(
        arg(args, "grpo_std_normalization", False)
    )
    for indices in grouped.values():
        active = [idx for idx in indices if not samples[idx].remove_sample]
        if not active:
            continue
        # If an off-policy teacher sample is inserted into the GRPO group, it
        # participates in the same advantage normalization as student samples.
        # Its additional Luffy objective is isolated by off_policy_loss_mask.
        normalizer = active
        normalizer_values = [raw_rewards[idx] for idx in normalizer]
        mean = sum(normalizer_values) / len(normalizer_values)
        centered = [raw_rewards[idx] - mean for idx in active]
        if use_std:
            if len(normalizer_values) > 1:
                normalizer_centered = [value - mean for value in normalizer_values]
                variance = sum(value * value for value in normalizer_centered) / (len(normalizer_values) - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0
            centered = [value / (std + 1e-6) for value in centered]
        for idx, value in zip(active, centered, strict=True):
            rewards[idx] = value
    credit_assignment.attach_process_advantages(args, samples, scalar_rewards=raw_rewards)
    _maybe_dump_reward_groups(args, samples, raw_rewards=raw_rewards, normalized_rewards=rewards, grouped=grouped)
    return raw_rewards, rewards


def _group_indices(samples: list[Sample], n_samples: int) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for idx, sample in enumerate(samples):
        group_key = int(sample.group_index) if sample.group_index is not None else idx // n_samples
        grouped.setdefault(group_key, []).append(idx)
    return grouped


def _maybe_dump_reward_groups(
    args: Any,
    samples: list[Sample],
    *,
    raw_rewards: list[float],
    normalized_rewards: list[float],
    grouped: dict[int, list[int]] | None = None,
) -> None:
    per_step_limit = dump.int_runtime_env(args, "AGENT_ENV_REWARD_GROUP_DUMP_N", "0")
    if per_step_limit <= 0:
        return
    total_limit = dump.int_runtime_env(args, "AGENT_ENV_REWARD_GROUP_DUMP_TOTAL_N", "0")
    n_samples = max(1, int(arg(args, "n_samples_per_prompt", 1) or 1))
    grouped = grouped or _group_indices(samples, n_samples)
    out_dir = _reward_group_dump_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reward_group_pid{os.getpid()}.jsonl"
    with out_path.open("a", encoding="utf-8") as handle:
        for group_key, indices in grouped.items():
            dump_step = _group_dump_step(samples, indices)
            slot = dump.reserve_dump_slot(
                namespace="reward_group",
                stage="group",
                dump_step=dump_step,
                per_step_limit=per_step_limit,
                total_limit=total_limit,
            )
            if slot is None:
                continue
            index_in_step, total_index = slot
            active = [idx for idx in indices if not samples[idx].remove_sample]
            active_raw = [raw_rewards[idx] for idx in active]
            active_normalized = [normalized_rewards[idx] for idx in active]
            payload = {
                "schema_version": "agent_env.reward_group_audit.v1",
                "time": time.time(),
                "pid": os.getpid(),
                "dump_step": dump_step,
                "index_in_step": index_in_step,
                "index": total_index,
                "group_key": group_key,
                "sample_count": len(indices),
                "active_count": len(active),
                "removed_count": len(indices) - len(active),
                "rewards_normalization": bool(arg(args, "rewards_normalization", False)),
                "grpo_std_normalization": bool(arg(args, "grpo_std_normalization", False)),
                "raw_reward_stats": _number_stats(active_raw),
                "normalized_reward_stats": _number_stats(active_normalized),
                "samples": [
                    _reward_group_sample_record(
                        sample=samples[idx],
                        index=idx,
                        raw_reward=raw_rewards[idx],
                        normalized_reward=normalized_rewards[idx],
                    )
                    for idx in indices
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _reward_group_dump_dir(args: Any) -> Path:
    raw = dump.runtime_env(args, "AGENT_ENV_REWARD_GROUP_DUMP_DIR", "").strip()
    if raw:
        return Path(raw)
    run_root = dump.runtime_env(args, "RUN_ROOT", "").strip()
    if run_root:
        return Path(run_root) / "reward_artifacts" / "reward_groups"
    return Path(os.getcwd()) / "reward_artifacts" / "reward_groups"


def _group_dump_step(samples: list[Sample], indices: list[int]) -> str:
    for idx in indices:
        label = dump.sample_dump_step_label(samples[idx])
        if label != "unknown":
            return label
    return "unknown"


def _number_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    mean = sum(values) / len(values)
    variance = 0.0
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(max(variance, 0.0))
    nonzero = sum(1 for value in values if abs(value) > 1e-12)
    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "nonzero_rate": nonzero / len(values),
        "positive_rate": positive / len(values),
        "negative_rate": negative / len(values),
    }


def _reward_group_sample_record(
    *,
    sample: Sample,
    index: int,
    raw_reward: float,
    normalized_reward: float,
) -> dict[str, Any]:
    sample_metadata = metadata(sample)
    rm_reward = sample_metadata.get("rm_reward")
    env = sample_metadata.get("env")
    credit = sample_metadata.get("credit_assignment")
    return {
        "index": index,
        "sample_index": getattr(sample, "index", None),
        "rollout_id": getattr(sample, "rollout_id", None) or sample_metadata.get("rollout_id"),
        "group_index": getattr(sample, "group_index", None),
        "task_id": sample_metadata.get("task_id"),
        "task_ref": sample_metadata.get("task_ref"),
        "source_name": sample_metadata.get("source_name"),
        "remove_sample": bool(getattr(sample, "remove_sample", False)),
        "status": str(getattr(sample, "status", "")),
        "response_length": int(getattr(sample, "response_length", 0) or 0),
        "loss_mask_sum": _sum_loss_mask(getattr(sample, "loss_mask", None)),
        "raw_reward": raw_reward,
        "normalized_reward": normalized_reward,
        "env_success": _nested_value(env, "env_success", sample_metadata.get("env_success")),
        "env_score": _nested_value(env, "env_score", sample_metadata.get("env_score")),
        "env_reward": _nested_value(env, "env_reward", sample_metadata.get("env_reward")),
        "rm_score": _nested_value(rm_reward, "score", sample_metadata.get("rm_score")),
        "process": _process_summary(credit),
    }


def _nested_value(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict) and key in container:
        return container.get(key)
    return default


def _sum_loss_mask(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(sum(int(item) for item in value))
    except TypeError:
        return 0


def _process_summary(credit: Any) -> dict[str, Any] | None:
    if not isinstance(credit, dict):
        return None
    keys = (
        "enabled",
        "advantage_mode",
        "beta",
        "normalization",
        "nonzero_tokens",
        "positive_tokens",
        "negative_tokens",
        "response_length",
        "nonzero_token_rate",
        "positive_token_rate",
        "negative_token_rate",
        "process_value_mean",
        "process_value_abs_mean",
        "process_delta_mean",
        "process_delta_abs_mean",
    )
    return {key: credit.get(key) for key in keys if key in credit}
