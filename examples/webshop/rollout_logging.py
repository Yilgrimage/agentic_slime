from __future__ import annotations

from typing import Any

from slime.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step


def _environment_metrics(samples: list[Any]) -> dict[str, float]:
    if not samples:
        return {}

    turn_counts = []
    format_error_counts = []
    success_count = 0
    env_rewards = []
    token_rewards = []

    for sample in samples:
        metadata = sample.metadata or {}
        turn_count = int(metadata.get("turn_count", 0) or 0)
        format_errors = int(metadata.get("format_errors", 0) or 0)
        turn_counts.append(turn_count)
        format_error_counts.append(format_errors)
        success_count += int(bool(metadata.get("env_success", False)))
        if "env_reward" in metadata:
            env_rewards.append(float(metadata["env_reward"]))
        if "token_rewards" in metadata:
            token_rewards.append(float(sum(metadata["token_rewards"])))

    total_turns = sum(turn_counts)
    total_format_errors = sum(format_error_counts)
    metrics = {
        "webshop/format_error_rate": total_format_errors / total_turns if total_turns else 0.0,
        "webshop/format_error_per_sample": total_format_errors / len(samples),
        "webshop/success_rate": success_count / len(samples),
        "webshop/turn_count_mean": total_turns / len(samples),
    }
    if env_rewards:
        metrics["webshop/env_reward_mean"] = sum(env_rewards) / len(env_rewards)
    if token_rewards:
        metrics["webshop/token_reward_mean"] = sum(token_rewards) / len(token_rewards)
    return metrics


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= _environment_metrics(samples)
    log_dict |= {f"rollout/{k}": v for k, v in compute_metrics_from_samples(args, samples).items()}
    log_dict |= {f"perf/{k}": v for k, v in compute_perf_metrics_from_samples(args, samples, rollout_time).items()}
    log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    if extra_metrics is None:
        return False
    for name, info in data.items():
        samples = info.get("samples") or []
        for key, value in _environment_metrics(samples).items():
            extra_metrics[f"eval/{name}/{key.removeprefix('webshop/')}"] = value
    return False
