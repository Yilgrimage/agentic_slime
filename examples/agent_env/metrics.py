from __future__ import annotations

from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", []):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reward_call_metrics(
    metrics: dict[str, float],
    *,
    prefix: str,
    role: str,
    calls: list[dict[str, Any]],
    sample_count: int,
) -> None:
    if not calls:
        return
    metrics[f"{prefix}/reward/llm/{role}_call_rate"] = len(calls) / sample_count if sample_count else 0.0
    metrics[f"{prefix}/reward/llm/{role}_latency_s_mean"] = _mean(
        [value for call in calls if (value := _float_or_none(call.get("latency_s"))) is not None]
    )
    metrics[f"{prefix}/reward/llm/{role}_prompt_chars_mean"] = _mean(
        [value for call in calls if (value := _float_or_none(call.get("prompt_chars"))) is not None]
    )
    for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [value for call in calls if (value := _float_or_none(call.get(token_key))) is not None]
        if values:
            metrics[f"{prefix}/reward/llm/{role}_{token_key}_mean"] = _mean(values)


def reward_metrics(samples: list[Any], *, prefix: str) -> dict[str, float]:
    if not samples:
        return {}

    real_samples = [
        sample
        for sample in samples
        if not bool((getattr(sample, "metadata", None) or {}).get("glm_padding_duplicate"))
    ]
    if not real_samples:
        real_samples = samples

    rm_scores = []
    reward_versions: dict[str, int] = {}
    rm_impls: dict[str, int] = {}
    component_values: dict[str, list[float]] = {}
    fallback_reasons: dict[str, int] = {}
    rubric_sources: dict[str, int] = {}
    judge_calls: list[dict[str, Any]] = []
    rubric_calls: list[dict[str, Any]] = []

    for sample in real_samples:
        sample_metadata = getattr(sample, "metadata", None) or {}
        impl = sample_metadata.get("rm_impl")
        if impl:
            rm_impls[str(impl)] = rm_impls.get(str(impl), 0) + 1
        rm_reward = _as_dict(sample_metadata.get("rm_reward"))
        score = _float_or_none(rm_reward.get("score"))
        if score is not None:
            rm_scores.append(score)
        version = rm_reward.get("reward_version")
        if version:
            reward_versions[str(version)] = reward_versions.get(str(version), 0) + 1
        for key, value in _as_dict(rm_reward.get("components")).items():
            scalar = _float_or_none(value)
            if scalar is not None:
                component_values.setdefault(str(key), []).append(scalar)
        raw = _as_dict(rm_reward.get("raw"))
        fallback = raw.get("fallback")
        if fallback:
            fallback_reasons[str(fallback)] = fallback_reasons.get(str(fallback), 0) + 1
        rubric_source = raw.get("rubric_source")
        if rubric_source:
            rubric_sources[str(rubric_source)] = rubric_sources.get(str(rubric_source), 0) + 1
        judge_call = _as_dict(raw.get("judge_call"))
        if judge_call:
            judge_calls.append(judge_call)
        rubric_call = _as_dict(raw.get("rubric_call"))
        if rubric_call:
            rubric_calls.append(rubric_call)

    total = len(real_samples)
    metrics: dict[str, float] = {}
    if rm_scores:
        metrics[f"{prefix}/reward/rm_score_mean"] = _mean(rm_scores)
        metrics[f"{prefix}/reward/rm_score_nonzero_rate"] = sum(1 for value in rm_scores if value != 0.0) / len(rm_scores)
    for impl, count in rm_impls.items():
        metrics[f"{prefix}/reward/impl_{impl}_rate"] = count / total
    for version, count in reward_versions.items():
        metrics[f"{prefix}/reward/version_{version}_rate"] = count / total
    for name, values in component_values.items():
        metrics[f"{prefix}/reward/component_{name}_mean"] = _mean(values)
    for reason, count in fallback_reasons.items():
        metrics[f"{prefix}/reward/fallback_{reason}_rate"] = count / total
    for source, count in rubric_sources.items():
        metrics[f"{prefix}/reward/ropd/rubric_source_{source}_rate"] = count / total
    if rubric_sources:
        metrics[f"{prefix}/reward/ropd/rubric_count"] = float(sum(rubric_sources.values()))
    _reward_call_metrics(metrics, prefix=prefix, role="judge", calls=judge_calls, sample_count=total)
    _reward_call_metrics(metrics, prefix=prefix, role="rubric", calls=rubric_calls, sample_count=total)
    return metrics


def environment_metrics(samples: list[Any], *, prefix: str) -> dict[str, float]:
    if not samples:
        return {}

    real_samples = [
        sample
        for sample in samples
        if not bool((getattr(sample, "metadata", None) or {}).get("glm_padding_duplicate"))
    ]
    if not real_samples:
        real_samples = samples

    turn_counts = []
    format_error_counts = []
    max_response_tokens_hit_counts = []
    success_count = 0
    env_rewards = []
    truncated_reasons: dict[str, int] = {}
    discard_reasons: dict[str, int] = {}

    for sample in real_samples:
        metadata = sample.metadata or {}
        turn_count = int(metadata.get("turn_count", 0) or 0)
        format_errors = int(metadata.get("format_errors", 0) or 0)
        max_response_tokens_hits = int(metadata.get("max_response_tokens_hits", 0) or 0)
        turn_counts.append(turn_count)
        format_error_counts.append(format_errors)
        max_response_tokens_hit_counts.append(max_response_tokens_hits)
        success_count += int(bool(metadata.get("env_success", False)))
        if "env_reward" in metadata:
            env_rewards.append(float(metadata["env_reward"]))
        reason = metadata.get("truncated_reason")
        if reason:
            truncated_reasons[str(reason)] = truncated_reasons.get(str(reason), 0) + 1
        discard_reason = metadata.get("discard_reason") if bool(getattr(sample, "remove_sample", False)) else None
        if discard_reason:
            discard_reasons[str(discard_reason)] = discard_reasons.get(str(discard_reason), 0) + 1

    total_turns = sum(turn_counts)
    total_format_errors = sum(format_error_counts)
    total_max_response_tokens_hits = sum(max_response_tokens_hit_counts)
    total_discards = sum(discard_reasons.values())
    metrics = {
        f"{prefix}/format_error_rate": total_format_errors / total_turns if total_turns else 0.0,
        f"{prefix}/max_response_tokens_hit_rate": total_max_response_tokens_hits / total_turns if total_turns else 0.0,
        f"{prefix}/discard_sample_rate": total_discards / len(real_samples),
        f"{prefix}/success_rate": success_count / len(real_samples),
        f"{prefix}/turn_count_mean": total_turns / len(real_samples),
    }
    if env_rewards:
        metrics[f"{prefix}/env_reward_mean"] = sum(env_rewards) / len(env_rewards)
    for reason, count in truncated_reasons.items():
        metrics[f"{prefix}/truncated_{reason}_rate"] = count / total_turns if total_turns else 0.0
    for reason, count in discard_reasons.items():
        metrics[f"{prefix}/discard_{reason}_rate"] = count / len(real_samples)
    return metrics


def log_rollout_data_for_env(prefix: str, rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    from slime.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= environment_metrics(samples, prefix=prefix)
    log_dict |= reward_metrics(samples, prefix=prefix)
    log_dict |= {f"rollout/{k}": v for k, v in compute_metrics_from_samples(args, samples).items()}
    log_dict |= {f"perf/{k}": v for k, v in compute_perf_metrics_from_samples(args, samples, rollout_time).items()}
    log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True


def log_eval_rollout_data_for_env(prefix: str, rollout_id, args, data, extra_metrics) -> bool:
    if extra_metrics is None:
        return False
    for name, info in data.items():
        samples = info.get("samples") or []
        for key, value in environment_metrics(samples, prefix=prefix).items():
            extra_metrics[f"eval/{name}/{key.removeprefix(prefix + '/')}"] = value
        for key, value in reward_metrics(samples, prefix=prefix).items():
            extra_metrics[f"eval/{name}/{key.removeprefix(prefix + '/')}"] = value
    return False
