from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import Any

logger = logging.getLogger(__name__)

_WANDB_DEFINED_STEP_METRICS: set[tuple[int, str, str, str]] = set()
_WANDB_METRIC_DEFINITION_WARNING_EMITTED = False
_GENERATED_ENV_PREFIX = "agent_env/generated/env/"
_GENERATED_REWARD_PREFIX = "agent_env/generated/reward/"


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


def _sum_numeric_dicts(values: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for item in values:
        for key, value in item.items():
            scalar = _float_or_none(value)
            if scalar is not None:
                totals[str(key)] = totals.get(str(key), 0.0) + scalar
    return totals


def _reward_call_metrics(
    metrics: dict[str, float],
    *,
    role: str,
    calls: list[dict[str, Any]],
    sample_count: int,
) -> None:
    if not calls:
        return
    metrics[f"reward/llm/{role}_call_rate"] = len(calls) / sample_count if sample_count else 0.0
    metrics[f"reward/llm/{role}_latency_s_mean"] = _mean(
        [value for call in calls if (value := _float_or_none(call.get("latency_s"))) is not None]
    )
    metrics[f"reward/llm/{role}_prompt_chars_mean"] = _mean(
        [value for call in calls if (value := _float_or_none(call.get("prompt_chars"))) is not None]
    )
    for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [value for call in calls if (value := _float_or_none(call.get(token_key))) is not None]
        if values:
            metrics[f"reward/llm/{role}_{token_key}_mean"] = _mean(values)


def reward_metrics(samples: list[Any]) -> dict[str, float]:
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
    fallback_reasons: dict[str, int] = {}
    rubric_sources: dict[str, int] = {}
    judge_calls: list[dict[str, Any]] = []
    rubric_calls: list[dict[str, Any]] = []
    ropd_student_points: list[float] = []
    ropd_teacher_points: list[float] = []
    ropd_answer_scores: list[float] = []
    ropd_teacher_answer_scores: list[float] = []
    ropd_rubric_scores: list[float] = []
    ropd_teacher_rubric_scores: list[float] = []
    ropd_train_scores: list[float] = []
    ropd_rubric_sizes: list[float] = []
    ropd_answer_rubric_sizes: list[float] = []
    ropd_process_rubric_sizes: list[float] = []
    ropd_answer_core_scores: list[float] = []
    ropd_answer_support_scores: list[float] = []
    ropd_process_scores: list[float] = []
    ropd_ca_train_scores: list[float] = []
    ropd_fatal_error_count = 0
    ropd_fatal_error_seen = False
    ropd_final_answer_quality: dict[str, int] = {}
    ropd_teacher_below_student = 0
    ropd_teacher_below_student_seen = False
    ropd_env_success_for_reward = 0
    ropd_env_success_for_reward_seen = False
    ca_nonzero_tokens: list[float] = []
    ca_nonzero_token_rates: list[float] = []
    ca_positive_token_rates: list[float] = []
    ca_negative_token_rates: list[float] = []
    ca_process_value_means: list[float] = []
    ca_process_value_abs_means: list[float] = []
    ca_process_delta_means: list[float] = []
    ca_process_delta_abs_means: list[float] = []
    ca_tasa_supported_token_rates: list[float] = []
    ca_tasa_unique_states: list[float] = []
    ca_tasa_state_reuse_rates: list[float] = []
    ca_tasa_supported_segment_rates: list[float] = []
    ca_tasa_peer_count_means: list[float] = []
    ca_tasa_state_reward_std_means: list[float] = []
    ca_tasa_local_outcome_corrs: list[float] = []
    ca_tasa_state_prior_means: list[float] = []
    ca_tasa_state_value_means: list[float] = []
    ca_tasa_td_delta_abs_means: list[float] = []
    ca_tasa_gae_abs_means: list[float] = []
    ca_tasa_prior_kappas: list[float] = []
    ca_tasa_lambdas: list[float] = []
    ca_tasa_teacher_weights: list[float] = []

    for sample in real_samples:
        sample_metadata = getattr(sample, "metadata", None) or {}
        ca = _as_dict(sample_metadata.get("credit_assignment"))
        if ca:
            nonzero_tokens = _float_or_none(ca.get("nonzero_tokens"))
            if nonzero_tokens is not None:
                ca_nonzero_tokens.append(nonzero_tokens)
            nonzero_rate = _float_or_none(ca.get("nonzero_token_rate"))
            if nonzero_rate is not None:
                ca_nonzero_token_rates.append(nonzero_rate)
            positive_rate = _float_or_none(ca.get("positive_token_rate"))
            if positive_rate is not None:
                ca_positive_token_rates.append(positive_rate)
            negative_rate = _float_or_none(ca.get("negative_token_rate"))
            if negative_rate is not None:
                ca_negative_token_rates.append(negative_rate)
            process_mean = _float_or_none(ca.get("process_value_mean"))
            if process_mean is not None:
                ca_process_value_means.append(process_mean)
            process_abs_mean = _float_or_none(ca.get("process_value_abs_mean"))
            if process_abs_mean is not None:
                ca_process_value_abs_means.append(process_abs_mean)
            process_delta_mean = _float_or_none(ca.get("process_delta_mean"))
            if process_delta_mean is not None:
                ca_process_delta_means.append(process_delta_mean)
            process_delta_abs_mean = _float_or_none(ca.get("process_delta_abs_mean"))
            if process_delta_abs_mean is not None:
                ca_process_delta_abs_means.append(process_delta_abs_mean)
            tasa_supported_token_rate = _float_or_none(ca.get("tasa_supported_token_rate"))
            if tasa_supported_token_rate is not None:
                ca_tasa_supported_token_rates.append(tasa_supported_token_rate)
            tasa_unique_states = _float_or_none(ca.get("tasa_group_unique_states"))
            if tasa_unique_states is not None:
                ca_tasa_unique_states.append(tasa_unique_states)
            tasa_state_reuse_rate = _float_or_none(ca.get("tasa_group_state_reuse_rate"))
            if tasa_state_reuse_rate is not None:
                ca_tasa_state_reuse_rates.append(tasa_state_reuse_rate)
            tasa_supported_segment_rate = _float_or_none(ca.get("tasa_group_supported_segment_rate"))
            if tasa_supported_segment_rate is not None:
                ca_tasa_supported_segment_rates.append(tasa_supported_segment_rate)
            tasa_peer_count_mean = _float_or_none(ca.get("tasa_group_peer_count_mean"))
            if tasa_peer_count_mean is not None:
                ca_tasa_peer_count_means.append(tasa_peer_count_mean)
            tasa_state_reward_std_mean = _float_or_none(ca.get("tasa_group_state_reward_std_mean"))
            if tasa_state_reward_std_mean is not None:
                ca_tasa_state_reward_std_means.append(tasa_state_reward_std_mean)
            tasa_local_outcome_corr = _float_or_none(ca.get("tasa_local_outcome_corr"))
            if tasa_local_outcome_corr is not None:
                ca_tasa_local_outcome_corrs.append(tasa_local_outcome_corr)
            tasa_state_prior_mean = _float_or_none(ca.get("tasa_state_prior_mean"))
            if tasa_state_prior_mean is not None:
                ca_tasa_state_prior_means.append(tasa_state_prior_mean)
            tasa_state_value_mean = _float_or_none(ca.get("tasa_state_value_mean"))
            if tasa_state_value_mean is not None:
                ca_tasa_state_value_means.append(tasa_state_value_mean)
            tasa_td_delta_abs_mean = _float_or_none(ca.get("tasa_td_delta_abs_mean"))
            if tasa_td_delta_abs_mean is not None:
                ca_tasa_td_delta_abs_means.append(tasa_td_delta_abs_mean)
            tasa_gae_abs_mean = _float_or_none(ca.get("tasa_gae_abs_mean"))
            if tasa_gae_abs_mean is not None:
                ca_tasa_gae_abs_means.append(tasa_gae_abs_mean)
            tasa_prior_kappa = _float_or_none(ca.get("tasa_prior_kappa"))
            if tasa_prior_kappa is not None:
                ca_tasa_prior_kappas.append(tasa_prior_kappa)
            tasa_lambda = _float_or_none(ca.get("tasa_lambda"))
            if tasa_lambda is not None:
                ca_tasa_lambdas.append(tasa_lambda)
            tasa_teacher_weight = _float_or_none(ca.get("tasa_teacher_weight"))
            if tasa_teacher_weight is not None:
                ca_tasa_teacher_weights.append(tasa_teacher_weight)
        rm_reward = _as_dict(sample_metadata.get("rm_reward"))
        score = _float_or_none(rm_reward.get("score"))
        if score is not None:
            rm_scores.append(score)
        raw = _as_dict(rm_reward.get("raw"))
        schema_mode = str(raw.get("ropd_schema_mode") or "")
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
        is_ca_compact = schema_mode == "ca_compact"
        student_score = _float_or_none(raw.get("student_score"))
        if student_score is not None and not is_ca_compact:
            ropd_student_points.append(student_score)
        answer_score = _float_or_none(raw.get("answer_score"))
        if answer_score is not None:
            ropd_answer_scores.append(answer_score)
        rubric_score = _float_or_none(raw.get("rubric_score"))
        if rubric_score is not None and not is_ca_compact:
            ropd_rubric_scores.append(rubric_score)
        reward_score = _float_or_none(raw.get("reward_score"))
        if reward_score is not None and is_ca_compact:
            ropd_ca_train_scores.append(reward_score)
        elif reward_score is not None:
            ropd_train_scores.append(reward_score)
        answer_core_score = _float_or_none(raw.get("answer_core_score"))
        if answer_core_score is not None:
            ropd_answer_core_scores.append(answer_core_score)
        answer_support_score = _float_or_none(raw.get("answer_support_score"))
        if answer_support_score is not None:
            ropd_answer_support_scores.append(answer_support_score)
        process_score = _float_or_none(raw.get("process_score"))
        if process_score is not None and not is_ca_compact:
            ropd_process_scores.append(process_score)
        if "fatal_error" in raw:
            ropd_fatal_error_seen = True
            ropd_fatal_error_count += int(bool(raw.get("fatal_error")))
        final_answer_quality = raw.get("final_answer_quality")
        if final_answer_quality:
            key = str(final_answer_quality)
            ropd_final_answer_quality[key] = ropd_final_answer_quality.get(key, 0) + 1
        maximum_score = _float_or_none(raw.get("maximum_score"))
        teacher_scores = raw.get("teacher_scores")
        if isinstance(teacher_scores, list):
            for item in teacher_scores:
                teacher_score = _float_or_none(item)
                if teacher_score is None:
                    continue
                ropd_teacher_points.append(teacher_score)
                if maximum_score is not None and maximum_score > 0:
                    normalized_teacher_score = max(0.0, min(1.0, teacher_score / maximum_score))
                    if schema_mode == "rubric_shaping":
                        ropd_teacher_rubric_scores.append(normalized_teacher_score)
                    elif not is_ca_compact:
                        ropd_teacher_answer_scores.append(normalized_teacher_score)
        if "teacher_below_student" in raw:
            ropd_teacher_below_student_seen = True
        if raw.get("teacher_below_student"):
            ropd_teacher_below_student += 1
        if "env_success_for_reward" in raw:
            ropd_env_success_for_reward_seen = True
            ropd_env_success_for_reward += int(bool(raw.get("env_success_for_reward")))
        rubric = _as_dict(raw.get("rubric"))
        rubric_items = rubric.get("rubrics")
        if isinstance(rubric_items, list):
            ropd_rubric_sizes.append(float(len(rubric_items)))
        answer_rubrics = rubric.get("answer_rubrics")
        if isinstance(answer_rubrics, list):
            ropd_answer_rubric_sizes.append(float(len(answer_rubrics)))
        process_rubrics = rubric.get("process_rubrics")
        if isinstance(process_rubrics, list):
            ropd_process_rubric_sizes.append(float(len(process_rubrics)))

    total = len(real_samples)
    metrics: dict[str, float] = {}
    if rm_scores:
        metrics["reward/rm_score_mean"] = _mean(rm_scores)
        metrics["reward/rm_score_nonzero_rate"] = sum(1 for value in rm_scores if value != 0.0) / len(rm_scores)
    for reason, count in fallback_reasons.items():
        metrics[f"reward/fallback_{reason}_rate"] = count / total
    for source, count in rubric_sources.items():
        metrics[f"reward/ropd/rubric_source_{source}_rate"] = count / total
    if ropd_student_points:
        metrics["reward/ropd/student_points_mean"] = _mean(ropd_student_points)
    if ropd_teacher_points:
        metrics["reward/ropd/teacher_points_mean"] = _mean(ropd_teacher_points)
    if ropd_answer_scores:
        metrics["reward/ropd/answer_score_mean"] = _mean(ropd_answer_scores)
    if ropd_rubric_scores:
        metrics["reward/ropd/rubric_score_mean"] = _mean(ropd_rubric_scores)
    if ropd_teacher_rubric_scores:
        metrics["reward/ropd/teacher_rubric_score_mean"] = _mean(ropd_teacher_rubric_scores)
    if ropd_teacher_answer_scores:
        metrics["reward/ropd/teacher_answer_score_mean"] = _mean(ropd_teacher_answer_scores)
    if ropd_train_scores:
        metrics["reward/ropd/train_score_mean"] = _mean(ropd_train_scores)
    if ropd_ca_train_scores:
        metrics["reward/ropd_ca/train_score_mean"] = _mean(ropd_ca_train_scores)
    if ropd_rubric_sizes:
        metrics["reward/ropd/rubric_size_mean"] = _mean(ropd_rubric_sizes)
    if ropd_answer_rubric_sizes:
        metrics["reward/ropd/answer_rubric_size_mean"] = _mean(ropd_answer_rubric_sizes)
    if ropd_process_rubric_sizes:
        metrics["reward/ropd/process_rubric_size_mean"] = _mean(ropd_process_rubric_sizes)
    if ropd_answer_core_scores:
        metrics["reward/ropd/answer_core_score_mean"] = _mean(ropd_answer_core_scores)
    if ropd_answer_support_scores:
        metrics["reward/ropd/answer_support_score_mean"] = _mean(ropd_answer_support_scores)
    if ropd_process_scores:
        metrics["reward/ropd/process_score_mean"] = _mean(ropd_process_scores)
    if ropd_fatal_error_seen:
        metrics["reward/ropd/fatal_error_rate"] = ropd_fatal_error_count / total
    for quality, count in ropd_final_answer_quality.items():
        metrics[f"reward/ropd/final_answer_quality_{quality}_rate"] = count / total
    if ropd_teacher_below_student_seen:
        metrics["reward/ropd/teacher_below_student_rate"] = ropd_teacher_below_student / total
    if ropd_env_success_for_reward_seen:
        metrics["reward/ropd/env_success_for_reward_rate"] = ropd_env_success_for_reward / total
    if ca_nonzero_tokens:
        metrics["reward/credit_assignment/nonzero_tokens_mean"] = _mean(ca_nonzero_tokens)
    if ca_nonzero_token_rates:
        metrics["reward/credit_assignment/nonzero_token_rate_mean"] = _mean(ca_nonzero_token_rates)
    if ca_positive_token_rates:
        metrics["reward/credit_assignment/positive_token_rate_mean"] = _mean(ca_positive_token_rates)
    if ca_negative_token_rates:
        metrics["reward/credit_assignment/negative_token_rate_mean"] = _mean(ca_negative_token_rates)
    if ca_process_value_means:
        metrics["reward/credit_assignment/process_value_mean"] = _mean(ca_process_value_means)
    if ca_process_value_abs_means:
        metrics["reward/credit_assignment/process_value_abs_mean"] = _mean(ca_process_value_abs_means)
    if ca_process_delta_means:
        metrics["reward/credit_assignment/process_delta_mean"] = _mean(ca_process_delta_means)
    if ca_process_delta_abs_means:
        metrics["reward/credit_assignment/process_delta_abs_mean"] = _mean(ca_process_delta_abs_means)
    if ca_tasa_supported_token_rates:
        metrics["reward/credit_assignment/tasa_supported_token_rate_mean"] = _mean(ca_tasa_supported_token_rates)
    if ca_tasa_unique_states:
        metrics["reward/credit_assignment/tasa_unique_states_mean"] = _mean(ca_tasa_unique_states)
    if ca_tasa_state_reuse_rates:
        metrics["reward/credit_assignment/tasa_state_reuse_rate_mean"] = _mean(ca_tasa_state_reuse_rates)
    if ca_tasa_supported_segment_rates:
        metrics["reward/credit_assignment/tasa_supported_segment_rate_mean"] = _mean(ca_tasa_supported_segment_rates)
    if ca_tasa_peer_count_means:
        metrics["reward/credit_assignment/tasa_peer_count_mean"] = _mean(ca_tasa_peer_count_means)
    if ca_tasa_state_reward_std_means:
        metrics["reward/credit_assignment/tasa_state_reward_std_mean"] = _mean(ca_tasa_state_reward_std_means)
    if ca_tasa_local_outcome_corrs:
        metrics["reward/credit_assignment/tasa_local_outcome_corr_mean"] = _mean(ca_tasa_local_outcome_corrs)
    if ca_tasa_state_prior_means:
        metrics["reward/credit_assignment/tasa_state_prior_mean"] = _mean(ca_tasa_state_prior_means)
    if ca_tasa_state_value_means:
        metrics["reward/credit_assignment/tasa_state_value_mean"] = _mean(ca_tasa_state_value_means)
    if ca_tasa_td_delta_abs_means:
        metrics["reward/credit_assignment/tasa_td_delta_abs_mean"] = _mean(ca_tasa_td_delta_abs_means)
    if ca_tasa_gae_abs_means:
        metrics["reward/credit_assignment/tasa_gae_abs_mean"] = _mean(ca_tasa_gae_abs_means)
    if ca_tasa_prior_kappas:
        metrics["reward/credit_assignment/tasa_prior_kappa"] = _mean(ca_tasa_prior_kappas)
    if ca_tasa_lambdas:
        metrics["reward/credit_assignment/tasa_lambda"] = _mean(ca_tasa_lambdas)
    if ca_tasa_teacher_weights:
        metrics["reward/credit_assignment/tasa_teacher_weight"] = _mean(ca_tasa_teacher_weights)
    _reward_call_metrics(metrics, role="judge", calls=judge_calls, sample_count=total)
    _reward_call_metrics(metrics, role="rubric", calls=rubric_calls, sample_count=total)
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
    env_scores = []
    env_rewards = []
    user_model_call_counts = []
    user_model_usage_totals: list[dict[str, Any]] = []
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
        if "env_score" in metadata:
            env_scores.append(float(metadata["env_score"]))
        if "env_reward" in metadata:
            env_rewards.append(float(metadata["env_reward"]))
        usage_totals = _as_dict(metadata.get("user_model_usage_totals"))
        if usage_totals:
            user_model_usage_totals.append(usage_totals)
        call_count = _float_or_none(metadata.get("user_model_call_count"))
        if call_count is None:
            usage_list = metadata.get("user_model_usage")
            if isinstance(usage_list, list):
                call_count = float(len([item for item in usage_list if isinstance(item, dict)]))
        if call_count is not None:
            user_model_call_counts.append(call_count)
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
    if env_scores:
        metrics[f"{prefix}/env_score_mean"] = sum(env_scores) / len(env_scores)
    if env_rewards:
        metrics[f"{prefix}/env_reward_mean"] = sum(env_rewards) / len(env_rewards)
    if user_model_call_counts:
        total_calls = sum(user_model_call_counts)
        metrics[f"{prefix}/usersim_call_count_total"] = total_calls
        metrics[f"{prefix}/usersim_call_count_mean"] = total_calls / len(real_samples)
    usage_sums = _sum_numeric_dicts(user_model_usage_totals)
    for key, total_value in usage_sums.items():
        metric_key = f"{prefix}/usersim_usage_{key}"
        metrics[f"{metric_key}_total"] = total_value
        metrics[f"{metric_key}_per_sample"] = total_value / len(real_samples)
        if user_model_call_counts and sum(user_model_call_counts) > 0:
            metrics[f"{metric_key}_per_call"] = total_value / sum(user_model_call_counts)
    for reason, count in truncated_reasons.items():
        metrics[f"{prefix}/truncated_{reason}_rate"] = count / total_turns if total_turns else 0.0
    for reason, count in discard_reasons.items():
        metrics[f"{prefix}/discard_{reason}_rate"] = count / len(real_samples)
    return metrics


def _flatten_sample_groups(groups: list[Any]) -> list[Any]:
    samples: list[Any] = []
    for group in groups:
        if group is None:
            continue
        if isinstance(group, list):
            for item in group:
                if isinstance(item, list):
                    samples.extend(item)
                else:
                    samples.append(item)
        else:
            samples.append(group)
    return samples


def _strip_metric_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {key.removeprefix(prefix): value for key, value in metrics.items() if key.startswith(prefix)}


def generated_train_scope_metrics(
    *,
    generated_groups: list[Any],
    train_groups: list[Any],
) -> dict[str, float]:
    """Compute generated-vs-train scope metrics for dynamic filtering.

    Dynamic sampling correctly removes all-correct/all-wrong groups from actor
    training, but those generated episodes are still part of the rollout
    distribution users need to monitor. Keep the full generated metrics in a
    private namespace; `log_rollout_data_for_env` promotes them to canonical
    W&B keys once it knows the environment prefix.
    """

    generated_samples = _flatten_sample_groups(generated_groups)
    generated_group_count = float(len(generated_groups))
    train_group_count = float(len(train_groups))
    dropped_group_count = max(0.0, generated_group_count - train_group_count)

    metrics: dict[str, float] = {
        "rollout/dynamic_filter/generated_groups": generated_group_count,
        "rollout/dynamic_filter/kept_groups": train_group_count,
        "rollout/dynamic_filter/dropped_groups": dropped_group_count,
        "rollout/dynamic_filter/drop_rate": dropped_group_count / generated_group_count
        if generated_group_count
        else 0.0,
    }

    env_generated = _strip_metric_prefix(environment_metrics(generated_samples, prefix="agent_env"), "agent_env/")
    for key, value in env_generated.items():
        metrics[f"{_GENERATED_ENV_PREFIX}{key}"] = value

    reward_generated = reward_metrics(generated_samples)
    for key, value in reward_generated.items():
        metrics[f"{_GENERATED_REWARD_PREFIX}{key.removeprefix('reward/')}"] = value

    return metrics


def _pop_generated_scope_metrics(metrics: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    generated_env: dict[str, float] = {}
    generated_reward: dict[str, float] = {}
    for key in list(metrics.keys()):
        value = metrics[key]
        if key.startswith(_GENERATED_ENV_PREFIX):
            generated_env[key.removeprefix(_GENERATED_ENV_PREFIX)] = float(value)
            del metrics[key]
        elif key.startswith(_GENERATED_REWARD_PREFIX):
            generated_reward[key.removeprefix(_GENERATED_REWARD_PREFIX)] = float(value)
            del metrics[key]
    return generated_env, generated_reward


def _define_wandb_step_metrics(args: Any, metric_names: Iterable[str], *, step_metric: str) -> None:
    """Bind every emitted metric to its explicit train/eval clock.

    W&B prefix globs such as ``rollout/*`` do not cover arbitrary nested or
    environment-specific names. Register exact keys here so resumed runs use
    the restored rollout/eval step instead of W&B's per-process ``_step``.
    """

    global _WANDB_METRIC_DEFINITION_WARNING_EMITTED
    if not bool(getattr(args, "use_wandb", False)):
        return
    try:
        import wandb

        run = wandb.run
        if run is None:
            return
        run_key = (id(run), str(getattr(run, "id", "")))
        definitions = [(step_metric, None)]
        definitions.extend((name, step_metric) for name in sorted(set(metric_names)) if name != step_metric)
        for metric_name, metric_step in definitions:
            cache_key = (*run_key, metric_name, step_metric)
            if cache_key in _WANDB_DEFINED_STEP_METRICS:
                continue
            if metric_step is None:
                wandb.define_metric(metric_name)
            else:
                wandb.define_metric(metric_name, step_metric=metric_step)
            _WANDB_DEFINED_STEP_METRICS.add(cache_key)
    except Exception:
        if not _WANDB_METRIC_DEFINITION_WARNING_EMITTED:
            logger.warning("Failed to bind W&B metrics to %s", step_metric, exc_info=True)
            _WANDB_METRIC_DEFINITION_WARNING_EMITTED = True


def log_rollout_data_for_env(prefix: str, rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    from slime.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step

    log_dict = {**(rollout_extra_metrics or {})}
    generated_env_metrics, generated_reward_metrics = _pop_generated_scope_metrics(log_dict)

    if generated_env_metrics:
        log_dict |= {f"{prefix}/{key}": value for key, value in generated_env_metrics.items()}
    else:
        log_dict |= environment_metrics(samples, prefix=prefix)

    if generated_reward_metrics:
        log_dict |= {f"reward/{key}": value for key, value in generated_reward_metrics.items()}
        # Credit-assignment metadata is attached in reward_post_process, after
        # generated-scope metrics are prepared. Keep generated reward metrics
        # canonical, but log CA diagnostics from the actual train samples.
        log_dict |= {
            key: value
            for key, value in reward_metrics(samples).items()
            if key.startswith("reward/credit_assignment/")
        }
    else:
        log_dict |= reward_metrics(samples)

    log_dict |= {f"rollout/{k}": v for k, v in compute_metrics_from_samples(args, samples).items()}
    log_dict |= {f"perf/{k}": v for k, v in compute_perf_metrics_from_samples(args, samples, rollout_time).items()}
    log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
    _define_wandb_step_metrics(args, log_dict, step_metric="rollout/step")
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True


def _sample_response_length(sample: Any) -> float:
    for attr in ("effective_response_length", "response_length"):
        value = _float_or_none(getattr(sample, attr, None))
        if value is not None:
            return value
    tokens = getattr(sample, "tokens", None)
    return float(len(tokens)) if tokens is not None else 0.0


def _sample_is_truncated(sample: Any) -> bool:
    status = getattr(sample, "status", None)
    name = str(getattr(status, "name", status))
    return name.upper() == "TRUNCATED"


def _eval_generation_metrics(samples: list[Any]) -> dict[str, float]:
    from slime.utils.metric_utils import compute_statistics, dict_add_prefix, has_repetition

    if not samples:
        return {}
    response_lengths = [_sample_response_length(sample) for sample in samples]
    metrics = dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    metrics["repetition_frac"] = _mean([float(has_repetition(str(getattr(sample, "response", "") or ""))) for sample in samples])
    metrics["truncated_ratio"] = _mean([float(_sample_is_truncated(sample)) for sample in samples])
    return metrics


def log_eval_rollout_data_for_env(prefix: str, rollout_id, args, data, extra_metrics) -> bool:
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, dict_add_prefix

    log_dict = {**(extra_metrics or {})}
    for name, info in data.items():
        rewards = info.get("rewards") or []
        if rewards:
            log_dict[f"eval/{name}"] = sum(rewards) / len(rewards)

        samples = info.get("samples") or []
        if samples:
            log_dict |= dict_add_prefix(_eval_generation_metrics(samples), f"eval/{name}/")
            for key, value in environment_metrics(samples, prefix=prefix).items():
                log_dict[f"eval/{name}/{key.removeprefix(prefix + '/')}"] = value
            for key, value in reward_metrics(samples).items():
                log_dict[f"eval/{name}/{key}"] = value

        truncated = info.get("truncated")
        if truncated:
            log_dict[f"eval/{name}-truncated_ratio"] = sum(truncated) / len(truncated)

        if rewards and bool(getattr(args, "log_passrate", False)):
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=getattr(args, "n_samples_per_eval_prompt", 1),
                ),
                f"eval/{name}-",
            )

    logger.info("eval %s: %s", rollout_id, log_dict)

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    _define_wandb_step_metrics(args, log_dict, step_metric="eval/step")
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True
