from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from examples.agent_env import dump
from examples.agent_env.rollout import arg, metadata
from examples.agent_env.rewards.config import reward_cfg_path
from examples.agent_env.rewards.extractors import bool_value, float_value, int_value


@dataclass(frozen=True)
class SegmentCreditRecord:
    """One assistant turn that can receive process credit.

    This is the shared intermediate representation for all CA modes. The only
    mode-specific choice is how these per-turn values are combined with scalar
    outcome rewards before being expanded back onto response tokens.
    """

    sample_index: int
    start: int
    end: int
    step: int
    value: float
    marked: bool
    state_key: str | None = None
    state_ids: tuple[str, ...] = ()
    peer_count: int | None = None
    state_reward_mean: float | None = None
    local_advantage: float | None = None
    supported: bool | None = None


@dataclass(frozen=True)
class StepMarkEvent:
    step: int
    source: str
    value: float
    previous_value: float
    new_value: float
    applied: bool
    criterion_id: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "source": self.source,
            "criterion_id": self.criterion_id,
            "value": self.value,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "applied": self.applied,
            "reason": self.reason,
        }


def config(args: Any) -> dict[str, Any]:
    value = reward_cfg_path(args, "credit_assignment", {})
    return dict(value) if isinstance(value, dict) else {}


def enabled(args: Any) -> bool:
    return bool_value(config(args).get("enable", False), False)


def request_process_step_evidence(args: Any) -> bool:
    if not enabled(args):
        return False
    return bool_value(config(args).get("request_process_step_evidence", True), True)


def beta(args: Any) -> float:
    value = float_value(config(args).get("beta", 0.05), 0.05)
    if value < 0:
        raise ValueError("reward.credit_assignment.beta must be non-negative")
    return value


def advantage_mode(args: Any) -> str:
    """Select how process credit is fused with scalar outcome credit.

    These modes intentionally live in reward.credit_assignment, because they
    change only the advantage construction for already-computed reward signals.
    They do not change rollout, RM scoring, or the reward scalar stored on the
    sample.
    """

    value = str(config(args).get("advantage_mode", "outcome_plus_process") or "outcome_plus_process").strip().lower()
    aliases = {
        "additive": "outcome_plus_process",
        "a": "outcome_plus_process",
        "sign_preserving": "outcome_reweight",
        "reweight": "outcome_reweight",
        "b": "outcome_reweight",
        "bangbang": "outcome_bangbang_reweight",
        "bangbang_b": "outcome_bangbang_reweight",
        "hard_b": "outcome_bangbang_reweight",
        "hard_bangbang": "outcome_bangbang_reweight",
        "segment_reward": "segment_reward_group_turn_norm",
        "reinforce": "segment_reward_group_turn_norm",
        "reinforce_plus_plus": "segment_reward_group_turn_norm",
        "segment_reward_global_norm": "segment_reward_group_turn_norm",
        "c": "segment_reward_group_turn_norm",
        "tasa": "teacher_anchored_state_aggregation",
        "tasa_grpo": "teacher_anchored_state_aggregation",
        "teacher_anchored": "teacher_anchored_state_aggregation",
        "teacher_anchored_state_aggregation": "teacher_anchored_state_aggregation",
    }
    value = aliases.get(value, value)
    valid = {
        "outcome_plus_process",
        "outcome_reweight",
        "outcome_bangbang_reweight",
        "segment_reward_group_turn_norm",
        "teacher_anchored_state_aggregation",
    }
    if value not in valid:
        raise ValueError(f"reward.credit_assignment.advantage_mode must be one of {sorted(valid)}")
    return value


def clip(args: Any) -> float:
    value = float_value(config(args).get("clip", 2.0), 2.0)
    if value <= 0:
        raise ValueError("reward.credit_assignment.clip must be positive")
    return value


def shaping_mode(args: Any) -> str:
    value = str(config(args).get("shaping_mode", "milestone")).strip().lower()
    if value != "milestone":
        raise ValueError("reward.credit_assignment.shaping_mode must be milestone")
    return value


def normalization(args: Any) -> str:
    raw = config(args).get("normalization", None)
    if raw in (None, ""):
        return "episode_turn_zscore" if advantage_mode(args) == "outcome_reweight" else "none"
    value = str(raw).strip().lower()
    aliases = {
        "turn_zscore": "group_turn_zscore",
        "group_turn_norm": "group_turn_zscore",
        "group_turn_normalization": "group_turn_zscore",
        "trace_turn_zscore": "episode_turn_zscore",
        "trajectory_turn_zscore": "episode_turn_zscore",
        "episode_turn_norm": "episode_turn_zscore",
    }
    value = aliases.get(value, value)
    if value not in {"none", "group_turn_zscore", "episode_turn_zscore"}:
        raise ValueError("reward.credit_assignment.normalization must be none, group_turn_zscore, or episode_turn_zscore")
    if advantage_mode(args) == "outcome_reweight" and value != "episode_turn_zscore":
        raise ValueError("reward.credit_assignment.advantage_mode=outcome_reweight requires normalization=episode_turn_zscore")
    if advantage_mode(args) == "outcome_bangbang_reweight" and value != "none":
        raise ValueError(
            "reward.credit_assignment.advantage_mode=outcome_bangbang_reweight requires normalization=none"
        )
    return value


def request_tasa_state_evidence(args: Any) -> bool:
    return enabled(args) and advantage_mode(args) == "teacher_anchored_state_aggregation"


def tasa_min_peer_support(args: Any) -> int:
    value = int_value(config(args).get("tasa_min_peer_support", 2), 2)
    if value < 1:
        raise ValueError("reward.credit_assignment.tasa_min_peer_support must be positive")
    return value


def tasa_local_normalization(args: Any) -> str:
    value = str(config(args).get("tasa_local_normalization", "group_segment_zscore") or "group_segment_zscore")
    value = value.strip().lower()
    aliases = {
        "zscore": "group_segment_zscore",
        "group_zscore": "group_segment_zscore",
        "segment_zscore": "group_segment_zscore",
        "group_segment_norm": "group_segment_zscore",
    }
    value = aliases.get(value, value)
    if value not in {"none", "group_segment_zscore"}:
        raise ValueError("reward.credit_assignment.tasa_local_normalization must be none or group_segment_zscore")
    return value


def milestone_reward(args: Any) -> float:
    return float_value(config(args).get("milestone_reward", 2.0), 2.0)


def predecessor_reward(args: Any) -> float:
    value = float_value(config(args).get("predecessor_reward", 0.5), 0.5)
    return max(0.0, value)


def predecessor_decay(args: Any) -> float:
    value = float_value(config(args).get("predecessor_decay", 1.0), 1.0)
    if value < 0 or value > 1:
        raise ValueError("reward.credit_assignment.predecessor_decay must be in [0, 1]")
    return value


def negative_reward(args: Any) -> float:
    return float_value(config(args).get("negative_reward", 0.0), 0.0)


def success_milestone_tail_steps(args: Any) -> int:
    value = int_value(config(args).get("success_milestone_tail_steps", 1), 1)
    if value < 0:
        raise ValueError("reward.credit_assignment.success_milestone_tail_steps must be non-negative")
    return value


def success_milestone_min_env_score(args: Any) -> float:
    value = float_value(config(args).get("success_milestone_min_env_score", 1.0), 1.0)
    if value < 0:
        raise ValueError("reward.credit_assignment.success_milestone_min_env_score must be non-negative")
    return value


def step_index_base(args: Any) -> int:
    value = int_value(config(args).get("step_index_base", 1), 1)
    if value not in (0, 1):
        raise ValueError("reward.credit_assignment.step_index_base must be 0 or 1")
    return value


def _dump_per_step_limit(args: Any) -> int:
    return dump.int_runtime_env(args, "AGENT_ENV_CREDIT_DUMP_N", "0")


def _dump_total_limit(args: Any) -> int:
    return dump.int_runtime_env(args, "AGENT_ENV_CREDIT_DUMP_TOTAL_N", "0")


def _dump_dir(args: Any) -> Path:
    raw = dump.runtime_env(args, "AGENT_ENV_CREDIT_DUMP_DIR", "").strip()
    if raw:
        return Path(raw)
    run_root = dump.runtime_env(args, "RUN_ROOT", "").strip()
    if run_root:
        return Path(run_root) / "reward_artifacts" / "credit_assignment"
    return Path(os.getcwd()) / "reward_artifacts" / "credit_assignment"


def _off_policy_mask_sum(sample: Sample) -> int:
    mask = getattr(sample, "off_policy_loss_mask", None)
    if mask is None:
        return 0
    return sum(int(value) for value in mask)


def _is_student_train_sample(sample: Sample) -> bool:
    sample_metadata = metadata(sample)
    if bool(sample_metadata.get("off_policy_sample", False)) or _off_policy_mask_sum(sample) > 0:
        return False
    if bool(getattr(sample, "remove_sample", False)) or sample.status == Sample.Status.ABORTED:
        return False
    mask = getattr(sample, "loss_mask", None)
    return bool(mask) and sum(int(value) for value in mask) > 0


def _rm_raw(sample: Sample) -> dict[str, Any]:
    sample_metadata = metadata(sample)
    rm_reward = sample_metadata.get("rm_reward")
    if isinstance(rm_reward, dict):
        raw = rm_reward.get("raw")
        if isinstance(raw, dict):
            return raw
    raw = sample_metadata.get("judge_raw")
    return raw if isinstance(raw, dict) else {}


def _step_list(value: Any) -> list[int]:
    if value in (None, "", []):
        return []
    if not isinstance(value, list):
        value = [value]
    steps: list[int] = []
    for item in value:
        try:
            step = int(item)
        except (TypeError, ValueError):
            continue
        if step >= 0:
            steps.append(step)
    return list(dict.fromkeys(steps))


def _mark_milestone(
    *,
    direct_positive_marks: dict[int, float],
    direct_negative_marks: dict[int, float],
    predecessor_marks: dict[int, float],
    step: int,
    available_steps: list[int],
    milestone_value: float,
    predecessor_value: float,
    predecessor_decay_value: float,
    events: list[StepMarkEvent] | None = None,
    source: str = "process_positive",
    criterion_id: str | None = None,
) -> None:
    _apply_priority_mark(
        direct_positive_marks=direct_positive_marks,
        direct_negative_marks=direct_negative_marks,
        predecessor_marks=predecessor_marks,
        step=step,
        value=milestone_value,
        kind="direct_positive",
        events=events,
        source=source,
        criterion_id=criterion_id,
    )
    for previous_step in available_steps:
        if previous_step >= step:
            break
        distance = step - previous_step
        decayed_predecessor = predecessor_value * (predecessor_decay_value ** max(0, distance - 1))
        _apply_priority_mark(
            direct_positive_marks=direct_positive_marks,
            direct_negative_marks=direct_negative_marks,
            predecessor_marks=predecessor_marks,
            step=previous_step,
            value=decayed_predecessor,
            kind="predecessor",
            events=events,
            source=f"{source}_predecessor",
            criterion_id=criterion_id,
        )


def _combined_step_value(
    *,
    step: int,
    direct_positive_marks: dict[int, float],
    direct_negative_marks: dict[int, float],
    predecessor_marks: dict[int, float],
) -> float:
    positive = direct_positive_marks.get(step, 0.0)
    negative = direct_negative_marks.get(step, 0.0)
    if positive != 0.0 or negative != 0.0:
        return positive + negative
    return predecessor_marks.get(step, 0.0)


def _final_step_marks(
    *,
    direct_positive_marks: dict[int, float],
    direct_negative_marks: dict[int, float],
    predecessor_marks: dict[int, float],
) -> dict[int, float]:
    steps = set(direct_positive_marks) | set(direct_negative_marks) | set(predecessor_marks)
    marks: dict[int, float] = {}
    for step in sorted(steps):
        value = _combined_step_value(
            step=step,
            direct_positive_marks=direct_positive_marks,
            direct_negative_marks=direct_negative_marks,
            predecessor_marks=predecessor_marks,
        )
        if value != 0.0:
            marks[step] = value
    return marks


def _apply_priority_mark(
    *,
    direct_positive_marks: dict[int, float],
    direct_negative_marks: dict[int, float],
    predecessor_marks: dict[int, float],
    step: int,
    value: float,
    kind: str,
    events: list[StepMarkEvent] | None,
    source: str,
    criterion_id: str | None = None,
) -> None:
    previous = _combined_step_value(
        step=step,
        direct_positive_marks=direct_positive_marks,
        direct_negative_marks=direct_negative_marks,
        predecessor_marks=predecessor_marks,
    )
    reason: str | None = None
    if kind == "direct_positive":
        direct_positive_marks[step] = max(direct_positive_marks.get(step, 0.0), value)
    elif kind == "direct_negative":
        direct_negative_marks[step] = min(direct_negative_marks.get(step, 0.0), value)
    elif kind == "predecessor":
        if step in direct_positive_marks or step in direct_negative_marks:
            reason = "direct_mark_already_present"
        else:
            predecessor_marks[step] = max(predecessor_marks.get(step, 0.0), value)
    else:
        raise ValueError(f"Unsupported mark kind: {kind}")
    new_value = _combined_step_value(
        step=step,
        direct_positive_marks=direct_positive_marks,
        direct_negative_marks=direct_negative_marks,
        predecessor_marks=predecessor_marks,
    )
    if events is not None:
        events.append(
            StepMarkEvent(
                step=step,
                source=source,
                criterion_id=criterion_id,
                value=value,
                previous_value=previous,
                new_value=new_value,
                applied=new_value != previous,
                reason=reason,
            )
        )


def _success_tail_milestone_steps(args: Any, sample: Sample, available_steps: list[int]) -> list[int]:
    sample_metadata = metadata(sample)
    if "env_score" in sample_metadata:
        try:
            strict_success = float(sample_metadata.get("env_score")) >= success_milestone_min_env_score(args)
        except (TypeError, ValueError):
            strict_success = False
    else:
        strict_success = bool(sample_metadata.get("env_success", False))
    if not strict_success:
        return []
    tail_steps = success_milestone_tail_steps(args)
    if tail_steps <= 0:
        return []
    return available_steps[-tail_steps:]


def _sample_step_marks(args: Any, sample: Sample, *, available_steps: list[int] | None = None) -> dict[int, float]:
    marks, _ = _sample_step_marks_and_events(args, sample, available_steps=available_steps)
    return marks


def _sample_step_marks_and_events(
    args: Any,
    sample: Sample,
    *,
    available_steps: list[int] | None = None,
) -> tuple[dict[int, float], list[StepMarkEvent]]:
    shaping_mode(args)
    raw = _rm_raw(sample)
    evidence = raw.get("process_step_evidence")
    available = sorted(dict.fromkeys(step for step in (available_steps or []) if step >= 0))
    milestone_value = milestone_reward(args)
    predecessor_value = predecessor_reward(args)
    predecessor_decay_value = predecessor_decay(args)
    negative_value = negative_reward(args)
    direct_positive_marks: dict[int, float] = {}
    direct_negative_marks: dict[int, float] = {}
    predecessor_marks: dict[int, float] = {}
    events: list[StepMarkEvent] = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            criterion_id = str(item.get("criterion_id") or "").strip() or None
            positive_steps = _step_list(item.get("positive_step_indices"))
            negative_steps = _step_list(item.get("negative_step_indices"))
            legacy_steps = _step_list(item.get("step_indices"))
            if legacy_steps:
                if bool(item.get("satisfied", True)):
                    positive_steps.extend(step for step in legacy_steps if step not in positive_steps)
                else:
                    negative_steps.extend(step for step in legacy_steps if step not in negative_steps)
            for step in positive_steps:
                _mark_milestone(
                    direct_positive_marks=direct_positive_marks,
                    direct_negative_marks=direct_negative_marks,
                    predecessor_marks=predecessor_marks,
                    step=step,
                    available_steps=available,
                    milestone_value=milestone_value,
                    predecessor_value=predecessor_value,
                    predecessor_decay_value=predecessor_decay_value,
                    events=events,
                    source="process_positive",
                    criterion_id=criterion_id,
                )
            if negative_value != 0.0:
                for step in negative_steps:
                    _apply_priority_mark(
                        direct_positive_marks=direct_positive_marks,
                        direct_negative_marks=direct_negative_marks,
                        predecessor_marks=predecessor_marks,
                        step=step,
                        value=negative_value,
                        kind="direct_negative",
                        events=events,
                        source="process_negative",
                        criterion_id=criterion_id,
                    )

    for step in _success_tail_milestone_steps(args, sample, available):
        _mark_milestone(
            direct_positive_marks=direct_positive_marks,
            direct_negative_marks=direct_negative_marks,
            predecessor_marks=predecessor_marks,
            step=step,
            available_steps=available,
            milestone_value=milestone_value,
            predecessor_value=predecessor_value,
            predecessor_decay_value=predecessor_decay_value,
            events=events,
            source="env_success_tail",
        )
    marks = _final_step_marks(
        direct_positive_marks=direct_positive_marks,
        direct_negative_marks=direct_negative_marks,
        predecessor_marks=predecessor_marks,
    )
    return marks, events


def _segment_turn(segment: dict[str, Any]) -> int | None:
    try:
        return int(segment.get("turn"))
    except (TypeError, ValueError):
        return None


def _segment_records_for_sample(args: Any, sample_index: int, sample: Sample) -> list[SegmentCreditRecord]:
    sample_metadata = metadata(sample)
    segments = sample_metadata.get("token_segments")
    if not isinstance(segments, list):
        return []
    base = step_index_base(args)
    candidates: list[tuple[int, int, int]] = []
    response_cursor = 0
    response_length = int(getattr(sample, "response_length", 0) or 0)
    for raw_segment in segments:
        if not isinstance(raw_segment, dict):
            continue
        try:
            token_count = int(raw_segment.get("token_count", 0) or 0)
        except (TypeError, ValueError):
            token_count = 0
        if token_count <= 0:
            continue
        kind = str(raw_segment.get("kind") or "")
        start = response_cursor
        end = min(response_length, response_cursor + token_count)
        if kind != "initial_prompt":
            response_cursor += token_count
        if kind != "assistant" or end <= start:
            continue
        try:
            loss_mask_sum = int(raw_segment.get("loss_mask_sum", 0) or 0)
        except (TypeError, ValueError):
            loss_mask_sum = 0
        if loss_mask_sum <= 0:
            continue
        turn = _segment_turn(raw_segment)
        if turn is None:
            continue
        rendered_step = turn + base
        candidates.append((start, end, rendered_step))

    marks = _sample_step_marks(args, sample, available_steps=[step for _, _, step in candidates])
    records: list[SegmentCreditRecord] = []
    for start, end, rendered_step in candidates:
        value = marks.get(rendered_step, 0.0)
        records.append(
            SegmentCreditRecord(
                sample_index=sample_index,
                start=start,
                end=end,
                step=rendered_step,
                value=value,
                marked=abs(value) > 0,
            )
        )
    return records


def _string_id_list(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if not isinstance(value, list):
        raise ValueError("TASA state id fields must be lists")
    ids: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        ids.append(text)
    return list(dict.fromkeys(ids))


def _tasa_schema_ids(raw: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    schema = raw.get("tasa_state_schema")
    if not isinstance(schema, dict):
        raise ValueError("TASA-GRPO requires judge raw.tasa_state_schema")
    milestones = schema.get("milestones")
    bad_flags = schema.get("bad_flags")
    if not isinstance(milestones, list) or not isinstance(bad_flags, list):
        raise ValueError("TASA-GRPO state schema requires milestones and bad_flags lists")

    def collect_ids(items: list[Any], *, prefix: str, field: str) -> tuple[str, ...]:
        ids: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"TASA-GRPO {field} entries must be objects")
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                raise ValueError(f"TASA-GRPO {field} entries require id")
            if not item_id.startswith(prefix):
                raise ValueError(f"TASA-GRPO {field} id must start with {prefix}")
            ids.append(item_id)
        if len(ids) != len(set(ids)):
            raise ValueError(f"TASA-GRPO {field} ids must be unique")
        return tuple(ids)

    milestone_ids = collect_ids(milestones, prefix="M", field="milestones")
    bad_flag_ids = collect_ids(bad_flags, prefix="B", field="bad_flags")
    if not milestone_ids and not bad_flag_ids:
        raise ValueError("TASA-GRPO state schema must contain at least one predicate")
    overlap = set(milestone_ids).intersection(bad_flag_ids)
    if overlap:
        raise ValueError(f"TASA-GRPO state schema has duplicate ids across predicate types: {sorted(overlap)}")
    return milestone_ids, bad_flag_ids


def _tasa_state_key(active_milestones: set[str], active_bad_flags: set[str]) -> tuple[str, tuple[str, ...]]:
    ids = tuple(sorted(active_milestones) + sorted(active_bad_flags))
    return "|".join(ids) if ids else "ROOT", ids


def _tasa_changes_by_step(raw: dict[str, Any], valid_ids: set[str]) -> dict[int, tuple[tuple[str, ...], tuple[str, ...]]]:
    changes = raw.get("tasa_state_changes")
    if not isinstance(changes, list):
        raise ValueError("TASA-GRPO requires judge raw.tasa_state_changes")
    by_step: dict[int, tuple[list[str], list[str]]] = {}
    for item in changes:
        if not isinstance(item, dict):
            raise ValueError("TASA-GRPO state change entries must be objects")
        try:
            step = int(item.get("step"))
        except (TypeError, ValueError):
            raise ValueError("TASA-GRPO state change entries require integer step") from None
        if step < 0:
            raise ValueError("TASA-GRPO state change step must be non-negative")
        set_ids = _string_id_list(item.get("set", []))
        unset_ids = _string_id_list(item.get("unset", []))
        unknown = sorted((set(set_ids) | set(unset_ids)) - valid_ids)
        if unknown:
            raise ValueError(f"TASA-GRPO state change references unknown predicate ids: {unknown}")
        current_set, current_unset = by_step.setdefault(step, ([], []))
        for item_id in set_ids:
            if item_id not in current_set:
                current_set.append(item_id)
        for item_id in unset_ids:
            if item_id not in current_unset:
                current_unset.append(item_id)
    return {step: (tuple(set_ids), tuple(unset_ids)) for step, (set_ids, unset_ids) in by_step.items()}


def _tasa_segment_records_for_sample(args: Any, sample_index: int, sample: Sample) -> list[SegmentCreditRecord]:
    sample_metadata = metadata(sample)
    segments = sample_metadata.get("token_segments")
    if not isinstance(segments, list):
        return []
    raw = _rm_raw(sample)
    milestone_ids, bad_flag_ids = _tasa_schema_ids(raw)
    valid_ids = set(milestone_ids) | set(bad_flag_ids)
    changes_by_step = _tasa_changes_by_step(raw, valid_ids)
    base = step_index_base(args)
    response_cursor = 0
    response_length = int(getattr(sample, "response_length", 0) or 0)
    active_milestones: set[str] = set()
    active_bad_flags: set[str] = set()
    records: list[SegmentCreditRecord] = []
    for raw_segment in segments:
        if not isinstance(raw_segment, dict):
            continue
        try:
            token_count = int(raw_segment.get("token_count", 0) or 0)
        except (TypeError, ValueError):
            token_count = 0
        if token_count <= 0:
            continue
        kind = str(raw_segment.get("kind") or "")
        start = response_cursor
        end = min(response_length, response_cursor + token_count)
        if kind != "initial_prompt":
            response_cursor += token_count
        if kind != "assistant" or end <= start:
            continue
        try:
            loss_mask_sum = int(raw_segment.get("loss_mask_sum", 0) or 0)
        except (TypeError, ValueError):
            loss_mask_sum = 0
        if loss_mask_sum <= 0:
            continue
        turn = _segment_turn(raw_segment)
        if turn is None:
            continue
        rendered_step = turn + base
        state_key, state_ids = _tasa_state_key(active_milestones, active_bad_flags)
        records.append(
            SegmentCreditRecord(
                sample_index=sample_index,
                start=start,
                end=end,
                step=rendered_step,
                value=0.0,
                marked=False,
                state_key=state_key,
                state_ids=state_ids,
            )
        )
        set_ids, unset_ids = changes_by_step.get(rendered_step, ((), ()))
        for item_id in unset_ids:
            active_milestones.discard(item_id)
            active_bad_flags.discard(item_id)
        for item_id in set_ids:
            if item_id in milestone_ids:
                active_milestones.add(item_id)
            else:
                active_bad_flags.add(item_id)
    return records


def sample_has_process_credit_signal(args: Any, sample: Sample) -> bool:
    if not enabled(args) or not _is_student_train_sample(sample):
        return False
    if advantage_mode(args) == "teacher_anchored_state_aggregation":
        return False
    return any(record.marked and abs(record.value) > 0 for record in _segment_records_for_sample(args, 0, sample))


def group_has_process_credit_signal(args: Any, samples: list[Sample]) -> bool:
    """Return whether a zero-scalar-reward group still has usable CA signal.

    Dynamic sampling filters scalar GRPO groups before actor training. For
    ROPD-CA, a group with identical scalar outcome rewards can still carry
    token-local process credit. Only non-zero credit marks count: if the judge
    found only bad behavior and `negative_reward=0`, this deliberately returns
    False so the zero-signal group may be dropped.
    """

    return any(sample_has_process_credit_signal(args, sample) for sample in samples)


def _sample_identifier(sample: Sample) -> dict[str, Any]:
    sample_metadata = metadata(sample)
    return {
        "index": getattr(sample, "index", None),
        "group_index": getattr(sample, "group_index", None),
        "rollout_id": getattr(sample, "rollout_id", None),
        "task_id": sample_metadata.get("task_id"),
        "task_ref": sample_metadata.get("task_ref"),
        "data_source": sample_metadata.get("data_source"),
    }


def _credit_record_payload(
    *,
    args: Any,
    sample_index: int,
    sample: Sample,
    records: list[SegmentCreditRecord],
    advantages: list[float],
) -> dict[str, Any]:
    sample_metadata = metadata(sample)
    rm_reward = sample_metadata.get("rm_reward") if isinstance(sample_metadata.get("rm_reward"), dict) else {}
    raw = rm_reward.get("raw") if isinstance(rm_reward.get("raw"), dict) else {}
    process_evidence = raw.get("process_step_evidence")
    step_mark_events: list[StepMarkEvent] = []
    if records:
        _, step_mark_events = _sample_step_marks_and_events(
            args,
            sample,
            available_steps=[record.step for record in records],
        )
    return {
        "schema_version": "agent_env.credit_assignment_audit.v1",
        "time": time.time(),
        "pid": os.getpid(),
        "dump_step": dump.sample_dump_step_label(sample),
        "sample_index_in_batch": sample_index,
        "sample": _sample_identifier(sample),
        "is_student_train_sample": _is_student_train_sample(sample),
        "response_length": len(advantages),
        "rm_reward": {
            "score": rm_reward.get("score"),
            "source": rm_reward.get("source"),
            "reason": raw.get("reward_reason") or raw.get("reason"),
            "reward_score": raw.get("reward_score"),
            "rubric_score": raw.get("rubric_score"),
            "env_success_for_reward": raw.get("env_success_for_reward"),
            "ropd_schema_mode": raw.get("ropd_schema_mode"),
        },
        "env": {
            "env_success": sample_metadata.get("env_success"),
            "env_score": sample_metadata.get("env_score"),
            "env_reward": sample_metadata.get("env_reward"),
        },
        "credit_assignment": sample_metadata.get("credit_assignment"),
        "step_marks": {record.step: record.value for record in records if record.marked},
        "step_mark_events": [event.as_dict() for event in step_mark_events],
        "process_step_evidence": process_evidence if isinstance(process_evidence, list) else [],
        "tasa_state_schema": raw.get("tasa_state_schema") if isinstance(raw.get("tasa_state_schema"), dict) else None,
        "tasa_state_changes": raw.get("tasa_state_changes") if isinstance(raw.get("tasa_state_changes"), list) else [],
        "segments": [
            {
                "start": record.start,
                "end": record.end,
                "step": record.step,
                "raw_step_value": record.value,
                "advantage_value": advantages[record.start] if 0 <= record.start < len(advantages) else 0.0,
                "marked": record.marked,
                "state_key": record.state_key,
                "state_ids": list(record.state_ids),
                "peer_count": record.peer_count,
                "state_reward_mean": record.state_reward_mean,
                "local_advantage": record.local_advantage,
                "supported": record.supported,
            }
            for record in records
        ],
    }


def _maybe_dump_credit_assignment(
    args: Any,
    samples: list[Sample],
    *,
    sample_advantages: list[list[float]],
    sample_records: list[list[SegmentCreditRecord]],
) -> None:
    per_step_limit = _dump_per_step_limit(args)
    if per_step_limit <= 0:
        return
    out_dir = _dump_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"credit_assignment_pid{os.getpid()}.jsonl"
    total_limit = _dump_total_limit(args)
    with out_path.open("a", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples):
            slot = dump.reserve_dump_slot(
                namespace="credit_assignment",
                stage="mask",
                dump_step=dump.sample_dump_step_label(sample),
                per_step_limit=per_step_limit,
                total_limit=total_limit,
            )
            if slot is None:
                continue
            index_in_step, total_index = slot
            payload = _credit_record_payload(
                args=args,
                sample_index=idx,
                sample=sample,
                records=sample_records[idx],
                advantages=sample_advantages[idx],
            )
            payload["index_in_step"] = index_in_step
            payload["index"] = total_index
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _group_indices(args: Any, samples: list[Sample]) -> dict[int, list[int]]:
    n_samples = max(1, int(arg(args, "n_samples_per_prompt", 1) or 1))
    grouped: dict[int, list[int]] = {}
    for idx, sample in enumerate(samples):
        group_key = int(sample.group_index) if sample.group_index is not None else idx // n_samples
        grouped.setdefault(group_key, []).append(idx)
    return grouped


def attach_process_advantages(args: Any, samples: list[Sample], *, scalar_rewards: list[float] | None = None) -> None:
    """Attach response-token-aligned credit tensors using one shared pipeline.

    All CA schemes first extract the same ``SegmentCreditRecord`` objects from
    token segments and ROPD step evidence. A/B write a normalized process-credit
    tensor and leave scalar GRPO outcome advantage to the actor hook. C combines
    raw scalar outcome reward with process credit at the turn level, normalizes
    those segment rewards inside each group, then writes that final tensor for
    the same actor hook to consume.
    """

    if not enabled(args):
        return

    sample_advantages: list[list[float]] = []
    sample_masks: list[list[float]] = []
    sample_records: list[list[SegmentCreditRecord]] = []
    for sample in samples:
        response_length = int(getattr(sample, "response_length", 0) or 0)
        sample_advantages.append([0.0] * max(0, response_length))
        sample_masks.append([0.0] * max(0, response_length))
        sample_records.append([])

    mode = advantage_mode(args)
    if mode == "segment_reward_group_turn_norm":
        if scalar_rewards is None:
            raise ValueError("segment_reward_group_turn_norm requires scalar_rewards from reward post-process")
        _attach_segment_reward_group_turn_norm(args, samples, sample_advantages, sample_records, scalar_rewards)
    elif mode == "teacher_anchored_state_aggregation":
        if scalar_rewards is None:
            raise ValueError("TASA-GRPO requires scalar_rewards from reward post-process")
        _attach_tasa_state_baselines(args, samples, sample_advantages, sample_masks, sample_records, scalar_rewards)
    else:
        _attach_process_credit(args, samples, sample_advantages, sample_records)

    for sample, values, support_values in zip(samples, sample_advantages, sample_masks, strict=True):
        sample_metadata = metadata(sample)
        nonzero = sum(1 for value in values if abs(value) > 0)
        positive = sum(1 for value in values if value > 0)
        negative = sum(1 for value in values if value < 0)
        response_length = len(values)
        process_mean = (sum(values) / response_length) if response_length else 0.0
        process_abs_mean = (sum(abs(value) for value in values) / response_length) if response_length else 0.0
        cfg_beta = beta(args)
        sample_metadata["process_advantages"] = values
        if mode == "teacher_anchored_state_aggregation":
            sample_metadata["process_advantage_masks"] = support_values
        credit_stats = {
            "enabled": True,
            "advantage_mode": mode,
            "beta": cfg_beta,
            "shaping_mode": shaping_mode(args),
            "normalization": normalization(args),
            "segment_reward_normalization": "group_turn_zscore"
            if mode == "segment_reward_group_turn_norm"
            else None,
            "tasa_local_normalization": tasa_local_normalization(args)
            if mode == "teacher_anchored_state_aggregation"
            else None,
            "tasa_min_peer_support": tasa_min_peer_support(args)
            if mode == "teacher_anchored_state_aggregation"
            else None,
            "milestone_reward": milestone_reward(args),
            "predecessor_reward": predecessor_reward(args),
            "predecessor_decay": predecessor_decay(args),
            "negative_reward": negative_reward(args),
            "success_milestone_tail_steps": success_milestone_tail_steps(args),
            "success_milestone_min_env_score": success_milestone_min_env_score(args),
            "clip": clip(args),
            "nonzero_tokens": nonzero,
            "positive_tokens": positive,
            "negative_tokens": negative,
            "response_length": response_length,
            "nonzero_token_rate": (nonzero / response_length) if response_length else 0.0,
            "positive_token_rate": (positive / response_length) if response_length else 0.0,
            "negative_token_rate": (negative / response_length) if response_length else 0.0,
            "process_value_mean": process_mean,
            "process_value_abs_mean": process_abs_mean,
            "process_delta_mean": cfg_beta * process_mean,
            "process_delta_abs_mean": cfg_beta * process_abs_mean,
        }
        tasa_stats = sample_metadata.pop("_tasa_credit_assignment", None)
        if isinstance(tasa_stats, dict):
            credit_stats.update(tasa_stats)
        sample_metadata["credit_assignment"] = credit_stats

    _maybe_dump_credit_assignment(args, samples, sample_advantages=sample_advantages, sample_records=sample_records)


def _write_record_value(target: list[float], record: SegmentCreditRecord, value: float) -> None:
    for offset in range(record.start, min(record.end, len(target))):
        target[offset] = value


def _mean_float(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean_float(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return max(variance, 0.0) ** 0.5


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) <= 1 or len(xs) != len(ys):
        return 0.0
    mean_x = _mean_float(xs)
    mean_y = _mean_float(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = sum(value * value for value in centered_x)
    denom_y = sum(value * value for value in centered_y)
    if denom_x <= 0 or denom_y <= 0:
        return 0.0
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / ((denom_x * denom_y) ** 0.5)


def _records_for_group(
    args: Any,
    samples: list[Sample],
    indices: list[int],
    sample_records: list[list[SegmentCreditRecord]],
) -> list[SegmentCreditRecord]:
    """Extract the common turn-credit records for one prompt group."""

    records: list[SegmentCreditRecord] = []
    for idx in indices:
        sample = samples[idx]
        if not _is_student_train_sample(sample):
            continue
        records.extend(_segment_records_for_sample(args, idx, sample))
    for record in records:
        sample_records[record.sample_index].append(record)
    return records


def _group_reward_advantages(sample_indices: list[int], scalar_rewards: list[float]) -> dict[int, float]:
    values = [float(scalar_rewards[idx]) for idx in sample_indices]
    mean = _mean_float(values)
    std = _sample_std(values)
    if std <= 0:
        return {idx: float(scalar_rewards[idx]) - mean for idx in sample_indices}
    return {idx: (float(scalar_rewards[idx]) - mean) / (std + 1e-6) for idx in sample_indices}


def _write_tasa_group_stats(
    *,
    samples: list[Sample],
    group_indices: list[int],
    records: list[SegmentCreditRecord],
    state_rewards: dict[str, dict[int, float]],
    local_values: list[float],
    base_values: list[float],
) -> None:
    unique_states = len(state_rewards)
    reused_states = sum(1 for values in state_rewards.values() if len(values) >= 2)
    state_reward_stds = [_sample_std(list(values.values())) for values in state_rewards.values() if len(values) >= 2]
    supported_records = [record for record in records if record.supported]
    records_with_any_peer = [record for record in records if (record.peer_count or 0) >= 1]
    group_stats = {
        "tasa_group_segment_count": float(len(records)),
        "tasa_group_supported_segment_count": float(len(supported_records)),
        "tasa_group_supported_segment_rate": len(supported_records) / len(records) if records else 0.0,
        "tasa_group_unique_states": float(unique_states),
        "tasa_group_reused_states": float(reused_states),
        "tasa_group_state_reuse_rate": reused_states / unique_states if unique_states else 0.0,
        "tasa_group_segment_peer_rate": len(records_with_any_peer) / len(records) if records else 0.0,
        "tasa_group_peer_count_mean": _mean_float([float(record.peer_count or 0) for record in records]),
        "tasa_group_state_reward_std_mean": _mean_float(state_reward_stds),
        "tasa_local_outcome_corr": _corr(local_values, base_values),
    }
    by_sample: dict[int, list[SegmentCreditRecord]] = {}
    for record in records:
        by_sample.setdefault(record.sample_index, []).append(record)
    for idx in group_indices:
        if not _is_student_train_sample(samples[idx]):
            continue
        sample_records = by_sample.get(idx, [])
        response_length = int(getattr(samples[idx], "response_length", 0) or 0)
        supported_tokens = sum(max(0, record.end - record.start) for record in sample_records if record.supported)
        sample_stats = {
            **group_stats,
            "tasa_sample_segment_count": float(len(sample_records)),
            "tasa_sample_supported_segment_count": float(len([record for record in sample_records if record.supported])),
            "tasa_supported_tokens": float(supported_tokens),
            "tasa_supported_token_rate": supported_tokens / response_length if response_length else 0.0,
        }
        metadata(samples[idx])["_tasa_credit_assignment"] = sample_stats


def _attach_process_credit(
    args: Any,
    samples: list[Sample],
    sample_advantages: list[list[float]],
    sample_records: list[list[SegmentCreditRecord]],
) -> None:
    """Attach raw or normalized process credit for schemes A/B.

    Scheme B intentionally uses episode-local normalization only: its process
    term is a strict within-trajectory redistribution of the scalar outcome
    advantage and must not depend on other sampled episodes.
    """

    cfg_clip = clip(args)
    cfg_normalization = normalization(args)
    for indices in _group_indices(args, samples).values():
        records = _records_for_group(args, samples, indices, sample_records)
        if cfg_normalization == "none":
            train_records = [record for record in records if record.marked]
            for record in train_records:
                normalized = max(-cfg_clip, min(cfg_clip, record.value))
                _write_record_value(sample_advantages[record.sample_index], record, normalized)
            continue
        if cfg_normalization == "group_turn_zscore":
            _write_group_turn_zscore(records, sample_advantages, cfg_clip)
            continue
        if cfg_normalization == "episode_turn_zscore":
            _write_episode_turn_zscore(records, sample_advantages, cfg_clip)
            continue
        raise ValueError(f"Unsupported credit assignment normalization={cfg_normalization!r}")


def _attach_tasa_state_baselines(
    args: Any,
    samples: list[Sample],
    sample_advantages: list[list[float]],
    sample_masks: list[list[float]],
    sample_records: list[list[SegmentCreditRecord]],
    scalar_rewards: list[float],
) -> None:
    """TASA-GRPO: build leave-one-out abstract-state baselines per group.

    The judge supplies task-level semantic predicates and per-student state
    change points. We reconstruct the abstract state before each assistant
    action, aggregate terminal outcome rewards over trajectories that reached
    the same state, and use ``R_i - V_{-i}(z)`` as a segment-local baseline.
    """

    cfg_clip = clip(args)
    cfg_min_peer_support = tasa_min_peer_support(args)
    cfg_normalization = tasa_local_normalization(args)
    for indices in _group_indices(args, samples).values():
        raw_records: list[SegmentCreditRecord] = []
        student_indices: list[int] = []
        for idx in indices:
            sample = samples[idx]
            if not _is_student_train_sample(sample):
                continue
            records = _tasa_segment_records_for_sample(args, idx, sample)
            if records:
                student_indices.append(idx)
                raw_records.extend(records)
        if not raw_records:
            continue

        state_rewards: dict[str, dict[int, float]] = {}
        for record in raw_records:
            state_key = record.state_key or "ROOT"
            state_rewards.setdefault(state_key, {})[record.sample_index] = float(scalar_rewards[record.sample_index])

        local_records: list[SegmentCreditRecord] = []
        raw_local_values: list[float] = []
        for record in raw_records:
            state_key = record.state_key or "ROOT"
            rewards_by_sample = state_rewards[state_key]
            peer_rewards = [
                reward
                for sample_index, reward in rewards_by_sample.items()
                if sample_index != record.sample_index
            ]
            peer_count = len(peer_rewards)
            supported = peer_count >= cfg_min_peer_support
            state_reward_mean = _mean_float(peer_rewards) if peer_rewards else 0.0
            local_advantage = float(scalar_rewards[record.sample_index]) - state_reward_mean if supported else 0.0
            if supported:
                raw_local_values.append(local_advantage)
            local_records.append(
                replace(
                    record,
                    value=local_advantage,
                    marked=supported and abs(local_advantage) > 0,
                    peer_count=peer_count,
                    state_reward_mean=state_reward_mean,
                    local_advantage=local_advantage,
                    supported=supported,
                )
            )

        if cfg_normalization == "group_segment_zscore":
            mean = _mean_float(raw_local_values)
            std = _sample_std(raw_local_values)
        else:
            mean = 0.0
            std = 0.0

        normalized_records: list[SegmentCreditRecord] = []
        train_local_values: list[float] = []
        train_base_values: list[float] = []
        base_by_sample = _group_reward_advantages(student_indices, scalar_rewards)
        normalization_has_scale = cfg_normalization != "group_segment_zscore" or std > 0
        for record in local_records:
            if record.supported and normalization_has_scale:
                if cfg_normalization == "group_segment_zscore":
                    normalized = (float(record.local_advantage or 0.0) - mean) / (std + 1e-6)
                else:
                    normalized = float(record.local_advantage or 0.0)
                normalized = max(-cfg_clip, min(cfg_clip, normalized))
                train_local_values.append(normalized)
                train_base_values.append(base_by_sample.get(record.sample_index, 0.0))
                final_record = replace(record, value=normalized, marked=abs(normalized) > 0)
                _write_record_value(sample_advantages[record.sample_index], record, normalized)
                _write_record_value(sample_masks[record.sample_index], record, 1.0)
            else:
                final_record = replace(record, value=0.0, marked=False, supported=False)
            normalized_records.append(final_record)
            sample_records[record.sample_index].append(final_record)

        _write_tasa_group_stats(
            samples=samples,
            group_indices=indices,
            records=normalized_records,
            state_rewards=state_rewards,
            local_values=train_local_values,
            base_values=train_base_values,
        )


def _write_group_turn_zscore(
    records: list[SegmentCreditRecord],
    sample_advantages: list[list[float]],
    cfg_clip: float,
) -> None:
    if not records:
        return
    values = [record.value for record in records]
    mean = sum(values) / len(values)
    if len(values) <= 1:
        std = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = max(variance, 0.0) ** 0.5
    for record in records:
        normalized = 0.0 if std <= 0 else (record.value - mean) / (std + 1e-6)
        normalized = max(-cfg_clip, min(cfg_clip, normalized))
        _write_record_value(sample_advantages[record.sample_index], record, normalized)


def _write_episode_turn_zscore(
    records: list[SegmentCreditRecord],
    sample_advantages: list[list[float]],
    cfg_clip: float,
) -> None:
    by_sample: dict[int, list[SegmentCreditRecord]] = {}
    for record in records:
        by_sample.setdefault(record.sample_index, []).append(record)
    for sample_records in by_sample.values():
        _write_group_turn_zscore(sample_records, sample_advantages, cfg_clip)


def _attach_segment_reward_group_turn_norm(
    args: Any,
    samples: list[Sample],
    sample_advantages: list[list[float]],
    sample_records: list[list[SegmentCreditRecord]],
    scalar_rewards: list[float],
) -> None:
    """Scheme C precomputation: normalize segment rewards over group turns.

    ``scalar_rewards`` is the raw scalar outcome/reward-model result before
    GRPO group normalization. Each assistant turn is one normalization sample:
    build ``raw_outcome_reward + beta * process_credit``, normalize those values
    over turns in the same group, then expand the turn value to its tokens.
    """

    cfg_clip = clip(args)
    cfg_beta = beta(args)
    for indices in _group_indices(args, samples).values():
        records = _records_for_group(args, samples, indices, sample_records)
        if not records:
            continue
        values = [float(scalar_rewards[record.sample_index]) + cfg_beta * record.value for record in records]
        mean = sum(values) / len(values)
        if len(values) <= 1:
            std = 0.0
        else:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            std = max(variance, 0.0) ** 0.5
        for record, value in zip(records, values, strict=True):
            normalized = value - mean if std <= 0 else (value - mean) / (std + 1e-6)
            normalized = max(-cfg_clip, min(cfg_clip, normalized))
            _write_record_value(sample_advantages[record.sample_index], record, normalized)
