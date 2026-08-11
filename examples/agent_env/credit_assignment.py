from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from examples.agent_env import dump
from examples.agent_env.rollout import arg, metadata
from examples.agent_env.rewards.config import reward_cfg_path
from examples.agent_env.rewards.extractors import bool_value, float_value, int_value


@dataclass(frozen=True)
class SegmentCreditRecord:
    sample_index: int
    start: int
    end: int
    step: int
    value: float
    marked: bool


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
    value = str(config(args).get("normalization", "none")).strip().lower()
    if value != "none":
        raise ValueError("reward.credit_assignment.normalization must be none")
    return value


def milestone_reward(args: Any) -> float:
    return float_value(config(args).get("milestone_reward", 2.0), 2.0)


def predecessor_reward(args: Any) -> float:
    value = float_value(config(args).get("predecessor_reward", 0.5), 0.5)
    return max(0.0, value)


def negative_reward(args: Any) -> float:
    return float_value(config(args).get("negative_reward", 0.0), 0.0)


def success_milestone_tail_steps(args: Any) -> int:
    value = int_value(config(args).get("success_milestone_tail_steps", 1), 1)
    if value < 0:
        raise ValueError("reward.credit_assignment.success_milestone_tail_steps must be non-negative")
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
    marks: dict[int, float],
    step: int,
    available_steps: list[int],
    milestone_value: float,
    predecessor_value: float,
) -> None:
    marks[step] = max(marks.get(step, 0.0), milestone_value)
    for previous_step in available_steps:
        if previous_step >= step:
            break
        marks[previous_step] = max(marks.get(previous_step, 0.0), predecessor_value)


def _success_tail_milestone_steps(args: Any, sample: Sample, available_steps: list[int]) -> list[int]:
    if not bool(metadata(sample).get("env_success", False)):
        return []
    tail_steps = success_milestone_tail_steps(args)
    if tail_steps <= 0:
        return []
    return available_steps[-tail_steps:]


def _sample_step_marks(args: Any, sample: Sample, *, available_steps: list[int] | None = None) -> dict[int, float]:
    shaping_mode(args)
    raw = _rm_raw(sample)
    evidence = raw.get("process_step_evidence")
    available = sorted(dict.fromkeys(step for step in (available_steps or []) if step >= 0))
    milestone_value = milestone_reward(args)
    predecessor_value = predecessor_reward(args)
    negative_value = negative_reward(args)
    marks: dict[int, float] = {}
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
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
                    marks=marks,
                    step=step,
                    available_steps=available,
                    milestone_value=milestone_value,
                    predecessor_value=predecessor_value,
                )
            if negative_value != 0.0:
                for step in negative_steps:
                    if step in marks and marks[step] > 0:
                        continue
                    marks[step] = min(marks.get(step, 0.0), negative_value)

    for step in _success_tail_milestone_steps(args, sample, available):
        _mark_milestone(
            marks=marks,
            step=step,
            available_steps=available,
            milestone_value=milestone_value,
            predecessor_value=predecessor_value,
        )
    return marks


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
        "process_step_evidence": process_evidence if isinstance(process_evidence, list) else [],
        "segments": [
            {
                "start": record.start,
                "end": record.end,
                "step": record.step,
                "raw_step_value": record.value,
                "advantage_value": advantages[record.start] if 0 <= record.start < len(advantages) else 0.0,
                "marked": record.marked,
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


def attach_process_advantages(args: Any, samples: list[Sample]) -> None:
    if not enabled(args):
        return

    sample_advantages: list[list[float]] = []
    sample_records: list[list[SegmentCreditRecord]] = []
    for sample in samples:
        response_length = int(getattr(sample, "response_length", 0) or 0)
        sample_advantages.append([0.0] * max(0, response_length))
        sample_records.append([])

    cfg_clip = clip(args)
    cfg_normalization = normalization(args)
    for indices in _group_indices(args, samples).values():
        records: list[SegmentCreditRecord] = []
        for idx in indices:
            sample = samples[idx]
            if not _is_student_train_sample(sample):
                continue
            records.extend(_segment_records_for_sample(args, idx, sample))
        for record in records:
            sample_records[record.sample_index].append(record)
        train_records = [record for record in records if record.marked]
        for record in train_records:
            if cfg_normalization != "none":
                raise ValueError("reward.credit_assignment.normalization must be none")
            normalized = max(-cfg_clip, min(cfg_clip, record.value))
            target = sample_advantages[record.sample_index]
            for offset in range(record.start, min(record.end, len(target))):
                target[offset] = normalized

    for sample, values in zip(samples, sample_advantages, strict=True):
        sample_metadata = metadata(sample)
        nonzero = sum(1 for value in values if abs(value) > 0)
        sample_metadata["process_advantages"] = values
        sample_metadata["credit_assignment"] = {
            "enabled": True,
            "beta": beta(args),
            "shaping_mode": shaping_mode(args),
            "normalization": normalization(args),
            "milestone_reward": milestone_reward(args),
            "predecessor_reward": predecessor_reward(args),
            "negative_reward": negative_reward(args),
            "success_milestone_tail_steps": success_milestone_tail_steps(args),
            "clip": clip(args),
            "nonzero_tokens": nonzero,
            "response_length": len(values),
            "nonzero_token_rate": (nonzero / len(values)) if values else 0.0,
        }

    _maybe_dump_credit_assignment(args, samples, sample_advantages=sample_advantages, sample_records=sample_records)
