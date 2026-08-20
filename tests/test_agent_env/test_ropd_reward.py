from argparse import Namespace
import json

import torch

from slime.utils.types import Sample

from examples.agent_env import credit_assignment
from examples.agent_env.advantage import segment_credit_assignment_advantage
from examples.agent_env.episode import _requires_structured_env_trace
from examples.agent_env.metrics import reward_metrics
from examples.agent_env.rewards.ropd import (
    _answer_for_judge,
    _build_ca_compact_verifier_prompt,
    _ca_compact_result,
    _parse_ca_compact_batch_masks,
    _select_train_score,
)


def _sample(*, env_success: bool) -> Sample:
    return Sample(
        group_index=0,
        reward=0.0,
        response_length=1,
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        metadata={"env_success": env_success},
    )


def test_answer_process_env_success_override_and_failure_shaping():
    args = Namespace(
        reward={
            "ropd": {
                "schema_mode": "answer_process_50_50",
                "env_success_overrides_reward": True,
                "failure_shaping_beta": 0.2,
            }
        }
    )

    success_score, success_reason = _select_train_score(
        args,
        sample=_sample(env_success=True),
        answer_score=0.1,
        group_stats={},
    )
    failure_score, failure_reason = _select_train_score(
        args,
        sample=_sample(env_success=False),
        answer_score=0.8,
        group_stats={},
    )

    assert success_score == 1.0
    assert success_reason == "env_success_else_answer_process_50_50"
    assert abs(failure_score - 0.16) < 1e-9
    assert failure_reason == "env_success_else_answer_process_50_50"


def test_appworld_ropd_trace_requires_env_owned_turns() -> None:
    args = Namespace(reward={"impl": "ropd", "ropd": {"answer_mode": "trace"}})
    appworld_spec = Namespace(name="appworld")

    assert _requires_structured_env_trace(args, appworld_spec)
    assert not _requires_structured_env_trace(
        Namespace(reward={"impl": "ropd", "ropd": {"answer_mode": "final"}}),
        appworld_spec,
    )
    assert not _requires_structured_env_trace(args, Namespace(name="webshop"))


def test_appworld_reward_trace_rejects_message_fallback_without_ambient_env_name():
    args = Namespace(
        reward={
            "ropd": {
                "answer_mode": "trace",
                "trace_compression": {
                    "strip_reasoning": True,
                    "strip_tool_call": 256,
                    "strip_tool_response": 120,
                    "strip_assistant_response": True,
                    "strip_system_prompt": True,
                },
            }
        }
    )
    sample = Sample(
        prompt="prompt",
        metadata={
            "appworld": {"task_id": "demo"},
            "task_prompt": "Update the calendar.",
            "messages": [
                {"role": "assistant", "content": "apis.calendar.create_event({'title': 'sync'})"},
                {"role": "user", "content": "Execution successful."},
            ],
        },
    )

    try:
        _answer_for_judge(args, sample)
    except ValueError as exc:
        assert "lacks valid structured execution evidence" in str(exc)
    else:
        raise AssertionError("AppWorld ROPD accepted a legacy message-only trace")


def test_ca_compact_parser_accepts_tasa_state_annotations():
    args = Namespace(
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_state_aggregation",
            }
        }
    )
    answer_items = (
        {"source": "teacher", "source_index": 0, "text": "Step 1\nteacher"},
        {"source": "student", "source_index": 0, "text": "Step 1\nstudent\nStep 2\nstudent"},
        {"source": "student", "source_index": 1, "text": "Step 1\nstudent"},
    )
    payload = {
        "schema_version": "ropd.tasa_state_batch_verifier.v1",
        "milestones": [
            {
                "id": "M1",
                "predicate": "first state",
                "requires": [],
                "progress": 1,
                "transitions_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "set_steps": [1], "unset_steps": []},
                    {"trajectory_id": "S1_STUDENT", "set_steps": [], "unset_steps": []},
                ],
            },
            {
                "id": "M2",
                "predicate": "second state",
                "requires": ["M1"],
                "progress": 2,
                "transitions_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "set_steps": [2], "unset_steps": []},
                    {"trajectory_id": "S1_STUDENT", "set_steps": [], "unset_steps": []},
                ],
            },
            {
                "id": "M3",
                "predicate": "third state",
                "requires": ["M2"],
                "progress": 3,
                "transitions_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "set_steps": [], "unset_steps": []},
                    {"trajectory_id": "S1_STUDENT", "set_steps": [], "unset_steps": []},
                ],
            },
        ],
    }

    scores = _parse_ca_compact_batch_masks(args, payload, answer_items=answer_items)

    assert [item["trajectory_id"] for item in scores] == ["S0_STUDENT", "S1_STUDENT"]
    assert scores[0]["tasa_state_schema"]["milestones"][0]["id"] == "M1"
    assert scores[0]["tasa_state_schema"]["milestones"][0]["progress"] == 1.0
    assert scores[0]["tasa_state_changes"][1]["set"] == ["M2"]
    assert scores[1]["tasa_state_changes"] == []
    assert scores[0]["behaviors"] == []

    try:
        _parse_ca_compact_batch_masks(
            args,
            {**payload, "trajectory_scores": []},
            answer_items=answer_items,
        )
    except ValueError as exc:
        assert "unsupported fields" in str(exc)
    else:
        raise AssertionError("ROPD compact CA accepted the removed trajectory_scores field")


def test_ca_compact_prompt_uses_teacher_only_as_reference():
    args = Namespace(reward={"ropd": {}})
    sample = _sample(env_success=False)
    sample.metadata["task_prompt"] = "Complete the requested task."
    answer_items = (
        {"source": "teacher", "source_index": 0, "answer_index": 2, "text": "Step 1\nteacher"},
        {"source": "student", "source_index": 0, "answer_index": 1, "text": "Step 1\nstudent"},
    )

    prompt = _build_ca_compact_verifier_prompt(args, sample, answer_items=answer_items)

    assert "[REFERENCE 1]" in prompt
    assert "[S0_STUDENT]" in prompt
    assert "answer_index" not in prompt
    assert "[Student Trajectory IDs]\nS0_STUDENT" in prompt
    assert "T0_REFERENCE" not in prompt
    assert "complete_task" not in prompt
    assert "[Additional Scoring Instructions]" not in prompt

    prompt_without_teacher = _build_ca_compact_verifier_prompt(args, sample, answer_items=answer_items[1:])
    assert "[Reference Trajectories]\nNone" in prompt_without_teacher
    assert "[Student Trajectory IDs]\nS0_STUDENT" in prompt_without_teacher


def test_tasa_prompt_uses_only_milestone_centric_schema():
    args = Namespace(
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
            }
        }
    )
    sample = _sample(env_success=False)
    sample.metadata["task_prompt"] = "Complete the requested task."
    answer_items = (
        {"source": "teacher", "source_index": 0, "text": "Step 1\nteacher"},
        {"source": "student", "source_index": 0, "text": "Step 1\nstudent"},
    )

    prompt = _build_ca_compact_verifier_prompt(args, sample, answer_items=answer_items)

    assert "ropd.tasa_state_batch_verifier.v1" in prompt
    assert "transitions_by_trajectory" in prompt
    assert '"requires": []' in prompt
    assert '"behaviors"' not in prompt
    assert "bad_flags" not in prompt


def test_tasa_parser_covers_sixteen_student_trajectories():
    args = Namespace(
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
            }
        }
    )
    answer_items = tuple(
        {"source": "student", "source_index": index, "text": "Step 1\naction"}
        for index in range(16)
    )
    transitions = [
        {"trajectory_id": f"S{index}_STUDENT", "set_steps": [], "unset_steps": []}
        for index in range(16)
    ]
    payload = {
        "schema_version": "ropd.tasa_state_batch_verifier.v1",
        "milestones": [
            {
                "id": f"M{index}",
                "predicate": f"state {index}",
                "requires": [] if index == 1 else [f"M{index - 1}"],
                "progress": index,
                "transitions_by_trajectory": transitions,
            }
            for index in range(1, 4)
        ],
    }

    scores = _parse_ca_compact_batch_masks(args, payload, answer_items=answer_items)

    assert len(scores) == 16
    assert [item["trajectory_id"] for item in scores] == [f"S{index}_STUDENT" for index in range(16)]


def test_ca_compact_result_uses_env_reward_without_synthetic_judge_score():
    args = Namespace(
        reward={
            "outcome": 10.0,
            "ropd": {
                "schema_mode": "ca_compact",
                "ca_scalar_reward_source": "env_success",
            },
        }
    )
    sample = _sample(env_success=True)
    item = {
        "trajectory_id": "S0_STUDENT",
        "process_step_evidence": [
            {"criterion_id": "g1", "positive_step_indices": [1], "negative_step_indices": []}
        ],
        "behaviors": [],
    }

    result = _ca_compact_result(
        args,
        sample=sample,
        rubric={"schema_version": "ropd.ca_compact_rubric.v2"},
        rubric_source="online",
        rubric_call=None,
        judge_call=None,
        student_item=item,
        student_position=1,
    )

    assert result.score == 10.0
    assert result.raw["reward_score"] == 1.0
    assert result.raw["process_step_evidence"] == item["process_step_evidence"]
    assert "student_score" not in result.raw
    assert "teacher_scores" not in result.raw
    assert "process_score" not in result.raw


def test_tasa_gae_requires_milestone_progress():
    args = Namespace(
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
            }
        }
    )
    answer_items = (
        {"source": "student", "source_index": 0, "text": "Step 1\nstudent"},
        {"source": "student", "source_index": 1, "text": "Step 1\nstudent"},
    )
    payload = {
        "schema_version": "ropd.tasa_state_batch_verifier.v1",
        "milestones": [
            {
                "id": f"M{index}",
                "predicate": f"state {index}",
                "requires": [] if index == 1 else [f"M{index - 1}"],
                **({"progress": index} if index != 2 else {}),
                "transitions_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "set_steps": [], "unset_steps": []},
                    {"trajectory_id": "S1_STUDENT", "set_steps": [], "unset_steps": []},
                ],
            }
            for index in range(1, 4)
        ],
    }

    try:
        _parse_ca_compact_batch_masks(args, payload, answer_items=answer_items)
    except ValueError as exc:
        assert "progress" in str(exc)
    else:
        raise AssertionError("TASA-GAE accepted a milestone without progress")


def test_tasa_schema_requires_explicit_prerequisite_list():
    args = Namespace(
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_state_aggregation",
            }
        }
    )
    answer_items = (
        {"source": "student", "source_index": 0, "text": "Step 1\nstudent"},
        {"source": "student", "source_index": 1, "text": "Step 1\nstudent"},
    )
    payload = {
        "schema_version": "ropd.tasa_state_batch_verifier.v1",
        "milestones": [
            {
                "id": f"M{index}",
                "predicate": f"state {index}",
                **({"requires": [] if index == 1 else [f"M{index - 1}"]} if index != 2 else {}),
                "progress": index,
                "transitions_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "set_steps": [], "unset_steps": []},
                    {"trajectory_id": "S1_STUDENT", "set_steps": [], "unset_steps": []},
                ],
            }
            for index in range(1, 4)
        ],
    }

    try:
        _parse_ca_compact_batch_masks(args, payload, answer_items=answer_items)
    except ValueError as exc:
        assert "explicit requires list" in str(exc)
    else:
        raise AssertionError("TASA accepted a milestone without an explicit requires list")


def _tasa_sample(
    index: int,
    *,
    changes: list[dict[str, object]],
    turns: int = 2,
    schema: dict[str, object] | None = None,
) -> Sample:
    if schema is None:
        schema = {
            "milestones": [
                {"id": "M1", "predicate": "first state", "requires": [], "progress": 1.0},
                {"id": "M2", "predicate": "second state", "requires": ["M1"], "progress": 2.0},
                {"id": "M3", "predicate": "third state", "requires": ["M1"], "progress": 3.0},
            ],
        }
    return Sample(
        group_index=0,
        index=index,
        rollout_id=index,
        response_length=turns * 2,
        loss_mask=[1] * (turns * 2),
        status=Sample.Status.COMPLETED,
        metadata={
            "token_segments": [
                {
                    "kind": "assistant",
                    "turn": turn,
                    "token_count": 2,
                    "loss_mask_sum": 2,
                    "text": f"action {turn + 1}",
                }
                for turn in range(turns)
            ],
            "rm_reward": {
                "raw": {
                    "tasa_state_schema": schema,
                    "tasa_state_changes": changes,
                }
            },
        },
    )


def test_tasa_gae_attaches_prior_posterior_segment_advantages():
    args = Namespace(
        n_samples_per_prompt=3,
        reward={
            "outcome": 1.0,
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
                "tasa_min_peer_support": 1,
                "tasa_local_normalization": "none",
                "tasa_prior_kappa": 4.0,
                "tasa_lambda": 0.0,
                "tasa_prior_root": 0.1,
                "tasa_prior_success": 0.9,
                "tasa_outcome_scale": 1.0,
            },
        },
    )
    samples = [
        _tasa_sample(
            0,
            changes=[
                {"step": 1, "set": ["M1"], "unset": []},
                {"step": 2, "set": ["M2"], "unset": []},
            ],
        ),
        _tasa_sample(1, changes=[{"step": 1, "set": ["M1"], "unset": []}]),
        _tasa_sample(2, changes=[]),
    ]

    credit_assignment.attach_process_advantages(args, samples, scalar_rewards=[1.0, 0.0, 0.0])

    first_adv = samples[0].metadata["process_advantages"]
    second_adv = samples[1].metadata["process_advantages"]
    assert first_adv[0] > 0
    assert first_adv[2] > 0
    assert second_adv[0] > 0
    assert second_adv[2] < 0
    stats = samples[0].metadata["credit_assignment"]
    assert stats["advantage_mode"] == "teacher_anchored_value_gae"
    assert stats["tasa_prior_kappa"] == 4.0
    assert stats["tasa_group_unique_states"] >= 3
    assert stats["tasa_td_delta_abs_mean"] > 0
    assert stats["tasa_group_teacher_prior_available_rate"] == 1.0
    assert stats["tasa_train_token_coverage_rate"] == 1.0


def test_tasa_no_teacher_skips_unsupported_state_and_uses_mc_anchor():
    args = Namespace(
        n_samples_per_prompt=4,
        reward={
            "outcome": 1.0,
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
                "tasa_min_peer_support": 2,
                "tasa_local_normalization": "none",
                "tasa_use_teacher_prior": False,
                "tasa_enforce_prerequisites": True,
                "tasa_lambda": 0.5,
                "tasa_outcome_scale": 1.0,
                "clip": 10.0,
            },
        },
    )
    samples = [
        _tasa_sample(
            0,
            turns=4,
            changes=[
                {"step": 1, "set": ["M1"], "unset": []},
                {"step": 2, "set": ["M2"], "unset": []},
                {"step": 3, "set": ["M3"], "unset": ["M2"]},
            ],
        ),
        _tasa_sample(
            1,
            turns=4,
            changes=[
                {"step": 1, "set": ["M1"], "unset": []},
                {"step": 3, "set": ["M3"], "unset": []},
            ],
        ),
        _tasa_sample(
            2,
            turns=4,
            changes=[
                {"step": 1, "set": ["M1"], "unset": []},
                {"step": 3, "set": ["M3"], "unset": []},
            ],
        ),
        _tasa_sample(3, turns=4, changes=[{"step": 1, "set": ["M1"], "unset": []}]),
    ]

    credit_assignment.attach_process_advantages(args, samples, scalar_rewards=[1.0, 0.0, 1.0, 0.0])

    first_adv = samples[0].metadata["process_advantages"]
    assert first_adv[4] > 0
    assert abs(first_adv[2] - 0.5 * first_adv[4]) < 1e-6
    stats = samples[0].metadata["credit_assignment"]
    assert stats["tasa_group_teacher_prior_available_rate"] == 0.0
    assert stats["tasa_group_mc_reliable_rate"] < 1.0
    assert stats["tasa_group_value_source_mc_rate"] > 0.0
    assert stats["tasa_group_value_source_none_rate"] > 0.0
    assert stats["tasa_train_token_coverage_rate"] == 1.0


def test_tasa_prerequisite_enforcement_filters_invalid_state_transition():
    schema = {
        "milestones": [
            {"id": "M1", "predicate": "first", "requires": [], "progress": 1.0},
            {"id": "M2", "predicate": "second", "requires": ["M1"], "progress": 2.0},
            {"id": "M3", "predicate": "third", "requires": ["M2"], "progress": 3.0},
        ]
    }
    sample = _tasa_sample(0, changes=[{"step": 1, "set": ["M3"], "unset": []}], schema=schema)
    strict_args = Namespace(
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
                "tasa_enforce_prerequisites": True,
            }
        }
    )
    loose_args = Namespace(
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
                "tasa_enforce_prerequisites": False,
            }
        }
    )

    strict_records = credit_assignment._tasa_segment_records_for_sample(strict_args, 0, sample)
    loose_records = credit_assignment._tasa_segment_records_for_sample(loose_args, 0, sample)

    assert strict_records[0].next_state_key == "ROOT"
    assert strict_records[0].prerequisites_satisfied is False
    assert loose_records[0].next_state_key == "M3"
    assert loose_records[0].prerequisites_satisfied is True


def test_tasa_segment_local_gae_stops_at_each_boundary():
    assert credit_assignment._segment_local_gae(2.0, 3, 0.0) == [0.0, 0.0, 2.0]
    assert credit_assignment._segment_local_gae(2.0, 3, 0.5) == [0.5, 1.0, 2.0]
    assert credit_assignment._segment_local_gae(2.0, 3, 1.0) == [2.0, 2.0, 2.0]


def test_tasa_rms_normalization_preserves_advantage_sign():
    records = [
        credit_assignment.SegmentCreditRecord(0, 0, 1, 1, 0.0, False, supported=True, gae_advantage=0.25),
        credit_assignment.SegmentCreditRecord(1, 0, 1, 1, 0.0, False, supported=True, gae_advantage=-1.0),
    ]

    normalized = credit_assignment._normalize_tasa_gae_values(records, "group_segment_rms", 10.0)

    assert normalized[(0, 0, 1)] > 0
    assert normalized[(1, 0, 1)] < 0


def test_tasa_value_source_masks_cover_full_and_no_teacher_modes():
    outcomes = {0: 1.0, 1: 0.0, 2: 1.0}

    blended = credit_assignment._tasa_state_value(
        prior=0.6,
        kappa=4.0,
        outcomes_by_sample=outcomes,
        exclude_sample_index=0,
        min_peer_support=2,
    )
    teacher_only = credit_assignment._tasa_state_value(
        prior=0.6,
        kappa=4.0,
        outcomes_by_sample=outcomes,
        exclude_sample_index=0,
        min_peer_support=3,
    )
    mc_only = credit_assignment._tasa_state_value(
        prior=None,
        kappa=4.0,
        outcomes_by_sample=outcomes,
        exclude_sample_index=0,
        min_peer_support=2,
    )
    unavailable = credit_assignment._tasa_state_value(
        prior=None,
        kappa=4.0,
        outcomes_by_sample=outcomes,
        exclude_sample_index=0,
        min_peer_support=3,
    )

    assert blended.teacher_prior_available and blended.mc_reliable and blended.source == "teacher_mc"
    assert teacher_only.value == 0.6 and teacher_only.source == "teacher"
    assert mc_only.value == 0.5 and mc_only.source == "mc"
    assert unavailable.value is None and not unavailable.anchor_eligible and unavailable.source == "none"


def test_tasa_metrics_report_masks_and_segment_coverage():
    sample = _sample(env_success=False)
    sample.metadata["credit_assignment"] = {
        "tasa_train_token_coverage_rate": 1.0,
        "tasa_group_anchor_eligible_rate": 0.75,
        "tasa_group_teacher_prior_available_rate": 0.0,
        "tasa_group_mc_reliable_rate": 0.5,
        "tasa_group_prerequisite_valid_rate": 0.875,
        "tasa_group_semantic_segment_count": 4.0,
        "tasa_group_semantic_segment_length_mean": 2.5,
        "tasa_group_value_source_teacher_rate": 0.0,
        "tasa_group_value_source_mc_rate": 0.5,
        "tasa_group_value_source_teacher_mc_rate": 0.0,
        "tasa_group_value_source_none_rate": 0.5,
    }

    metrics = reward_metrics([sample])

    assert metrics["reward/credit_assignment/tasa_train_token_coverage_rate_mean"] == 1.0
    assert metrics["reward/credit_assignment/tasa_mc_reliable_rate_mean"] == 0.5
    assert metrics["reward/credit_assignment/tasa_value_source_none_rate_mean"] == 0.5


def test_tasa_debug_dump_exposes_state_value_and_segment_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_CREDIT_DUMP_N", "1")
    monkeypatch.setenv("AGENT_ENV_CREDIT_DUMP_TOTAL_N", "4")
    monkeypatch.setenv("AGENT_ENV_CREDIT_DUMP_DIR", str(tmp_path))
    args = Namespace(
        n_samples_per_prompt=3,
        reward={
            "outcome": 1.0,
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
                "tasa_min_peer_support": 1,
                "tasa_local_normalization": "none",
                "tasa_use_teacher_prior": True,
                "tasa_lambda": 0.5,
                "tasa_outcome_scale": 1.0,
            },
        },
    )
    samples = [
        _tasa_sample(0, changes=[{"step": 1, "set": ["M1"], "unset": []}]),
        _tasa_sample(1, changes=[{"step": 1, "set": ["M1"], "unset": []}]),
        _tasa_sample(2, changes=[]),
    ]
    samples[0].metadata["env_success"] = True

    credit_assignment.attach_process_advantages(args, samples, scalar_rewards=[1.0, 0.0, 0.0])

    dump_file = next(tmp_path.glob("credit_assignment_pid*.jsonl"))
    payload = json.loads(dump_file.read_text().splitlines()[0])
    segment = payload["segments"][0]
    assert payload["schema_version"] == "agent_env.credit_assignment_audit.v2"
    assert payload["step_marks"] == {}
    assert payload["step_mark_events"] == []
    assert isinstance(payload["credit_assignment"]["tasa_group_peer_count_histogram"], dict)
    for key in (
        "action",
        "state_key",
        "next_state_key",
        "state_prior",
        "peer_count",
        "state_value",
        "td_delta",
        "gae_advantage",
        "teacher_prior_available",
        "mc_reliable",
        "value_source",
        "anchor_state_key",
        "boundary_state_key",
        "segment_index",
        "distance_to_boundary",
    ):
        assert key in segment


def test_tasa_train_token_coverage_excludes_environment_tokens():
    args = Namespace(
        n_samples_per_prompt=3,
        reward={
            "outcome": 1.0,
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
                "tasa_min_peer_support": 1,
                "tasa_local_normalization": "none",
                "tasa_use_teacher_prior": True,
                "tasa_lambda": 0.5,
                "tasa_outcome_scale": 1.0,
            },
        },
    )
    samples = [
        _tasa_sample(index, changes=[{"step": 1, "set": ["M1"], "unset": []}])
        for index in range(3)
    ]
    for sample in samples:
        sample.response_length = 6
        sample.loss_mask = [1, 1, 0, 0, 1, 1]
        sample.metadata["token_segments"] = [
            {"kind": "assistant", "turn": 0, "token_count": 2, "loss_mask_sum": 2, "text": "action 1"},
            {"kind": "environment", "turn": 0, "token_count": 2, "loss_mask_sum": 0, "text": "result"},
            {"kind": "assistant", "turn": 1, "token_count": 2, "loss_mask_sum": 2, "text": "action 2"},
        ]

    credit_assignment.attach_process_advantages(args, samples, scalar_rewards=[1.0, 0.0, 0.0])

    stats = samples[0].metadata["credit_assignment"]
    assert stats["tasa_supported_token_rate"] == 4 / 6
    assert stats["tasa_train_token_coverage_rate"] == 1.0


def test_tasa_gae_train_hook_replaces_supported_tokens_and_scales_teacher():
    args = Namespace(
        advantage_estimator="grpo",
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "teacher_anchored_value_gae",
                "tasa_teacher_weight": 0.5,
            }
        },
    )
    rollout_data = {
        "kl": [torch.zeros(2), torch.zeros(2)],
        "rewards": [1.0, 2.0],
        "process_advantages": [torch.tensor([9.0, 9.0]), torch.tensor([9.0, 9.0])],
        "process_advantage_masks": [torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])],
        "loss_masks": [torch.tensor([1, 1]), torch.tensor([1, 1])],
        "off_policy_loss_masks": [torch.tensor([0, 0]), torch.tensor([1, 1])],
    }

    segment_credit_assignment_advantage(args, rollout_data)

    assert torch.allclose(rollout_data["advantages"][0], torch.tensor([9.0, 9.0]))
    assert torch.allclose(rollout_data["advantages"][1], torch.tensor([1.0, 1.0]))
