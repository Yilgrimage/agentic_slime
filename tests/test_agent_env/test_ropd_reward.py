from argparse import Namespace

import torch

from slime.utils.types import Sample

from examples.agent_env import credit_assignment
from examples.agent_env.advantage import segment_credit_assignment_advantage
from examples.agent_env.rewards.ropd import (
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
        "schema_version": "ropd.ca_compact_batch_verifier.v2",
        "behaviors": [
            {
                "behavior_id": "g1",
                "polarity": "good",
                "description": "good action",
                "hits_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "step_indices": [1]},
                    {"trajectory_id": "S1_STUDENT", "step_indices": []},
                ],
            },
            {
                "behavior_id": "g2",
                "polarity": "good",
                "description": "good followup",
                "hits_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "step_indices": [2]},
                    {"trajectory_id": "S1_STUDENT", "step_indices": []},
                ],
            },
            {
                "behavior_id": "b1",
                "polarity": "bad",
                "description": "bad action",
                "hits_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "step_indices": []},
                    {"trajectory_id": "S1_STUDENT", "step_indices": [1]},
                ],
            },
        ],
        "tasa_state_schema": {
            "milestones": [
                {"id": "M1", "predicate": "first state", "requires": []},
                {"id": "M2", "predicate": "second state", "requires": ["M1"]},
            ],
            "bad_flags": [{"id": "B1", "predicate": "wrong state"}],
        },
        "tasa_state_changes": [
            {
                "trajectory_id": "S0_STUDENT",
                "changes": [
                    {"step": 1, "set": ["M1"], "unset": []},
                    {"step": 2, "set": ["M2"], "unset": []},
                ],
            },
            {
                "trajectory_id": "S1_STUDENT",
                "changes": [{"step": 1, "set": ["B1"], "unset": []}],
            },
        ],
    }

    scores = _parse_ca_compact_batch_masks(args, payload, answer_items=answer_items)

    assert [item["trajectory_id"] for item in scores] == ["S0_STUDENT", "S1_STUDENT"]
    assert scores[0]["tasa_state_schema"]["milestones"][0]["id"] == "M1"
    assert scores[0]["tasa_state_schema"]["milestones"][0]["progress"] == 1.0
    assert scores[0]["tasa_state_changes"][1]["set"] == ["M2"]
    assert scores[1]["tasa_state_changes"][0]["set"] == ["B1"]

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
        "schema_version": "ropd.ca_compact_batch_verifier.v2",
        "behaviors": [
            {
                "behavior_id": "g1",
                "polarity": "good",
                "description": "good action",
                "hits_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "step_indices": [1]},
                    {"trajectory_id": "S1_STUDENT", "step_indices": []},
                ],
            },
            {
                "behavior_id": "g2",
                "polarity": "good",
                "description": "good followup",
                "hits_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "step_indices": []},
                    {"trajectory_id": "S1_STUDENT", "step_indices": [1]},
                ],
            },
            {
                "behavior_id": "b1",
                "polarity": "bad",
                "description": "bad action",
                "hits_by_trajectory": [
                    {"trajectory_id": "S0_STUDENT", "step_indices": []},
                    {"trajectory_id": "S1_STUDENT", "step_indices": []},
                ],
            },
        ],
        "tasa_state_schema": {
            "milestones": [{"id": "M1", "predicate": "state reached", "requires": []}],
            "bad_flags": [
                {"id": "B1", "predicate": "wrong state"},
                {"id": "B2", "predicate": "worse state"},
            ],
        },
        "tasa_state_changes": [
            {"trajectory_id": "S0_STUDENT", "changes": [{"step": 1, "set": ["M1"], "unset": []}]},
            {"trajectory_id": "S1_STUDENT", "changes": [{"step": 1, "set": ["B1"], "unset": []}]},
        ],
    }

    try:
        _parse_ca_compact_batch_masks(args, payload, answer_items=answer_items)
    except ValueError as exc:
        assert "progress" in str(exc)
    else:
        raise AssertionError("TASA-GAE accepted a milestone without progress")


def _tasa_sample(index: int, *, changes: list[dict[str, object]]) -> Sample:
    schema = {
        "milestones": [
            {"id": "M1", "predicate": "first state", "requires": [], "progress": 1.0},
            {"id": "M2", "predicate": "second state", "requires": ["M1"], "progress": 2.0},
        ],
        "bad_flags": [{"id": "B1", "predicate": "wrong state"}],
    }
    return Sample(
        group_index=0,
        index=index,
        rollout_id=index,
        response_length=4,
        loss_mask=[1, 1, 1, 1],
        status=Sample.Status.COMPLETED,
        metadata={
            "token_segments": [
                {"kind": "assistant", "turn": 0, "token_count": 2, "loss_mask_sum": 2},
                {"kind": "assistant", "turn": 1, "token_count": 2, "loss_mask_sum": 2},
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
                "tasa_bad_flag_penalty": 0.2,
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
        "process_advantage_masks": [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])],
        "off_policy_loss_masks": [torch.tensor([0, 0]), torch.tensor([1, 1])],
    }

    segment_credit_assignment_advantage(args, rollout_data)

    assert torch.allclose(rollout_data["advantages"][0], torch.tensor([9.0, 1.0]))
    assert torch.allclose(rollout_data["advantages"][1], torch.tensor([1.0, 1.0]))
