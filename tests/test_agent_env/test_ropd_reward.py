from argparse import Namespace

from slime.utils.types import Sample

from examples.agent_env.rewards.ropd import _parse_ca_compact_batch_scores, _select_train_score


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
        "schema_version": "ropd.ca_compact_batch_verifier.v1",
        "behaviors": [
            {
                "behavior_id": "g1",
                "polarity": "good",
                "description": "good action",
                "weight": 1.0,
                "hits_by_trajectory": [
                    {"trajectory_id": "T0_REFERENCE", "step_indices": [1]},
                    {"trajectory_id": "S0_STUDENT", "step_indices": [1]},
                    {"trajectory_id": "S1_STUDENT", "step_indices": []},
                ],
            },
            {
                "behavior_id": "g2",
                "polarity": "good",
                "description": "good followup",
                "weight": 1.0,
                "hits_by_trajectory": [
                    {"trajectory_id": "T0_REFERENCE", "step_indices": []},
                    {"trajectory_id": "S0_STUDENT", "step_indices": [2]},
                    {"trajectory_id": "S1_STUDENT", "step_indices": []},
                ],
            },
            {
                "behavior_id": "b1",
                "polarity": "bad",
                "description": "bad action",
                "weight": 1.0,
                "hits_by_trajectory": [
                    {"trajectory_id": "T0_REFERENCE", "step_indices": []},
                    {"trajectory_id": "S0_STUDENT", "step_indices": []},
                    {"trajectory_id": "S1_STUDENT", "step_indices": [1]},
                ],
            },
        ],
        "trajectory_scores": [
            {"trajectory_id": "T0_REFERENCE", "process_score": 1.0, "quality": "strong"},
            {"trajectory_id": "S0_STUDENT", "process_score": 0.8, "quality": "useful"},
            {"trajectory_id": "S1_STUDENT", "process_score": 0.1, "quality": "weak"},
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

    scores = _parse_ca_compact_batch_scores(args, payload, answer_items=answer_items)

    student_scores = [item for item in scores if item["trajectory_id"].startswith("S")]
    assert student_scores[0]["tasa_state_schema"]["milestones"][0]["id"] == "M1"
    assert student_scores[0]["tasa_state_changes"][1]["set"] == ["M2"]
    assert student_scores[1]["tasa_state_changes"][0]["set"] == ["B1"]
