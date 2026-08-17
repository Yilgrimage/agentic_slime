from argparse import Namespace

from slime.utils.types import Sample

from examples.agent_env.rewards.ropd import _select_train_score


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
