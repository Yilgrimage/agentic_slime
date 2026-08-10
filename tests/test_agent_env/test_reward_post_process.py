from argparse import Namespace

import torch

from slime.utils.types import Sample

from examples.agent_env.advantage import segment_credit_assignment_advantage
from examples.agent_env.reward_post_process import check_reward_nonzero_std, post_process_rewards


def _sample(reward: float, *, group_index: int = 0, off_policy: bool = False) -> Sample:
    metadata = {"off_policy_sample": True} if off_policy else {}
    return Sample(
        group_index=group_index,
        reward=reward,
        response_length=1,
        loss_mask=[1],
        off_policy_loss_mask=[1] if off_policy else None,
        status=Sample.Status.COMPLETED,
        metadata=metadata,
    )


def test_luffy_teacher_sample_uses_same_grpo_normalizer_as_students():
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=8,
        reward_key=None,
    )
    samples = [_sample(1.0) for _ in range(7)]
    samples.append(_sample(1.078, off_policy=True))

    raw_rewards, rewards = post_process_rewards(args, samples)

    assert raw_rewards == [1.0] * 7 + [1.078]
    assert max(abs(value) for value in rewards) < 3.0
    assert abs(rewards[-1] - 2.474873) < 1e-5


def test_credit_assignment_attaches_response_aligned_process_advantages():
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        reward_key=None,
        reward={
            "credit_assignment": {
                "enable": True,
                "beta": 0.05,
                "clip": 2.0,
                "step_index_base": 1,
            },
        },
    )
    sample = Sample(
        group_index=0,
        reward=0.0,
        response_length=5,
        loss_mask=[1, 1, 0, 1, 1],
        status=Sample.Status.COMPLETED,
        metadata={
            "token_segments": [
                {"kind": "initial_prompt", "turn": 0, "token_count": 3, "loss_mask_sum": 0, "text": "prompt"},
                {"kind": "assistant", "turn": 0, "token_count": 2, "loss_mask_sum": 2, "text": "a1"},
                {"kind": "environment", "turn": 0, "token_count": 1, "loss_mask_sum": 0, "text": "obs"},
                {"kind": "assistant", "turn": 1, "token_count": 2, "loss_mask_sum": 2, "text": "a2"},
            ],
            "rm_reward": {
                "score": 0.0,
                "raw": {
                    "process_step_evidence": [
                        {
                            "criterion_id": "r1",
                            "positive_step_indices": [1],
                            "negative_step_indices": [2],
                        }
                    ]
                },
            },
        },
    )
    other = _sample(0.0)

    post_process_rewards(args, [sample, other])

    process_advantages = sample.metadata["process_advantages"]
    assert len(process_advantages) == sample.response_length
    assert process_advantages[0] > 0
    assert process_advantages[1] > 0
    assert process_advantages[2] == 0
    assert process_advantages[3] < 0
    assert process_advantages[4] < 0
    assert sample.metadata["credit_assignment"]["nonzero_tokens"] == 4


def test_credit_assignment_process_signal_keeps_zero_reward_group():
    args = Namespace(
        n_samples_per_prompt=2,
        reward_key=None,
        reward={
            "credit_assignment": {
                "enable": True,
                "beta": 0.05,
                "clip": 2.0,
                "step_index_base": 1,
            },
        },
    )
    sample = Sample(
        group_index=0,
        reward=0.0,
        response_length=4,
        loss_mask=[1, 1, 1, 1],
        status=Sample.Status.COMPLETED,
        metadata={
            "token_segments": [
                {"kind": "initial_prompt", "turn": 0, "token_count": 2, "loss_mask_sum": 0, "text": "prompt"},
                {"kind": "assistant", "turn": 0, "token_count": 2, "loss_mask_sum": 2, "text": "a1"},
                {"kind": "assistant", "turn": 1, "token_count": 2, "loss_mask_sum": 2, "text": "a2"},
            ],
            "rm_reward": {
                "score": 0.0,
                "raw": {
                    "process_step_evidence": [
                        {
                            "criterion_id": "r1",
                            "positive_step_indices": [1],
                            "negative_step_indices": [2],
                        }
                    ]
                },
            },
        },
    )
    output = check_reward_nonzero_std(args, [sample, _sample(0.0)])

    assert output.keep


def test_segment_credit_assignment_advantage_adds_process_term():
    args = Namespace(
        advantage_estimator="grpo",
        reward={"credit_assignment": {"enable": True, "beta": 0.5}},
    )
    rollout_data = {
        "kl": [torch.zeros(3, dtype=torch.float32)],
        "rewards": [2.0],
        "process_advantages": [torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)],
    }

    segment_credit_assignment_advantage(args, rollout_data)

    assert torch.allclose(rollout_data["advantages"][0], torch.tensor([2.5, 2.0, 1.5]))
    assert rollout_data["returns"][0] is rollout_data["advantages"][0]
