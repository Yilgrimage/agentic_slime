from argparse import Namespace

from slime.utils.types import Sample

from examples.agent_env.reward_post_process import post_process_rewards


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
