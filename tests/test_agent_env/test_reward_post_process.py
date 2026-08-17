from argparse import Namespace
import json

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
    assert abs(rewards[-1] - 2.474873) < 1e-4


def test_reward_group_dump_records_raw_and_normalized_rewards(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_ENV_REWARD_GROUP_DUMP_N", "10")
    monkeypatch.setenv("AGENT_ENV_REWARD_GROUP_DUMP_TOTAL_N", "10")
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        reward_key=None,
    )
    samples = [_sample(1.0), _sample(3.0)]

    raw_rewards, rewards = post_process_rewards(args, samples)

    assert raw_rewards == [1.0, 3.0]
    assert rewards == [-1.0, 1.0]
    dump_files = list((tmp_path / "reward_artifacts" / "reward_groups").glob("reward_group_pid*.jsonl"))
    assert len(dump_files) == 1
    payload = json.loads(dump_files[0].read_text().splitlines()[0])
    assert payload["sample_count"] == 2
    assert payload["active_count"] == 2
    assert payload["raw_reward_stats"]["mean"] == 2.0
    assert payload["normalized_reward_stats"]["mean"] == 0.0
    assert [row["normalized_reward"] for row in payload["samples"]] == [-1.0, 1.0]


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
                "normalization": "group_turn_zscore",
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


def test_segment_credit_assignment_advantage_reweights_outcome_sign():
    args = Namespace(
        advantage_estimator="grpo",
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "outcome_reweight",
                "beta": 0.5,
            }
        },
    )
    rollout_data = {
        "kl": [torch.zeros(2, dtype=torch.float32)],
        "rewards": [2.0],
        "process_advantages": [torch.tensor([2.0, -2.0], dtype=torch.float32)],
    }

    segment_credit_assignment_advantage(args, rollout_data)

    expected = 2.0 * torch.clamp(1.0 + 0.5 * torch.tensor([2.0, -2.0]), min=1e-6)
    assert torch.allclose(rollout_data["advantages"][0], expected)
    assert torch.all(rollout_data["advantages"][0] > 0)


def test_segment_credit_assignment_advantage_reweight_clamps_sign():
    args = Namespace(
        advantage_estimator="grpo",
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "outcome_reweight",
                "beta": 10.0,
            }
        },
    )
    rollout_data = {
        "kl": [torch.zeros(2, dtype=torch.float32)],
        "rewards": [2.0],
        "process_advantages": [torch.tensor([-10.0, 10.0], dtype=torch.float32)],
    }

    segment_credit_assignment_advantage(args, rollout_data)

    assert torch.all(rollout_data["advantages"][0] > 0)


def test_credit_assignment_reweight_mode_uses_episode_local_process_normalization():
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        reward_key=None,
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "outcome_reweight",
                "beta": 0.5,
                "clip": 10.0,
                "step_index_base": 1,
            },
        },
    )
    first = Sample(
        group_index=0,
        reward=0.0,
        response_length=4,
        loss_mask=[1, 1, 1, 1],
        status=Sample.Status.COMPLETED,
        metadata={
            "token_segments": [
                {"kind": "initial_prompt", "turn": 0, "token_count": 1, "loss_mask_sum": 0},
                {"kind": "assistant", "turn": 0, "token_count": 2, "loss_mask_sum": 2},
                {"kind": "assistant", "turn": 1, "token_count": 2, "loss_mask_sum": 2},
            ],
            "rm_reward": {
                "score": 0.0,
                "raw": {"process_step_evidence": [{"criterion_id": "m1", "positive_step_indices": [1]}]},
            },
        },
    )
    second = Sample(
        group_index=0,
        reward=0.0,
        response_length=4,
        loss_mask=[1, 1, 1, 1],
        status=Sample.Status.COMPLETED,
        metadata={
            "token_segments": [
                {"kind": "initial_prompt", "turn": 0, "token_count": 1, "loss_mask_sum": 0},
                {"kind": "assistant", "turn": 0, "token_count": 2, "loss_mask_sum": 2},
                {"kind": "assistant", "turn": 1, "token_count": 2, "loss_mask_sum": 2},
            ],
            "rm_reward": {"score": 0.0, "raw": {}},
        },
    )

    post_process_rewards(args, [first, second])

    first_adv = torch.tensor(first.metadata["process_advantages"], dtype=torch.float32)
    second_adv = torch.tensor(second.metadata["process_advantages"], dtype=torch.float32)
    assert torch.all(first_adv[:2] > 0)
    assert torch.all(first_adv[2:] < 0)
    assert torch.allclose(second_adv, torch.zeros_like(second_adv))
    assert first.metadata["credit_assignment"]["normalization"] == "episode_turn_zscore"


def test_segment_credit_assignment_advantage_uses_precomputed_segment_rewards_for_mode_c():
    args = Namespace(
        advantage_estimator="grpo",
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "segment_reward_group_turn_norm",
                "beta": 0.0,
            }
        },
    )
    precomputed = torch.tensor([0.7, -0.7], dtype=torch.float32)
    rollout_data = {
        "kl": [torch.zeros(2, dtype=torch.float32)],
        "rewards": [999.0],
        "process_advantages": [precomputed],
    }

    segment_credit_assignment_advantage(args, rollout_data)

    assert torch.allclose(rollout_data["advantages"][0], precomputed)
    assert rollout_data["returns"][0] is rollout_data["advantages"][0]


def test_credit_assignment_mode_c_builds_group_turn_segment_rewards_in_post_process():
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        reward_key=None,
        reward={
            "credit_assignment": {
                "enable": True,
                "advantage_mode": "segment_reward_group_turn_norm",
                "beta": 1.0,
                "clip": 10.0,
                "step_index_base": 1,
            },
        },
    )
    strong = Sample(
        group_index=0,
        reward=1.0,
        response_length=2,
        loss_mask=[1, 1],
        status=Sample.Status.COMPLETED,
        metadata={
            "token_segments": [
                {"kind": "initial_prompt", "turn": 0, "token_count": 1, "loss_mask_sum": 0},
                {"kind": "assistant", "turn": 0, "token_count": 2, "loss_mask_sum": 2},
            ],
            "rm_reward": {
                "score": 1.0,
                "raw": {
                    "process_step_evidence": [
                        {"criterion_id": "m1", "positive_step_indices": [1]},
                    ]
                },
            },
        },
    )
    weak = Sample(
        group_index=0,
        reward=0.0,
        response_length=2,
        loss_mask=[1, 1],
        status=Sample.Status.COMPLETED,
        metadata={
            "token_segments": [
                {"kind": "initial_prompt", "turn": 0, "token_count": 1, "loss_mask_sum": 0},
                {"kind": "assistant", "turn": 0, "token_count": 2, "loss_mask_sum": 2},
            ],
            "rm_reward": {"score": 0.0, "raw": {}},
        },
    )

    post_process_rewards(args, [strong, weak])

    strong_adv = torch.tensor(strong.metadata["process_advantages"], dtype=torch.float32)
    weak_adv = torch.tensor(weak.metadata["process_advantages"], dtype=torch.float32)
    assert torch.all(strong_adv > 0)
    assert torch.all(weak_adv < 0)
    all_values = torch.cat([strong_adv, weak_adv])
    assert abs(float(all_values.mean())) < 1e-6


def test_segment_credit_assignment_advantage_dumps_token_level_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_ADVANTAGE_DUMP_N", "1")
    monkeypatch.setenv("AGENT_ENV_ADVANTAGE_DUMP_TOKEN_IDS", "1")
    monkeypatch.setenv("AGENT_ENV_ADVANTAGE_DUMP_DIR", str(tmp_path))
    args = Namespace(
        advantage_estimator="grpo",
        reward={"credit_assignment": {"enable": True, "beta": 0.5}},
    )
    rollout_data = {
        "kl": [torch.zeros(2, dtype=torch.float32)],
        "rewards": [2.0],
        "process_advantages": [torch.tensor([1.0, -1.0], dtype=torch.float32)],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "response_lengths": [2],
        "tokens": [torch.tensor([11, 22, 33, 44], dtype=torch.long)],
        "sample_indices": [7],
        "rollout_ids": [3],
        "source_names": ["appworld"],
    }

    segment_credit_assignment_advantage(args, rollout_data)

    paths = list(tmp_path.glob("advantage_pid*.jsonl"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "agent_env.token_advantage_audit.v1"
    assert payload["sample_index"] == 7
    assert payload["response_token_ids"] == [33, 44]
    assert payload["tokens"][0]["process_advantage"] == 1.0
    assert payload["tokens"][0]["final_advantage"] == 2.5
    assert payload["tokens"][1]["process_advantage"] == -1.0
    assert payload["tokens"][1]["final_advantage"] == 1.5
