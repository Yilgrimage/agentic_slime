import asyncio
from types import SimpleNamespace

from slime.utils.types import Sample

import examples.alfworld.generate_with_alfworld as alf_gen


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 1000 for ch in text]


class FakeEnv:
    def reset(self):
        return ["You are in a kitchen. Your task is to pick up the apple."], {
            "admissible_commands": [["look", "take apple"]],
            "won": [False],
        }

    def step(self, actions):
        assert actions == ["take apple"]
        return ["You picked up the apple."], [1.0], [True], {
            "admissible_commands": [[]],
            "won": [True],
        }

    def close(self):
        pass


async def fake_policy(args, sample, sampling_params):
    return "take apple", [101, 102], [-0.1, -0.2], "stop"


async def main():
    alf_gen._tokenizer = lambda args: FakeTokenizer()
    alf_gen._build_env = lambda args, sample: FakeEnv()
    alf_gen._call_policy = fake_policy

    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_context_len=512,
        alfworld_max_turns=2,
        alfworld_action_max_tokens=8,
        alfworld_stop=[chr(10)],
        alfworld_include_admissible_actions=True,
        alfworld_restrict_to_admissible=False,
        alfworld_invalid_action_fallback="model",
        alfworld_reward_source="won",
        alfworld_skip_to_task=False,
        alfworld_direct_game_file=True,
        alfworld_return_logprob=True,
        alfworld_split="train",
        use_opd=False,
        opd_type=None,
    )
    sample = Sample(prompt="", metadata={"task_index": 0})
    result = await alf_gen.generate(args, sample, sampling_params={})

    assert result.status == Sample.Status.COMPLETED
    assert result.reward == 1.0
    assert result.metadata["alfworld_success"] is True
    assert result.metadata["alfworld_actions"] == ["take apple"]
    assert len(result.loss_mask) == result.response_length
    assert len(result.rollout_log_probs) == result.response_length
    assert result.rollout_log_probs[:2] == [-0.1, -0.2]
    assert all(logp == 0.0 for logp, mask in zip(result.rollout_log_probs, result.loss_mask) if mask == 0)
    assert sum(result.loss_mask) == 2
    print("ALFWorld smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
