import asyncio
from types import SimpleNamespace

from slime.utils.types import Sample

import examples.alfworld.generate_with_alfworld as alf_gen


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 1000 for ch in text]


async def fake_policy(args, sample, sampling_params):
    return "<think>pick up the visible apple</think><action>take apple</action>", [101, 102], [-0.1, -0.2], "stop"


async def fake_allocate(args, sample):
    return {"lease_id": "lease-0"}


async def fake_reset(args, sample, lease_id):
    return {
        "observation": "You are in a kitchen. Your task is to pick up the apple.",
        "info": {"admissible_commands": [["look", "take apple"]], "won": [False]},
        "split": "train",
        "game_file": "/tmp/game.tw-pddl",
    }


async def fake_step(args, lease_id, action):
    assert action == "take apple"
    return {
        "observation": "You picked up the apple.",
        "score": 1.0,
        "done": True,
        "info": {"admissible_commands": [[]], "won": [True]},
    }


async def fake_evaluate(args, lease_id):
    return {"score": 1.0, "success": True}


async def fake_close(args, lease_id):
    return None


async def main():
    alf_gen._tokenizer = lambda args: FakeTokenizer()
    alf_gen._call_policy = fake_policy
    alf_gen._allocate_env = fake_allocate
    alf_gen._reset_env = fake_reset
    alf_gen._step_env = fake_step
    alf_gen._evaluate_env = fake_evaluate
    alf_gen._close_env = fake_close

    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_context_len=512,
        max_turns=2,
        action_max_tokens=128,
        generation_stop=None,
        include_admissible_actions=True,
        restrict_to_admissible=False,
        invalid_action_fallback="model",
        reward_source="won",
        outcome_reward=10.0,
        return_logprob=True,
        env_split="train",
        format_reward=0.0,
        format_penalty=-0.1,
        use_opd=False,
        opd_type=None,
    )
    sample = Sample(prompt="", metadata={"task_index": 0})
    result = await alf_gen.generate(args, sample, sampling_params={})

    assert result.status == Sample.Status.COMPLETED
    assert result.reward == 10.0
    assert result.metadata["env_success"] is True
    assert result.metadata["actions"] == ["take apple"]
    assert result.metadata["alfworld"]["game_file"] == "/tmp/game.tw-pddl"
    assert len(result.metadata["token_rewards"]) == result.response_length
    assert sum(result.metadata["token_rewards"]) == result.reward
    assert len(result.loss_mask) == result.response_length
    assert len(result.rollout_log_probs) == result.response_length
    assert "<think>" not in result.response
    assert "<action>take apple</action>" in result.response
    assert all(logp == 0.0 for logp, mask in zip(result.rollout_log_probs, result.loss_mask) if mask == 0)
    assert sum(result.loss_mask) == len(FakeTokenizer().encode("<action>take apple</action>"))
    print("ALFWorld smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
