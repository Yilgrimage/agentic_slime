import asyncio
from types import SimpleNamespace

from slime.utils.types import Sample

import examples.agent_env.alfworld.rollout as alf_gen
import examples.agent_env.rollout as agent_rollout


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
    agent_rollout.tokenizer = lambda args: FakeTokenizer()
    agent_rollout.call_policy = lambda args, spec, sample, sampling_params: fake_policy(args, sample, sampling_params)
    agent_rollout.allocate_env = lambda args, spec, sample: fake_allocate(args, sample)
    agent_rollout.reset_env = lambda args, spec, sample, lease_id, extra_payload=None: fake_reset(args, sample, lease_id)
    agent_rollout.step_env = lambda args, spec, lease_id, action: fake_step(args, lease_id, action)
    agent_rollout.evaluate_env = lambda args, spec, lease_id: fake_evaluate(args, lease_id)
    agent_rollout.close_env = lambda args, spec, lease_id: fake_close(args, lease_id)

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
        keep_think_in_context=False,
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

    keep_args = SimpleNamespace(**vars(args))
    keep_args.keep_think_in_context = True
    keep_result = await alf_gen.generate(keep_args, Sample(prompt="", metadata={"task_index": 0}), sampling_params={})
    assert "<think>" in keep_result.response
    assert "<action>take apple</action>" in keep_result.response
    print("ALFWorld smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
