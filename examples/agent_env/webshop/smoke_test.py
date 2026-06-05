import asyncio
from types import SimpleNamespace

from slime.utils.types import Sample

import examples.agent_env.webshop.rollout as rollout
import examples.agent_env.rollout as agent_rollout


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 1000 for ch in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


async def fake_policy(args, sample, sampling_params):
    return (
        "<think>search for the requested red mug</think><action>search[red ceramic mug]</action>",
        [101, 102],
        [-0.1, -0.2],
        "stop",
    )


async def fake_allocate(args, sample):
    return {"lease_id": "lease-0"}


async def fake_reset(args, sample, lease_id):
    return {
        "observation": "Instruction: find a red ceramic mug.",
        "info": {"available_actions": {"has_search_bar": True, "clickables": ["Back to Search"]}},
        "split": "train",
    }


async def fake_step(args, lease_id, action):
    assert action == "search[red ceramic mug]"
    return {
        "observation": "Search results contain a red ceramic mug.",
        "score": 1.0,
        "done": True,
        "info": {"available_actions": {"has_search_bar": False, "clickables": []}, "done": True},
    }


async def fake_evaluate(args, lease_id):
    return {"score": 1.0, "success": True, "done": True}


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
        rollout_max_context_len=4096,
        max_turns=2,
        action_max_tokens=128,
        generation_stop=None,
        include_available_actions=True,
        restrict_to_available=False,
        invalid_action_fallback="model",
        reward_source="score",
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
    result = await rollout.generate(args, sample, sampling_params={})

    if result.status != Sample.Status.COMPLETED:
        raise AssertionError(f"unexpected status={result.status} metadata={result.metadata}")
    assert result.status == Sample.Status.COMPLETED
    assert result.reward == 10.0
    assert result.metadata["env_success"] is True
    assert result.metadata["actions"] == ["search[red ceramic mug]"]
    assert result.metadata["webshop"]["task_index"] == 0
    assert len(result.metadata["token_rewards"]) == result.response_length
    assert sum(result.metadata["token_rewards"]) == result.reward
    assert len(result.loss_mask) == result.response_length
    assert len(result.rollout_log_probs) == result.response_length
    assert "<think>" not in result.response
    assert "<action>search[red ceramic mug]</action>" in result.response

    keep_args = SimpleNamespace(**vars(args))
    keep_args.keep_think_in_context = True
    keep_result = await rollout.generate(keep_args, Sample(prompt="", metadata={"task_index": 0}), sampling_params={})
    assert "<think>" in keep_result.response
    assert "<action>search[red ceramic mug]</action>" in keep_result.response
    print("WebShop smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
