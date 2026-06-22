import asyncio
from types import SimpleNamespace

from slime.utils.types import Sample

import examples.agent_env.webshop.rollout as rollout
import examples.agent_env.rollout as agent_rollout


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        data = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            data["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return data

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=True, tools=None, add_generation_prompt=False, **kwargs):
        pieces = []
        for message in messages:
            role = message["role"]
            if role in {"system", "user", "assistant"}:
                pieces.append(f"<|im_start|>{role}\n{message.get('content', '')}<|im_end|>\n")
            elif role == "tool":
                pieces.append(f"<|im_start|>user\n<tool_response>\n{message.get('content', '')}\n</tool_response><|im_end|>\n")
        if add_generation_prompt:
            pieces.append("<|im_start|>assistant\n<think>\n")
        rendered = "".join(pieces)
        return self.encode(rendered) if tokenize else rendered


async def fake_policy(args, sample, input_ids, sampling_params):
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
    agent_rollout.call_policy = lambda args, spec, sample, input_ids, sampling_params: fake_policy(args, sample, input_ids, sampling_params)
    agent_rollout.allocate_env = lambda args, spec, sample: fake_allocate(args, sample)
    agent_rollout.reset_env = lambda args, spec, sample, lease_id, extra_payload=None: fake_reset(args, sample, lease_id)
    agent_rollout.step_env = lambda args, spec, lease_id, action: fake_step(args, lease_id, action)
    agent_rollout.evaluate_env = lambda args, spec, lease_id: fake_evaluate(args, lease_id)
    agent_rollout.close_env = lambda args, spec, lease_id: fake_close(args, lease_id)

    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_context_len=4096,
        rollout_max_response_len=128,
        max_turns=2,
        task={"split": "train"},
        timeouts={"policy_s": 120, "env_request_s": 660},
        interaction={"mode": "text_action", "text_action": {"tag": "action"}},
        observation={"include_actions": True},
        action={"restrict_to_available": False, "invalid_fallback": "model"},
        reward={"source": "score", "outcome": 10.0, "format": {"valid": 0.0, "invalid": -0.1}},
        generation={"stop": None},
        loss_mask_type="qwen3_5",
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
    assert len(result.loss_mask) == result.response_length
    assert result.rollout_log_probs is None
    assert "<think>" in result.response
    assert "<action>search[red ceramic mug]</action>" in result.response

    print("WebShop smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
