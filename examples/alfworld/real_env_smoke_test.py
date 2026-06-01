import argparse
import asyncio
from types import SimpleNamespace

from slime.utils.types import Sample

import examples.alfworld.generate_with_alfworld as alf_gen


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 1000 for ch in text]


async def fake_policy(args, sample, sampling_params):
    return "<think>inspect the current room</think><action>look</action>", [101], [-0.1], "stop"


async def run(data_dir: str):
    alf_gen._tokenizer = lambda args: FakeTokenizer()
    alf_gen._call_policy = fake_policy

    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_context_len=4096,
        alfworld_data_dir=data_dir,
        alfworld_config_path=None,
        alfworld_config_overrides={},
        alfworld_env_type="AlfredTWEnv",
        alfworld_split="train",
        alfworld_num_train_games=2,
        max_turns=1,
        action_max_tokens=128,
        generation_stop=None,
        include_admissible_actions=True,
        restrict_to_admissible=False,
        invalid_action_fallback="model",
        reward_source="won",
        outcome_reward=10.0,
        alfworld_skip_to_task=False,
        alfworld_direct_game_file=True,
        return_logprob=True,
        format_reward=0.0,
        format_penalty=-0.1,
        use_opd=False,
        opd_type=None,
    )

    results = []
    for task_index in range(2):
        sample = Sample(prompt="", metadata={"task_index": task_index, "split": "train"})
        result = await alf_gen.generate(args, sample, sampling_params={})
        results.append(result)

        assert result.status == Sample.Status.COMPLETED
        assert result.metadata["actions"] == ["look"]
        assert result.metadata["alfworld"]["game_file"]
        assert len(result.metadata["token_rewards"]) == result.response_length
        assert sum(result.metadata["token_rewards"]) == result.reward
        assert len(result.loss_mask) == result.response_length
        assert len(result.rollout_log_probs) == result.response_length
        assert "<think>" not in result.response
        assert "<action>look</action>" in result.response
        assert sum(result.loss_mask) == len(FakeTokenizer().encode("<action>look</action>"))
        assert result.reward in (0.0, 10.0)

    assert results[0].metadata["alfworld"]["game_file"] != results[1].metadata["alfworld"]["game_file"]
    print("ALFWorld real env adapter smoke test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/bn/jixf-nas-lq/mlf/data/alfworld")
    args = parser.parse_args()
    asyncio.run(run(args.data_dir))


if __name__ == "__main__":
    main()
