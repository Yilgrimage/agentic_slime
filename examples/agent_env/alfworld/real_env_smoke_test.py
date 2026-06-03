import argparse
import asyncio
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from types import SimpleNamespace

from slime.utils.types import Sample

import examples.agent_env.alfworld.rollout as alf_gen
import examples.agent_env.rollout as agent_rollout


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 1000 for ch in text]


async def fake_policy(args, sample, sampling_params):
    return "<think>inspect the current room</think><action>look</action>", [101], [-0.1], "stop"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(base_url: str, proc: subprocess.Popen, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ALFWorld server exited early with code {proc.returncode}")
        try:
            with _OPENER.open(f"{base_url}/health", timeout=2) as resp:
                payload = json.loads(resp.read().decode())
            if payload.get("ok"):
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"ALFWorld server did not become healthy: {last_error}")


def _start_alfworld_server(data_dir: str) -> tuple[subprocess.Popen, str, str]:
    env_bin = os.environ.get("ALFWORLD_ENV_BIN", "/tmp/mlf-envs/alfworld/bin/python")
    port = _free_port()
    config = f"""
alfworld_data_dir: {data_dir}
alfworld_server_pool_size: 1
alfworld_server_prewarm_splits: [train]
alfworld_num_train_games: 2
alfworld_server_worker_start_timeout_s: 120
max_turns: 1
"""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    tmp.write(config)
    tmp.close()
    env = dict(os.environ)
    repo_dir = os.getcwd()
    env["PYTHONPATH"] = repo_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen(
        [
            env_bin,
            "-m",
            "examples.agent_env.alfworld.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--config",
            tmp.name,
        ],
        cwd=repo_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    _wait_health(base_url, proc)
    return proc, base_url, tmp.name


async def run(data_dir: str):
    agent_rollout.tokenizer = lambda args: FakeTokenizer()
    agent_rollout.call_policy = lambda args, spec, sample, sampling_params: fake_policy(args, sample, sampling_params)

    proc, base_url, config_path = _start_alfworld_server(data_dir)

    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_context_len=4096,
        alfworld_env_server_url=base_url,
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

    try:
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
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        try:
            os.unlink(config_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/bn/jixf-nas-lq/mlf/data/alfworld")
    args = parser.parse_args()
    asyncio.run(run(args.data_dir))


if __name__ == "__main__":
    main()
