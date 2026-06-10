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
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        data = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            data["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return data

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

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
alfworld:
  data_dir: {data_dir}
  num_train_games: 2
max_turns: 1
env_server:
  pool_size: 1
  prewarm_splits: [train]
  worker_start_timeout_s: 120
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
    agent_rollout.call_policy = lambda args, spec, sample, input_ids, sampling_params: fake_policy(args, sample, input_ids, sampling_params)

    proc, base_url, config_path = _start_alfworld_server(data_dir)

    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_context_len=4096,
        rollout_max_response_len=128,
        alfworld_env_server_url=base_url,
        task={"split": "train"},
        timeouts={"policy_s": 120, "env_request_s": 660},
        max_turns=1,
        interaction={"mode": "text_action", "text_action": {"tag": "action"}},
        observation={"include_actions": True},
        action={"restrict_to_available": False, "invalid_fallback": "model"},
        reward={"source": "won", "outcome": 10.0, "format": {"valid": 0.0, "invalid": -0.1}},
        generation={"stop": None},
        alfworld={"direct_game_file": True, "skip_to_task": False, "num_tasks": None},
        loss_mask_type="qwen3_5",
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
            assert len(result.loss_mask) == result.response_length
            assert result.rollout_log_probs is None
            assert "<think>" in result.response
            assert "<action>look</action>" in result.response
            assert sum(result.loss_mask) > 0
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
