import asyncio
import json
import os
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from slime.utils.types import Sample

import examples.agent_env.episode as episode
import examples.agent_env.webshop.rollout as rollout


class FakeTokenizer:
    eos_token_id = 0
    chat_template = "fake-qwen"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        data = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            data["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return data

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 251 + 1 for ch in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr((int(token_id) - 1) % 251) for token_id in token_ids if int(token_id) > 0)

    def apply_chat_template(self, messages, tokenize=True, tools=None, add_generation_prompt=False, **kwargs):
        rendered = "".join(f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return self.encode(rendered) if tokenize else rendered


async def fake_policy(args, spec, sample, input_ids, sampling_params):
    text = "<think>search for the requested red mug</think><action>search[red ceramic mug]</action>"
    token_ids = [111, 112, 113]
    return text, token_ids, [-0.1, -0.2, -0.3], "stop"


class FakeWebShopEpisodeServer(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        assert self.path == "/run_episode"
        policy = payload["policy"]
        req = urllib.request.Request(
            policy["base_url"] + policy["chat_completions_path"],
            data=json.dumps(
                {
                    "messages": [{"role": "user", "content": "Observation:\nInstruction: find a red ceramic mug.\n"}],
                    "max_tokens": payload.get("max_response_tokens", 128),
                }
            ).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": "Bearer " + policy["api_key"],
                **policy.get("headers", {}),
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=10) as response:
            completion = json.loads(response.read().decode("utf-8"))
        content = completion["choices"][0]["message"]["content"]
        match = re.search(r"<action>\s*(.*?)\s*</action>", content, flags=re.DOTALL)
        action = match.group(1).strip() if match else ""
        body = json.dumps(
            {
                "ok": True,
                "status": "completed",
                "done": True,
                "score": 1.0,
                "success": True,
                "info": {"done": True},
                "task_index": payload.get("task_index", 0),
                "metadata": {
                    "turn_count": 1,
                    "actions": [action],
                    "format_checks": [{"turn": 0, "valid": action == "search[red ceramic mug]", "parse_mode": "action_tag"}],
                    "action_parse_modes": ["action_tag"],
                    "format_errors": 0,
                    "format_ok": True,
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


async def main():
    os.environ["AGENT_ENV_POLICY_HOST"] = "127.0.0.1"
    episode.tokenizer = lambda args: FakeTokenizer()
    episode.call_policy = fake_policy

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeWebShopEpisodeServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        args = SimpleNamespace(
            partial_rollout=False,
            rollout_max_context_len=4096,
            rollout_max_response_len=128,
            max_turns=2,
            env_server_url=f"http://127.0.0.1:{server.server_port}",
            task={"split": "train"},
            timeouts={"policy_s": 120, "env_request_s": 20},
            interaction={"mode": "text_action", "text_action": {"tag": "action"}},
            observation={"include_actions": True},
            action={"restrict_to_available": False, "invalid_fallback": "model"},
            reward={"source": "score", "outcome": 10.0, "format": {"valid": 0.0, "invalid": -0.1}},
            generation={"stop": None},
            loss_mask_type="qwen3_5",
            use_opd=False,
            opd_type=None,
            train_env_vars={},
            router_policy=None,
            hf_checkpoint="fake",
        )
        sample = Sample(prompt="", metadata={"task_index": 0})
        result = await rollout.generate(args, sample, sampling_params={})

        assert result.status == Sample.Status.COMPLETED
        assert result.reward is None
        assert result.metadata["env_reward"] == 10.0
        assert result.metadata["env_success"] is True
        assert result.metadata["actions"] == ["search[red ceramic mug]"]
        assert result.metadata["webshop"]["task_index"] == 0
        assert len(result.metadata["token_rewards"]) == result.response_length
        assert len(result.loss_mask) == result.response_length
        assert result.rollout_log_probs is not None
        assert len(result.rollout_log_probs) == result.response_length
        assert "<action>search[red ceramic mug]</action>" in result.response
        print("WebShop smoke test passed")
    finally:
        server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
