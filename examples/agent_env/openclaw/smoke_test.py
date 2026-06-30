from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from examples.agent_env.openclaw.server import OpenClawBackend


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _PolicyHandler(BaseHTTPRequestHandler):
    def _read(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _write(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        body = self._read()
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        self._write(
            {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "model": body.get("model", "smoke"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_smoke",
                                    "type": "function",
                                    "function": {"name": "finish", "arguments": json.dumps({"answer": "blue"})},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )

    def log_message(self, fmt: str, *args) -> None:
        return


class _HarnessHandler(BaseHTTPRequestHandler):
    def _read(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _write(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        payload = self._read()
        if self.path == "/run_episode":
            policy = payload["policy"]
            url = policy["base_url"].rstrip("/") + policy.get("chat_completions_path", "/v1/chat/completions")
            body = {
                "model": policy.get("model", "smoke"),
                "messages": [
                    {"role": "system", "content": payload.get("policy_instruction", "")},
                    {"role": "user", "content": payload["task"]["prompt"]},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Submit the final answer.",
                            "parameters": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                            },
                        },
                    }
                ],
            }
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {policy['api_key']}"}
            headers.update(policy.get("headers") or {})
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
            with _OPENER.open(req, timeout=10) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            call = response["choices"][0]["message"]["tool_calls"][0]
            arguments = json.loads(call["function"]["arguments"])
            success = str(arguments.get("answer") or "").strip().lower() == "blue"
            self._write(
                {
                    "status": "completed",
                    "observation": "finished",
                    "done": True,
                    "success": success,
                    "score": 1.0 if success else 0.0,
                    "info": {"policy_called": True, "tool_name": call["function"]["name"]},
                }
            )
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return


def _start_server(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def main() -> None:
    policy_server, policy_thread, policy_url = _start_server(_PolicyHandler)
    harness_server, harness_thread, harness_url = _start_server(_HarnessHandler)
    backend = OpenClawBackend(
        "smoke",
        "train",
        {
            "name": "openclaw_smoke",
            "backend": "http_harness",
            "base_url": harness_url,
            "episode_path": "/run_episode",
            "close_path": "",
            "request_timeout_s": 10.0,
            "headers": {},
            "policy": "Use OpenClaw tools.",
            "tasks": [{"id": "smoke", "prompt": "Answer blue.", "target": "blue"}],
            "max_turns": 1,
        },
    )
    try:
        print("start", backend.start())
        result = backend.run_episode(
            {
                "task_index": 0,
                "split": "train",
                "policy": {
                    "base_url": policy_url,
                    "chat_completions_path": "/v1/chat/completions",
                    "api_key": "smoke-key",
                    "session_id": "smoke-session",
                    "model": "smoke",
                    "headers": {"X-Agent-Env-Policy-Session": "smoke-session"},
                },
            }
        )
        print("episode done", result["done"], "score", result["score"], "success", result["success"])
        print("info", result["info"]["policy_called"], result["info"]["tool_name"])
    finally:
        backend.close()
        harness_server.shutdown()
        policy_server.shutdown()
        harness_thread.join(timeout=5)
        policy_thread.join(timeout=5)


if __name__ == "__main__":
    main()
