from __future__ import annotations

from examples.agent_env.mcp_server.server import MCPServerBackend, _load_config


def main() -> None:
    env_config, _server_config = _load_config("examples/agent_env/mcp_server/env_config.yaml")
    backend = MCPServerBackend("smoke", "train", env_config)
    print("start", backend.start())
    reset = backend.reset({"task_index": 0, "split": "train"})
    print("reset", reset["info"].get("task_id"), reset["info"].get("tools"))
    step1 = backend.step({"action": {"type": "tool_call", "name": "lookup", "arguments": {"key": "color"}}})
    print("lookup", step1["observation"], step1["done"], step1["score"])
    step2 = backend.step({"action": {"type": "tool_call", "name": "finish", "arguments": {"answer": "blue"}}})
    print("finish", step2["done"], step2["success"], step2["score"])
    assert step2["done"] is True
    assert step2["success"] is True
    assert step2["score"] == 1.0
    backend.close()


if __name__ == "__main__":
    main()
