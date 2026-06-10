from __future__ import annotations

import os

from examples.agent_env.tau2.server import Tau2Backend


def main() -> None:
    backend = Tau2Backend(
        "smoke",
        "train",
        {
            "data_dir": os.environ.get("TAU2_DATA_DIR", ""),
            "domain": os.environ.get("TAU2_DOMAIN", "mock"),
            "task_set": os.environ.get("TAU2_TASK_SET", "mock"),
            "split": None,
            "num_tasks": 1,
            "solo_mode": False,
            "max_turns": 5,
            "include_policy": True,
            "include_tools": True,
            "evaluation_type": "action",
        },
    )
    print("start", backend.start())
    reset = backend.reset({"task_index": 0, "split": "train"})
    print("reset task", reset["info"].get("task_id"), "tools", reset["info"].get("tools")[:5])
    step = backend.step(
        {
            "action": {
                "type": "assistant_message",
                "content": "Smoke test final response.",
            }
        }
    )
    print("done", step["done"], "score", step["score"], "success", step["success"])
    backend.close()


if __name__ == "__main__":
    main()
