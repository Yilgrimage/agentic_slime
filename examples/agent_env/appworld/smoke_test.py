from __future__ import annotations

import os

from examples.agent_env.appworld.server import AppWorldBackend


def main() -> None:
    backend = AppWorldBackend(
        "smoke",
        "train",
        {
            "root": os.environ.get("APPWORLD_ROOT", ""),
            "dataset_name": os.environ.get("APPWORLD_DATASET", "train"),
            "eval_dataset_name": os.environ.get("APPWORLD_EVAL_DATASET", "dev"),
            "difficulty": None,
            "num_tasks_per_scenario": None,
            "only_tagged": None,
            "num_tasks": 1,
            "max_interactions": 3,
            "raise_on_failure": False,
            "experiment_prefix": "slime_agent_env_smoke",
            "include_api_overview": True,
        },
    )
    print("start", backend.start())
    reset = backend.reset({"task_index": 0, "split": "train"})
    print("reset task", reset["info"].get("task_id"))
    step = backend.step(
        {
            "action": {
                "type": "tool_call",
                "name": "execute",
                "arguments": {"code": "print(dir(apis.supervisor))"},
            }
        }
    )
    print("execute done", step["done"], "obs", step["observation"][:300])
    finish = backend.step(
        {
            "action": {
                "type": "tool_call",
                "name": "finish",
                "arguments": {"submit": False},
            }
        }
    )
    print("finish done", finish["done"], "score", finish["score"], "success", finish["success"])
    backend.close()


if __name__ == "__main__":
    main()
