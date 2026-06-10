from __future__ import annotations

import argparse
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import yaml

from examples.agent_env.server import serve_process_pool

logger = logging.getLogger(__name__)


def _deep_get(raw: dict, section: str, key: str, default: Any = None) -> Any:
    value = raw.get(key)
    if value is not None:
        return value
    nested = raw.get(section)
    if isinstance(nested, dict):
        return nested.get(key, default)
    return default


def _server_config(raw: dict) -> dict:
    return {
        "pool_size": int(_deep_get(raw, "env_server", "pool_size", 4)),
        "acquire_timeout_s": float(_deep_get(raw, "env_server", "acquire_timeout_s", 600.0)),
        "lease_ttl_s": float(_deep_get(raw, "env_server", "lease_ttl_s", 1800.0)),
        "idempotency_ttl_s": float(_deep_get(raw, "env_server", "idempotency_ttl_s", 300.0)),
        "worker_start_timeout_s": float(_deep_get(raw, "env_server", "worker_start_timeout_s", 300.0)),
        "worker_request_timeout_s": float(_deep_get(raw, "env_server", "worker_request_timeout_s", 180.0)),
        "prewarm_splits": list(_deep_get(raw, "env_server", "prewarm_splits", ["train"])),
        "reuse_workers": bool(_deep_get(raw, "env_server", "reuse_workers", True)),
        "reset_on_release": bool(_deep_get(raw, "env_server", "reset_on_release", False)),
        "shared_pool": bool(_deep_get(raw, "env_server", "shared_pool", True)),
    }


def _env_path(value: Any, envvar: str) -> str:
    text = str(value or "").strip()
    if text in {"", f"${{{envvar}}}"}:
        return os.environ.get(envvar, "")
    return os.path.expandvars(text)


def _environment_config(raw: dict) -> dict:
    root = _env_path(_deep_get(raw, "appworld", "root", os.environ.get("APPWORLD_ROOT", "")), "APPWORLD_ROOT")
    return {
        "root": root,
        "dataset_name": str(_deep_get(raw, "appworld", "dataset_name", "train")),
        "eval_dataset_name": _deep_get(raw, "appworld", "eval_dataset_name", None),
        "difficulty": _deep_get(raw, "appworld", "difficulty", None),
        "num_tasks_per_scenario": _deep_get(raw, "appworld", "num_tasks_per_scenario", None),
        "only_tagged": _deep_get(raw, "appworld", "only_tagged", None),
        "num_tasks": _deep_get(raw, "appworld", "num_tasks", None),
        "max_interactions": int(_deep_get(raw, "appworld", "max_interactions", 20)),
        "raise_on_failure": bool(_deep_get(raw, "appworld", "raise_on_failure", False)),
        "experiment_prefix": str(_deep_get(raw, "appworld", "experiment_prefix", "slime_agent_env")),
        "include_api_overview": bool(_deep_get(raw, "appworld", "include_api_overview", True)),
    }


def _load_config(path: str) -> tuple[dict, dict]:
    with Path(path).expanduser().open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _environment_config(raw), _server_config(raw)


def _tool_action(payload: Any) -> tuple[str, dict[str, Any], bool]:
    if isinstance(payload, dict) and payload.get("type") == "tool_call":
        name = str(payload.get("name") or "").strip()
        arguments = payload.get("arguments") or {}
        return name, arguments if isinstance(arguments, dict) else {}, True
    if isinstance(payload, str):
        return payload.strip(), {}, False
    return str(payload), {}, False


def _split_dataset(config: dict[str, Any], split: str) -> str:
    if split in {"eval", "validation", "val", "dev", "test"}:
        return str(config.get("eval_dataset_name") or ("dev" if split in {"eval", "validation", "val"} else split))
    return str(config.get("dataset_name") or split)


class AppWorldBackend:
    def __init__(self, worker_id: str, split: str, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.split = split
        self.config = config
        self.dataset_name = _split_dataset(config, split)
        self.task_ids: list[str] = []
        self.world: Any | None = None
        self.task_id = ""
        self.experiment_name = ""
        self.task_index = 0
        self.reset_count = 0
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.last_info: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        root = self.config.get("root")
        if root:
            os.environ["APPWORLD_ROOT"] = str(root)
            os.environ.setdefault("HOME", str(root))
        self.task_ids = self._load_task_ids(self.dataset_name)
        return {"num_tasks": len(self.task_ids), "dataset_name": self.dataset_name}

    def _load_task_ids(self, dataset_name: str) -> list[str]:
        from appworld.task import load_task_ids

        ids = load_task_ids(
            dataset_name=dataset_name,
            difficulty=self.config.get("difficulty"),
            num_tasks_per_scenario=self.config.get("num_tasks_per_scenario"),
            only_tagged=self.config.get("only_tagged"),
        )
        num_tasks = self.config.get("num_tasks")
        if num_tasks is not None:
            ids = ids[: int(num_tasks)]
        return list(ids)

    def _close_world(self) -> None:
        if self.world is not None:
            try:
                self.world.close()
            except Exception:
                logger.debug("Failed to close AppWorld task %s", self.task_id, exc_info=True)
        self.world = None

    def _api_overview(self) -> str:
        if self.world is None or not self.config.get("include_api_overview", True):
            return ""
        api_docs = getattr(getattr(self.world, "task", None), "api_docs", None)
        if api_docs is None:
            return ""
        app_names = [
            name
            for name in dir(api_docs)
            if not name.startswith("_") and name not in {"api_docs"} and not callable(getattr(api_docs, name, None))
        ]
        app_names = [name for name in app_names if name not in {"show_app_descriptions", "show_api_descriptions", "show_api_doc", "search_api_docs"}]
        supervisor_doc = str(getattr(api_docs, "supervisor", "")).strip()
        parts = []
        if app_names:
            parts.append("Apps available through the `apis` object: " + ", ".join(sorted(app_names)))
        if supervisor_doc:
            parts.append("Supervisor API includes `apis.supervisor.show_active_task()` and `apis.supervisor.complete_task(answer=..., status='success')`.")
        return "\n".join(parts)

    def _initial_observation(self) -> str:
        assert self.world is not None
        instruction = str(getattr(self.world.task, "instruction", "")).strip()
        parts = [f"Task id: {self.task_id}", f"Instruction:\n{instruction}"]
        overview = self._api_overview().strip()
        if overview:
            parts.append(overview)
        parts.append(
            "Execute Python snippets against the AppWorld `apis` object. "
            "Inspect apps with `dir(apis.<app>)`, then call app APIs. "
            "When complete, call `apis.supervisor.complete_task(...)` or use the finish tool."
        )
        return "\n\n".join(parts)

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        split = str(payload.get("split") or self.split)
        dataset = _split_dataset(self.config, split)
        if dataset != self.dataset_name:
            self.dataset_name = dataset
            self.task_ids = self._load_task_ids(dataset)
        self.task_index = int(payload.get("task_index") or 0) % max(1, len(self.task_ids))
        self.task_id = self.task_ids[self.task_index]
        self._close_world()
        from appworld.environment import AppWorld

        self.experiment_name = f"{self.config['experiment_prefix']}_{self.worker_id}_{uuid.uuid4().hex[:8]}"
        self.world = AppWorld(
            self.task_id,
            experiment_name=self.experiment_name,
            max_interactions=int(self.config.get("max_interactions", 20)),
            raise_on_failure=bool(self.config.get("raise_on_failure", False)),
        )
        self.split = split
        self.reset_count += 1
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.last_info = {
            "task_id": self.task_id,
            "dataset_name": self.dataset_name,
            "experiment_name": self.experiment_name,
            "tools": ["execute", "finish"],
        }
        return {
            "observation": self._initial_observation(),
            "info": self.last_info,
            "split": self.split,
            "task_index": self.task_index,
            "num_tasks": len(self.task_ids),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def _evaluate(self) -> dict[str, Any]:
        from appworld.evaluator import evaluate_task

        tracker = evaluate_task(
            self.task_id,
            experiment_name=self.experiment_name,
            suppress_errors=True,
            save_report=False,
        )
        score = float(getattr(tracker, "pass_percentage", 0.0) or 0.0) / 100.0
        success = bool(getattr(tracker, "success", False))
        self.final_score = 1.0 if success else score
        return {
            "success": success,
            "pass_count": int(getattr(tracker, "pass_count", 0) or 0),
            "fail_count": int(getattr(tracker, "fail_count", 0) or 0),
            "num_tests": int(getattr(tracker, "num_tests", 0) or 0),
            "pass_percentage": float(getattr(tracker, "pass_percentage", 0.0) or 0.0),
        }

    def _finish(self, arguments: dict[str, Any]) -> str:
        assert self.world is not None
        answer = arguments.get("answer", arguments.get("message", None))
        status = str(arguments.get("status", "success"))
        if answer is None and not arguments.get("submit", False):
            return "Finish requested without submitting an answer. Evaluating current AppWorld state."
        code = f"print(apis.supervisor.complete_task(answer={answer!r}, status={status!r}))"
        return str(self.world.execute(code))

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.world is not None
        name, arguments, structured = _tool_action(payload.get("action"))
        self.step_count += 1
        info = dict(self.last_info)
        info["last_action"] = name
        info["structured_action"] = structured

        if name in {"execute", "python", "python_exec"}:
            code = str(arguments.get("code") or arguments.get("python") or arguments.get("command") or "")
            observation = str(self.world.execute(code))
        elif name in {"finish", "final_response", "submit"}:
            observation = self._finish(arguments)
            self.done = True
        elif name in {"format_error", "invalid_format"}:
            observation = (
                "Invalid response format. Respond with exactly one <code>...</code> block "
                "containing Python code to execute."
            )
            info["format_error"] = True
        else:
            observation = f"Unknown AppWorld tool `{name}`. Respond with a <code>...</code> block that calls AppWorld APIs."
            info["tool_error"] = observation

        if not self.done:
            try:
                self.done = bool(self.world.task_completed())
            except Exception:
                self.done = False
        if self.done:
            info.update(self._evaluate())
        info["done"] = self.done
        self.last_info = info
        return self._result(observation, info)

    def _result(self, observation: str, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation": observation,
            "score": self.final_score,
            "done": self.done,
            "success": bool(info.get("success", False)) or self.final_score >= 1.0,
            "info": info,
            "task_index": self.task_index,
            "num_tasks": len(self.task_ids),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        info = dict(self.last_info)
        if self.world is not None and (self.done or payload.get("force", False)):
            info.update(self._evaluate())
        return self._result("", info)

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"reset_count": self.reset_count, "step_count": self.step_count}

    def close(self) -> dict[str, Any]:
        self._close_world()
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a process-isolated AppWorld environment server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18183)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    env_config, server_config = _load_config(args.config)
    serve_process_pool(
        host=args.host,
        port=args.port,
        backend_cls=AppWorldBackend,
        env_config=env_config,
        server_config=server_config,
        env_name="appworld",
    )


if __name__ == "__main__":
    main()
