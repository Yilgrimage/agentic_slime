from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from datetime import datetime, timezone
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
        "pool_size": int(_deep_get(raw, "env_server", "pool_size", 8)),
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
    data_dir = _env_path(_deep_get(raw, "tau2", "data_dir", os.environ.get("TAU2_DATA_DIR", "")), "TAU2_DATA_DIR")
    return {
        "data_dir": data_dir,
        "domain": str(_deep_get(raw, "tau2", "domain", "retail")),
        "task_set": str(_deep_get(raw, "tau2", "task_set", _deep_get(raw, "tau2", "domain", "retail"))),
        "split": _deep_get(raw, "tau2", "split", None),
        "num_tasks": _deep_get(raw, "tau2", "num_tasks", None),
        "solo_mode": bool(_deep_get(raw, "tau2", "solo_mode", False)),
        "max_turns": int(_deep_get(raw, "tau2", "max_turns", 20)),
        "include_policy": bool(_deep_get(raw, "tau2", "include_policy", True)),
        "include_tools": bool(_deep_get(raw, "tau2", "include_tools", True)),
        "evaluation_type": str(_deep_get(raw, "tau2", "evaluation_type", "action")),
    }


def _load_config(path: str) -> tuple[dict, dict]:
    with Path(path).expanduser().open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _environment_config(raw), _server_config(raw)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _tool_action(payload: Any) -> tuple[str, dict[str, Any], bool]:
    if isinstance(payload, dict) and payload.get("type") == "tool_call":
        name = str(payload.get("name") or "").strip()
        arguments = payload.get("arguments") or {}
        return name, arguments if isinstance(arguments, dict) else {}, True
    if isinstance(payload, str):
        return payload.strip(), {}, False
    return str(payload), {}, False


class Tau2Backend:
    def __init__(self, worker_id: str, split: str, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.split = split
        self.config = config
        self.tasks: list[Any] = []
        self.env: Any | None = None
        self.task: Any | None = None
        self.messages: list[Any] = []
        self.task_index = 0
        self.reset_count = 0
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.last_info: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        data_dir = self.config.get("data_dir")
        if data_dir:
            os.environ["TAU2_DATA_DIR"] = str(data_dir)
        from tau2.run import get_tasks

        self.tasks = get_tasks(
            self.config["task_set"],
            task_split_name=self.config.get("split"),
            num_tasks=self.config.get("num_tasks"),
        )
        return {"num_tasks": len(self.tasks), "domain": self.config["domain"], "task_set": self.config["task_set"]}

    def _build_env(self) -> Any:
        from tau2.run import build_environment

        return build_environment(self.config["domain"], solo_mode=bool(self.config.get("solo_mode", False)))

    def _tools_description(self) -> str:
        if self.env is None or not self.config.get("include_tools", True):
            return ""
        try:
            return str(self.env.get_tools_description("assistant"))
        except Exception:
            tools = []
            for tool in self.env.get_tools():
                tools.append(getattr(tool, "name", str(tool)))
            return "\n".join(tools)

    def _initial_observation(self) -> str:
        assert self.task is not None
        parts = [f"Task id: {getattr(self.task, 'id', self.task_index)}"]
        description = str(getattr(self.task, "description", "") or "").strip()
        if description:
            parts.append(f"Task:\n{description}")
        user_scenario = str(getattr(self.task, "user_scenario", "") or "").strip()
        if user_scenario:
            parts.append(f"User scenario:\n{user_scenario}")
        ticket = str(getattr(self.task, "ticket", "") or "").strip()
        if ticket:
            parts.append(f"Ticket:\n{ticket}")
        if self.config.get("include_policy", True) and self.env is not None:
            try:
                policy = str(self.env.get_policy()).strip()
                if policy:
                    parts.append(f"Policy:\n{policy}")
            except Exception:
                pass
        tool_docs = self._tools_description().strip()
        if tool_docs:
            parts.append(f"Available tools:\n{tool_docs}")
        parts.append("Use tools to solve the task. End with the respond tool when you are done.")
        return "\n\n".join(parts)

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.split = str(payload.get("split") or self.split)
        self.task_index = int(payload.get("task_index") or 0) % max(1, len(self.tasks))
        self.task = self.tasks[self.task_index]
        self.env = self._build_env()
        self.messages = []
        self.reset_count += 1
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.last_info = {
            "domain": self.config["domain"],
            "task_set": self.config["task_set"],
            "task_id": getattr(self.task, "id", None),
            "tools": [getattr(tool, "name", str(tool)) for tool in self.env.get_tools()],
            "finish_tools": ["respond", "final_response", "finish"],
        }
        return {
            "observation": self._initial_observation(),
            "info": self.last_info,
            "split": self.split,
            "task_index": self.task_index,
            "num_tasks": len(self.tasks),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def _record_tool_call(self, name: str, arguments: dict[str, Any], result: Any, error: bool = False) -> None:
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

        call_id = f"call-{uuid.uuid4().hex[:12]}"
        self.messages.append(
            AssistantMessage.text("", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments, requestor="assistant")])
        )
        self.messages.append(ToolMessage(id=call_id, role="tool", content=_json_text(result), requestor="assistant", error=error))

    def _finish(self, message: str) -> tuple[float, dict[str, Any]]:
        from tau2.data_model.message import AssistantMessage
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        self.messages.append(AssistantMessage.text(message))
        simulation = SimulationRun(
            id=f"agent-env-{uuid.uuid4().hex[:12]}",
            task_id=str(getattr(self.task, "id", self.task_index)),
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            duration=0.0,
            termination_reason=TerminationReason.AGENT_STOP,
            messages=self.messages,
        )
        evaluation_name = str(self.config.get("evaluation_type", "action")).upper()
        evaluation_type = getattr(EvaluationType, evaluation_name, EvaluationType.ACTION)
        reward_info = evaluate_simulation(
            simulation,
            self.task,
            evaluation_type=evaluation_type,
            solo_mode=bool(self.config.get("solo_mode", False)),
            domain=self.config["domain"],
        )
        reward = float(getattr(reward_info, "reward", 0.0) or 0.0)
        return reward, {"reward_info": reward_info.model_dump() if hasattr(reward_info, "model_dump") else str(reward_info)}

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.env is not None and self.task is not None
        name, arguments, structured = _tool_action(payload.get("action"))
        self.step_count += 1
        info = dict(self.last_info)
        info["last_action"] = name
        info["structured_action"] = structured

        if name in {"respond", "final_response", "finish"}:
            message = str(arguments.get("message") or arguments.get("answer") or arguments.get("response") or "")
            self.final_score, eval_info = self._finish(message)
            self.done = True
            info.update(eval_info)
            info["done"] = True
            self.last_info = info
            return self._result(f"Final response submitted: {message}", info)

        try:
            result = self.env.use_tool(name, **arguments)
            self._record_tool_call(name, arguments, result, error=False)
            observation = f"Tool result for {name}:\n{_json_text(result)}"
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            self._record_tool_call(name, arguments, result, error=True)
            observation = f"Tool call failed for {name}:\n{_json_text(result)}"
            info["tool_error"] = result["error"]
        info["done"] = self.done
        self.last_info = info
        return self._result(observation, info)

    def _result(self, observation: str, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation": observation,
            "score": self.final_score,
            "done": self.done,
            "success": self.final_score > 0,
            "info": info,
            "task_index": self.task_index,
            "num_tasks": len(self.tasks),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._result("", self.last_info)

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"reset_count": self.reset_count, "step_count": self.step_count}

    def close(self) -> dict[str, Any]:
        self.env = None
        self.task = None
        self.messages = []
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a process-isolated tau2 environment server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18182)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    env_config, server_config = _load_config(args.config)
    serve_process_pool(
        host=args.host,
        port=args.port,
        backend_cls=Tau2Backend,
        env_config=env_config,
        server_config=server_config,
        env_name="tau2",
    )


if __name__ == "__main__":
    main()
