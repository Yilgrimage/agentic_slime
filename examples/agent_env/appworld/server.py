from __future__ import annotations

import argparse
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import yaml

from examples.agent_env.env_episode import (
    call_policy_chat,
    finish_reason_is_length,
    parse_text_action,
    policy_context_limit_reached,
)
from examples.agent_env.prompting import require_prompt
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
    pool_size = _deep_get(raw, "env_server", "pool_size", None)
    if pool_size is None:
        raise ValueError("Missing env_server.pool_size in AppWorld env_config.yaml")
    return {
        "pool_size": int(pool_size),
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
    interaction = raw.get("interaction") if isinstance(raw.get("interaction"), dict) else {}
    text_action = interaction.get("text_action") if isinstance(interaction.get("text_action"), dict) else {}
    action = raw.get("action") if isinstance(raw.get("action"), dict) else {}
    return {
        "root": root,
        "max_turns": int(raw.get("max_turns") or _deep_get(raw, "appworld", "max_interactions", 40)),
        "text_action_tag": str(text_action.get("tag") or "code"),
        "legacy_text_as_code": bool(action.get("legacy_text_as_code", False)),
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

    def _initial_messages(self, prompt: str, observation: str, info: dict[str, Any]) -> list[dict[str, str]]:
        system_prompt = require_prompt(prompt, env_name="AppWorld", source="run_episode.prompt")
        user_prompt = self._observation_text(observation, info).strip()
        if not user_prompt:
            raise ValueError("AppWorld initial user prompt is empty after reset")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _observation_text(self, observation: str, info: dict[str, Any]) -> str:
        return f"Observation:\n{observation.strip()}\n"

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        split = str(payload.get("split") or self.split)
        dataset = str(payload.get("dataset_name") or _split_dataset(self.config, split))
        if dataset != self.dataset_name:
            self.dataset_name = dataset
            self.task_ids = self._load_task_ids(dataset)
        requested_task_id = str(payload.get("task_id") or "").strip()
        if requested_task_id:
            if requested_task_id not in self.task_ids:
                raise KeyError(
                    f"AppWorld task_id from prompt data is not available in dataset={dataset}: {requested_task_id}"
                )
            self.task_index = self.task_ids.index(requested_task_id)
        else:
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

    def run_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        if not policy:
            raise ValueError("AppWorld run_episode requires policy endpoint")

        reset = self.reset(payload)
        observation = str(reset.get("observation", ""))
        info = reset.get("info") if isinstance(reset.get("info"), dict) else {}
        messages = self._initial_messages(str(payload.get("prompt") or ""), observation, info)
        metadata: dict[str, Any] = {
            "actions": [],
            "action_parse_modes": [],
            "format_checks": [],
            "format_errors": 0,
            "policy_usage": [],
            "turn_count": 0,
        }
        include_trace = bool(payload.get("include_trace", False))
        if include_trace:
            metadata["messages"] = messages
            metadata["turns"] = []

        runtime = self.config
        max_turns = int(payload.get("max_turns") or runtime.get("max_turns") or runtime.get("max_interactions") or 40)
        sampling_params = payload.get("sampling_params") if isinstance(payload.get("sampling_params"), dict) else {}
        max_tokens = int(payload.get("max_response_tokens") or 1024)
        timeout_s = float((payload.get("timeouts") or {}).get("policy_s") or 120)
        tag = str(runtime.get("text_action_tag") or "code")
        legacy_text_as_code = bool(runtime.get("legacy_text_as_code", False))
        status = "truncated"
        truncated_reason = "max_turns"
        final_score = 0.0
        success = False
        last_step: dict[str, Any] = reset

        for turn in range(max_turns):
            turn_trace: dict[str, Any] = {"turn": turn} if include_trace else {}
            reply = call_policy_chat(
                policy=policy,
                messages=messages,
                sampling_params=sampling_params,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            assistant_message = reply.message
            messages.append(assistant_message)
            metadata["policy_usage"].append(reply.usage)
            if policy_context_limit_reached(reply):
                metadata["context_limit_hits"] = int(metadata.get("context_limit_hits", 0) or 0) + 1
                truncated_reason = "context_limit_after_observation"
                if include_trace:
                    turn_trace.update(
                        {
                            "assistant_message": assistant_message,
                            "format_valid": False,
                            "parse_mode": "context_limit",
                            "finish_reason": reply.finish_reason,
                            "truncated_reason": truncated_reason,
                        }
                    )
                    metadata["turns"].append(turn_trace)
                break
            if finish_reason_is_length(reply):
                metadata["max_response_tokens_hits"] = int(metadata.get("max_response_tokens_hits", 0) or 0) + 1

            content = str(assistant_message.get("content") or "")
            code, valid, parse_mode = parse_text_action(content, tag=tag)
            metadata["action_parse_modes"].append(parse_mode)
            metadata["format_checks"].append({"turn": turn, "valid": bool(valid), "parse_mode": parse_mode})
            if not valid:
                metadata["format_errors"] = int(metadata.get("format_errors", 0) or 0) + 1
            if valid or legacy_text_as_code:
                action: dict[str, Any] = {"type": "tool_call", "name": "execute", "arguments": {"code": code}}
            else:
                action = {"type": "tool_call", "name": "format_error", "arguments": {"response": content[:500]}}
            metadata["actions"].append(action)
            step = self.step({"action": action})
            last_step = step
            observation = str(step.get("observation", ""))
            info = step.get("info") if isinstance(step.get("info"), dict) else {}
            final_score = float(step.get("score", 0.0) or 0.0)
            done = bool(step.get("done", False))
            success = bool(step.get("success", False))
            if include_trace:
                turn_trace.update(
                    {
                        "assistant_message": assistant_message,
                        "action": action,
                        "format_valid": bool(valid),
                        "parse_mode": parse_mode,
                        "finish_reason": reply.finish_reason,
                        "env_step": step,
                    }
                )
                metadata["turns"].append(turn_trace)
            if done:
                status = "completed"
                truncated_reason = ""
                break
            messages.append({"role": "user", "content": self._observation_text(observation, info)})

        metadata["turn_count"] = len(metadata["actions"])
        metadata["format_ok"] = int(metadata.get("format_errors", 0) or 0) == 0
        if truncated_reason:
            metadata["truncated_reason"] = truncated_reason
        return {
            "status": status,
            "observation": observation,
            "score": final_score,
            "done": status == "completed",
            "success": success,
            "info": last_step.get("info") if isinstance(last_step.get("info"), dict) else {},
            "split": self.split,
            "task_index": self.task_index,
            "num_tasks": len(self.task_ids),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
            "metadata": metadata,
        }

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
