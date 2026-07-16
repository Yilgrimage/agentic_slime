from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import inspect
import urllib.error
import urllib.request

import yaml

from examples.agent_env.env_episode import (
    call_policy_chat,
    choose_tool_action,
    environment_messages_from_step,
    extract_tool_action,
    extract_tool_actions,
    finish_reason_is_length,
    mark_assistant_messages_untrained,
    policy_context_limit_reached,
)
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
        raise ValueError("Missing env_server.pool_size in tau2 env_config.yaml")
    return {
        "pool_size": int(pool_size),
        "acquire_timeout_s": float(_deep_get(raw, "env_server", "acquire_timeout_s", 600.0)),
        "lease_ttl_s": float(_deep_get(raw, "env_server", "lease_ttl_s", 1800.0)),
        "idempotency_ttl_s": float(_deep_get(raw, "env_server", "idempotency_ttl_s", 300.0)),
        "worker_start_timeout_s": float(_deep_get(raw, "env_server", "worker_start_timeout_s", 300.0)),
        "worker_request_timeout_s": float(_deep_get(raw, "env_server", "worker_request_timeout_s", 180.0)),
        "worker_episode_timeout_s": float(_deep_get(raw, "env_server", "worker_episode_timeout_s", _deep_get(raw, "env_server", "worker_request_timeout_s", 180.0))),
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


def _env_value(value: Any, *envvars: str, default: str = "") -> str:
    if value is not None and not isinstance(value, str):
        return str(value)
    text = str(value or "").strip()
    for envvar in envvars:
        if text and text != f"${{{envvar}}}":
            break
        env_value = os.environ.get(envvar)
        if env_value:
            return env_value
    if text:
        return os.path.expandvars(text)
    return default


def _default_tau2_data_dir() -> str:
    agent_data_dir = os.environ.get("AGENT_ENV_DATA_DIR")
    if agent_data_dir:
        return str(Path(agent_data_dir).expanduser() / "data")
    return os.environ.get("TAU2_DATA_DIR", "")


def _csv_values(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _flatten_numeric_usage(usage: Any, prefix: str = "") -> dict[str, float]:
    if not isinstance(usage, dict):
        return {}
    output: dict[str, float] = {}
    for key, value in usage.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            output.update(_flatten_numeric_usage(value, name))
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            output[name] = output.get(name, 0.0) + float(value)
    return output


def _sum_usage(usages: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for usage in usages:
        for key, value in _flatten_numeric_usage(usage).items():
            totals[key] = totals.get(key, 0.0) + value
    return totals


def _merge_user_model_usage(metadata: dict[str, Any], info: dict[str, Any]) -> None:
    usages = info.get("user_model_usage")
    if not isinstance(usages, list) or not usages:
        return
    cleaned = [usage for usage in usages if isinstance(usage, dict)]
    if not cleaned:
        return
    metadata.setdefault("user_model_usage", []).extend(cleaned)
    metadata["user_model_usage_totals"] = _sum_usage(metadata["user_model_usage"])
    metadata["user_model_call_count"] = len(metadata["user_model_usage"])


_USER_SIM_TRANSIENT_INFO_KEYS = (
    "user_model_calls",
    "user_model_latency_s",
    "user_model_usage",
    "user_model_usage_totals",
    "user_sim_trace",
)


def _clear_user_sim_transient_info(info: dict[str, Any]) -> None:
    for key in _USER_SIM_TRANSIENT_INFO_KEYS:
        info.pop(key, None)


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def _worker_route_seed(worker_id: str) -> int:
    match = re.search(r"(?:^|[-_])(\d+)$", worker_id)
    if match:
        return int(match.group(1))
    return _stable_int(worker_id)


def _environment_config(raw: dict) -> dict:
    data_dir = _env_path(_deep_get(raw, "tau2", "data_dir", _default_tau2_data_dir()), "TAU2_DATA_DIR")
    return {
        "_agent_env_runtime": {
            "max_turns": int(raw.get("max_turns", _deep_get(raw, "tau2", "max_turns", 20))),
            "timeouts": raw.get("timeouts") if isinstance(raw.get("timeouts"), dict) else {},
            "interaction": raw.get("interaction") if isinstance(raw.get("interaction"), dict) else {},
            "observation": raw.get("observation") if isinstance(raw.get("observation"), dict) else {},
            "action": raw.get("action") if isinstance(raw.get("action"), dict) else {},
        },
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
        "user_sim_enabled": bool(_deep_get(raw, "tau2", "user_sim_enabled", False)),
        "user_model_provider": _env_value(
            _deep_get(raw, "tau2", "user_model_provider", ""),
            "AUX_ENDPOINT_PROVIDER",
            "TAU2_USER_MODEL_PROVIDER",
            default="sglang",
        ),
        "user_model": _env_value(
            _deep_get(raw, "tau2", "user_model", ""),
            "AUX_ENDPOINT_MODEL",
            "TAU2_USER_MODEL",
            default="local-user-sim",
        ),
        "user_model_base_url": _env_value(
            _deep_get(raw, "tau2", "user_model_base_url", ""),
            "AUX_ENDPOINT_BASE_URL",
            "TAU2_USER_MODEL_BASE_URL",
        ),
        "user_model_api_key": _env_value(
            _deep_get(raw, "tau2", "user_model_api_key", ""),
            "AUX_ENDPOINT_API_KEY",
            "TAU2_USER_MODEL_API_KEY",
        ),
        "user_model_api_key_path": _env_value(
            _deep_get(raw, "tau2", "user_model_api_key_path", ""),
            "AUX_ENDPOINT_API_KEY_PATH",
            "TAU2_USER_MODEL_API_KEY_PATH",
        ),
        "user_model_timeout_s": float(_env_value(_deep_get(raw, "tau2", "user_model_timeout_s", ""), "AUX_ENDPOINT_TIMEOUT_S", default="120")),
        "user_model_max_tokens": int(_env_value(_deep_get(raw, "tau2", "user_model_max_tokens", ""), "AUX_ENDPOINT_MAX_TOKENS", default="512")),
        "user_model_temperature": float(_env_value(_deep_get(raw, "tau2", "user_model_temperature", ""), "AUX_ENDPOINT_TEMPERATURE", default="0.0")),
        "user_model_top_p": float(_env_value(_deep_get(raw, "tau2", "user_model_top_p", ""), "AUX_ENDPOINT_TOP_P", default="1.0")),
        "user_model_enable_thinking": _env_value(
            _deep_get(raw, "tau2", "user_model_enable_thinking", ""),
            "AUX_ENDPOINT_ENABLE_THINKING",
            default="0",
        ).strip().lower()
        in {"1", "true", "yes", "y", "on"},
        "user_model_separate_reasoning": _env_value(
            _deep_get(raw, "tau2", "user_model_separate_reasoning", ""),
            "AUX_ENDPOINT_SEPARATE_REASONING",
            default="1",
        ).strip().lower()
        in {"1", "true", "yes", "y", "on"},
        "user_model_reasoning_effort": _env_value(
            _deep_get(raw, "tau2", "user_model_reasoning_effort", ""),
            "AUX_ENDPOINT_REASONING_EFFORT",
        ),
        "max_user_tool_rounds": int(_deep_get(raw, "tau2", "max_user_tool_rounds", 20)),
        "user_sim_trace": bool(_deep_get(raw, "tau2", "user_sim_trace", False)),
        "ignore_user_stop_before_assistant_tool": bool(
            _deep_get(raw, "tau2", "ignore_user_stop_before_assistant_tool", True)
        ),
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


def _strip_model_artifacts(text: str) -> str:
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(token, "")
    return text.strip()


def _visible_model_text(text: str) -> str:
    """Strip chat boundary artifacts from text visible to the user simulator."""
    return _strip_model_artifacts(str(text or ""))


_USER_CONTROL_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")


def _user_control_tokens(text: str) -> list[str]:
    return [token for token in _USER_CONTROL_TOKENS if token in text]


def _remove_user_control_tokens(text: str, tokens: list[str]) -> str:
    for token in tokens:
        text = text.replace(token, "")
    return _strip_model_artifacts(text)


_USER_CONTROL_TOKEN_PROTOCOL = """
Protocol for task-control tokens:
- ###STOP###, ###TRANSFER###, and ###OUT-OF-SCOPE### are plain text tokens in the user message content.
- Never emit these control tokens as tool calls or function calls.
- User tools are only for real user-side actions requested by the agent.
- Do not include control tokens while giving requested information, identifiers, confirmations, or troubleshooting answers that require the agent to continue.
- When the task is complete, transferred, or out of scope, send a normal text message containing the control token and no tool call.
""".strip()


def _tool_action(payload: Any) -> tuple[str, dict[str, Any], bool]:
    if isinstance(payload, dict) and payload.get("type") == "tool_call":
        name = str(payload.get("name") or "").strip()
        arguments = payload.get("arguments") or {}
        return name, arguments if isinstance(arguments, dict) else {}, True
    if isinstance(payload, dict) and payload.get("type") == "assistant_message":
        return "__assistant_message__", {"content": str(payload.get("content") or "")}, True
    if isinstance(payload, str):
        return "__assistant_message__", {"content": payload.strip()}, False
    return str(payload), {}, False


def _parameter_schema(name: str, param: inspect.Parameter) -> dict[str, Any]:
    annotation = param.annotation
    if annotation in (int, "int"):
        schema_type = "integer"
    elif annotation in (float, "float"):
        schema_type = "number"
    elif annotation in (bool, "bool"):
        schema_type = "boolean"
    elif annotation in (list, tuple, "list", "tuple"):
        schema_type = "array"
    elif annotation in (dict, "dict"):
        schema_type = "object"
    else:
        schema_type = "string"
    return {"type": schema_type, "description": name.replace("_", " ")}


def _callable_tool_schema(fn: Any, name: str | None = None, description: str | None = None) -> dict[str, Any]:
    tool_name = str(name or getattr(fn, "__name__", "tool"))
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    try:
        signature = inspect.signature(fn)
        for param_name, param in signature.parameters.items():
            if param_name in {"self", "cls"} or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                continue
            parameters["properties"][param_name] = _parameter_schema(param_name, param)
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)
    except Exception:
        pass
    if not parameters["required"]:
        parameters.pop("required", None)
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": str(description or getattr(fn, "__doc__", "") or tool_name).strip(),
            "parameters": parameters,
        },
    }


def _openai_tool_schema(tool: Any) -> dict[str, Any] | None:
    if isinstance(tool, dict):
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            return tool
        name = tool.get("name") or tool.get("tool_name")
        if name:
            return {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": str(tool.get("description") or name),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
    for method_name in ("to_openai_tool", "openai_schema", "schema"):
        value = getattr(tool, method_name, None)
        if isinstance(value, dict):
            return _openai_tool_schema(value)
        if callable(value):
            try:
                return _openai_tool_schema(value())
            except Exception:
                pass
    fn = getattr(tool, "function", None) or getattr(tool, "func", None) or getattr(tool, "callable", None)
    name = getattr(tool, "name", None) or getattr(fn, "__name__", None)
    description = getattr(tool, "description", None) or getattr(tool, "__doc__", None)
    if callable(fn):
        return _callable_tool_schema(fn, name=name, description=description)
    if name:
        return {
            "type": "function",
            "function": {
                "name": str(name),
                "description": str(description or name),
                "parameters": {"type": "object", "properties": {}},
            },
        }
    return None


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def _openai_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if not isinstance(arguments, dict):
        arguments = {}
    return json.dumps(arguments, ensure_ascii=False)


def _make_openai_user_simulator(
    *,
    llm: str,
    instructions: str,
    tools: list[Any],
    call_model: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]],
    message_to_openai: Callable[[Any], dict[str, Any] | None],
    tool_call_from_openai: Callable[[dict[str, Any], str], Any],
    separate_reasoning: bool,
) -> Any:
    from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
    from tau2.user.user_simulator import UserSimulator

    class OpenAIUserSimulator(UserSimulator):
        @property
        def system_prompt(self) -> str:
            return f"{super().system_prompt}\n\n{_USER_CONTROL_TOKEN_PROTOCOL}"

        def _generate_next_message(self, message: Any, state: Any) -> Any:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            elif isinstance(message, ToolMessage):
                state.messages.append(message)
            elif message.has_content() or message.is_tool_call():
                state.messages.append(message)

            openai_messages = []
            for item in state.system_messages + state.flip_roles():
                converted = message_to_openai(item)
                if converted is not None:
                    openai_messages.append(converted)

            openai_tools = []
            for tool in self.tools or []:
                schema = _openai_tool_schema(tool)
                if schema is not None:
                    openai_tools.append(schema)
            raw_message = call_model(openai_messages, openai_tools)
            content = str(raw_message.get("content") or "")
            if raw_message.get("reasoning_content") and separate_reasoning:
                content = _strip_model_artifacts(content)
            else:
                content = _visible_model_text(content)

            user_message = UserMessage(
                role="user",
                content=content,
                usage=raw_message.get("_usage"),
                raw_data={k: v for k, v in raw_message.items() if not str(k).startswith("_")},
                generation_time_seconds=raw_message.get("_latency_s"),
            )
            raw_tool_calls = raw_message.get("tool_calls") or []
            if raw_tool_calls:
                tool_calls = [
                    tool_call_from_openai(raw_tool_call, "user")
                    for raw_tool_call in raw_tool_calls
                    if isinstance(raw_tool_call, dict)
                ]
                if tool_calls:
                    user_message.tool_calls = tool_calls
            return user_message

    return OpenAIUserSimulator(
        llm=llm,
        instructions=instructions,
        tools=tools or None,
        llm_args={},
    )


class Tau2Backend:
    def __init__(self, worker_id: str, split: str, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.split = split
        self.config = config
        self.tasks_cache: dict[tuple[str, str | None], list[Any]] = {}
        self.tasks: list[Any] = []
        self.env: Any | None = None
        self.task: Any | None = None
        self.messages: list[Any] = []
        self.domain = str(config["domain"])
        self.task_set = str(config["task_set"])
        self.task_index = 0
        self.reset_count = 0
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.last_info: dict[str, Any] = {}
        self.user_simulator: Any | None = None
        self.user_state: Any | None = None
        self.user_model_call_index = _worker_route_seed(worker_id)
        self.user_model_endpoint: dict[str, str] | None = None
        self.initial_agent_openai_message_count = 0

    @property
    def runtime(self) -> dict[str, Any]:
        value = self.config.get("_agent_env_runtime")
        return value if isinstance(value, dict) else {}

    @property
    def user_sim_enabled(self) -> bool:
        return bool(self.config.get("user_sim_enabled", False)) and not bool(self.config.get("solo_mode", False))

    def start(self) -> dict[str, Any]:
        data_dir = self.config.get("data_dir")
        if data_dir:
            os.environ["TAU2_DATA_DIR"] = str(data_dir)
        self.tasks = self._tasks_for(self.task_set, self.config.get("split"))
        return {"num_tasks": len(self.tasks), "domain": self.domain, "task_set": self.task_set}

    def _tasks_for(self, task_set: str, split: str | None) -> list[Any]:
        key = (task_set, split)
        if key not in self.tasks_cache:
            from tau2.run import get_tasks

            self.tasks_cache[key] = list(
                get_tasks(
                    task_set,
                    task_split_name=split,
                    num_tasks=self.config.get("num_tasks"),
                )
            )
        return self.tasks_cache[key]

    def _build_env(self, domain: str) -> Any:
        from tau2.run import build_environment

        return build_environment(domain, solo_mode=bool(self.config.get("solo_mode", False)))

    def _resolve_task_ref_path(self, task_ref: dict[str, Any]) -> Path:
        path = Path(os.path.expandvars(str(task_ref.get("path") or ""))).expanduser()
        if not path.is_absolute():
            root = Path(os.path.expandvars(str(task_ref.get("root") or self.config.get("data_dir") or "."))).expanduser()
            path = root / path
        return path

    def _load_file_task(self, task_ref: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        from tau2.data_model.tasks import Task

        path = self._resolve_task_ref_path(task_ref)
        raw = json.loads(path.read_text(encoding="utf-8"))
        task_payload = raw.get("task") if isinstance(raw.get("task"), dict) else raw
        task_payload = dict(task_payload)
        if isinstance(task_payload.get("evaluation_criteria"), str):
            criteria_text = task_payload["evaluation_criteria"]
            try:
                task_payload["evaluation_criteria"] = json.loads(criteria_text)
            except json.JSONDecodeError:
                pass
        return Task.model_validate(task_payload), {"task_file": str(path), "task_root": str(path.parent), "raw": raw}

    def _load_db(self, domain: str, db_path: str | None, base_dir: str | None = None) -> Any | None:
        if not db_path:
            return None
        path = Path(os.path.expandvars(str(db_path))).expanduser()
        if not path.is_absolute():
            path = Path(os.path.expandvars(str(base_dir or self.config.get("data_dir") or "."))).expanduser() / path
        if domain == "retail":
            from tau2.domains.retail.data_model import RetailDB

            return RetailDB.load(path)
        if domain == "airline":
            from tau2.domains.airline.data_model import FlightDB

            return FlightDB.load(path)
        if domain == "telecom":
            from tau2.domains.telecom.data_model import TelecomDB

            return TelecomDB.load(path)
        return None

    def _build_env_for_task(self, domain: str, task_context: dict[str, Any] | None = None) -> Any:
        if not task_context:
            return self._build_env(domain)
        raw = task_context.get("raw") or {}
        base_dir = raw.get("_data_root") or task_context.get("task_root")
        db = self._load_db(domain, raw.get("db_path"), base_dir=base_dir)
        if db is None:
            return self._build_env(domain)
        from tau2.run import build_environment

        return build_environment(
            domain,
            solo_mode=bool(self.config.get("solo_mode", False)),
            env_kwargs={"db": db},
        )

    def _apply_initial_state(self) -> None:
        if self.env is None or self.task is None:
            return
        initial_state = getattr(self.task, "initial_state", None)
        if initial_state is None:
            return
        self.env.set_state(
            initial_state.initialization_data,
            initial_state.initialization_actions,
            list(initial_state.message_history or []),
        )

    def _initial_message_history(self) -> list[Any]:
        if self.task is None:
            return []
        initial_state = getattr(self.task, "initial_state", None)
        if initial_state is None:
            return []
        return list(initial_state.message_history or [])

    def _tools_description(self) -> str:
        if self.env is None or not self.config.get("include_tools", True):
            return ""
        try:
            descriptions = [str(tool).strip() for tool in self.env.get_tools()]
            return "\n\n".join(description for description in descriptions if description)
        except Exception:
            try:
                return str(self.env.get_tools_description("assistant"))
            except Exception:
                tools = []
                for tool in self.env.get_tools():
                    tools.append(getattr(tool, "name", str(tool)))
                return "\n".join(tools)

    def _available_tool_names(self, info: dict[str, Any]) -> list[str]:
        tools = info.get("tools") if isinstance(info, dict) else None
        if isinstance(tools, list):
            return [str(tool) for tool in tools]
        return []

    def _agent_messages_for_policy(self) -> list[dict[str, Any]]:
        agent_messages = self._agent_openai_messages()
        initial_count = max(0, min(self.initial_agent_openai_message_count, len(agent_messages)))
        if not initial_count:
            return agent_messages
        return mark_assistant_messages_untrained(agent_messages[:initial_count]) + agent_messages[initial_count:]

    def _policy_messages(self, observation: str, info: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
        policy = str(info.get("policy") or "").strip()
        messages: list[dict[str, Any]] = []
        if policy:
            messages.append({"role": "system", "content": policy})
        agent_messages = self._agent_messages_for_policy()
        if agent_messages:
            messages.extend(agent_messages)
            return messages
        if str(observation).strip():
            messages.append({"role": "user", "content": str(observation).strip()})
            return messages
        if str(prompt).strip():
            messages.append({"role": "user", "content": str(prompt).strip()})
            return messages
        raise ValueError("tau2 initial messages are empty: policy, env messages, observation, and prompt are all missing")


    def _policy_doc(self) -> str:
        if not self.env or not self.config.get("include_policy", True):
            return ""
        policy = str(self.env.get_policy()).strip()
        if not policy:
            raise RuntimeError("tau2 include_policy=true but env.get_policy() returned an empty policy")
        try:
            from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

            return SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION).strip()
        except Exception as exc:
            raise RuntimeError("Failed to load tau2.agent.llm_agent system prompt templates") from exc

    def _user_tool_schemas(self) -> list[dict[str, Any]]:
        if self.env is None or not self.config.get("include_tools", True):
            return []
        try:
            user_tools = self.env.get_user_tools()
        except Exception:
            return []
        schemas = []
        for tool in user_tools or []:
            schema = _openai_tool_schema(tool)
            if schema is not None:
                schemas.append(schema)
        return schemas

    def _user_tools(self) -> list[Any]:
        if self.env is None or not self.config.get("include_tools", True):
            return []
        try:
            return list(self.env.get_user_tools() or [])
        except Exception:
            return []

    def _tool_schema_names(self, tools: list[dict[str, Any]]) -> set[str]:
        names = set()
        for tool in tools:
            name = str((tool.get("function") or {}).get("name") or "")
            if name:
                names.add(name)
        return names

    def _tool_schemas(self) -> list[dict[str, Any]]:
        if self.env is None or not self.config.get("include_tools", True):
            return []
        schemas = []
        for tool in self.env.get_tools():
            schema = _openai_tool_schema(tool)
            if schema is not None:
                schemas.append(schema)
        seen = set()
        unique = []
        for schema in schemas:
            name = str((schema.get("function") or {}).get("name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            unique.append(schema)
        return unique

    def _episode_tool_schemas(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        raw_tools = info.get("tool_schemas") if isinstance(info, dict) else None
        if isinstance(raw_tools, list) and raw_tools:
            return [tool for tool in raw_tools if isinstance(tool, dict)]
        return self._tool_schemas()

    def _initial_observation(self) -> str:
        assert self.task is not None
        parts = [f"Task id: {getattr(self.task, 'id', self.task_index)}"]
        if self.user_sim_enabled:
            parts.append("A user has contacted customer support. Start the conversation and help the user according to the policy.")
            return "\n\n".join(parts)
        description = str(getattr(self.task, "description", "") or "").strip()
        if description:
            parts.append(f"Task:\n{description}")
        user_scenario = str(getattr(self.task, "user_scenario", "") or "").strip()
        if user_scenario:
            parts.append(f"User scenario:\n{user_scenario}")
        ticket = str(getattr(self.task, "ticket", "") or "").strip()
        if ticket:
            parts.append(f"Ticket:\n{ticket}")
        if self.user_sim_enabled:
            parts.append("The user will interact with you. Send normal assistant messages to talk to the user. Use tools when needed.")
        else:
            parts.append("Use tools to solve the task.")
        return "\n\n".join(parts)

    def _api_key(self, explicit_key: str | None = None, key_path: str | None = None) -> str:
        explicit = str(explicit_key if explicit_key is not None else self.config.get("user_model_api_key") or "").strip()
        if explicit:
            return explicit
        path = str(key_path if key_path is not None else self.config.get("user_model_api_key_path") or "").strip()
        if path:
            try:
                return Path(path).expanduser().read_text(encoding="utf-8").strip()
            except Exception:
                logger.warning("Failed to read tau2 user model api key path: %s", path, exc_info=True)
        return os.environ.get("TAU2_USER_MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"

    def _chat_completions_url(self, base_url: str | None = None, provider_name: str | None = None) -> str:
        base = str(base_url if base_url is not None else self.config.get("user_model_base_url") or os.environ.get("TAU2_USER_MODEL_BASE_URL") or "").strip()
        if not base:
            raise ValueError("tau2.user_model_base_url or TAU2_USER_MODEL_BASE_URL is required when user_sim_enabled=true")
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        provider = str(provider_name if provider_name is not None else self.config.get("user_model_provider") or "").strip().lower()
        if provider == "ark":
            return f"{base}/chat/completions"
        if provider == "deepseek" and not base.endswith("/v1"):
            return f"{base}/chat/completions"
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _select_user_model_endpoint(self) -> dict[str, str]:
        providers = _csv_values(self.config.get("user_model_provider") or "sglang")
        models = _csv_values(self.config.get("user_model") or "local-user-sim")
        base_urls = _csv_values(self.config.get("user_model_base_url") or os.environ.get("TAU2_USER_MODEL_BASE_URL") or "")
        api_keys = _csv_values(self.config.get("user_model_api_key") or "")
        api_key_paths = _csv_values(self.config.get("user_model_api_key_path") or "")
        width = max(len(providers), len(models), len(base_urls), len(api_keys), len(api_key_paths), 1)
        index = self.user_model_call_index % width
        self.user_model_call_index += 1

        def pick(values: list[str], default: str = "") -> str:
            if not values:
                return default
            return values[index % len(values)]

        return {
            "provider": pick(providers, "sglang").lower(),
            "model": pick(models, "local-user-sim"),
            "base_url": pick(base_urls),
            "api_key": pick(api_keys),
            "api_key_path": pick(api_key_paths),
            "route_index": str(index),
        }

    def _user_model_endpoint_info(self) -> dict[str, str]:
        endpoint = self.user_model_endpoint or {}
        return {
            "provider": str(endpoint.get("provider") or ""),
            "model": str(endpoint.get("model") or ""),
            "base_url": str(endpoint.get("base_url") or ""),
            "route_index": str(endpoint.get("route_index") or ""),
        }

    def _tau2_message_to_openai(self, message: Any) -> dict[str, Any] | None:
        from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage

        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content or ""}
        if isinstance(message, UserMessage):
            item: dict[str, Any] = {"role": "user", "content": message.content or ""}
            if message.tool_calls:
                item["tool_calls"] = [self._tool_call_to_openai(tool_call) for tool_call in message.tool_calls]
            return _strip_none(item)
        if isinstance(message, AssistantMessage):
            item = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                item["tool_calls"] = [self._tool_call_to_openai(tool_call) for tool_call in message.tool_calls]
            return _strip_none(item)
        if isinstance(message, ToolMessage):
            return {"role": "tool", "tool_call_id": message.id, "content": message.content or ""}
        return None

    def _agent_openai_messages(self, messages: list[Any] | None = None) -> list[dict[str, Any]]:
        from tau2.agent.base_agent import is_valid_agent_history_message

        output = []
        for message in messages if messages is not None else self.messages:
            if not is_valid_agent_history_message(message):
                continue
            item = self._tau2_message_to_openai(message)
            if item is not None:
                output.append(item)
        return output

    def _tool_call_to_openai(self, tool_call: Any) -> dict[str, Any]:
        return {
            "id": getattr(tool_call, "id", "") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": str(getattr(tool_call, "name", "")),
                "arguments": _openai_tool_arguments(getattr(tool_call, "arguments", {}) or {}),
            },
        }

    def _openai_tool_call_to_tau2(self, raw: dict[str, Any], requestor: str) -> Any:
        from tau2.data_model.message import ToolCall

        fn = raw.get("function") or {}
        arguments = fn.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(
            id=str(raw.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
            name=str(fn.get("name") or raw.get("name") or ""),
            arguments=arguments,
            requestor=requestor,
        )

    def _call_user_model(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if self.user_model_endpoint is None:
            self.user_model_endpoint = self._select_user_model_endpoint()
        endpoint = self.user_model_endpoint
        provider = endpoint["provider"]
        enable_thinking = bool(self.config.get("user_model_enable_thinking", False))
        payload: dict[str, Any] = {
            "model": endpoint["model"],
            "messages": messages,
            "max_tokens": int(self.config.get("user_model_max_tokens", 512)),
        }
        if provider != "deepseek" or not enable_thinking:
            payload["temperature"] = float(self.config.get("user_model_temperature", 0.0))
            payload["top_p"] = float(self.config.get("user_model_top_p", 1.0))
        if provider in {"sglang", "vllm", "aux", "local"}:
            payload["separate_reasoning"] = bool(self.config.get("user_model_separate_reasoning", True))
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        elif provider == "deepseek":
            payload["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
            effort = str(self.config.get("user_model_reasoning_effort") or "").strip()
            if enable_thinking and effort:
                payload["reasoning_effort"] = effort
        if tools:
            payload["tools"] = tools
        data = json.dumps(_strip_none(payload), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._chat_completions_url(endpoint["base_url"], provider),
            data=data,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key(endpoint['api_key'], endpoint['api_key_path'])}",
            },
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=float(self.config.get("user_model_timeout_s", 120.0))) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"user model HTTP {exc.code}: {body[:1000]}") from exc
        result = json.loads(body)
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        message["_latency_s"] = time.perf_counter() - start
        usage = result.get("usage")
        if usage:
            message["_usage"] = usage
        return message

    def _build_user_simulator(self) -> Any:
        return _make_openai_user_simulator(
            llm=str(self.config.get("user_model") or "local-user-sim"),
            instructions=str(getattr(self.task, "user_scenario", "") or ""),
            tools=self._user_tools(),
            call_model=self._call_user_model,
            message_to_openai=self._tau2_message_to_openai,
            tool_call_from_openai=self._openai_tool_call_to_tau2,
            separate_reasoning=bool(self.config.get("user_model_separate_reasoning", True)),
        )

    def _has_assistant_tool_call(self) -> bool:
        for message in self.messages:
            if str(getattr(message, "role", "")) != "assistant":
                continue
            if getattr(message, "tool_calls", None):
                return True
        return False

    def _normalize_user_control_response(self, content: str, done: bool) -> tuple[str, bool, dict[str, Any]]:
        tokens = _user_control_tokens(content)
        if not done or "###STOP###" not in tokens:
            return content, done, {}
        if not bool(self.config.get("ignore_user_stop_before_assistant_tool", True)):
            return content, done, {}
        if self._has_assistant_tool_call():
            return content, done, {}

        cleaned = _remove_user_control_tokens(content, ["###STOP###"])
        if not cleaned:
            return content, done, {}
        remaining_tokens = [token for token in _user_control_tokens(cleaned) if token != "###STOP###"]
        return cleaned, bool(remaining_tokens), {
            "user_control_tokens_ignored": ["###STOP###"],
            "user_control_stop_ignored": True,
        }

    def _reset_user_simulator(self) -> None:
        self.user_simulator = None
        self.user_state = None
        if not self.user_sim_enabled:
            return
        from tau2.user.user_simulator_base import is_valid_user_history_message

        self.user_simulator = self._build_user_simulator()
        history = [message for message in self.messages if is_valid_user_history_message(message)]
        self.user_state = self.user_simulator.get_init_state(message_history=history)

    def _generate_user_reply(self, assistant_text: str) -> tuple[str, bool, dict[str, Any]]:
        from tau2.data_model.message import AssistantMessage, MultiToolMessage

        if self.user_simulator is None or self.user_state is None:
            self._reset_user_simulator()
        if self.user_simulator is None or self.user_state is None:
            raise RuntimeError("tau2 user simulator is not initialized")

        assistant_message = AssistantMessage.text(_visible_model_text(assistant_text))
        self.messages.append(assistant_message)
        valid_user_tool_names = self._tool_schema_names(self._user_tool_schemas())
        latencies: list[float] = []
        usages: list[dict[str, Any]] = []
        trace_enabled = bool(self.config.get("user_sim_trace", False))
        user_sim_trace: list[dict[str, Any]] = []
        next_user_input: Any = assistant_message
        for round_index in range(max(1, int(self.config.get("max_user_tool_rounds", 20)))):
            user_message, self.user_state = self.user_simulator.generate_next_message(next_user_input, self.user_state)
            if user_message.generation_time_seconds is not None:
                latencies.append(float(user_message.generation_time_seconds))
            if isinstance(getattr(user_message, "usage", None), dict):
                usages.append(dict(user_message.usage))
            trace_entry: dict[str, Any] = {
                "round": round_index,
                "content": str(user_message.content or ""),
                "tool_calls": [self._tool_call_to_openai(tool_call) for tool_call in (user_message.tool_calls or [])],
            }
            if user_message.generation_time_seconds is not None:
                trace_entry["generation_time_s"] = float(user_message.generation_time_seconds)
            if isinstance(getattr(user_message, "usage", None), dict):
                trace_entry["usage"] = dict(user_message.usage)
            if user_message.tool_calls:
                invalid_tool_names = []
                for tool_call in user_message.tool_calls:
                    name = str(getattr(tool_call, "name", ""))
                    if name not in valid_user_tool_names:
                        invalid_tool_names.append(name)
                if invalid_tool_names:
                    trace_entry["invalid_tool_names"] = invalid_tool_names
                    if trace_enabled:
                        user_sim_trace.append(trace_entry)
                    return f"User simulator produced invalid tool call: {', '.join(invalid_tool_names)}", True, {
                        "user_model_calls": len(latencies),
                        "user_model_latency_s": sum(latencies),
                        "user_model_usage": list(usages),
                        "user_model_usage_totals": _sum_usage(usages),
                        "user_invalid_tool_call": True,
                        "user_invalid_tool_names": invalid_tool_names,
                        "user_model_content": user_message.content,
                        "user_model_tool_calls": [
                            self._tool_call_to_openai(tool_call) for tool_call in user_message.tool_calls
                        ],
                        "discard_sample": True,
                        "discard_reason": "user_invalid_tool_call",
                        **({"user_sim_trace": user_sim_trace} if trace_enabled else {}),
                    }
                self.messages.append(user_message)
                tool_messages = []
                for tool_call in user_message.tool_calls:
                    tool_result = self.env.get_response(tool_call)
                    self.messages.append(tool_result)
                    tool_messages.append(tool_result)
                if trace_enabled:
                    trace_entry["tool_results"] = [
                        self._tau2_message_to_openai(tool_message) for tool_message in tool_messages
                    ]
                    user_sim_trace.append(trace_entry)
                next_user_input = (
                    tool_messages[0]
                    if len(tool_messages) == 1
                    else MultiToolMessage(role="tool", tool_messages=tool_messages)
                )
                continue
            done = self.user_simulator.is_stop(user_message)
            content = str(user_message.content or "")
            content, done, control_info = self._normalize_user_control_response(content, done)
            if content != str(user_message.content or ""):
                try:
                    user_message.content = content
                except Exception:
                    object.__setattr__(user_message, "content", content)
            self.messages.append(user_message)
            if trace_enabled:
                trace_entry["done"] = done
                if control_info:
                    trace_entry.update(control_info)
                user_sim_trace.append(trace_entry)
            return content, done, {
                "user_model_calls": len(latencies),
                "user_model_latency_s": sum(latencies),
                "user_model_usage": list(usages),
                "user_model_usage_totals": _sum_usage(usages),
                **control_info,
                **({"user_sim_trace": user_sim_trace} if trace_enabled else {}),
            }
        fallback = "User simulator exceeded the configured tool-call round limit."
        return fallback, True, {
            "user_model_calls": len(latencies),
            "user_model_latency_s": sum(latencies),
            "user_model_usage": list(usages),
            "user_model_usage_totals": _sum_usage(usages),
            "user_tool_round_limit": True,
            "discard_sample": True,
            "discard_reason": "user_tool_round_limit",
            **({"user_sim_trace": user_sim_trace} if trace_enabled else {}),
        }

    def _prime_user_simulator(self, info: dict[str, Any]) -> str:
        if not self.user_sim_enabled or self.messages:
            return self._initial_observation()
        from tau2.data_model.simulation import TerminationReason
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

        user_message, user_done, user_info = self._generate_user_reply(str(DEFAULT_FIRST_AGENT_MESSAGE.content or ""))
        self.done = bool(user_done)
        info.update(user_info)
        info["done"] = self.done
        if self.done and not bool(info.get("discard_sample", False)):
            self.final_score, eval_info = self._finish("", TerminationReason.USER_STOP)
            info.update(eval_info)
            info["done"] = True
        return f"User response:\n{user_message}"

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.split = str(payload.get("split") or self.split)
        self.domain = str(payload.get("domain") or self.config["domain"])
        self.task_set = str(payload.get("task_set") or self.config["task_set"])
        task_ref = payload.get("task_ref") if isinstance(payload.get("task_ref"), dict) else None
        task_context: dict[str, Any] | None = None
        if task_ref and task_ref.get("type") == "file":
            self.task, task_context = self._load_file_task(task_ref)
            self.task_index = int(payload.get("task_index") or 0)
            self.tasks = [self.task]
            requested_task_id = str(payload.get("task_id") or "").strip()
            actual_task_id = str(getattr(self.task, "id", "") or "").strip()
            if requested_task_id and actual_task_id and requested_task_id != actual_task_id:
                raise ValueError(f"tau2 task_ref loaded task_id={actual_task_id}, expected {requested_task_id}")
        else:
            self.tasks = self._tasks_for(self.task_set, None if self.split == "all" else self.split)
            requested_task_id = str(payload.get("task_id") or "").strip()
            if requested_task_id:
                matches = [
                    index
                    for index, task in enumerate(self.tasks)
                    if str(getattr(task, "id", "") or "").strip() == requested_task_id
                ]
                if not matches:
                    raise KeyError(
                        f"tau2 task_id from prompt data is not available in task_set={self.task_set} "
                        f"split={self.split}: {requested_task_id}"
                    )
                self.task_index = matches[0]
            else:
                self.task_index = int(payload.get("task_index") or 0) % max(1, len(self.tasks))
            self.task = self.tasks[self.task_index]
        self.env = self._build_env_for_task(self.domain, task_context)
        self._apply_initial_state()
        self.messages = self._initial_message_history()
        self.initial_agent_openai_message_count = 0
        self.user_model_endpoint = self._select_user_model_endpoint() if self.user_sim_enabled else None
        self._reset_user_simulator()
        self.reset_count += 1
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.last_info = {
            "domain": self.domain,
            "task_set": self.task_set,
            "task_id": getattr(self.task, "id", None),
            "data_source": payload.get("data_source") or (task_ref or {}).get("source"),
            "task_ref": task_ref,
            "task_file": (task_context or {}).get("task_file"),
            "tools": [getattr(tool, "name", str(tool)) for tool in self.env.get_tools()],
            "policy": self._policy_doc(),
            "tool_schemas": self._tool_schemas(),
            "user_sim_enabled": self.user_sim_enabled,
            "user_model_endpoint": self._user_model_endpoint_info(),
        }
        observation = self._prime_user_simulator(self.last_info)
        self.initial_agent_openai_message_count = len(self._agent_openai_messages())
        self.last_info["agent_messages"] = self._agent_openai_messages()
        return {
            "observation": observation,
            "info": self.last_info,
            "split": self.split,
            "task_index": self.task_index,
            "num_tasks": len(self.tasks),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def _record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        error: bool = False,
        call_id: str = "",
        assistant_content: str = "",
    ) -> list[dict[str, Any]]:
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

        call_id = str(call_id or "").strip() or f"call-{uuid.uuid4().hex[:12]}"
        delta = [
            AssistantMessage.text(
                assistant_content,
                tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments, requestor="assistant")],
            ),
            ToolMessage(id=call_id, role="tool", content=_json_text(result), requestor="assistant", error=error),
        ]
        self.messages.extend(delta)
        return self._agent_openai_messages(delta[1:])

    def _execute_tool_actions(
        self,
        actions: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        info: dict[str, Any],
    ) -> dict[str, Any]:
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

        self.step_count += 1
        next_info = dict(info)
        for transient_key in ("tool_error",):
            next_info.pop(transient_key, None)
        _clear_user_sim_transient_info(next_info)
        next_info["last_action"] = ",".join(str(action.get("name") or "") for action in actions)
        next_info["structured_action"] = True

        assistant_content = str(assistant_message.get("content") or "")
        tool_calls = []
        tool_messages = []
        observations = []
        tool_errors = []
        for action in actions:
            name = str(action.get("name") or "").strip()
            arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
            call_id = str(action.get("tool_call_id") or "").strip() or f"call-{uuid.uuid4().hex[:12]}"
            try:
                result = self.env.use_tool(name, **arguments)
                error = False
                observations.append(f"Tool result for {name}:\n{_json_text(result)}")
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
                error = True
                tool_errors.append(result["error"])
                observations.append(f"Tool call failed for {name}:\n{_json_text(result)}")
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments, requestor="assistant"))
            tool_messages.append(
                ToolMessage(id=call_id, role="tool", content=_json_text(result), requestor="assistant", error=error)
            )

        self.messages.append(AssistantMessage.text(assistant_content, tool_calls=tool_calls))
        self.messages.extend(tool_messages)
        if tool_errors:
            next_info["tool_error"] = "\n".join(tool_errors)
        next_info["message_updates"] = self._agent_openai_messages(tool_messages)
        observation = "\n\n".join(observations)
        next_info["done"] = self.done
        next_info["agent_messages"] = self._agent_openai_messages()
        self.last_info = next_info
        return self._result(observation, next_info)

    def _finish(self, message: str, termination_reason: Any | None = None) -> tuple[float, dict[str, Any]]:
        from tau2.data_model.message import AssistantMessage
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        if message:
            self.messages.append(AssistantMessage.text(message))
        if termination_reason is None:
            termination_reason = TerminationReason.USER_STOP if self.user_sim_enabled else TerminationReason.AGENT_STOP
        simulation = SimulationRun(
            id=f"agent-env-{uuid.uuid4().hex[:12]}",
            task_id=str(getattr(self.task, "id", self.task_index)),
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            duration=0.0,
            termination_reason=termination_reason,
            messages=self.messages,
        )
        evaluation_name = str(self.config.get("evaluation_type", "action")).upper()
        evaluation_type = getattr(EvaluationType, evaluation_name, EvaluationType.ACTION)
        reward_info = evaluate_simulation(
            simulation,
            self.task,
            evaluation_type=evaluation_type,
            solo_mode=bool(self.config.get("solo_mode", False)),
            domain=self.domain,
        )
        reward = float(getattr(reward_info, "reward", 0.0) or 0.0)
        return reward, {"reward_info": reward_info.model_dump() if hasattr(reward_info, "model_dump") else str(reward_info)}

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.env is not None and self.task is not None
        raw_action = payload.get("action")
        name, arguments, structured = _tool_action(raw_action)
        call_id = str(raw_action.get("tool_call_id") or "") if isinstance(raw_action, dict) else ""
        assistant_content = str(raw_action.get("content") or "") if isinstance(raw_action, dict) else ""
        self.step_count += 1
        info = dict(self.last_info)
        for transient_key in ("tool_error",):
            info.pop(transient_key, None)
        _clear_user_sim_transient_info(info)
        info["last_action"] = name
        info["structured_action"] = structured

        if name == "__assistant_message__":
            message = str(arguments.get("content") or "")
            if self.user_sim_enabled:
                try:
                    before = len(self.messages)
                    user_message, user_done, user_info = self._generate_user_reply(message)
                    self.done = bool(user_done)
                    observation = f"User response:\n{user_message}"
                    info.update(user_info)
                    info["done"] = self.done
                    updates = self._agent_openai_messages(self.messages[before:])
                    if updates and updates[0].get("role") == "assistant":
                        updates = updates[1:]
                    info["message_updates"] = updates
                    if self.done and not bool(info.get("discard_sample", False)):
                        from tau2.data_model.simulation import TerminationReason

                        self.final_score, eval_info = self._finish("", TerminationReason.USER_STOP)
                        info.update(eval_info)
                        info["done"] = True
                    info["agent_messages"] = self._agent_openai_messages()
                    self.last_info = info
                    return self._result(observation, info)
                except Exception as exc:
                    observation = f"User simulator failed:\n{type(exc).__name__}: {exc}"
                    self.done = True
                    info["user_sim_error"] = observation
                    info["discard_sample"] = True
                    info["discard_reason"] = "user_sim_error"
                    info["done"] = True
                    info["agent_messages"] = self._agent_openai_messages()
                    self.last_info = info
                    return self._result(observation, info)
            observation = "Assistant messages require user_sim_enabled=true in tau2."
            info["agent_message_error"] = observation
            info["done"] = False
            info["agent_messages"] = self._agent_openai_messages()
            self.last_info = info
            return self._result(observation, info)

        try:
            result = self.env.use_tool(name, **arguments)
            info["message_updates"] = self._record_tool_call(
                name,
                arguments,
                result,
                error=False,
                call_id=call_id,
                assistant_content=assistant_content,
            )
            observation = f"Tool result for {name}:\n{_json_text(result)}"
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            info["message_updates"] = self._record_tool_call(
                name,
                arguments,
                result,
                error=True,
                call_id=call_id,
                assistant_content=assistant_content,
            )
            observation = f"Tool call failed for {name}:\n{_json_text(result)}"
            info["tool_error"] = result["error"]
        info["done"] = self.done
        info["agent_messages"] = self._agent_openai_messages()
        self.last_info = info
        return self._result(observation, info)

    def run_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        if not policy:
            raise ValueError("tau2 run_episode requires policy endpoint")
        reset = self.reset(payload)
        observation = str(reset.get("observation", ""))
        info = reset.get("info") if isinstance(reset.get("info"), dict) else {}
        tools = self._episode_tool_schemas(info)
        runtime = self.runtime
        action_cfg = runtime.get("action") if isinstance(runtime.get("action"), dict) else {}
        prompt = str(payload.get("prompt") or "")
        metadata: dict[str, Any] = {
            "actions": [],
            "policy_usage": [],
            "turn_count": 0,
        }
        _merge_user_model_usage(metadata, info)
        include_trace = bool(payload.get("include_trace", False))
        include_messages = bool(payload.get("include_messages", False)) or include_trace
        policy_messages = self._policy_messages(observation, info, prompt)
        if include_messages:
            metadata["messages"] = list(policy_messages)
        if include_trace:
            metadata["turns"] = []

        max_turns = int(payload.get("max_turns") or runtime.get("max_turns") or self.config.get("max_turns") or 20)
        sampling_params = payload.get("sampling_params") if isinstance(payload.get("sampling_params"), dict) else {}
        max_tokens = int(payload.get("max_response_tokens") or 512)
        timeout_s = float((payload.get("timeouts") or {}).get("policy_s") or (runtime.get("timeouts") or {}).get("policy_s") or 300)
        final_score = 0.0
        success = False
        status = "truncated"
        truncated_reason = "max_turns"
        last_step: dict[str, Any] = reset

        for turn in range(max_turns):
            turn_trace: dict[str, Any] = {"turn": turn} if include_trace else {}
            reply = call_policy_chat(
                policy=policy,
                messages=policy_messages,
                tools=tools,
                sampling_params=sampling_params,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            assistant_message = reply.message
            metadata["policy_usage"].append(reply.usage)
            if policy_context_limit_reached(reply):
                metadata["context_limit_hits"] = int(metadata.get("context_limit_hits", 0) or 0) + 1
                truncated_reason = "context_limit_after_observation"
                if include_trace:
                    turn_trace.update(
                        {
                            "assistant_message": assistant_message,
                            "finish_reason": reply.finish_reason,
                            "truncated_reason": truncated_reason,
                        }
                    )
                    metadata["turns"].append(turn_trace)
                break
            if finish_reason_is_length(reply):
                metadata["max_response_tokens_hits"] = int(metadata.get("max_response_tokens_hits", 0) or 0) + 1
            tool_actions = extract_tool_actions(assistant_message)
            if tool_actions:
                if bool(action_cfg.get("restrict_to_available", False)):
                    available_names = set(self._available_tool_names(info))
                    for item in tool_actions:
                        name = str(item.get("name") or "")
                        if available_names and name not in available_names:
                            metadata.setdefault("invalid_actions", []).append(name)
                action: Any = tool_actions[0] if len(tool_actions) == 1 else tool_actions
            else:
                action, _, _ = extract_tool_action(assistant_message)
                action = choose_tool_action(
                    action,
                    self._available_tool_names(info),
                    restrict_to_available=bool(action_cfg.get("restrict_to_available", False)),
                    metadata=metadata,
                )
            metadata["actions"].append(action)
            policy_messages.append(assistant_message)
            if tool_actions:
                step = self._execute_tool_actions(tool_actions, assistant_message, info)
            else:
                step = self.step({"action": action})
            last_step = step
            observation = str(step.get("observation", ""))
            info = step.get("info") if isinstance(step.get("info"), dict) else {}
            _merge_user_model_usage(metadata, info)
            final_score = float(step.get("score", 0.0) or 0.0)
            done = bool(step.get("done", False))
            success = bool(step.get("success", final_score >= 1.0))
            discard_reason = info.get("discard_reason") if isinstance(info, dict) else None
            mode = "tool_call" if tool_actions else "assistant_message"
            env_messages = environment_messages_from_step(
                mode=mode,
                action=action,
                assistant_message=assistant_message,
                observation=observation,
                info=info,
                done=done,
                env_text=observation,
            )
            policy_messages.extend(env_messages)
            if bool(info.get("discard_sample", False)):
                metadata["discard_sample"] = True
                metadata["discard_reason"] = discard_reason or "tau2_env_discard"
                status = "failed"
                truncated_reason = ""
                break
            if include_trace:
                turn_trace.update(
                    {
                        "assistant_message": assistant_message,
                        "action": action,
                        "finish_reason": reply.finish_reason,
                        "env_step": step,
                        "env_messages": env_messages,
                    }
                )
                metadata["turns"].append(turn_trace)
            if done:
                status = "completed"
                truncated_reason = ""
                break

        metadata["turn_count"] = len(metadata["actions"])
        if include_messages:
            metadata["messages"] = policy_messages
        if truncated_reason:
            metadata["truncated_reason"] = truncated_reason
        return {
            "status": status,
            "observation": observation,
            "score": final_score,
            "done": status == "completed",
            "success": success,
            "info": last_step.get("info") if isinstance(last_step.get("info"), dict) else {},
            "task_index": self.task_index,
            "num_tasks": len(self.tasks),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
            "metadata": metadata,
        }

    def _result(self, observation: str, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation": observation,
            "score": self.final_score,
            "done": self.done,
            "success": self.final_score >= 1.0,
            "info": info,
            "task_index": self.task_index,
            "num_tasks": len(self.tasks),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.user_sim_enabled and not self.done and self.messages:
            try:
                from tau2.data_model.simulation import TerminationReason

                self.final_score, eval_info = self._finish("", TerminationReason.MAX_STEPS)
                info = dict(self.last_info)
                info.update(eval_info)
                info["done"] = False
                self.last_info = info
            except Exception:
                logger.debug("Failed to evaluate unfinished tau2 user-sim trajectory", exc_info=True)
        return self._result("", self.last_info)

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"reset_count": self.reset_count, "step_count": self.step_count}

    def close(self) -> dict[str, Any]:
        self.env = None
        self.task = None
        self.messages = []
        self.initial_agent_openai_message_count = 0
        self.user_simulator = None
        self.user_state = None
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
