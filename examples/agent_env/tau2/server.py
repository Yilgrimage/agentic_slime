from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import inspect
import urllib.error
import urllib.request

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
        "user_sim_enabled": bool(_deep_get(raw, "tau2", "user_sim_enabled", False)),
        "user_model": _env_path(_deep_get(raw, "tau2", "user_model", "local-user-sim"), "TAU2_USER_MODEL") or "local-user-sim",
        "user_model_base_url": _env_path(_deep_get(raw, "tau2", "user_model_base_url", ""), "TAU2_USER_MODEL_BASE_URL"),
        "user_model_api_key": _env_path(_deep_get(raw, "tau2", "user_model_api_key", ""), "TAU2_USER_MODEL_API_KEY"),
        "user_model_api_key_path": _env_path(_deep_get(raw, "tau2", "user_model_api_key_path", ""), "TAU2_USER_MODEL_API_KEY_PATH"),
        "user_model_timeout_s": float(_deep_get(raw, "tau2", "user_model_timeout_s", 120.0)),
        "user_model_max_tokens": int(_deep_get(raw, "tau2", "user_model_max_tokens", 512)),
        "user_model_temperature": float(_deep_get(raw, "tau2", "user_model_temperature", 0.0)),
        "user_model_top_p": float(_deep_get(raw, "tau2", "user_model_top_p", 1.0)),
        "user_model_enable_thinking": bool(_deep_get(raw, "tau2", "user_model_enable_thinking", False)),
        "user_model_separate_reasoning": bool(_deep_get(raw, "tau2", "user_model_separate_reasoning", True)),
        "max_user_tool_rounds": int(_deep_get(raw, "tau2", "max_user_tool_rounds", 4)),
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
    """Best-effort fallback for APIs that do not return reasoning_content."""
    return _strip_model_artifacts(str(text or ""))


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
        path = Path(str(task_ref.get("path") or "")).expanduser()
        if not path.is_absolute():
            root = Path(str(task_ref.get("root") or self.config.get("data_dir") or ".")).expanduser()
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
        path = Path(db_path).expanduser()
        if not path.is_absolute():
            path = Path(str(base_dir or self.config.get("data_dir") or ".")).expanduser() / path
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

    def _policy_doc(self) -> str:
        if not self.env or not self.config.get("include_policy", True):
            return ""
        try:
            policy = str(self.env.get_policy()).strip()
        except Exception:
            return ""
        try:
            from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

            return SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION).strip()
        except Exception:
            return (
                "You are a customer service agent that helps the user according to the <policy> provided below.\n"
                "In each turn you can either send a message to the user or make a tool call. "
                "You cannot do both at the same time.\n\n"
                f"<policy>\n{policy}\n</policy>"
            ).strip()

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

    def _api_key(self) -> str:
        explicit = str(self.config.get("user_model_api_key") or "").strip()
        if explicit:
            return explicit
        path = str(self.config.get("user_model_api_key_path") or "").strip()
        if path:
            try:
                return Path(path).expanduser().read_text(encoding="utf-8").strip()
            except Exception:
                logger.warning("Failed to read tau2 user model api key path: %s", path, exc_info=True)
        return os.environ.get("TAU2_USER_MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"

    def _chat_completions_url(self) -> str:
        base = str(self.config.get("user_model_base_url") or os.environ.get("TAU2_USER_MODEL_BASE_URL") or "").strip()
        if not base:
            raise ValueError("tau2.user_model_base_url or TAU2_USER_MODEL_BASE_URL is required when user_sim_enabled=true")
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

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
                "arguments": getattr(tool_call, "arguments", {}) or {},
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

    def _user_messages_for_llm(self) -> list[dict[str, Any]]:
        from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
        from tau2.user.user_simulator import SYSTEM_PROMPT, get_global_user_sim_guidelines

        instructions = str(getattr(self.task, "user_scenario", "") or "")
        guidelines = get_global_user_sim_guidelines(use_tools=bool(self._user_tool_schemas()))
        system_prompt = SYSTEM_PROMPT.format(
            global_user_sim_guidelines_with_persona=guidelines.replace("<PERSONA_GUIDELINES>", ""),
            instructions=instructions,
        ).strip()
        messages: list[Any] = [SystemMessage(role="system", content=system_prompt)]
        for message in self.messages:
            if isinstance(message, UserMessage):
                messages.append(AssistantMessage(role="assistant", content=message.content, tool_calls=message.tool_calls))
            elif isinstance(message, AssistantMessage):
                if not message.tool_calls:
                    messages.append(UserMessage(role="user", content=message.content))
            elif isinstance(message, ToolMessage) and message.requestor == "user":
                messages.append(ToolMessage(id=message.id, role="tool", content=message.content, requestor="user", error=message.error))
        return [item for message in messages if (item := self._tau2_message_to_openai(message)) is not None]

    def _call_user_model(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": str(self.config.get("user_model") or "local-user-sim"),
            "messages": messages,
            "max_tokens": int(self.config.get("user_model_max_tokens", 512)),
            "temperature": float(self.config.get("user_model_temperature", 0.0)),
            "top_p": float(self.config.get("user_model_top_p", 1.0)),
            "separate_reasoning": bool(self.config.get("user_model_separate_reasoning", True)),
            "chat_template_kwargs": {
                "enable_thinking": bool(self.config.get("user_model_enable_thinking", False)),
            },
        }
        if tools:
            payload["tools"] = tools
        data = json.dumps(_strip_none(payload), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._chat_completions_url(),
            data=data,
            headers={"content-type": "application/json", "authorization": f"Bearer {self._api_key()}"},
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

    def _generate_user_reply(self, assistant_text: str) -> tuple[str, bool, dict[str, Any]]:
        from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
        from tau2.user.user_simulator import UserSimulator

        self.messages.append(AssistantMessage.text(_visible_model_text(assistant_text)))
        tools = self._user_tool_schemas()
        latencies: list[float] = []
        for _ in range(max(1, int(self.config.get("max_user_tool_rounds", 4)))):
            raw_message = self._call_user_model(self._user_messages_for_llm(), tools)
            if raw_message.get("_latency_s") is not None:
                latencies.append(float(raw_message["_latency_s"]))
            content = str(raw_message.get("content") or "")
            if raw_message.get("reasoning_content") and bool(self.config.get("user_model_separate_reasoning", True)):
                content = _strip_model_artifacts(content)
            else:
                content = _visible_model_text(content)
            raw_tool_calls = raw_message.get("tool_calls") or []
            if raw_tool_calls:
                tool_calls = [self._openai_tool_call_to_tau2(raw, "user") for raw in raw_tool_calls if isinstance(raw, dict)]
                user_message = UserMessage(role="user", content=content, tool_calls=tool_calls)
                self.messages.append(user_message)
                for tool_call in tool_calls:
                    tool_result = self.env.get_response(tool_call)
                    self.messages.append(tool_result)
                continue
            user_message = UserMessage.text(content)
            self.messages.append(user_message)
            done = UserSimulator.is_stop(user_message)
            return content, done, {"user_model_calls": len(latencies), "user_model_latency_s": sum(latencies)}
        fallback = "I am unable to continue. ###STOP###"
        self.messages.append(UserMessage.text(fallback))
        return fallback, True, {"user_model_calls": len(latencies), "user_model_latency_s": sum(latencies), "user_tool_round_limit": True}

    def _prime_user_simulator(self, info: dict[str, Any]) -> str:
        if not self.user_sim_enabled or self.messages:
            return self._initial_observation()
        from tau2.data_model.simulation import TerminationReason
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

        user_message, user_done, user_info = self._generate_user_reply(str(DEFAULT_FIRST_AGENT_MESSAGE.content or ""))
        self.done = bool(user_done)
        info.update(user_info)
        info["done"] = self.done
        if self.done:
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
        else:
            self.tasks = self._tasks_for(self.task_set, None if self.split == "all" else self.split)
            self.task_index = int(payload.get("task_index") or 0) % max(1, len(self.tasks))
            self.task = self.tasks[self.task_index]
        self.env = self._build_env_for_task(self.domain, task_context)
        self._apply_initial_state()
        self.messages = self._initial_message_history()
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
        }
        observation = self._prime_user_simulator(self.last_info)
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

    def _record_tool_call(self, name: str, arguments: dict[str, Any], result: Any, error: bool = False) -> list[dict[str, Any]]:
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

        call_id = f"call-{uuid.uuid4().hex[:12]}"
        delta = [
            AssistantMessage.text("", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments, requestor="assistant")]),
            ToolMessage(id=call_id, role="tool", content=_json_text(result), requestor="assistant", error=error),
        ]
        self.messages.extend(delta)
        return self._agent_openai_messages(delta)

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
        name, arguments, structured = _tool_action(payload.get("action"))
        self.step_count += 1
        info = dict(self.last_info)
        for transient_key in ("tool_error",):
            info.pop(transient_key, None)
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
                    info["message_updates"] = self._agent_openai_messages(self.messages[before:])
                    if self.done:
                        from tau2.data_model.simulation import TerminationReason

                        self.final_score, eval_info = self._finish("", TerminationReason.USER_STOP)
                        info.update(eval_info)
                        info["done"] = True
                    info["agent_messages"] = self._agent_openai_messages()
                    self.last_info = info
                    return self._result(observation, info)
                except Exception as exc:
                    observation = f"User simulator failed:\n{type(exc).__name__}: {exc}"
                    info["user_sim_error"] = observation
                    info["done"] = False
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
            info["message_updates"] = self._record_tool_call(name, arguments, result, error=False)
            observation = f"Tool result for {name}:\n{_json_text(result)}"
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            info["message_updates"] = self._record_tool_call(name, arguments, result, error=True)
            observation = f"Tool call failed for {name}:\n{_json_text(result)}"
            info["tool_error"] = result["error"]
        info["done"] = self.done
        info["agent_messages"] = self._agent_openai_messages()
        self.last_info = info
        return self._result(observation, info)

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
