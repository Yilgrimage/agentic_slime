from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import uuid
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import anyio
import yaml
try:
    from mcp import ClientSession, StdioServerParameters, stdio_client
    from mcp.client.sse import sse_client
    from mcp.types import JSONRPCNotification
except ImportError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    sse_client = None
    JSONRPCNotification = None

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
        raise ValueError("Missing env_server.pool_size in mcp_server env_config.yaml")
    return {
        "pool_size": int(pool_size),
        "acquire_timeout_s": float(_deep_get(raw, "env_server", "acquire_timeout_s", 600.0)),
        "lease_ttl_s": float(_deep_get(raw, "env_server", "lease_ttl_s", 1800.0)),
        "idempotency_ttl_s": float(_deep_get(raw, "env_server", "idempotency_ttl_s", 300.0)),
        "worker_start_timeout_s": float(_deep_get(raw, "env_server", "worker_start_timeout_s", 300.0)),
        "worker_request_timeout_s": float(_deep_get(raw, "env_server", "worker_request_timeout_s", 300.0)),
        "prewarm_splits": list(_deep_get(raw, "env_server", "prewarm_splits", ["train"])),
        "reuse_workers": bool(_deep_get(raw, "env_server", "reuse_workers", True)),
        "reset_on_release": bool(_deep_get(raw, "env_server", "reset_on_release", False)),
        "shared_pool": bool(_deep_get(raw, "env_server", "shared_pool", True)),
    }


def _expand_text(value: Any) -> str:
    return os.path.expandvars(str(value or "")).strip()


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            lines = list(f)
        for line in lines:
            line = line.strip()
            if line:
                value = json.loads(line, strict=False)
                if isinstance(value, dict):
                    rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("tasks"), list):
        return [item for item in value["tasks"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported task file format: {path}")


def _load_mcp_server_config(path: Path, server_name: str = "") -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    servers = value.get("mcpServers") if isinstance(value, dict) else None
    if not isinstance(servers, dict) or not servers:
        raise ValueError(f"{path} must contain a non-empty mcpServers object")
    name = server_name or next(iter(servers))
    server = servers.get(name)
    if not isinstance(server, dict):
        raise ValueError(f"MCP server {name!r} not found in {path}")
    loaded = dict(server)
    loaded.setdefault("name", name)
    loaded["transport"] = str(loaded.pop("type", loaded.get("transport", "stdio"))).strip().lower()
    return loaded


def _resolve_server_config(cfg: dict[str, Any]) -> dict[str, Any]:
    server = dict(cfg.get("server") or {})
    config_path = _expand_text(server.pop("config_path", "") or cfg.get("mcp_servers_config_path", ""))
    if config_path:
        loaded = _load_mcp_server_config(Path(config_path).expanduser(), str(server.pop("server_name", "") or cfg.get("server_name", "")))
        loaded.update({key: value for key, value in server.items() if value is not None})
        server = loaded
    return server


def _environment_config(raw: dict) -> dict:
    cfg = dict(raw.get("mcp_server") or {})
    tasks = cfg.get("tasks") or raw.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError("mcp_server.tasks must be a list")
    task_file = _expand_text(cfg.get("task_file") or raw.get("task_file"))
    if task_file:
        loaded = _load_json_or_jsonl(Path(task_file).expanduser())
        tasks = [*tasks, *loaded]
    if not tasks:
        tasks = [
            {
                "id": "default",
                "prompt": "Use available MCP tools to solve the task, then call finish.",
            }
        ]
    server = _resolve_server_config(cfg)
    if not server:
        raise ValueError("mcp_server.server is required")
    policy = str(cfg.get("policy") or "")
    policy_file = _expand_text(cfg.get("policy_file") or "")
    if policy_file:
        policy = Path(policy_file).expanduser().read_text(encoding="utf-8")
    terminal_tools = cfg["terminal_tools"] if "terminal_tools" in cfg else ["finish", "submit", "final_answer"]
    return {
        "name": str(cfg.get("name") or "mcp_server"),
        "policy": policy,
        "server": server,
        "tasks": tasks,
        "tool_allowlist": [str(item) for item in (cfg.get("tool_allowlist") or [])],
        "connect_timeout_s": float(cfg.get("connect_timeout_s", 30.0)),
        "request_timeout_s": float(cfg.get("request_timeout_s", 120.0)),
        "terminal_tools": [str(item) for item in (terminal_tools or [])],
        "allow_assistant_final": bool(cfg.get("allow_assistant_final", True)),
        "assistant_final_success_without_target": bool(cfg.get("assistant_final_success_without_target", True)),
        "max_turns": int(_deep_get(raw, "mcp_server", "max_turns", raw.get("max_turns", 20))),
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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(by_alias=True)
    return {}


def _tool_schema(tool: Any) -> dict[str, Any]:
    raw = _as_dict(tool)
    name = str(raw.get("name") or getattr(tool, "name", ""))
    description = str(raw.get("description") or getattr(tool, "description", "") or name)
    input_schema = raw.get("inputSchema") or raw.get("input_schema") or getattr(tool, "inputSchema", None)
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema,
        },
    }


def _content_text(item: Any) -> str:
    raw = _as_dict(item)
    if raw.get("type") == "text" and raw.get("text") is not None:
        return str(raw["text"])
    text = getattr(item, "text", None)
    if text is not None:
        return str(text)
    if raw:
        return _json_text(raw)
    return str(item)


def _result_payload(result: Any) -> tuple[str, dict[str, Any], bool]:
    raw = _as_dict(result)
    structured = raw.get("structuredContent") or raw.get("structured_content")
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    structured_dict = structured if isinstance(structured, dict) else {}
    content = raw.get("content")
    if content is None:
        content = getattr(result, "content", None)
    parts = [_content_text(item) for item in (content or [])]
    if structured_dict:
        observation = str(structured_dict.get("observation") or structured_dict.get("message") or "").strip()
        if not observation:
            observation = _json_text(structured_dict)
    else:
        observation = "\n".join(part for part in parts if part).strip()
    is_error = bool(raw.get("isError") or raw.get("is_error") or getattr(result, "isError", False))
    return observation, structured_dict, is_error


def _is_keepalive_notification(message: Any) -> bool:
    try:
        root = message.message.root
    except Exception:
        return False
    if JSONRPCNotification is not None and isinstance(root, JSONRPCNotification):
        return root.method == "notifications/keepalive"
    return getattr(root, "method", None) == "notifications/keepalive"


@asynccontextmanager
async def _filtered_sse_client(
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = 5,
    sse_read_timeout: float = 60 * 5,
):
    if sse_client is None:
        raise RuntimeError("SSE MCP transport requires the `mcp` Python package")
    async with sse_client(
        url,
        headers=headers,
        timeout=timeout,
        sse_read_timeout=sse_read_timeout,
    ) as (read_stream, write_stream):
        filtered_send, filtered_recv = anyio.create_memory_object_stream(0)

        async def _forward() -> None:
            try:
                async with read_stream:
                    async for item in read_stream:
                        if _is_keepalive_notification(item):
                            continue
                        await filtered_send.send(item)
            finally:
                await filtered_send.aclose()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_forward)
            try:
                yield filtered_recv, write_stream
            finally:
                task_group.cancel_scope.cancel()


class MCPClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="mcp-client-loop", daemon=True)
        self.stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.psm_client: Any | None = None
        self.tools: list[Any] = []
        self.request_timeout_s = float(config.get("request_timeout_s", 120.0))

    def _future_result(self, coro: Any, timeout_s: float | None = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=timeout_s or self.request_timeout_s)

    def start(self) -> list[Any]:
        self.thread.start()
        self.tools = self._future_result(self._connect(), timeout_s=float(self.config.get("connect_timeout_s", 30.0)))
        return self.tools

    async def _connect(self) -> list[Any]:
        server = dict(self.config["server"])
        transport = str(server.get("transport") or "stdio").strip().lower()
        self.stack = AsyncExitStack()
        if transport == "stdio":
            if stdio_client is None or StdioServerParameters is None:
                raise RuntimeError("stdio MCP transport requires the `mcp` Python package")
            command = _expand_text(server.get("command")) or sys.executable
            if command in {"python", "python3"} or "$" in command:
                command = sys.executable
            args = [_expand_text(item) for item in (server.get("args") or [])]
            cwd_text = _expand_text(server.get("cwd") or "")
            cwd = Path(cwd_text).expanduser() if cwd_text else None
            raw_env = server.get("env") or {}
            if raw_env and not isinstance(raw_env, dict):
                raise ValueError("mcp_server.server.env must be a mapping")
            env = os.environ.copy()
            env.update({str(key): _expand_text(value) for key, value in raw_env.items()})
            params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
            read_stream, write_stream = await self.stack.enter_async_context(stdio_client(params))
        elif transport == "sse":
            url = _expand_text(server.get("url"))
            if not url:
                raise ValueError("mcp_server.server.url is required for SSE transport")
            headers = server.get("headers")
            if headers is not None and not isinstance(headers, dict):
                raise ValueError("mcp_server.server.headers must be a mapping")
            read_stream, write_stream = await self.stack.enter_async_context(
                _filtered_sse_client(
                    url,
                    headers={str(key): _expand_text(value) for key, value in (headers or {}).items()} or None,
                    timeout=float(server.get("connect_timeout_s") or self.config.get("connect_timeout_s", 30.0)),
                    sse_read_timeout=float(server.get("sse_read_timeout_s") or self.request_timeout_s),
                )
            )
        elif transport == "psm":
            raw_env = server.get("env") or {}
            if raw_env and not isinstance(raw_env, dict):
                raise ValueError("mcp_server.server.env must be a mapping")
            for key, value in raw_env.items():
                os.environ[str(key)] = _expand_text(value)

            raw_psms = server.get("psms") or server.get("psm_list") or server.get("psm")
            if isinstance(raw_psms, str):
                psm_list = [part.strip() for part in raw_psms.split(",") if part.strip()]
            elif isinstance(raw_psms, (list, tuple)):
                psm_list = [str(item).strip() for item in raw_psms if str(item).strip()]
            else:
                psm_list = []
            if not psm_list:
                raise ValueError("mcp_server.server.psm is required for PSM transport")

            region = _expand_text(server.get("mcp_gateway_region") or os.environ.get("MCP_GATEWAY_REGION") or "i18n")
            secret = _expand_text(server.get("secret_key") or "")
            secret_path = _expand_text(server.get("secret_key_path") or os.environ.get("MCP_SERVICE_ACCOUNT_SECRET_PATH") or "")
            if not secret and secret_path:
                secret = Path(secret_path).expanduser().read_text(encoding="utf-8").strip()

            from bytedance.mcp.mcp_client import byted_mcp_client_with_server_psm

            self.psm_client = await byted_mcp_client_with_server_psm(
                psm_list,
                transport=str(server.get("psm_transport") or server.get("transport_type") or "http"),
                mcp_gateway_region=region,
                connect_timeout=float(server.get("connect_timeout_s") or self.config.get("connect_timeout_s", 30.0)),
                request_timeout=float(server.get("request_timeout_s") or self.request_timeout_s),
            )
            if secret:
                jwt_server_name = str(server.get("jwt_server_name") or "*")
                await self.psm_client.with_jwt_token_setter(secret, region, server_name=jwt_server_name)
            await self.psm_client.connect_to_servers()
            return list(self.psm_client.list_tools())
        else:
            raise ValueError(f"Unsupported MCP transport={transport!r}; expected stdio, sse, or psm")
        if ClientSession is None:
            raise RuntimeError("MCP ClientSession requires the `mcp` Python package")
        timeout = timedelta(seconds=self.request_timeout_s)
        self.session = await self.stack.enter_async_context(ClientSession(read_stream, write_stream, read_timeout_seconds=timeout))
        await self.session.initialize()
        tools = []
        cursor = None
        while True:
            result = await self.session.list_tools(cursor=cursor)
            tools.extend(list(getattr(result, "tools", []) or []))
            cursor = getattr(result, "nextCursor", None) or getattr(result, "next_cursor", None)
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.psm_client is not None:
            return self._future_result(self.psm_client.call_tool(name, arguments))
        if self.session is None:
            raise RuntimeError("MCP session is not started")
        timeout = timedelta(seconds=self.request_timeout_s)
        return self._future_result(self.session.call_tool(name, arguments, read_timeout_seconds=timeout))

    def close(self) -> None:
        if self.loop.is_closed():
            return
        if self.stack is not None:
            try:
                self._future_result(self.stack.aclose(), timeout_s=10.0)
            except Exception:
                logger.debug("Failed to close MCP client stack", exc_info=True)
        if self.psm_client is not None:
            try:
                self._future_result(self.psm_client.__aexit__(None, None, None), timeout_s=10.0)
            except Exception:
                logger.debug("Failed to close PSM MCP client", exc_info=True)
            self.psm_client = None
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        self.loop.close()


class MCPServerBackend:
    def __init__(self, worker_id: str, split: str, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.split = split
        self.config = config
        self.client: MCPClient | None = None
        self.tools: list[Any] = []
        self.tool_schemas: list[dict[str, Any]] = []
        self.tasks = list(config.get("tasks") or [])
        self.task: dict[str, Any] = {}
        self.task_index = 0
        self.reset_count = 0
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.success = False
        self.last_info: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        self.client = MCPClient(self.config)
        self.tools = self.client.start()
        tool_schemas = [_tool_schema(tool) for tool in self.tools]
        allowlist = set(self.config.get("tool_allowlist") or [])
        if allowlist:
            tool_schemas = [schema for schema in tool_schemas if schema["function"]["name"] in allowlist]
        self.tool_schemas = tool_schemas
        return {
            "num_tasks": len(self.tasks),
            "tools": [schema["function"]["name"] for schema in self.tool_schemas],
            "mcp_server": self.config["name"],
        }

    def _base_info(self) -> dict[str, Any]:
        return {
            "task_id": self.task.get("id") or self.task.get("task_id") or self.task_index,
            "split": self.split,
            "mcp_server": self.config["name"],
            "tools": [schema["function"]["name"] for schema in self.tool_schemas],
            "tool_schemas": self.tool_schemas,
            "policy": self.config.get("policy") or "",
            "success": self.success,
            "done": self.done,
        }

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.split = str(payload.get("split") or self.split)
        self.task_index = int(payload.get("task_index") or 0) % max(1, len(self.tasks))
        self.task = dict(self.tasks[self.task_index])
        self.reset_count += 1
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.success = False
        self.last_info = self._base_info()
        observation = str(
            self.task.get("prompt")
            or self.task.get("query")
            or self.task.get("task_question")
            or "Use MCP tools to solve the task."
        ).strip()
        return {
            "observation": observation,
            "info": self.last_info,
            "split": self.split,
            "task_index": self.task_index,
            "num_tasks": len(self.tasks),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def _finish_from_answer(self, answer: str, info: dict[str, Any]) -> str:
        target = str(self.task.get("target") or "").strip()
        self.done = True
        if target:
            self.success = answer.strip().lower() == target.lower()
            self.final_score = 1.0 if self.success else 0.0
        else:
            self.success = bool(self.config.get("assistant_final_success_without_target", True))
            self.final_score = 1.0 if self.success else 0.0
        info["final_answer"] = answer
        return "finished"

    def _apply_structured_result(self, tool_name: str, structured: dict[str, Any], info: dict[str, Any]) -> None:
        if "score" in structured:
            try:
                self.final_score = float(structured["score"])
            except (TypeError, ValueError):
                pass
        if "success" in structured:
            self.success = bool(structured["success"])
        if "done" in structured:
            self.done = bool(structured["done"])
        if tool_name in self.config.get("terminal_tools", []):
            self.done = True
            if "success" not in structured and self.final_score > 0:
                self.success = True
        if self.done and "success" not in structured:
            self.success = self.final_score > 0
        info["mcp_structured_result"] = structured

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        self.step_count += 1
        info = self._base_info()
        info["structured_action"] = isinstance(action, dict)
        if isinstance(action, dict) and action.get("type") == "assistant_message":
            answer = str(action.get("content") or "")
            if not bool(self.config.get("allow_assistant_final", True)):
                observation = "Assistant messages are disabled for this MCP env; use a tool call."
                info["assistant_message_error"] = observation
            else:
                observation = self._finish_from_answer(answer, info)
            self.last_info = {**info, "done": self.done, "success": self.success}
            return self._result(observation, self.last_info)

        if not isinstance(action, dict) or action.get("type") != "tool_call":
            observation = "Invalid action. Expected a tool_call action."
            info["format_error"] = observation
            self.last_info = info
            return self._result(observation, info)

        tool_name = str(action.get("name") or "")
        arguments = action.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        info["last_action"] = tool_name
        info["last_arguments"] = arguments
        if tool_name in {"__assistant_message__", "assistant_message"}:
            observation = self._finish_from_answer(str(arguments.get("content") or ""), info)
            self.last_info = {**info, "done": self.done, "success": self.success}
            return self._result(observation, self.last_info)

        try:
            if self.client is None:
                raise RuntimeError("MCP client is not started")
            result = self.client.call_tool(tool_name, arguments)
            observation, structured, is_error = _result_payload(result)
            if is_error:
                info["tool_error"] = True
            if structured:
                self._apply_structured_result(tool_name, structured, info)
            elif tool_name in self.config.get("terminal_tools", []):
                self.done = True
                self.success = self.final_score > 0
            if not observation:
                observation = _json_text(structured) if structured else "MCP tool returned no content."
        except Exception as exc:
            observation = f"MCP tool call failed for {tool_name}: {type(exc).__name__}: {exc}"
            info["tool_error"] = observation
        info["done"] = self.done
        info["success"] = self.success
        self.last_info = info
        return self._result(observation, info)

    def _result(self, observation: str, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation": observation,
            "score": self.final_score,
            "done": self.done,
            "success": self.success,
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
        if self.client is not None:
            self.client.close()
            self.client = None
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a process-isolated MCP environment server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18185)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    env_config, server_config = _load_config(args.config)
    serve_process_pool(
        host=args.host,
        port=args.port,
        backend_cls=MCPServerBackend,
        env_config=env_config,
        server_config=server_config,
        env_name="mcp_server",
    )


if __name__ == "__main__":
    main()
