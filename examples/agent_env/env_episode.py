from __future__ import annotations

import copy
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyReply:
    message: dict[str, Any]
    finish_reason: str
    usage: dict[str, Any]
    raw: dict[str, Any]
    latency_s: float


class PolicyCallError(RuntimeError):
    # A policy transport failure aborts the episode, but it does not corrupt
    # the process-isolated environment. The pool can safely reuse that worker;
    # the next run_episode call resets the backend before use.
    recoverable_worker = True

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _chat_completions_url(policy: dict[str, Any]) -> str:
    base_url = str(policy.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("episode policy.base_url is required")
    path = str(policy.get("chat_completions_path") or "/v1/chat/completions").strip()
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base_url}{path}"


def _policy_headers(policy: dict[str, Any]) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    api_key = str(policy.get("api_key") or "").strip()
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    for key, value in (policy.get("headers") or {}).items():
        if value is not None:
            headers[str(key)] = str(value)
    return headers


def _sampling_body(sampling_params: dict[str, Any], max_tokens: int | None) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if max_tokens is not None:
        body["max_tokens"] = int(max_tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        body["max_tokens"] = int(sampling_params["max_new_tokens"])
    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("frequency_penalty", "frequency_penalty"),
        ("presence_penalty", "presence_penalty"),
        ("stop", "stop"),
    ):
        if sampling_params.get(src) is not None:
            body[dst] = sampling_params[src]
    return body


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    item: dict[str, Any] = {
        "role": str(message.get("role") or "assistant"),
        "content": content,
    }
    if message.get("tool_calls"):
        calls = []
        for raw_call in message.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            fn = raw_call.get("function") or {}
            calls.append(
                {
                    "id": str(raw_call.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(fn.get("name") or raw_call.get("name") or ""),
                        "arguments": _json_object(fn.get("arguments")),
                    },
                }
            )
        if calls:
            item["tool_calls"] = calls
    if message.get("tool_call_id"):
        item["tool_call_id"] = str(message["tool_call_id"])
    if message.get("step_loss_mask") is not None:
        item["step_loss_mask"] = message["step_loss_mask"]
    return item


def _wire_message(message: dict[str, Any]) -> dict[str, Any]:
    item = _normalize_message(message)
    for raw_call in item.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            continue
        fn = raw_call.get("function")
        if not isinstance(fn, dict):
            continue
        arguments = fn.get("arguments")
        if not isinstance(arguments, str):
            fn["arguments"] = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False, sort_keys=True)
    return item


def valid_message_updates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        output.append(_normalize_message(item))
    return output


def mark_assistant_messages_untrained(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [_normalize_message(item) for item in messages]
    for item in output:
        if item.get("role") == "assistant":
            item["step_loss_mask"] = 0
    return output


def call_policy_chat(
    *,
    policy: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    sampling_params: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    timeout_s: float = 120.0,
) -> PolicyReply:
    sampling_params = sampling_params or {}
    payload = {
        "model": str(policy.get("model") or "agent-env-policy"),
        "messages": [_wire_message(item) for item in messages],
        **_sampling_body(sampling_params, max_tokens),
    }
    if tools:
        payload["tools"] = copy.deepcopy(tools)
    if policy.get("parallel_tool_calls") is not None:
        payload["parallel_tool_calls"] = bool(policy.get("parallel_tool_calls"))
    # Keep message/tool-call payloads byte-for-byte semantic. Tool arguments may
    # legitimately contain JSON nulls; recursively stripping None mutates the
    # policy history and breaks the gateway's append-only ledger check.
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        _chat_completions_url(policy),
        data=data,
        headers=_policy_headers(policy),
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout_s) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise PolicyCallError(f"policy HTTP {exc.code}: {body[:1000]}", status=exc.code, body=body) from exc
    except Exception as exc:
        raise PolicyCallError(f"{type(exc).__name__}: {exc}") from exc
    result = json.loads(raw_body)
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if "content" not in message and choice.get("text") is not None:
        message = {"role": "assistant", "content": str(choice.get("text") or "")}
    return PolicyReply(
        message=_normalize_message(message),
        finish_reason=str(choice.get("finish_reason") or "stop"),
        usage=result.get("usage") or {},
        raw=result,
        latency_s=time.perf_counter() - start,
    )


def strip_outer_code_fences(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return value


def parse_text_action(response_text: str, tag: str = "action") -> tuple[str, bool, str]:
    text = strip_outer_code_fences(response_text)
    escaped_tag = re.escape(tag)
    action_match = re.search(rf"<{escaped_tag}>\s*(.*?)\s*</{escaped_tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if action_match:
        action = action_match.group(1).strip().strip(chr(34)).strip(chr(39))
        if action:
            return action, True, f"{tag}_tag"
        return "", False, f"empty_{tag}_tag"

    unterminated = re.search(rf"<{escaped_tag}>\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if unterminated:
        lines = unterminated.group(1).strip().splitlines()
        if not lines:
            return "", False, f"empty_unterminated_{tag}_tag"
        action = lines[0].strip().strip(chr(34)).strip(chr(39))
        if action:
            return action, False, f"unterminated_{tag}_tag"
        return "", False, f"empty_unterminated_{tag}_tag"

    text = text.strip()
    for line in text.splitlines() or [text]:
        line = line.strip().strip(chr(34)).strip(chr(39))
        if ":" in line and line.split(":", 1)[0].strip().lower() in {"action", "act"}:
            line = line.split(":", 1)[1].strip()
        line = line.lstrip("-*0123456789. ").strip()
        if line:
            return line, False, "legacy"
    # Keep text-action envs executable, but preserve the invalid parse signal.
    return "look", False, "fallback"


def _tool_action_from_call(message: dict[str, Any], raw_call: dict[str, Any]) -> dict[str, Any] | None:
    fn = raw_call.get("function") or {}
    name = str(fn.get("name") or raw_call.get("name") or "").strip()
    if not name:
        return None
    return {
        "type": "tool_call",
        "name": name,
        "arguments": _json_object(fn.get("arguments")),
        "tool_call_id": str(raw_call.get("id") or ""),
        "content": str(message.get("content") or ""),
    }


def extract_tool_actions(message: dict[str, Any]) -> list[dict[str, Any]]:
    actions = []
    for raw_call in message.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            continue
        action = _tool_action_from_call(message, raw_call)
        if action is not None:
            actions.append(action)
    return actions


def extract_tool_action(message: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    calls = message.get("tool_calls") or []
    if calls:
        actions = extract_tool_actions(message)
        if actions:
            return actions[0], True, "tool_call"
        return {"type": "assistant_message", "content": str(message.get("content") or "")}, False, "empty_tool_name"
    return {"type": "assistant_message", "content": str(message.get("content") or "")}, False, "assistant_message"


def choose_text_action(
    action: str,
    available_actions: list[str],
    *,
    restrict_to_available: bool,
    invalid_fallback: str,
    metadata: dict[str, Any],
) -> str:
    if not restrict_to_available or not available_actions:
        return action
    by_lower = {item.lower(): item for item in available_actions}
    if action.lower() in by_lower:
        return by_lower[action.lower()]
    metadata.setdefault("invalid_actions", []).append(action)
    if invalid_fallback in {"first_available", "first_admissible"}:
        return available_actions[0]
    if invalid_fallback == "look" and "look" in by_lower:
        return by_lower["look"]
    return action


def choose_tool_action(
    action: dict[str, Any],
    available_tool_names: list[str],
    *,
    restrict_to_available: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if action.get("type") != "tool_call" or not restrict_to_available or not available_tool_names:
        return action
    name = str(action.get("name") or "")
    if name in set(available_tool_names):
        return action
    metadata.setdefault("invalid_actions", []).append(name)
    return {"type": "assistant_message", "content": f"Unable to use unavailable tool: {name}"}


def tool_result_message(assistant_message: dict[str, Any], observation: str) -> dict[str, Any]:
    calls = assistant_message.get("tool_calls") or []
    call_id = calls[0].get("id") if calls and isinstance(calls[0], dict) else ""
    return {"role": "tool", "tool_call_id": str(call_id), "content": str(observation)}


def environment_messages_from_step(
    *,
    mode: str,
    action: Any,
    assistant_message: dict[str, Any],
    observation: str,
    info: dict[str, Any],
    done: bool,
    env_text: str,
) -> list[dict[str, Any]]:
    updates = valid_message_updates(info.get("message_updates"))
    if updates:
        if updates[0].get("role") == "assistant":
            raise ValueError("env message_updates must contain only environment-side messages")
        return updates
    if mode == "tool_call" and isinstance(action, dict) and action.get("type") == "tool_call":
        return [tool_result_message(assistant_message, observation)]
    if not done:
        return [{"role": "user", "content": env_text}]
    return []


def _policy_timeout(payload: dict[str, Any], default: float = 120.0) -> float:
    timeouts = payload.get("timeouts") if isinstance(payload.get("timeouts"), dict) else {}
    try:
        return float(timeouts.get("policy_s", default))
    except (TypeError, ValueError):
        return default


def _max_response_tokens(payload: dict[str, Any], default: int) -> int:
    try:
        return int(payload.get("max_response_tokens") or default)
    except (TypeError, ValueError):
        return default


def finish_reason_is_length(reply: PolicyReply) -> bool:
    return reply.finish_reason == "length"


def policy_context_limit_reached(reply: PolicyReply) -> bool:
    return bool((reply.usage or {}).get("agent_env_context_limit_reached", False))
