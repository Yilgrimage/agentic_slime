from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "appworld.execution_evidence.v1"
TRACE_LABEL = "AppWorld execution evidence"

_SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:access_?token|auth(?:orization)?|client_?secret|password|secret|session_?token)(?:$|_)",
    flags=re.IGNORECASE,
)
_REQUEST_CONTROL_FIELDS = {
    "_app_name",
    "_api_name",
    "_system_datetime",
    "client",
    "raise_on_failure",
    "show",
    "track",
}
_STATE_CHANGING_API_PREFIXES = (
    "add_",
    "approve_",
    "cancel_",
    "create_",
    "delete_",
    "deny_",
    "disable_",
    "enable_",
    "login",
    "logout",
    "mark_",
    "move_",
    "remove_",
    "remind_",
    "reset_",
    "send_",
    "set_",
    "signup",
    "transfer_",
    "update_",
    "upload_",
    "verify_",
    "withdraw_",
    "write_",
)
_STATE_CHANGE_ARGUMENT_SAMPLE_LIMIT = 16


def api_arguments(call_args: tuple[Any, ...], call_kwargs: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Extract the public AppWorld API identity and evaluated arguments."""

    app_name = str(call_kwargs.get("_app_name") or (call_args[0] if call_args else "")).strip()
    api_name = str(call_kwargs.get("_api_name") or (call_args[1] if len(call_args) > 1 else "")).strip()
    arguments = {
        str(key): value
        for key, value in call_kwargs.items()
        if key not in _REQUEST_CONTROL_FIELDS
    }
    return app_name, api_name, arguments


def build_api_call_evidence(
    *,
    app_name: str,
    api_name: str,
    arguments: dict[str, Any],
    result: Any = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "api": f"{app_name}.{api_name}" if app_name and api_name else (api_name or app_name or "unknown"),
        "status": "raised" if error is not None else "returned",
    }
    normalized_arguments = normalize_evidence_value(arguments)
    if normalized_arguments not in ({}, [], "", None):
        evidence["arguments"] = normalized_arguments
    if error is not None:
        evidence["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    elif not _is_sensitive_call(evidence):
        normalized_result = normalize_evidence_value(result)
        if normalized_result not in ({}, [], "", None):
            evidence["result_summary"] = normalized_result
    return evidence


def build_execution_evidence(*, observation: str, api_calls: list[dict[str, Any]]) -> dict[str, Any]:
    output = str(observation or "").strip()
    failed = output.startswith("Execution failed.") or output.startswith("Tool execution error")
    if failed:
        output_kind = "error"
    elif not output or output == "Execution successful.":
        output_kind = "no_stdout"
    else:
        output_kind = "stdout"
    return {
        "schema_version": SCHEMA_VERSION,
        "code_execution": "error" if failed else "ok",
        "output_kind": output_kind,
        "api_calls": list(api_calls),
    }


def normalize_evidence_value(value: Any, *, _key: str = "", _seen: set[int] | None = None) -> Any:
    """Make captured evidence JSON-safe and redact secrets without truncating it."""

    if _key and _SECRET_KEY_RE.search(_key):
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return "[CYCLE]"
    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            return {
                str(key): normalize_evidence_value(item, _key=str(key), _seen=seen)
                for key, item in value.items()
            }
        finally:
            seen.remove(value_id)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seen.add(value_id)
        try:
            return [normalize_evidence_value(item, _seen=seen) for item in value]
        finally:
            seen.remove(value_id)
    if hasattr(value, "model_dump"):
        try:
            return normalize_evidence_value(value.model_dump(), _key=_key, _seen=seen)
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return normalize_evidence_value(value.dict(), _key=_key, _seen=seen)
        except Exception:
            pass
    return str(value)


def render_execution_evidence(
    evidence: Any,
    *,
    max_chars: int | None = None,
    observation: Any = "",
) -> str:
    if not isinstance(evidence, dict) or evidence.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("AppWorld reward evidence is missing or has an unsupported schema_version")
    calls = evidence.get("api_calls")
    if not isinstance(calls, list):
        raise ValueError("AppWorld reward evidence api_calls must be a list")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "code_execution": str(evidence.get("code_execution") or "unknown"),
        "output_kind": str(evidence.get("output_kind") or "unknown"),
        "api_calls": _aggregate_calls([_sanitize_call_for_render(call) for call in calls]),
    }
    output_summary = _safe_output_summary(observation, payload["api_calls"])
    if output_summary:
        payload["output_summary"] = output_summary
    rendered = _json(payload)
    if max_chars is None or max_chars <= 0 or len(rendered) <= max_chars:
        return f"{TRACE_LABEL}:\n{rendered}"

    structured_calls = [_structured_call_summary(item) for item in payload["api_calls"]]
    structured_payload = {
        **payload,
        "api_calls": structured_calls,
        "details_compacted": "structured",
    }
    if output_summary:
        structured_payload["output_summary"] = _compact_string(output_summary, 160)
    rendered = _json(structured_payload)
    if len(rendered) <= max_chars:
        return f"{TRACE_LABEL}:\n{rendered}"

    # Preserve every API identity and terminal submission before optional details.
    reduced_calls = []
    for item in payload["api_calls"]:
        reduced = {
            key: item[key]
            for key in ("api", "status", "count", "distinct_argument_count")
            if key in item
        }
        if "count" in item:
            for key in ("arguments", "argument_samples"):
                if key in item:
                    reduced[key] = _compact_call_detail(item, key)
        if _is_terminal_call(item) or _is_state_changing_call(item):
            for key in (
                "arguments",
                "argument_samples",
                "result_summary",
                "result_samples",
                "error",
            ):
                if key in item:
                    reduced[key] = _compact_call_detail(item, key)
        reduced_calls.append(reduced)
    reduced_payload = {**payload, "api_calls": reduced_calls, "details_compacted": "identity"}
    if output_summary:
        reduced_payload["output_summary"] = _compact_string(output_summary, 160)
    rendered = _json(reduced_payload)
    if len(rendered) <= max_chars:
        return f"{TRACE_LABEL}:\n{rendered}"

    terminal_calls = [item for item in reduced_calls if _is_terminal_call(item)]
    state_changing_calls = [
        item
        for item in reduced_calls
        if _is_state_changing_call(item) and not _is_terminal_call(item)
    ]
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "code_execution": payload["code_execution"],
        "output_kind": payload["output_kind"],
        "api_sequence": [
            {
                key: item[key]
                for key in (
                    "api",
                    "status",
                    "count",
                    "distinct_argument_count",
                    "arguments",
                    "argument_samples",
                )
                if key in item
            }
            for item in payload["api_calls"]
        ],
        "terminal_calls": terminal_calls,
        "state_changing_calls": state_changing_calls,
        "details_compacted": "identity",
        "configured_budget_exceeded": True,
    }
    if output_summary:
        identity_payload["output_summary"] = _compact_string(output_summary, 160)
    # Never character-truncate API names or submitted answers to satisfy a bad budget.
    return f"{TRACE_LABEL}:\n{_json(identity_payload)}"


def has_execution_evidence(text: Any) -> bool:
    return TRACE_LABEL in str(text or "")


def parse_execution_evidence_trace(text: Any) -> tuple[dict[str, Any], ...]:
    prefix = TRACE_LABEL + ":\n"
    chunks = str(text or "").split(prefix)[1:]
    if not chunks:
        raise ValueError("AppWorld reward trace contains no structured execution evidence")
    blocks: list[dict[str, Any]] = []
    for chunk in chunks:
        line = chunk.splitlines()[0].strip() if chunk.splitlines() else ""
        if not line:
            raise ValueError("AppWorld execution evidence block is empty")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("AppWorld execution evidence block is not valid JSON") from exc
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("AppWorld execution evidence block has an unsupported schema_version")
        calls = value.get("api_calls", value.get("api_sequence"))
        if not isinstance(calls, list):
            raise ValueError("AppWorld execution evidence block lacks an API call sequence")
        blocks.append(value)
    return tuple(blocks)


def _aggregate_calls(calls: list[Any]) -> list[dict[str, Any]]:
    normalized = [dict(call) for call in calls if isinstance(call, dict)]
    groups: list[list[dict[str, Any]]] = []
    for call in normalized:
        if groups and _call_group_key(groups[-1][-1]) == _call_group_key(call):
            groups[-1].append(call)
        else:
            groups.append([call])
    return [_aggregate_call_group(group) for group in groups]


def _structured_call_summary(call: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: call[key]
        for key in ("api", "status", "count", "distinct_argument_count")
        if key in call
    }
    for key in ("arguments", "argument_samples", "result_summary", "result_samples", "error"):
        if key in call:
            summary[key] = _compact_call_detail(call, key)
    if _is_terminal_call(call) and "arguments" in call:
        summary["arguments"] = call["arguments"]
    return summary


def _compact_call_detail(call: dict[str, Any], key: str) -> Any:
    value = call[key]
    if key != "argument_samples" or not _is_state_changing_call(call):
        return _compact_render_value(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return _compact_render_value(value)
    return [_compact_render_value(item, _depth=1) for item in value]


def _compact_render_value(value: Any, *, _depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _compact_string(value, 96)
    if _depth >= 2:
        return _shape_summary(value)
    if isinstance(value, Mapping):
        items = list(value.items())
        selected = items if len(items) <= 10 else [*items[:5], *items[-5:]]
        compact = {
            str(key): _compact_render_value(item, _depth=_depth + 1)
            for key, item in selected
        }
        if len(selected) < len(items):
            compact["__omitted_fields__"] = len(items) - len(selected)
        return compact
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) <= 2:
            return [_compact_render_value(item, _depth=_depth + 1) for item in items]
        return {
            "type": "array",
            "item_count": len(items),
            "samples": [
                _compact_render_value(items[0], _depth=_depth + 1),
                _compact_render_value(items[-1], _depth=_depth + 1),
            ],
        }
    return _compact_string(str(value), 96)


def _aggregate_call_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    first = group[0]
    aggregated = {
        "api": str(first.get("api") or "unknown"),
        "status": str(first.get("status") or "unknown"),
    }
    if len(group) > 1:
        aggregated["count"] = len(group)
    for field, plural in (("arguments", "argument_samples"), ("result_summary", "result_samples")):
        values = [item.get(field) for item in group if item.get(field) not in (None, "", {}, [])]
        if not values:
            continue
        sample_limit = (
            _STATE_CHANGE_ARGUMENT_SAMPLE_LIMIT
            if field == "arguments" and _is_state_changing_call(aggregated)
            else 4
        )
        if all(value == values[0] for value in values):
            aggregated[field] = values[0]
        elif len(values) <= sample_limit:
            aggregated[plural] = values
        else:
            edge_count = sample_limit // 2
            aggregated[plural] = [
                *values[:edge_count],
                {"__omitted_calls__": len(values) - sample_limit},
                *values[-edge_count:],
            ]
        if field == "arguments":
            aggregated["distinct_argument_count"] = len({_json(value) for value in values})
    errors = [item.get("error") for item in group if item.get("error")]
    if errors:
        aggregated["error"] = errors[0] if len(errors) == 1 else errors
    return aggregated


def _call_group_key(call: dict[str, Any]) -> tuple[str, str]:
    return str(call.get("api") or ""), str(call.get("status") or "")


def _is_terminal_call(call: dict[str, Any]) -> bool:
    return str(call.get("api") or "").endswith(".complete_task")


def _is_state_changing_call(call: dict[str, Any]) -> bool:
    api_name = str(call.get("api") or "").rsplit(".", 1)[-1].strip().lower()
    return api_name.startswith(_STATE_CHANGING_API_PREFIXES)


def _safe_output_summary(observation: Any, calls: list[dict[str, Any]]) -> str:
    text = str(observation or "").strip()
    if not text or text == "Execution successful.":
        return ""
    if any(_is_sensitive_call(call) for call in calls):
        return ""
    return text


def _is_sensitive_call(call: dict[str, Any]) -> bool:
    api = str(call.get("api") or "").strip().lower()
    name = api.rsplit(".", 1)[-1]
    return name == "login" or any(token in name for token in ("password", "credential", "authenticate"))


def _sanitize_call_for_render(call: Any) -> Any:
    if not isinstance(call, dict):
        return call
    sanitized = dict(call)
    if _is_sensitive_call(call):
        sanitized.pop("result_summary", None)
        sanitized.pop("result_samples", None)
        return sanitized
    for key in ("result_summary", "result_samples"):
        if key not in sanitized:
            continue
        value = _sanitize_result_for_render(sanitized[key])
        if value in (None, "", {}, []):
            sanitized.pop(key)
        else:
            sanitized[key] = value
    return sanitized


def _sanitize_result_for_render(value: Any) -> Any:
    """Remove AppWorld's generic execution acknowledgement from reward evidence."""

    if isinstance(value, str):
        return None if value.strip() == "Execution successful." else value
    if isinstance(value, Mapping):
        sanitized = {
            str(key): item
            for key, raw in value.items()
            if (item := _sanitize_result_for_render(raw)) not in (None, "", {}, [])
        }
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            item
            for raw in value
            if (item := _sanitize_result_for_render(raw)) not in (None, "", {}, [])
        ]
    return value


def _shape_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {"type": "object", "keys": [str(key) for key in list(value)[:24]], "field_count": len(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"type": "array", "item_count": len(value)}
    return {"type": type(value).__name__}


def _compact_string(value: str, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    marker = f"...[{len(text) - max_chars} chars omitted]..."
    budget = max(0, max_chars - len(marker))
    head = budget // 2
    return text[:head] + marker + text[-(budget - head) :]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
