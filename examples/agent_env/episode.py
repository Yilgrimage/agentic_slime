from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.prompting import require_prompt
from examples.agent_env.rollout import (
    AgentEnvSpec,
    AgentTokenLedger,
    _sample_case_dump_enabled,
    arg,
    call_policy,
    cfg_path,
    decode_token_ids,
    dump_completed_sample_case,
    enable_thinking_for_rollout,
    ensure_rollout_shapes,
    env_server_url,
    infer_tool_call_parser_name,
    interaction_mode,
    lease_request_id,
    messages_for_chat_template,
    metadata,
    normalize_openai_tool,
    outcome_reward,
    parse_policy_text_view,
    parse_standard_tool_call,
    parse_text_action,
    post_env,
    record_env_metadata,
    task_key,
    task_payload,
    text_action_tag,
    tokenizer,
    turn_params,
    visible_assistant_text,
)

logger = logging.getLogger(__name__)


class AgentPolicyThreadingHTTPServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _format_error(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _recover_text_action_content(args: Any, raw_text: str) -> str:
    tag = text_action_tag(args)
    action, valid, parse_mode = parse_text_action(raw_text, tag=tag)
    if valid and parse_mode == f"{tag}_tag":
        return f"<{tag}>{action}</{tag}>"
    if not valid:
        return ""
    return ""


def _http_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _first_routable_host() -> str:
    configured = os.environ.get("AGENT_ENV_POLICY_HOST", "").strip()
    if configured:
        return configured
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=2)
        candidates = [item for item in output.split() if item and not item.startswith("127.") and item != "::1"]
        for item in candidates:
            if ":" not in item:
                return item
        for item in output.split():
            if item and not item.startswith("127.") and item != "::1":
                return item
    except Exception:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except Exception:
        return "127.0.0.1"


def _openai_tool_call_message(action: dict[str, Any], content: str = "") -> dict[str, Any]:
    arguments = action.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": str(action.get("name") or ""),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _normalize_tools(value: Any) -> list[dict[str, Any]]:
    tools = []
    for item in value if isinstance(value, list) else []:
        schema = normalize_openai_tool(item)
        if schema is not None:
            tools.append(schema)
    return tools


def _prefix_len(expected_prefix: list[dict[str, Any]], full: list[dict[str, Any]]) -> int:
    if len(full) < len(expected_prefix):
        return -1
    for idx, message in enumerate(expected_prefix):
        if full[idx] != message:
            return -1
    return len(expected_prefix)


def _message_debug_summary(message: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {"type": type(message).__name__}
    summary: dict[str, Any] = {
        "role": message.get("role"),
        "content": str(message.get("content") or "")[:200],
    }
    if "step_loss_mask" in message:
        summary["step_loss_mask"] = message.get("step_loss_mask")
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        summary["tool_calls"] = [
            {
                "id": call.get("id") if isinstance(call, dict) else None,
                "name": ((call.get("function") or {}).get("name") if isinstance(call, dict) else None),
                "arguments": str(((call.get("function") or {}).get("arguments") if isinstance(call, dict) else ""))[:200],
            }
            for call in calls[:2]
        ]
    if message.get("role") == "tool":
        summary["tool_call_id"] = message.get("tool_call_id")
    return summary


def _canonical_debug_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_debug_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_debug_json(value).encode("utf-8")).hexdigest()[:16]


def _prefix_mismatch_debug(expected_prefix: list[dict[str, Any]], full: list[dict[str, Any]]) -> dict[str, Any]:
    common = 0
    for expected, got in zip(expected_prefix, full):
        if expected != got:
            break
        common += 1
    expected_item = expected_prefix[common] if common < len(expected_prefix) else None
    got_item = full[common] if common < len(full) else None
    return {
        "expected_len": len(expected_prefix),
        "received_len": len(full),
        "first_diff": common,
        "expected_hash": _canonical_debug_hash(expected_item),
        "received_hash": _canonical_debug_hash(got_item),
        "expected": _message_debug_summary(expected_item),
        "received": _message_debug_summary(got_item),
        "expected_full": expected_item,
        "received_full": got_item,
    }


def _finish_reason(finish_type: str, assistant_message: dict[str, Any]) -> str:
    if finish_type == "length":
        return "length"
    if assistant_message.get("tool_calls"):
        return "tool_calls"
    return "stop"


@dataclass
class PolicySession:
    session_id: str
    args: Any
    spec: AgentEnvSpec
    sample: Sample
    sampling_params: dict[str, Any]
    tok: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    ledger: AgentTokenLedger | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    enable_thinking: bool | None = None
    turn_count: int = 0
    format_errors: int = 0
    max_response_tokens_hits: int = 0
    parse_modes: list[str] = field(default_factory=list)
    response_texts: list[str] = field(default_factory=list)
    context_limit_hits: int = 0

    def _request_sampling_params(self, body: dict[str, Any]) -> dict[str, Any]:
        params = copy.deepcopy(self.sampling_params)
        if "temperature" in body and body["temperature"] is not None:
            params["temperature"] = body["temperature"]
        if "top_p" in body and body["top_p"] is not None:
            params["top_p"] = body["top_p"]
        if "max_tokens" in body and body["max_tokens"] is not None:
            params["max_new_tokens"] = int(body["max_tokens"])
        elif "max_new_tokens" in body and body["max_new_tokens"] is not None:
            params["max_new_tokens"] = int(body["max_new_tokens"])
        remaining = self.ledger.remaining_context() if self.ledger is not None else None
        return turn_params(self.args, self.spec, params, remaining)

    def _context_limit_response(self, body: dict[str, Any]) -> dict[str, Any]:
        self.context_limit_hits += 1
        prompt_tokens = len(self.ledger.tokens) if self.ledger is not None else 0
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(body.get("model") or "agent-env-policy"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
                "agent_env_context_limit_reached": True,
            },
        }

    def _prepare_ledger(self, body: dict[str, Any]) -> bool:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        normalized_messages = messages_for_chat_template(messages)
        request_tools = _normalize_tools(body.get("tools"))
        if self.ledger is None:
            self.tools = request_tools
            self.enable_thinking = enable_thinking_for_rollout(self.args, self.spec)
            self.ledger = AgentTokenLedger(
                args=self.args,
                spec=self.spec,
                tok=self.tok,
                messages=normalized_messages,
                tools=self.tools,
                enable_thinking=self.enable_thinking,
            )
            self.sample.prompt = self.ledger.prompt_text
            self.sample.tokens = list(self.ledger.tokens)
            self.sample.response = ""
            self.sample.response_length = 0
            self.sample.loss_mask = []
            self.sample.rollout_log_probs = None
            return True

        prefix = _prefix_len(self.ledger.messages, normalized_messages)
        if prefix < 0:
            debug = _prefix_mismatch_debug(self.ledger.messages, normalized_messages)
            raise ValueError(
                "policy gateway received a message history that is not an append-only extension "
                f"of the recorded session: {json.dumps(debug, ensure_ascii=False)}"
            )
        new_messages = normalized_messages[prefix:]
        if new_messages:
            fits, _total = self.ledger.can_append_environment_messages(
                new_messages,
                add_generation_prompt=True,
            )
            if not fits:
                return False
            self.ledger.append_environment_messages(
                turn=self.turn_count,
                messages=new_messages,
                add_generation_prompt=True,
            )
        return True

    async def _chat_completion_async(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self._prepare_ledger(body):
            return self._context_limit_response(body)
        assert self.ledger is not None
        params = self._request_sampling_params(body)
        if int(params.get("max_new_tokens", 0) or 0) <= 0:
            return self._context_limit_response(body)

        response_text, token_ids, log_probs, finish_type = await call_policy(
            self.args,
            self.spec,
            self.sample,
            self.ledger.tokens,
            params,
        )
        if response_text and not token_ids:
            if bool(arg(self.args, "allow_policy_retokenize_fallback", False)):
                token_ids = self.tok(response_text, add_special_tokens=False)["input_ids"]
            else:
                raise RuntimeError("SGLang did not return output token ids for non-empty policy text")

        decoded_raw = decode_token_ids(self.tok, token_ids, skip_special_tokens=False)
        raw_response_text = response_text or decoded_raw
        text_view = parse_policy_text_view(self.args, self.spec, raw_response_text)
        parser_text = text_view.content_text
        assistant_message: dict[str, Any]
        parse_mode = "assistant_message"
        format_valid = True
        if interaction_mode(self.args, self.spec) == "tool_call" and self.tools:
            parser_name = infer_tool_call_parser_name(self.tok)
            action, format_valid, parse_mode = parse_standard_tool_call(parser_text, self.tools, parser_name)
            if not format_valid and not parser_text:
                action, format_valid, parse_mode = parse_standard_tool_call(raw_response_text, self.tools, parser_name)
            if not format_valid and parse_mode == "no_standard_tool_call" and self.spec.allow_assistant_message:
                content = visible_assistant_text(parser_text)
                if content:
                    action = {"type": "assistant_message", "content": content}
                    format_valid = True
                    parse_mode = "assistant_message"
            if format_valid and isinstance(action, dict) and action.get("type") == "tool_call":
                assistant_message = _openai_tool_call_message(action, str(action.get("content") or ""))
            else:
                content = visible_assistant_text(parser_text)
                assistant_message = {"role": "assistant", "content": content}
        else:
            content = visible_assistant_text(parser_text)
            if not content:
                content = _recover_text_action_content(self.args, raw_response_text)
            assistant_message = {"role": "assistant", "content": content}

        self.parse_modes.append(parse_mode)
        if not format_valid:
            self.format_errors += 1
        if finish_type == "length":
            self.max_response_tokens_hits += 1

        self.ledger.append_assistant_generation(
            turn=self.turn_count,
            message=assistant_message,
            token_ids=token_ids,
            text=raw_response_text,
            log_probs=log_probs,
        )
        self.turn_count += 1
        self.response_texts.append(raw_response_text)

        created = int(time.time())
        prompt_tokens = max(0, len(self.ledger.tokens) - len(token_ids))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": str(body.get("model") or "agent-env-policy"),
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": _finish_reason(finish_type, assistant_message),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(token_ids),
                "total_tokens": prompt_tokens + len(token_ids),
            },
        }

    def chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            return asyncio.run(self._chat_completion_async(body))

    def materialize(self, sample: Sample, sample_metadata: dict[str, Any]) -> None:
        if self.ledger is None:
            sample.status = Sample.Status.FAILED
            sample.remove_sample = True
            sample_metadata["discard_sample"] = True
            sample_metadata["discard_reason"] = "episode_no_policy_calls"
            return
        self.ledger.materialize(sample, sample_metadata, include_trace=True)
        sample_metadata["token_audit"] = self.ledger.audit()
        sample_metadata["turn_count"] = self.turn_count
        sample_metadata["format_errors"] = self.format_errors
        sample_metadata["format_ok"] = self.format_errors == 0
        sample_metadata["action_parse_modes"] = list(self.parse_modes)
        if self.max_response_tokens_hits:
            sample_metadata["max_response_tokens_hits"] = self.max_response_tokens_hits
        if self.context_limit_hits:
            sample_metadata["context_limit_hits"] = self.context_limit_hits
        sample_metadata["policy_gateway_session_id"] = self.session_id


class PolicyGatewayHandler(BaseHTTPRequestHandler):
    gateway: "PolicyGateway"

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path in ("/health", "/healthz"):
            _json_response(self, 200, {"ok": True})
            return
        _json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        try:
            if self.path != "/v1/chat/completions":
                _json_response(self, 404, {"ok": False, "error": "not found"})
                return
            self.gateway.check_auth(self.headers.get("authorization"))
            body = self._read_json()
            session_id = (
                self.headers.get("x-agent-env-policy-session")
                or self.headers.get("x-session-id")
                or body.get("agent_env_policy_session_id")
                or body.get("session_id")
            )
            if not session_id:
                raise ValueError("missing X-Agent-Env-Policy-Session header")
            session = self.gateway.session(str(session_id))
            _json_response(self, 200, session.chat_completion(body))
        except Exception as exc:
            logger.exception("policy gateway request failed path=%s", self.path)
            _json_response(self, 500, {"ok": False, "error": _format_error(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)


class PolicyGateway:
    def __init__(self) -> None:
        self.bind_host = os.environ.get("AGENT_ENV_POLICY_BIND_HOST", "0.0.0.0")
        self.public_host = _first_routable_host()
        self.api_key = os.environ.get("AGENT_ENV_POLICY_API_KEY") or f"agent-env-{uuid.uuid4().hex}"
        self.sessions: dict[str, PolicySession] = {}
        self.lock = threading.Lock()
        self.server: AgentPolicyThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port = 0

    def start(self) -> None:
        if self.server is not None:
            return
        PolicyGatewayHandler.gateway = self
        self.server = AgentPolicyThreadingHTTPServer((self.bind_host, 0), PolicyGatewayHandler)
        self.port = int(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, name="agent-env-policy-gateway", daemon=True)
        self.thread.start()
        logger.info("agent-env policy gateway listening on %s", self.base_url)

    @property
    def base_url(self) -> str:
        return f"http://{_http_host(self.public_host)}:{self.port}"

    def check_auth(self, authorization: str | None) -> None:
        expected = f"Bearer {self.api_key}"
        if authorization != expected:
            raise PermissionError("invalid policy gateway bearer token")

    def create_session(self, args: Any, spec: AgentEnvSpec, sample: Sample, sampling_params: dict[str, Any]) -> PolicySession:
        self.start()
        session_id = f"policy-{uuid.uuid4().hex[:16]}"
        sample.session_id = session_id
        session = PolicySession(
            session_id=session_id,
            args=args,
            spec=spec,
            sample=sample,
            sampling_params=copy.deepcopy(sampling_params),
            tok=tokenizer(args),
        )
        with self.lock:
            self.sessions[session_id] = session
        return session

    def session(self, session_id: str) -> PolicySession:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown policy session: {session_id}")
        return session

    def pop_session(self, session_id: str) -> PolicySession | None:
        with self.lock:
            return self.sessions.pop(session_id, None)


_GATEWAY: PolicyGateway | None = None
_GATEWAY_LOCK = threading.Lock()


def get_policy_gateway() -> PolicyGateway:
    global _GATEWAY
    with _GATEWAY_LOCK:
        if _GATEWAY is None:
            _GATEWAY = PolicyGateway()
            _GATEWAY.start()
        return _GATEWAY


def _status_from_episode(result: dict[str, Any]) -> Sample.Status:
    status = str(result.get("status") or "").strip().lower()
    if status in {"aborted", "abort"}:
        return Sample.Status.ABORTED
    if status in {"failed", "failure", "error"}:
        return Sample.Status.FAILED
    if status in {"truncated", "timeout"}:
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED if bool(result.get("done", True)) else Sample.Status.TRUNCATED


def _requested_task_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("env", "data_source", "domain", "task_set", "dataset_name", "split", "task_id", "task_index", "seed"):
        value = payload.get(key)
        if value not in (None, "", []):
            metadata[key] = value
    task_ref = payload.get("task_ref")
    if isinstance(task_ref, dict):
        metadata["task_ref"] = copy.deepcopy(task_ref)
    return metadata


async def generate_server_episode_rollout(
    args: Any,
    sample: Sample,
    sampling_params: dict,
    *,
    spec: AgentEnvSpec,
    episode_payload: dict[str, Any] | None = None,
) -> Sample:
    assert not arg(args, "partial_rollout", False), f"{spec.name} rollout does not support partial rollout yet."

    sample_metadata = metadata(sample)
    if sample.status == Sample.Status.ABORTED:
        sample.status = Sample.Status.PENDING
    sample.remove_sample = False
    gateway = get_policy_gateway()
    session = gateway.create_session(args, spec, sample, sampling_params)
    final_score = 0.0
    success = False
    split = sample_metadata.get("split") or cfg_path(args, "task.split", spec.default_split)

    try:
        request_task_payload = task_payload(sample, spec)
        requested_task = _requested_task_metadata(request_task_payload)
        if requested_task:
            sample_metadata["requested_task"] = requested_task
        response_max_tokens = arg(args, "rollout_max_response_len", None)
        if response_max_tokens is None:
            response_max_tokens = spec.default_response_max_tokens
        payload = {
            **request_task_payload,
            **(episode_payload or {}),
            "split": split,
            "task_key": task_key(sample, spec),
            "request_id": lease_request_id(sample),
            "release_on_done": True,
            "include_trace": _sample_case_dump_enabled(args),
            "prompt": require_prompt(sample.prompt, env_name=spec.name, source="sample.prompt"),
            "max_turns": int(cfg_path(args, "max_turns", spec.default_max_turns)),
            "max_response_tokens": int(response_max_tokens),
            "sampling_params": copy.deepcopy(sampling_params),
            "timeouts": {
                "policy_s": float(cfg_path(args, "timeouts.policy_s", 120.0)),
            },
            "policy": {
                "base_url": gateway.base_url,
                "chat_completions_path": "/v1/chat/completions",
                "api_key": gateway.api_key,
                "session_id": session.session_id,
                "model": str(getattr(args, "served_model_name", None) or getattr(args, "hf_checkpoint", None) or "agent-env-policy"),
                "headers": {"X-Agent-Env-Policy-Session": session.session_id},
            },
        }
        result = await post_env(args, spec, "/run_episode", payload, max_retries=1)
        info = result.get("info") if isinstance(result.get("info"), dict) else {}
        episode_metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        final_score = float(result.get("score", 0.0) or 0.0)
        success = bool(result.get("success", final_score > 0))
        session.materialize(sample, sample_metadata)
        for key in (
            "actions",
            "action_parse_modes",
            "format_checks",
            "format_errors",
            "format_ok",
            "max_response_tokens_hits",
            "messages",
            "message_delta_modes",
            "policy_usage",
            "token_count_estimate",
            "truncated_reason",
            "turn_count",
            "turns",
        ):
            if key in episode_metadata:
                sample_metadata[key] = episode_metadata[key]
        if isinstance(sample_metadata.get("format_checks"), list):
            checks = [item for item in sample_metadata["format_checks"] if isinstance(item, dict)]
            sample_metadata["format_errors"] = sum(1 for item in checks if not bool(item.get("valid", False)))
            sample_metadata["format_ok"] = sample_metadata["format_errors"] == 0
        discard_reason = episode_metadata.get("discard_reason") or info.get("discard_reason")
        if bool(episode_metadata.get("discard_sample", False)) or bool(info.get("discard_sample", False)):
            sample.remove_sample = True
            sample_metadata["discard_sample"] = True
            if discard_reason:
                sample_metadata["discard_reason"] = discard_reason
        if bool(sample_metadata.get("discard_sample", False)):
            sample.status = Sample.Status.FAILED
            sample.reward = 0.0
        else:
            sample.status = _status_from_episode(result)
        sample_metadata.update(
            {
                "env_score": final_score,
                "env_success": success,
                "env_reward": outcome_reward(args, spec, success, final_score),
                "episode_result": result,
                "interaction_mode": "server_episode",
            }
        )
        env_meta = {
            "task_index": result.get("task_index", request_task_payload.get("task_index", 0)),
            "task_id": result.get("task_id") or info.get("task_id"),
            "split": result.get("split", split),
            "lease_id": result.get("lease_id"),
            "server_url": env_server_url(args, spec),
        }
        if requested_task.get("task_id") not in (None, "", []):
            env_meta["requested_task_id"] = requested_task["task_id"]
        for key in ("game_file", "domain", "task_set", "data_source", "task_ref", "num_tasks"):
            if result.get(key) not in (None, "", []):
                env_meta[key] = result.get(key)
        env_meta.update({key: value for key, value in info.items() if key not in env_meta})
        record_env_metadata(sample_metadata, spec, env_meta)
        if not bool(sample_metadata.get("discard_sample", False)):
            sample.reward = None if sample.status != Sample.Status.ABORTED else 0.0
        ensure_rollout_shapes(args, sample, spec)
        dump_completed_sample_case(args, spec, sample, tokenizer(args))
        return sample
    except Exception as exc:
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        sample.remove_sample = True
        sample_metadata["discard_sample"] = True
        sample_metadata["discard_reason"] = "server_episode_failure"
        sample_metadata.setdefault("error", repr(exc))
        ensure_rollout_shapes(args, sample, spec)
        dump_completed_sample_case(args, spec, sample, tokenizer(args))
        logger.exception("%s server-episode rollout failed", spec.name)
        return sample
    finally:
        gateway.pop_session(session.session_id)
