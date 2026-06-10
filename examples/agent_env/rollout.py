from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
_SAMPLE_DUMP_COUNTS: dict[str, int] = {}
_DELTA_BASE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


InfoFn = Callable[[dict], list[str]]
TextFn = Callable[[Any, str, dict], str]
PromptFn = Callable[[Any, Sample, str, dict], str]
ChooseFn = Callable[[Any, Any, list[str], Sample], Any]
ParseFn = Callable[[str], tuple[Any, bool, str]]
SuccessFn = Callable[[dict, float], bool]
MetadataFn = Callable[[dict, int, str, str | None], dict]


@dataclass(frozen=True)
class AgentEnvSpec:
    name: str
    default_env_url: str
    env_url_arg: str
    env_url_envvar: str
    default_split: str
    info_actions: InfoFn
    observation_text: TextFn
    initial_prompt: PromptFn
    choose_action: ChooseFn
    success: SuccessFn
    env_metadata: MetadataFn
    default_max_turns: int = 20
    default_response_max_tokens: int = 128
    default_reward_source: str = "score"
    default_interaction_mode: str = "text_action"
    parse_action_fn: ParseFn | None = None
    allow_assistant_message: bool = False


def arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def cfg_path(args: Any, path: str, default: Any = None) -> Any:
    value: Any = args
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return default
    return value


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def interaction_mode(args: Any, spec: AgentEnvSpec) -> str:
    mode = str(cfg_path(args, "interaction.mode", spec.default_interaction_mode) or "")
    mode = mode.strip().lower().replace("-", "_")
    aliases = {
        "action": "text_action",
        "text": "text_action",
        "text_action": "text_action",
        "tool": "tool_call",
        "tool_use": "tool_call",
        "tool_call": "tool_call",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported interaction_mode={mode!r}; expected text_action or tool_call")
    return aliases[mode]


def text_action_tag(args: Any) -> str:
    tag = str(cfg_path(args, "interaction.text_action.tag", "action") or "action").strip()
    if not tag:
        raise ValueError("interaction.text_action.tag must not be empty")
    return tag


def _tokenizer_chat_template(tok: Any) -> str:
    template = getattr(tok, "chat_template", None)
    if template:
        return str(template)
    init_kwargs = getattr(tok, "init_kwargs", None)
    if isinstance(init_kwargs, dict) and init_kwargs.get("chat_template"):
        return str(init_kwargs["chat_template"])
    return ""


def infer_tool_call_parser_name(tok: Any) -> str:
    template = _tokenizer_chat_template(tok)
    if "<function=" in template and "<parameter=" in template:
        return "qwen3_coder"
    if "<tool_call>" in template and '"name"' in template and '"arguments"' in template:
        return "qwen"
    return "qwen"


def reasoning_parser_name(args: Any, spec: AgentEnvSpec) -> str:
    value = cfg_path(args, "interaction.reasoning.parser", None)
    if value is None:
        return ""
    parser = str(value or "").strip()
    if parser.lower() in {"", "none", "off", "false", "0"}:
        return ""
    return parser


def first(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def metadata(sample: Sample) -> dict:
    if sample.metadata is None:
        sample.metadata = {}
    return sample.metadata


def _runtime_env(args: Any, name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    train_env_vars = getattr(args, "train_env_vars", None) or {}
    if isinstance(train_env_vars, dict):
        value = train_env_vars.get(name)
        if value:
            return str(value)
    return default


def _sample_case_dump_enabled(args: Any) -> bool:
    try:
        limit = int(_runtime_env(args, "AGENT_ENV_ROLLOUT_DUMP_N", "0") or 0)
    except (TypeError, ValueError):
        limit = 0
    return limit > 0 and bool(_runtime_env(args, "RUN_ROOT") or _runtime_env(args, "MLF_RUN_ROOT"))


def _dump_trace_mode(args: Any) -> str:
    mode = str(_runtime_env(args, "AGENT_ENV_ROLLOUT_DUMP_TRACE", "both") or "both").strip().lower()
    aliases = {
        "message": "messages",
        "messages": "messages",
        "token": "tokens",
        "tokens": "tokens",
        "both": "both",
        "all": "both",
        "none": "none",
        "off": "none",
        "0": "none",
    }
    if mode not in aliases:
        logger.warning("Unknown AGENT_ENV_ROLLOUT_DUMP_TRACE=%r; using both", mode)
        return "both"
    return aliases[mode]


def _decoded_sample_token_traces(tok: Any, sample: Sample) -> dict[str, Any]:
    if tok is None:
        return {}
    token_ids = list(getattr(sample, "tokens", []) or [])
    if not token_ids:
        return {}
    traces: dict[str, Any] = {
        "decoded_token_trace": decode_token_ids(tok, token_ids, skip_special_tokens=False),
    }
    try:
        response_length = int(getattr(sample, "response_length", 0) or 0)
    except (TypeError, ValueError):
        response_length = 0
    if 0 < response_length <= len(token_ids):
        prompt_tokens = token_ids[:-response_length]
        response_tokens = token_ids[-response_length:]
        traces["decoded_prompt_token_trace"] = decode_token_ids(tok, prompt_tokens, skip_special_tokens=False)
        traces["decoded_response_token_trace"] = decode_token_ids(tok, response_tokens, skip_special_tokens=False)
    return traces


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _is_env_infra_exception(exc: BaseException) -> bool:
    """Classify transport/capacity failures separately from model behavior.

    These failures mean the rollout did not actually run in the environment,
    so treating them as zero-reward policy samples corrupts GRPO groups.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    exc_name = type(exc).__name__.lower()
    text = repr(exc).lower()
    markers = (
        "capacityerror",
        "connecterror",
        "connecttimeout",
        "httpstatuserror",
        "no env worker could allocate",
        "pooltimeout",
        "readtimeout",
        "remoteprotocolerror",
        "service unavailable",
        "timed out",
        "worker available",
    )
    return any(marker in exc_name or marker in text for marker in markers)


def _record_rollout_infra_failure(
    args: Any,
    spec: AgentEnvSpec,
    sample: Sample,
    phase: str,
    exc: BaseException,
    tok: Any | None,
) -> Sample:
    sample_metadata = metadata(sample)
    sample_metadata["infra_error"] = {
        "phase": phase,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    sample_metadata.setdefault("error", repr(exc))
    sample.reward = 0.0
    sample.remove_sample = True
    # Allocation failures are pure capacity/backpressure events; in fully async
    # mode ABORTED groups are requeued instead of being shipped to training.
    sample.status = Sample.Status.ABORTED if phase == "allocate" else Sample.Status.FAILED
    ensure_rollout_shapes(args, sample, spec)
    dump_completed_sample_case(args, spec, sample, tok)
    logger.warning(
        "%s rollout infra failure phase=%s status=%s remove_sample=%s error=%r",
        spec.name,
        phase,
        sample.status.value,
        sample.remove_sample,
        exc,
    )
    return sample


def dump_completed_sample_case(args: Any, spec: AgentEnvSpec, sample: Sample, tok: Any | None = None) -> None:
    limit = int(_runtime_env(args, "AGENT_ENV_ROLLOUT_DUMP_N", "0") or 0)
    if limit <= 0:
        return
    run_root = _runtime_env(args, "RUN_ROOT") or _runtime_env(args, "MLF_RUN_ROOT")
    if not run_root:
        return
    count = _SAMPLE_DUMP_COUNTS.get(spec.name, 0)
    if count >= limit:
        return
    _SAMPLE_DUMP_COUNTS[spec.name] = count + 1
    sample_metadata = sample.metadata or {}
    output_dir = Path(run_root) / "rollout_cases" / spec.name / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sample_{count:04d}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
    trace_mode = _dump_trace_mode(args)
    record = {
        "sample_index": count,
        "status": getattr(getattr(sample, "status", None), "name", str(getattr(sample, "status", ""))),
        "reward": getattr(sample, "reward", None),
        "response_length": getattr(sample, "response_length", None),
        "effective_response_length": getattr(sample, "effective_response_length", None),
        "total_token_length": len(getattr(sample, "tokens", []) or []),
        "response": getattr(sample, "response", None),
        "turn_count": sample_metadata.get("turn_count"),
        "format_errors": sample_metadata.get("format_errors"),
        "max_response_tokens_hits": sample_metadata.get("max_response_tokens_hits"),
        "truncated_reason": sample_metadata.get("truncated_reason"),
        "env_score": sample_metadata.get("env_score"),
        "env_success": sample_metadata.get("env_success"),
        "env_reward": sample_metadata.get("env_reward"),
        "env_metadata": sample_metadata.get(spec.name),
        "actions": sample_metadata.get("actions"),
        "turns": _json_safe(sample_metadata.get("turns")),
        "action_parse_modes": sample_metadata.get("action_parse_modes"),
        "dump_trace_mode": trace_mode,
        "token_audit": sample_metadata.get("token_audit"),
        "env_evaluate": sample_metadata.get("env_evaluate"),
        "error": sample_metadata.get("error"),
    }
    if trace_mode in {"messages", "both"}:
        record["messages"] = sample_metadata.get("messages")
    if trace_mode in {"tokens", "both"}:
        record["token_segments"] = sample_metadata.get("token_segments")
        record.update(_decoded_sample_token_traces(tok, sample))
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_index(sample: Sample) -> int:
    sample_metadata = metadata(sample)
    if "task_index" in sample_metadata:
        return int(sample_metadata["task_index"])
    if sample.group_index is not None:
        return int(sample.group_index)
    if sample.index is not None:
        return int(sample.index)
    return 0


def task_payload(sample: Sample, spec: AgentEnvSpec) -> dict[str, Any]:
    sample_metadata = metadata(sample)
    split = sample_metadata.get("split") or spec.default_split
    payload: dict[str, Any] = {
        "split": split,
        "task_index": task_index(sample),
    }
    for key in ("env", "data_source", "domain", "task_set", "task_id", "seed", "task_ref"):
        if key in sample_metadata and sample_metadata[key] is not None:
            payload[key] = sample_metadata[key]
    return payload


def task_key(sample: Sample, spec: AgentEnvSpec) -> str:
    payload = task_payload(sample, spec)
    parts = []
    for key in ("env", "domain", "task_set", "split", "task_id", "task_index"):
        value = payload.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    task_ref = payload.get("task_ref")
    if isinstance(task_ref, dict):
        ref_type = task_ref.get("type")
        ref_path = task_ref.get("path")
        if ref_type:
            parts.append(f"task_ref_type={ref_type}")
        if ref_path:
            parts.append(f"task_ref_path={ref_path}")
    return "|".join(parts)


def tokenizer(args: Any):
    from slime.rollout.sglang_rollout import GenerateState

    return GenerateState(args).tokenizer


def _normalize_tool_call_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _parse_json_object(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            fn = tool_call.get("function")
            if not isinstance(fn, dict):
                continue
            fn["arguments"] = _normalize_tool_call_arguments(fn.get("arguments"))
    return normalized


def apply_chat_template_ids(
    tok: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
) -> list[int]:
    def normalize_ids(value: Any) -> list[int]:
        if isinstance(value, dict) and "input_ids" in value:
            value = value["input_ids"]
        if hasattr(value, "data") and isinstance(getattr(value, "data"), dict) and "input_ids" in value.data:
            value = value.data["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError(f"Expected a single chat-template input_ids row, got {len(value)} rows")
            value = value[0]
        if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
            raise TypeError(f"Expected chat-template token ids as list[int], got {type(value)} with head={str(value)[:120]}")
        return value

    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = copy.deepcopy(tools)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    messages = messages_for_chat_template(messages)
    try:
        return normalize_ids(tok.apply_chat_template(messages, **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return normalize_ids(tok.apply_chat_template(messages, **kwargs))


def apply_chat_template_text(
    tok: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = copy.deepcopy(tools)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    messages = messages_for_chat_template(messages)
    try:
        return str(tok.apply_chat_template(messages, **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return str(tok.apply_chat_template(messages, **kwargs))


def tool_call_message(action: dict[str, Any], content: str | None = None) -> dict[str, Any]:
    call_id = f"call_{uuid.uuid4().hex[:24]}"
    name = str(action.get("name") or "")
    arguments = action.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        ],
    }


def valid_message_updates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    updates = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        updates.append(messages_for_chat_template([item])[0])
    return updates


def mark_messages_untrained(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = messages_for_chat_template(messages)
    for message in output:
        if message.get("role") == "assistant":
            message["step_loss_mask"] = 0
    return output


def tool_result_message(assistant_message: dict[str, Any], observation: str) -> dict[str, Any]:
    tool_calls = assistant_message.get("tool_calls") or []
    call_id = tool_calls[0].get("id") if tool_calls and isinstance(tool_calls[0], dict) else f"call_{uuid.uuid4().hex[:24]}"
    return {"role": "tool", "tool_call_id": call_id, "content": observation}


def policy_message_for_generation(mode: str, response_text: str, action: Any) -> dict[str, Any]:
    if mode == "tool_call" and isinstance(action, dict) and action.get("type") == "tool_call":
        return tool_call_message(action, content="")

    if isinstance(action, dict) and action.get("type") == "assistant_message":
        return {"role": "assistant", "content": str(action.get("content") or response_text or "")}

    return {"role": "assistant", "content": response_text}


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
        # Env servers may return the full step delta, including the policy
        # assistant message we already appended from raw generation tokens.
        if updates[0].get("role") == "assistant":
            return updates[1:]
        return updates
    if mode == "tool_call" and isinstance(action, dict) and action.get("type") == "tool_call":
        return [tool_result_message(assistant_message, observation)]
    if not done:
        return [{"role": "user", "content": env_text}]
    return []


def initial_messages_for_rollout(
    args: Any,
    spec: AgentEnvSpec,
    sample: Sample,
    observation: str,
    info: dict[str, Any],
    prompt: str,
) -> list[dict[str, Any]]:
    if interaction_mode(args, spec) == "tool_call":
        policy = str(info.get("policy") or "").strip()
        messages = []
        if policy:
            messages.append({"role": "system", "content": policy})
        agent_messages = valid_message_updates(info.get("agent_messages") or info.get("initial_messages"))
        if agent_messages:
            messages.extend(mark_messages_untrained(agent_messages))
            return messages
        if observation.strip():
            messages.append({"role": "user", "content": observation.strip()})
            return messages
    return [{"role": "user", "content": prompt}]


def tool_schemas_for_rollout(args: Any, spec: AgentEnvSpec, info: dict[str, Any]) -> list[dict[str, Any]]:
    if interaction_mode(args, spec) != "tool_call":
        return []
    raw_tools = info.get("tool_schemas") or info.get("tools") or []
    schemas = []
    for tool in raw_tools if isinstance(raw_tools, list) else []:
        schema = normalize_openai_tool(tool)
        if schema is not None:
            schemas.append(schema)
    seen = set()
    unique = []
    for schema in schemas:
        name = openai_tool_name(schema)
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(schema)
    return unique


def enable_thinking_for_rollout(args: Any, spec: AgentEnvSpec) -> bool | None:
    value = cfg_path(args, "interaction.enable_thinking", None)
    if value is None:
        return True if interaction_mode(args, spec) == "tool_call" else None
    return _bool_value(value)


def token_prefix_length(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return idx
    return limit


def has_prefix(values: list[int], prefix: list[int]) -> bool:
    return len(values) >= len(prefix) and values[: len(prefix)] == prefix


def decode_token_ids(tok: Any, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
    if not token_ids:
        return ""
    decode = getattr(tok, "decode", None)
    if callable(decode):
        try:
            return str(decode(token_ids, skip_special_tokens=skip_special_tokens))
        except TypeError:
            return str(decode(token_ids))
    return ""


def strip_chat_boundary_tokens(text: str) -> str:
    text = str(text or "")
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(token, "")
    return text.strip()


def visible_assistant_text(text: str) -> str:
    """Text that should be visible to the environment/user side.

    The token ledger keeps the raw model output, including reasoning and chat
    boundary tokens. Reasoning separation is handled by the configured SGLang
    reasoning parser; this helper only removes chat boundary artifacts.
    """
    return strip_chat_boundary_tokens(text)


@dataclass(frozen=True)
class PolicyTextView:
    raw_text: str
    content_text: str
    reasoning_text: str = ""
    reasoning_parser: str = ""
    reasoning_ok: bool = True
    reasoning_error: str | None = None


def parse_policy_text_view(args: Any, spec: AgentEnvSpec, raw_text: str) -> PolicyTextView:
    raw_text = str(raw_text or "")
    parser_name = reasoning_parser_name(args, spec)
    if not parser_name:
        return PolicyTextView(raw_text=raw_text, content_text=strip_chat_boundary_tokens(raw_text))

    try:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        parser = ReasoningParser(parser_name)
        reasoning_text, content_text = parser.parse_non_stream(raw_text)
        return PolicyTextView(
            raw_text=raw_text,
            content_text=strip_chat_boundary_tokens(content_text or ""),
            reasoning_text=str(reasoning_text or ""),
            reasoning_parser=parser_name,
            reasoning_ok=True,
        )
    except Exception as exc:
        logger.debug("SGLang reasoning parser failed: parser=%s", parser_name, exc_info=True)
        if _bool_value(cfg_path(args, "interaction.reasoning.fallback_to_raw", False)):
            content_text = strip_chat_boundary_tokens(raw_text)
        else:
            content_text = ""
        return PolicyTextView(
            raw_text=raw_text,
            content_text=content_text,
            reasoning_text="",
            reasoning_parser=parser_name,
            reasoning_ok=False,
            reasoning_error=f"{type(exc).__name__}: {exc}",
        )


@dataclass
class TokenSegment:
    kind: str
    role: str | None
    turn: int
    token_count: int
    loss_mask_sum: int
    text: str


class AgentTokenLedger:
    """Token-in/token-out conversation state for agent-env rollouts.

    `tokens` is the training truth. `messages` is a semantic/logging view and is
    only used to render newly introduced non-model messages or for audit.
    """

    def __init__(
        self,
        *,
        args: Any,
        spec: AgentEnvSpec,
        tok: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        enable_thinking: bool | None,
    ) -> None:
        self.args = args
        self.spec = spec
        self.tok = tok
        self.tools = tools or []
        self.enable_thinking = enable_thinking
        self.messages = messages_for_chat_template(messages)

        self.prompt_tokens = apply_chat_template_ids(
            tok,
            self.messages,
            tools=self.tools or None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        self.prompt_text = apply_chat_template_text(
            tok,
            self.messages,
            tools=self.tools or None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt_without_generation = apply_chat_template_ids(
            tok,
            self.messages,
            tools=self.tools or None,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        self.generation_prompt_ids = (
            self.prompt_tokens[len(prompt_without_generation) :]
            if has_prefix(self.prompt_tokens, prompt_without_generation)
            else []
        )

        self.tokens = list(self.prompt_tokens)
        self.response_tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.token_rewards: list[float] = []
        self.rollout_log_probs: list[float] | None = None
        self.assistant_response_texts: list[str] = []
        self.segments: list[TokenSegment] = [
            TokenSegment(
                kind="initial_prompt",
                role=None,
                turn=0,
                token_count=len(self.prompt_tokens),
                loss_mask_sum=0,
                text=self.prompt_text,
            )
        ]

    @property
    def response_length(self) -> int:
        return len(self.response_tokens)

    def _append_segment(
        self,
        *,
        kind: str,
        role: str | None,
        turn: int,
        tokens: list[int],
        loss_mask_value: int,
        text: str | None = None,
    ) -> None:
        if not tokens:
            return
        mask = [int(loss_mask_value)] * len(tokens)
        self.tokens.extend(tokens)
        self.response_tokens.extend(tokens)
        self.loss_mask.extend(mask)
        self.token_rewards.extend([0.0] * len(tokens))
        if self.rollout_log_probs is not None:
            self.rollout_log_probs.extend([0.0] * len(tokens))
        self.segments.append(
            TokenSegment(
                kind=kind,
                role=role,
                turn=turn,
                token_count=len(tokens),
                loss_mask_sum=sum(mask),
                text=text if text is not None else decode_token_ids(self.tok, tokens, skip_special_tokens=False),
            )
        )

    def _render_delta_tokens(
        self,
        new_messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[list[int], str, str]:
        normalized = messages_for_chat_template(new_messages)
        try:
            base_ids = apply_chat_template_ids(
                self.tok,
                _DELTA_BASE_MESSAGES,
                tools=self.tools or None,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
            )
            with_ids = apply_chat_template_ids(
                self.tok,
                _DELTA_BASE_MESSAGES + normalized,
                tools=self.tools or None,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=self.enable_thinking,
            )
            if has_prefix(with_ids, base_ids):
                delta = with_ids[len(base_ids) :]
                return delta, decode_token_ids(self.tok, delta, skip_special_tokens=False), "short_base"
        except Exception:
            logger.debug("short-base message delta rendering failed; falling back to full-prefix delta", exc_info=True)

        before = apply_chat_template_ids(
            self.tok,
            self.messages,
            tools=self.tools or None,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        after = apply_chat_template_ids(
            self.tok,
            self.messages + normalized,
            tools=self.tools or None,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
        )
        if has_prefix(after, before):
            delta = after[len(before) :]
            return delta, decode_token_ids(self.tok, delta, skip_special_tokens=False), "full_prefix"

        common = token_prefix_length(after, before)
        delta = after[common:]
        return delta, decode_token_ids(self.tok, delta, skip_special_tokens=False), "full_common_prefix"

    def remaining_context(self) -> int | None:
        max_len = arg(self.args, "rollout_max_context_len")
        if max_len is None:
            return None
        return int(max_len) - len(self.tokens) - 1

    def append_assistant_generation(
        self,
        *,
        turn: int,
        message: dict[str, Any],
        token_ids: list[int],
        text: str,
        log_probs: list[float] | None = None,
    ) -> None:
        self.messages.append(messages_for_chat_template([message])[0])
        self._append_segment(
            kind="assistant",
            role="assistant",
            turn=turn,
            tokens=list(token_ids),
            loss_mask_value=1,
            text=text,
        )
        self.assistant_response_texts.append(text)

    def append_environment_messages(
        self,
        *,
        turn: int,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> tuple[int, str]:
        if not messages and not add_generation_prompt:
            return 0, "none"
        delta, text, mode = self._render_delta_tokens(messages, add_generation_prompt=add_generation_prompt)
        self.messages.extend(messages_for_chat_template(messages))
        self._append_segment(
            kind="environment",
            role="+".join(str(message.get("role", "")) for message in messages) or None,
            turn=turn,
            tokens=delta,
            loss_mask_value=0,
            text=text,
        )
        return len(delta), mode

    def can_append_environment_messages(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> tuple[bool, int]:
        delta, _text, _mode = self._render_delta_tokens(messages, add_generation_prompt=add_generation_prompt)
        total = len(self.tokens) + len(delta)
        max_len = arg(self.args, "rollout_max_context_len")
        if max_len is None:
            return True, total
        return total <= int(max_len), total

    def add_reward_to_last_token(self, value: float) -> bool:
        if value == 0:
            return True
        if not self.token_rewards:
            return False
        self.token_rewards[-1] += float(value)
        return True

    def materialize(self, sample: Sample, sample_metadata: dict[str, Any], *, include_trace: bool = False) -> None:
        sample.tokens = list(self.tokens)
        sample.response_length = self.response_length
        sample.loss_mask = list(self.loss_mask)
        sample.response = "".join(self.assistant_response_texts)
        sample.rollout_log_probs = None
        sample_metadata["token_rewards"] = list(self.token_rewards)
        if include_trace:
            sample_metadata["messages"] = copy.deepcopy(self.messages)
            sample_metadata["token_segments"] = [segment.__dict__.copy() for segment in self.segments]

    def audit(self) -> dict[str, Any]:
        try:
            full_ids = apply_chat_template_ids(
                self.tok,
                self.messages,
                tools=self.tools or None,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
            )
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}
        common = token_prefix_length(full_ids, self.tokens)
        return {
            "ok": full_ids == self.tokens,
            "ledger_tokens": len(self.tokens),
            "full_template_tokens": len(full_ids),
            "common_prefix_tokens": common,
            "first_diff": None if full_ids == self.tokens else common,
        }


def _strip_outer_code_fences(text: str) -> str:
    text = text.strip()
    fence = chr(96) * 3
    if not text.startswith(fence) or not text.endswith(fence):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text.strip(fence).strip()
    return "\n".join(lines[1:-1]).strip()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = _strip_outer_code_fences(text)
    if text.lower().startswith("json\n"):
        text = text.split("\n", 1)[1].strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _json_schema_for_parameter(name: str, param: Any) -> dict[str, Any]:
    annotation = getattr(param, "annotation", None)
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


def callable_to_openai_tool(fn: Callable, name: str | None = None, description: str | None = None) -> dict[str, Any]:
    import inspect

    tool_name = name or getattr(fn, "__name__", "tool")
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    try:
        signature = inspect.signature(fn)
        for param_name, param in signature.parameters.items():
            if param_name in {"self", "cls"} or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            parameters["properties"][param_name] = _json_schema_for_parameter(param_name, param)
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)
    except Exception:
        pass
    if not parameters["required"]:
        parameters.pop("required", None)
    return {
        "type": "function",
        "function": {
            "name": str(tool_name),
            "description": str(description or getattr(fn, "__doc__", "") or tool_name).strip(),
            "parameters": parameters,
        },
    }


def normalize_openai_tool(tool: Any) -> dict[str, Any] | None:
    if isinstance(tool, dict):
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            return tool
        name = tool.get("name") or tool.get("tool_name")
        description = tool.get("description") or ""
        parameters = tool.get("parameters") or tool.get("args_schema") or {"type": "object", "properties": {}}
        if name:
            return {
                "type": "function",
                "function": {"name": str(name), "description": str(description), "parameters": parameters},
            }
    for method_name in ("to_openai_tool", "openai_schema", "schema"):
        method = getattr(tool, method_name, None)
        if callable(method):
            try:
                return normalize_openai_tool(method())
            except Exception:
                pass
    fn = getattr(tool, "function", None) or getattr(tool, "func", None) or getattr(tool, "callable", None)
    name = getattr(tool, "name", None) or getattr(fn, "__name__", None)
    description = getattr(tool, "description", None) or getattr(tool, "__doc__", None)
    if callable(fn):
        return callable_to_openai_tool(fn, name=name, description=description)
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


def openai_tool_name(tool: dict[str, Any]) -> str:
    return str((tool.get("function") or {}).get("name") or "")


def _parse_tool_call(response_text: str) -> tuple[dict[str, Any] | None, bool, str]:
    call_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, flags=re.IGNORECASE | re.DOTALL)
    if not call_match:
        args_match = re.search(r"<arguments>\s*(.*?)\s*</arguments>", response_text, flags=re.IGNORECASE | re.DOTALL)
        if args_match:
            arguments = _parse_json_object(args_match.group(1))
            if isinstance(arguments, dict):
                name_match = re.search(r"<name>\s*(.*?)\s*</name>", response_text, flags=re.IGNORECASE | re.DOTALL)
                name = str(name_match.group(1)).strip() if name_match else ""
                if not name:
                    if any(key in arguments for key in ("code", "python", "command")):
                        name = "execute"
                    elif any(key in arguments for key in ("answer", "message", "status")):
                        name = "finish"
                if name:
                    return {"type": "tool_call", "name": name, "arguments": arguments}, False, "orphan_tool_arguments"
        return None, False, "no_tool_call"

    body = call_match.group(1).strip()
    object_call = _parse_json_object(body)
    if object_call is not None and "name" in object_call:
        name = str(object_call.get("name") or "").strip()
        arguments = object_call.get("arguments", {})
        if not name:
            return None, False, "empty_tool_name"
        if not isinstance(arguments, dict):
            return None, False, "invalid_tool_arguments"
        return {"type": "tool_call", "name": name, "arguments": arguments}, True, "tool_call_json"

    name_match = re.search(r"<name>\s*(.*?)\s*</name>", body, flags=re.IGNORECASE | re.DOTALL)
    args_match = re.search(r"<arguments>\s*(.*?)\s*</arguments>", body, flags=re.IGNORECASE | re.DOTALL)
    if not name_match:
        return None, False, "missing_tool_name"

    name = name_match.group(1).strip()
    if not name:
        return None, False, "empty_tool_name"

    args_text = args_match.group(1) if args_match else "{}"
    arguments = _parse_json_object(args_text)
    if arguments is None:
        return None, False, "invalid_tool_arguments"

    return {"type": "tool_call", "name": name, "arguments": arguments}, True, "tool_call"


def parse_text_action(response_text: str, tag: str = "action") -> tuple[Any, bool, str]:
    text = _strip_outer_code_fences(response_text)
    escaped_tag = re.escape(tag)
    action_match = re.search(rf"<{escaped_tag}>\s*(.*?)\s*</{escaped_tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if action_match:
        action = action_match.group(1).strip().strip(chr(34)).strip(chr(39))
        if action:
            return action, True, f"{tag}_tag"
        return "", False, f"empty_{tag}_tag"

    unterminated_action = re.search(rf"<{escaped_tag}>\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if unterminated_action:
        action_lines = unterminated_action.group(1).strip().splitlines()
        if not action_lines:
            return "", False, f"empty_unterminated_{tag}_tag"
        action = action_lines[0].strip().strip(chr(34)).strip(chr(39))
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
    return "look", False, "fallback"


def parse_tool_call_action(response_text: str) -> tuple[Any, bool, str]:
    tool_action, tool_valid, tool_mode = _parse_tool_call(response_text)
    if tool_action is not None or tool_mode != "no_tool_call":
        return tool_action or "", tool_valid, tool_mode
    return "", False, "no_tool_call"


def parse_standard_tool_call(response_text: str, tools: list[dict[str, Any]], parser_name: str = "qwen") -> tuple[Any, bool, str]:
    try:
        from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
        from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    except Exception:
        from sglang.srt.managers.io_struct import Function as SglFunction
        from sglang.srt.managers.io_struct import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    try:
        sgl_tools = [
            SglTool(type=tool["type"], function=SglFunction(**tool["function"]))
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        ]
        parser = FunctionCallParser(sgl_tools, parser_name)
        if not parser.has_tool_call(response_text):
            return "", False, "no_standard_tool_call"
        normal_text, calls = parser.parse_non_stream(response_text)
    except Exception as exc:
        logger.debug("standard tool-call parser failed", exc_info=True)
        return "", False, f"tool_parser_error:{type(exc).__name__}"

    if not calls:
        return "", False, "empty_standard_tool_call"
    call = calls[0]
    name = str(getattr(call, "name", "") or "")
    parameters = getattr(call, "parameters", {}) or {}
    if isinstance(parameters, str):
        parsed_parameters = _parse_json_object(parameters)
        parameters = parsed_parameters if parsed_parameters is not None else {}
    if not isinstance(parameters, dict):
        parameters = {}
    if not name:
        return "", False, "empty_standard_tool_name"
    return {
        "type": "tool_call",
        "name": name,
        "arguments": parameters,
        "content": strip_chat_boundary_tokens(str(normal_text or "")),
    }, True, f"standard_tool_call:{parser_name}"


def action_parser(args: Any, spec: AgentEnvSpec) -> ParseFn:
    if spec.parse_action_fn is not None:
        return spec.parse_action_fn
    mode = interaction_mode(args, spec)
    if mode == "tool_call":
        return parse_tool_call_action
    tag = text_action_tag(args)
    return lambda response_text: parse_text_action(response_text, tag=tag)


def ensure_rollout_shapes(args: Any, sample: Sample, spec: AgentEnvSpec) -> None:
    token_count = len(sample.tokens)
    if token_count == 0:
        sample.tokens = [0]
        sample.remove_sample = True
        metadata(sample)["shape_inserted_dummy_token"] = True
        token_count = 1
    response_length = int(sample.response_length or 0)
    if response_length >= token_count and token_count > 0:
        corrected = max(0, token_count - 1)
        metadata(sample)["shape_corrected_response_length"] = {
            "old": response_length,
            "new": corrected,
            "token_count": token_count,
        }
        response_length = corrected
        sample.response_length = corrected
    sample_metadata = metadata(sample)
    token_rewards = list(sample_metadata.get("token_rewards") or [])
    if len(token_rewards) > response_length:
        token_rewards = token_rewards[-response_length:] if response_length > 0 else []
    if len(token_rewards) < response_length:
        token_rewards.extend([0.0] * (response_length - len(token_rewards)))
    sample_metadata["token_rewards"] = token_rewards[:response_length]

    if sample.loss_mask is None:
        sample.loss_mask = [0] * response_length
    elif len(sample.loss_mask) > response_length:
        sample.loss_mask = sample.loss_mask[-response_length:] if response_length > 0 else []
    elif len(sample.loss_mask) < response_length:
        sample.loss_mask.extend([0] * (response_length - len(sample.loss_mask)))

    if sample.rollout_log_probs is not None:
        if len(sample.rollout_log_probs) > response_length:
            sample.rollout_log_probs = sample.rollout_log_probs[-response_length:] if response_length > 0 else []
        elif len(sample.rollout_log_probs) < response_length:
            sample.rollout_log_probs.extend([0.0] * (response_length - len(sample.rollout_log_probs)))


def turn_params(args: Any, spec: AgentEnvSpec, sampling_params: dict, remaining: int | None) -> dict:
    params = copy.deepcopy(sampling_params)
    max_tokens = params.get("max_new_tokens")
    if max_tokens is None:
        max_tokens = arg(args, "rollout_max_response_len", None)
    if max_tokens is None:
        max_tokens = spec.default_response_max_tokens
    max_tokens = int(max_tokens)
    if remaining is not None:
        max_tokens = max(0, min(max_tokens, remaining))
    params["max_new_tokens"] = max_tokens
    stop = cfg_path(args, "generation.stop", None)
    if stop is not None:
        params["stop"] = stop
    # The token ledger needs SGLang's raw generated ids, including stop/special
    # boundary tokens when the engine emits them. Action parsers operate on a
    # cleaned decode view instead of mutating these training tokens.
    params["no_stop_trim"] = True
    params["skip_special_tokens"] = False
    return params


async def call_policy(
    args: Any,
    spec: AgentEnvSpec,
    sample: Sample,
    input_ids: list[int],
    sampling_params: dict,
) -> tuple[str, list[int], list[float], str]:
    from slime.rollout.sglang_rollout import get_model_url

    url = get_model_url(args, "actor", "/generate")
    headers = None
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": sample.session_id}
    payload = {"input_ids": input_ids, "sampling_params": sampling_params, "return_logprob": True}
    output = await asyncio.wait_for(
        post(url, payload, headers=headers),
        timeout=float(cfg_path(args, "timeouts.policy_s", 60.0)),
    )
    text = output.get("text", "")
    meta = output.get("meta_info", {})
    token_logprobs = meta.get("output_token_logprobs") or []
    output_ids = output.get("output_ids") or meta.get("output_ids") or []
    if output_ids and hasattr(output_ids, "tolist"):
        output_ids = output_ids.tolist()
    if output_ids and isinstance(output_ids[0], list):
        output_ids = output_ids[0]
    finish_type = meta.get("finish_reason", {}).get("type", "stop")
    if token_logprobs:
        return text, [item[1] for item in token_logprobs], [item[0] for item in token_logprobs], finish_type
    if output_ids:
        return text, [int(item) for item in output_ids], [], finish_type
    return text, [], [], finish_type


def env_server_url(args: Any, spec: AgentEnvSpec) -> str:
    return str(os.environ.get(spec.env_url_envvar) or arg(args, spec.env_url_arg, None) or spec.default_env_url).rstrip("/")


async def post_env(args: Any, spec: AgentEnvSpec, endpoint: str, payload: dict, max_retries: int = 60) -> dict:
    timeout_s = float(cfg_path(args, "timeouts.env_request_s", 30.0))
    return await asyncio.wait_for(
        post(f"{env_server_url(args, spec)}{endpoint}", payload, max_retries=max_retries),
        timeout=timeout_s,
    )


def lease_request_id(sample: Sample) -> str:
    # Stable across HTTP retries for this in-memory rollout call only.
    return f"sample-{sample.index}-group-{sample.group_index}-obj-{id(sample)}"


async def allocate_env(args: Any, spec: AgentEnvSpec, sample: Sample) -> dict:
    payload = task_payload(sample, spec)
    split = payload.get("split") or cfg_path(args, "task.split", spec.default_split)
    index = task_index(sample)
    logger.debug(
        "%s allocate split=%s task_index=%s url=%s envvar=%s arg_%s=%s",
        spec.name,
        split,
        index,
        env_server_url(args, spec),
        os.environ.get(spec.env_url_envvar),
        spec.env_url_arg,
        arg(args, spec.env_url_arg, None),
    )
    return await post_env(
        args,
        spec,
        "/allocate",
        {**payload, "split": split, "task_key": task_key(sample, spec), "request_id": lease_request_id(sample)},
    )


async def reset_env(args: Any, spec: AgentEnvSpec, sample: Sample, lease_id: str, extra_payload: dict | None = None) -> dict:
    payload = {
        **task_payload(sample, spec),
        "lease_id": lease_id,
        "seed": metadata(sample).get("seed", sample.group_index if sample.group_index is not None else sample.index),
    }
    if extra_payload:
        payload.update(extra_payload)
    return await post_env(args, spec, "/reset", payload)


async def step_env(args: Any, spec: AgentEnvSpec, lease_id: str, action: Any) -> dict:
    return await post_env(args, spec, "/step", {"lease_id": lease_id, "action": action})


async def evaluate_env(args: Any, spec: AgentEnvSpec, lease_id: str | None) -> dict:
    if not lease_id:
        return {}
    return await post_env(args, spec, "/evaluate", {"lease_id": lease_id}, max_retries=3)


async def close_env(args: Any, spec: AgentEnvSpec, lease_id: str | None) -> None:
    if not lease_id:
        return
    try:
        await post_env(args, spec, "/close", {"lease_id": lease_id}, max_retries=3)
    except Exception:
        logger.debug("Failed to close %s server lease %s", spec.name, lease_id, exc_info=True)


def outcome_reward(args: Any, spec: AgentEnvSpec, success: bool, score: float) -> float:
    reward = float(cfg_path(args, "reward.outcome", 10.0))
    source = cfg_path(args, "reward.source", spec.default_reward_source)
    if source == "score":
        return float(score) * reward
    return reward if success else 0.0


def format_reward(args: Any, valid: bool) -> float:
    if valid:
        return float(cfg_path(args, "reward.format.valid", 0.0))
    return float(cfg_path(args, "reward.format.invalid", -0.1))


async def generate_agent_rollout(
    args: Any,
    sample: Sample,
    sampling_params: dict,
    *,
    spec: AgentEnvSpec,
    reset_payload: dict | None = None,
) -> Sample:
    assert not arg(args, "partial_rollout", False), f"{spec.name} rollout does not support partial rollout yet."

    tok = tokenizer(args)
    actions: list[str] = []
    final_score = 0.0
    success = False
    index = task_index(sample)
    lease_id = None
    sample_metadata = metadata(sample)
    split = sample_metadata.get("split") or cfg_path(args, "task.split", spec.default_split)
    phase = "start"
    if sample.status == Sample.Status.ABORTED:
        sample.status = Sample.Status.PENDING
    sample.remove_sample = False

    try:
        phase = "allocate"
        lease = await allocate_env(args, spec, sample)
        lease_id = lease["lease_id"]
        phase = "reset"
        reset = await reset_env(args, spec, sample, lease_id, reset_payload)
        observation = str(reset.get("observation", ""))
        info = reset.get("info") or {}
        split = reset.get("split") or split

        prompt = spec.initial_prompt(args, sample, observation, info)
        mode = interaction_mode(args, spec)
        tools = tool_schemas_for_rollout(args, spec, info)
        enable_thinking = enable_thinking_for_rollout(args, spec)
        messages = initial_messages_for_rollout(args, spec, sample, observation, info, prompt)
        ledger = AgentTokenLedger(
            args=args,
            spec=spec,
            tok=tok,
            messages=messages,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        trace_turns = _sample_case_dump_enabled(args)
        sample.prompt = ledger.prompt_text
        sample.tokens = list(ledger.tokens)
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = None
        sample_metadata["token_rewards"] = ledger.token_rewards
        sample_metadata["actions"] = actions
        sample_metadata["format_errors"] = 0
        sample_metadata["interaction_mode"] = mode
        if trace_turns:
            sample_metadata["messages"] = ledger.messages
        if tools:
            sample_metadata["tool_schemas"] = tools

        max_turns = int(cfg_path(args, "max_turns", spec.default_max_turns))
        for _turn in range(max_turns):
            turn_trace: dict[str, Any] | None = None
            if trace_turns:
                turn_trace = {
                    "turn": _turn,
                    "token_count_before_generation": len(ledger.tokens),
                }
            remaining = ledger.remaining_context()
            if remaining is not None and remaining <= 0:
                sample.status = Sample.Status.TRUNCATED
                sample_metadata["truncated_reason"] = "context_limit"
                break

            params = turn_params(args, spec, sampling_params, remaining)
            if params["max_new_tokens"] <= 0:
                sample.status = Sample.Status.TRUNCATED
                sample_metadata["truncated_reason"] = "context_limit"
                break

            phase = "policy"
            response_text, _response_token_ids, _response_log_probs, finish_type = await call_policy(
                args,
                spec,
                sample,
                ledger.tokens,
                params,
            )
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break
            if not _response_token_ids and response_text:
                if bool(arg(args, "allow_policy_retokenize_fallback", False)):
                    _response_token_ids = tok(response_text, add_special_tokens=False)["input_ids"]
                    sample_metadata["policy_retokenize_fallbacks"] = int(
                        sample_metadata.get("policy_retokenize_fallbacks", 0)
                    ) + 1
                else:
                    sample.status = Sample.Status.FAILED
                    sample.remove_sample = True
                    sample_metadata["error"] = "SGLang did not return output token ids for non-empty policy text"
                    break

            decoded_raw = decode_token_ids(tok, _response_token_ids, skip_special_tokens=False)
            raw_response_text = response_text or decoded_raw
            text_view = parse_policy_text_view(args, spec, raw_response_text)
            parser_text = text_view.content_text
            if response_text and decoded_raw and response_text != decoded_raw:
                sample_metadata.setdefault("policy_text_token_mismatches", 0)
                sample_metadata["policy_text_token_mismatches"] = int(sample_metadata["policy_text_token_mismatches"]) + 1
            if text_view.reasoning_parser:
                sample_metadata["reasoning_parser"] = text_view.reasoning_parser
                if not text_view.reasoning_ok:
                    sample_metadata["reasoning_parse_errors"] = int(sample_metadata.get("reasoning_parse_errors", 0)) + 1
                    sample_metadata.setdefault("reasoning_parse_error", text_view.reasoning_error)
            if turn_trace is not None:
                turn_trace.update(
                    {
                        "finish_type": finish_type,
                        "response_text": response_text,
                        "decoded_raw_response": decoded_raw,
                        "parser_text": parser_text,
                        "reasoning_parser": text_view.reasoning_parser,
                        "reasoning_ok": text_view.reasoning_ok,
                        "reasoning_error": text_view.reasoning_error,
                        "reasoning_text": text_view.reasoning_text,
                        "response_token_count": len(_response_token_ids),
                    }
                )

            if mode == "tool_call":
                parser_name = infer_tool_call_parser_name(tok)
                sample_metadata["tool_call_parser"] = parser_name
                action, format_valid, parse_mode = parse_standard_tool_call(parser_text, tools, parser_name)
                if not format_valid and parse_mode == "no_standard_tool_call" and spec.allow_assistant_message:
                    content = visible_assistant_text(parser_text)
                    if content:
                        action = {"type": "assistant_message", "content": content}
                        format_valid = True
                        parse_mode = "assistant_message"
            else:
                parser = action_parser(args, spec)
                action, format_valid, parse_mode = parser(parser_text)
            sample_metadata.setdefault("action_parse_modes", []).append(parse_mode)
            if not format_valid:
                sample_metadata["format_errors"] = int(sample_metadata.get("format_errors", 0)) + 1

            if finish_type == "length":
                sample_metadata["max_response_tokens_hits"] = int(
                    sample_metadata.get("max_response_tokens_hits", 0)
                ) + 1

            action = spec.choose_action(args, action, spec.info_actions(info), sample)
            actions.append(action)
            if turn_trace is not None:
                turn_trace.update({"parse_mode": parse_mode, "format_valid": bool(format_valid), "action": action})

            if mode == "tool_call":
                assistant_message = policy_message_for_generation(
                    mode,
                    parser_text,
                    action,
                )
            else:
                assistant_text = parser_text if text_view.reasoning_parser else (parser_text or raw_response_text)
                assistant_message = {"role": "assistant", "content": visible_assistant_text(assistant_text)}
            if turn_trace is not None:
                turn_trace["assistant_message"] = assistant_message

            ledger.append_assistant_generation(
                turn=_turn,
                message=assistant_message,
                token_ids=_response_token_ids,
                text=raw_response_text,
                log_probs=_response_log_probs,
            )
            if not ledger.add_reward_to_last_token(format_reward(args, format_valid)):
                sample_metadata["unassigned_token_reward"] = float(
                    sample_metadata.get("unassigned_token_reward", 0.0)
                ) + float(format_reward(args, format_valid))
            if turn_trace is not None:
                turn_trace["token_count_after_assistant"] = len(ledger.tokens)

            phase = "step"
            step = await step_env(args, spec, lease_id, action)
            observation = str(step.get("observation", ""))
            final_score = float(step.get("score", 0.0) or 0.0)
            done = bool(step.get("done", False))
            info = step.get("info") or {}
            success = spec.success(info, final_score)
            if turn_trace is not None:
                turn_trace.update(
                    {
                        "done": done,
                        "score": final_score,
                        "success": success,
                        "env_step": step,
                        "observation": observation,
                    }
                )
            env_text = spec.observation_text(args, observation, info)
            env_messages = environment_messages_from_step(
                mode=mode,
                action=action,
                assistant_message=assistant_message,
                observation=observation,
                info=info,
                done=done,
                env_text=env_text,
            )
            if turn_trace is not None:
                turn_trace["env_messages"] = env_messages

            if env_messages:
                add_next_generation_prompt = not done
                fits_final, final_token_count = ledger.can_append_environment_messages(
                    env_messages,
                    add_generation_prompt=add_next_generation_prompt,
                )
                if fits_final:
                    _delta_len, delta_mode = ledger.append_environment_messages(
                        turn=_turn,
                        messages=env_messages,
                        add_generation_prompt=add_next_generation_prompt,
                    )
                    sample_metadata.setdefault("message_delta_modes", []).append(delta_mode)
                elif not done:
                    sample.status = Sample.Status.TRUNCATED
                    sample_metadata["truncated_reason"] = "context_limit_after_observation"
                    sample_metadata["context_limit_token_count"] = final_token_count
                    if turn_trace is not None:
                        turn_trace["context_limit_token_count"] = final_token_count
                        turn_trace["token_count_after_environment"] = len(ledger.tokens)
                        sample_metadata.setdefault("turns", []).append(turn_trace)
                    break
            if turn_trace is not None:
                turn_trace["token_count_after_environment"] = len(ledger.tokens)
                sample_metadata.setdefault("turns", []).append(turn_trace)

            if done:
                sample.status = Sample.Status.COMPLETED
                break
        else:
            sample.status = Sample.Status.TRUNCATED
            sample_metadata["truncated_reason"] = "max_turns"

        env_reward = outcome_reward(args, spec, success, final_score)
        if not ledger.add_reward_to_last_token(env_reward):
            sample_metadata["unassigned_token_reward"] = float(sample_metadata.get("unassigned_token_reward", 0.0)) + float(
                env_reward
            )
        env_meta = spec.env_metadata(reset, index, split, lease_id)
        env_meta.setdefault("server_url", env_server_url(args, spec))
        ledger.materialize(sample, sample_metadata, include_trace=trace_turns)
        sample_metadata["token_audit"] = ledger.audit()
        sample_metadata.update(
            {
                "turn_count": len(actions),
                "format_ok": int(sample_metadata.get("format_errors", 0)) == 0,
                "env_score": final_score,
                "env_success": success,
                "env_reward": env_reward,
                "interaction_mode": sample_metadata["interaction_mode"],
                spec.name: env_meta,
            }
        )

        if sample.status == Sample.Status.ABORTED:
            sample.reward = 0.0
        elif arg(args, "use_opd", False) and arg(args, "opd_type") == "sglang":
            sample.reward = None
        else:
            sample.reward = float(sum(sample_metadata.get("token_rewards", []))) + float(
                sample_metadata.get("unassigned_token_reward", 0.0)
            )
        dump_completed_sample_case(args, spec, sample, tok)
        ensure_rollout_shapes(args, sample, spec)
        return sample
    except Exception as exc:
        if phase in {"allocate", "reset", "step"} and _is_env_infra_exception(exc):
            return _record_rollout_infra_failure(args, spec, sample, phase, exc, tok)
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        ensure_rollout_shapes(args, sample, spec)
        metadata(sample).setdefault("error", repr(exc))
        dump_completed_sample_case(args, spec, sample, tok)
        logger.exception("%s rollout failed", spec.name)
        return sample
    finally:
        if lease_id is not None:
            try:
                eval_payload = await evaluate_env(args, spec, lease_id)
                if eval_payload:
                    metadata(sample).setdefault("env_evaluate", eval_payload)
            except Exception:
                logger.debug("Failed to evaluate %s lease %s before close", spec.name, lease_id, exc_info=True)
        await close_env(args, spec, lease_id)
