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

from examples.agent_env.dump import int_runtime_env, reserve_dump_slot, runtime_env, safe_filename_part, sample_dump_step_label

logger = logging.getLogger(__name__)
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
    env_url_arg: str
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
    return runtime_env(args, name, default)


def _sample_case_dump_enabled(args: Any) -> bool:
    limits = (
        _case_dump_limit(args, "samples"),
        _case_dump_limit(args, "discarded"),
        _case_dump_limit(args, "format_errors"),
    )
    return any(limit > 0 for limit in limits) and bool(_runtime_env(args, "RUN_ROOT"))


def _int_runtime_env(args: Any, name: str, default: str = "0") -> int:
    return int_runtime_env(args, name, default)


def _case_dump_buckets(sample: Sample) -> list[str]:
    sample_metadata = sample.metadata or {}
    buckets: list[str] = []
    if bool(getattr(sample, "remove_sample", False)) or bool(sample_metadata.get("discard_sample", False)):
        buckets.append("discarded")
    else:
        buckets.append("samples")
    if int(sample_metadata.get("format_errors", 0) or 0) > 0:
        buckets.append("format_errors")
    return buckets


def _case_dump_file_stem(bucket: str) -> str:
    stems = {
        "samples": "sample",
        "discarded": "discarded",
        "format_errors": "format_error",
    }
    return stems.get(bucket, bucket.rstrip("s") or "case")


def _case_dump_limit(args: Any, bucket: str) -> int:
    fallback = str(_int_runtime_env(args, "AGENT_ENV_ROLLOUT_DUMP_N", "0"))
    if bucket == "discarded":
        return _int_runtime_env(args, "AGENT_ENV_ROLLOUT_DUMP_DISCARD_N", fallback)
    if bucket == "format_errors":
        return _int_runtime_env(args, "AGENT_ENV_ROLLOUT_DUMP_FORMAT_N", "0")
    return _int_runtime_env(args, "AGENT_ENV_ROLLOUT_DUMP_N", "0")


def _case_dump_total_limit(args: Any) -> int:
    return _int_runtime_env(args, "AGENT_ENV_ROLLOUT_DUMP_TOTAL_N", "0")


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
    exc_text = repr(exc).lower()
    reason_kind = "timeout" if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timed out" in exc_text else "failure"
    discard_prefix = "policy" if phase == "policy" else f"env_{phase}"
    discard_reason = f"{discard_prefix}_{reason_kind}"
    sample_metadata["infra_error"] = {
        "phase": phase,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    sample_metadata["discard_sample"] = True
    sample_metadata["discard_reason"] = discard_reason
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


def _env_discard_reason(info: dict[str, Any]) -> str:
    if bool(info.get("discard_sample", False)):
        return str(info.get("discard_reason") or "env_discard_sample")
    if bool(info.get("user_tool_round_limit", False)):
        return "user_tool_round_limit"
    return ""


def dump_completed_sample_case(args: Any, spec: AgentEnvSpec, sample: Sample, tok: Any | None = None) -> None:
    run_root = _runtime_env(args, "RUN_ROOT")
    if not run_root:
        return
    sample_metadata = sample.metadata or {}
    trace_mode = _dump_trace_mode(args)
    dump_step = sample_dump_step_label(sample)
    safe_dump_step = safe_filename_part(dump_step)
    for bucket in _case_dump_buckets(sample):
        limit = _case_dump_limit(args, bucket)
        if limit <= 0:
            continue
        total_limit = _case_dump_total_limit(args)
        slot = reserve_dump_slot(
            namespace=f"rollout_cases:{spec.name}",
            stage=bucket,
            dump_step=dump_step,
            per_step_limit=limit,
            total_limit=total_limit,
        )
        if slot is None:
            continue
        count, total_count = slot
        output_dir = Path(run_root) / "rollout_cases" / spec.name / bucket
        output_dir.mkdir(parents=True, exist_ok=True)
        path = (
            output_dir
            / f"step_{safe_dump_step}_{_case_dump_file_stem(bucket)}_{count:04d}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
        )
        record = {
            "sample_index": getattr(sample, "index", None),
            "dump_step": dump_step,
            "dump_bucket": bucket,
            "dump_index_in_step": count,
            "dump_index_total": total_count,
            "status": getattr(getattr(sample, "status", None), "name", str(getattr(sample, "status", ""))),
            "remove_sample": bool(getattr(sample, "remove_sample", False)),
            "discard_sample": bool(sample_metadata.get("discard_sample", False)),
            "discard_reason": sample_metadata.get("discard_reason"),
            "reward": getattr(sample, "reward", None),
            "response_length": getattr(sample, "response_length", None),
            "effective_response_length": getattr(sample, "effective_response_length", None),
            "total_token_length": len(getattr(sample, "tokens", []) or []),
            "response": getattr(sample, "response", None),
            "turn_count": sample_metadata.get("turn_count"),
            "format_errors": sample_metadata.get("format_errors"),
            "format_checks": sample_metadata.get("format_checks"),
            "max_response_tokens_hits": sample_metadata.get("max_response_tokens_hits"),
            "truncated_reason": sample_metadata.get("truncated_reason"),
            "env_score": sample_metadata.get("env_score"),
            "env_success": sample_metadata.get("env_success"),
            "env_reward": sample_metadata.get("env_reward"),
            "user_model_call_count": sample_metadata.get("user_model_call_count"),
            "user_model_usage_totals": sample_metadata.get("user_model_usage_totals"),
            "rm_impl": sample_metadata.get("rm_impl"),
            "rm_reward": sample_metadata.get("rm_reward"),
            "rm_reward_for_train": sample_metadata.get("rm_reward_for_train"),
            "reward_components": sample_metadata.get("reward_components"),
            "raw_reward": sample_metadata.get("raw_reward"),
            "judge_score": sample_metadata.get("judge_score"),
            "judge_reason": sample_metadata.get("judge_reason"),
            "requested_task": sample_metadata.get("requested_task"),
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
    for key in (
        "env",
        "data_source",
        "domain",
        "task_set",
        "dataset_name",
        "task_id",
        "seed",
        "task_ref",
        "task",
        "query",
        "task_prompt",
        "instruction",
        "question",
        "task_question",
        "instruction_text",
    ):
        if key in sample_metadata and sample_metadata[key] is not None:
            payload[key] = sample_metadata[key]
    return payload


def task_key(sample: Sample, spec: AgentEnvSpec) -> str:
    payload = task_payload(sample, spec)
    parts = []
    for key in ("env", "domain", "task_set", "dataset_name", "split", "task_id", "task_index"):
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


def _normalize_task_metadata_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _record_env_metadata_value(
    sample_metadata: dict[str, Any],
    *,
    key: str,
    value: Any,
    spec: AgentEnvSpec,
    validate: bool,
) -> None:
    if value in (None, "", []):
        return
    existing = sample_metadata.get(key)
    if validate and existing not in (None, "", []):
        if _normalize_task_metadata_value(existing) != _normalize_task_metadata_value(value):
            raise ValueError(
                f"{spec.name} env returned metadata mismatch for {key}: "
                f"sample={existing!r} env={value!r}. "
                "Prompt data and env reset must describe the same task."
            )
    sample_metadata[key] = value


def record_env_metadata(sample_metadata: dict[str, Any], spec: AgentEnvSpec, env_meta: dict[str, Any]) -> None:
    sample_metadata[spec.name] = env_meta
    for key in (
        "task_id",
        "task_ref",
        "domain",
        "data_source",
        "task_set",
        "dataset_name",
        "query",
        "task_prompt",
        "instruction",
        "question",
        "task_question",
        "instruction_text",
    ):
        value = env_meta.get(key)
        _record_env_metadata_value(
            sample_metadata,
            key=key,
            value=value,
            spec=spec,
            validate=key in {"task_id", "query", "task_prompt", "instruction", "question", "task_question", "instruction_text"},
        )


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
    """Canonical message shape for tokenizer chat templates and trajectory match.

    OpenAI tool-call ids are wire-only correlation fields. They should stay in
    env/server HTTP traffic, but not in training history: clients may rewrite
    them, and chat templates do not need them.
    """
    normalized = copy.deepcopy(messages)
    for message in normalized:
        content = message.get("content")
        if content not in (None, "") and not isinstance(content, str):
            message["content"] = json.dumps(content, ensure_ascii=False, sort_keys=True)
        if message.get("role") == "tool":
            message.pop("tool_call_id", None)
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call.pop("id", None)
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
        if updates[0].get("role") == "assistant":
            raise ValueError("env message_updates must contain only environment-side messages")
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

    Agent-env trains one complete environment episode as one Slime sample. This
    ledger keeps token accounting local to that contract instead of inheriting
    generic trajectory forking semantics from core Slime adapters.
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
        log_probs: list[float] | None = None,
    ) -> None:
        if not tokens:
            return
        mask = [int(loss_mask_value)] * len(tokens)
        self.tokens.extend(tokens)
        self.response_tokens.extend(tokens)
        self.loss_mask.extend(mask)
        self.token_rewards.extend([0.0] * len(tokens))
        if self.rollout_log_probs is not None:
            if log_probs is None:
                segment_log_probs = [0.0] * len(tokens)
            else:
                segment_log_probs = [float(value) for value in log_probs[: len(tokens)]]
                if len(segment_log_probs) < len(tokens):
                    segment_log_probs.extend([0.0] * (len(tokens) - len(segment_log_probs)))
            self.rollout_log_probs.extend(segment_log_probs)
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
        if self.rollout_log_probs is None:
            self.rollout_log_probs = [0.0] * len(self.response_tokens)
        segment_log_probs = list(log_probs or [])
        self._append_segment(
            kind="assistant",
            role="assistant",
            turn=turn,
            tokens=list(token_ids),
            loss_mask_value=1,
            text=text,
            log_probs=segment_log_probs,
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

    def materialize(self, sample: Sample, sample_metadata: dict[str, Any], *, include_trace: bool = False) -> None:
        sample.tokens = list(self.tokens)
        sample.response_length = self.response_length
        sample.loss_mask = list(self.loss_mask)
        sample.response = "".join(self.assistant_response_texts)
        sample.rollout_log_probs = list(self.rollout_log_probs) if self.rollout_log_probs is not None else None
        sample_metadata["token_rewards"] = list(self.token_rewards)
        if include_trace:
            sample_metadata["messages"] = copy.deepcopy(self.messages)
            sample_metadata["token_segments"] = [segment.__dict__.copy() for segment in self.segments]

    def audit(self) -> dict[str, Any]:
        segment_tokens = sum(segment.token_count for segment in self.segments)
        segment_loss_mask = sum(segment.loss_mask_sum for segment in self.segments)
        structural_ok = segment_tokens == len(self.tokens) and segment_loss_mask == sum(self.loss_mask)
        try:
            full_ids = apply_chat_template_ids(
                self.tok,
                self.messages,
                tools=self.tools or None,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
            )
        except Exception as exc:
            return {
                "ok": structural_ok,
                "structural_ok": structural_ok,
                "message_replay_ok": False,
                "message_replay_error": repr(exc),
                "ledger_tokens": len(self.tokens),
                "segment_tokens": segment_tokens,
            }
        common = token_prefix_length(full_ids, self.tokens)
        message_replay_ok = full_ids == self.tokens
        return {
            "ok": structural_ok,
            "structural_ok": structural_ok,
            "message_replay_ok": message_replay_ok,
            "message_replay_expected_to_differ": not message_replay_ok
            and any(segment.kind == "assistant" for segment in self.segments),
            "ledger_tokens": len(self.tokens),
            "segment_tokens": segment_tokens,
            "full_template_tokens": len(full_ids),
            "common_prefix_tokens": common,
            "first_diff": None if message_replay_ok else common,
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
    # Keep text-action envs executable, but preserve the invalid parse signal.
    return "look", False, "fallback"


def parse_tool_call_action(response_text: str) -> tuple[Any, bool, str]:
    tool_action, tool_valid, tool_mode = _parse_tool_call(response_text)
    if tool_action is not None or tool_mode != "no_tool_call":
        return tool_action or "", tool_valid, tool_mode
    return "", False, "no_tool_call"


def parse_standard_tool_calls(response_text: str, tools: list[dict[str, Any]], parser_name: str = "qwen") -> tuple[list[dict[str, Any]], bool, str]:
    try:
        from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
        from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    except Exception:
        from sglang.srt.managers.io_struct import Function as SglFunction
        from sglang.srt.managers.io_struct import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    sgl_tools = [
        SglTool(type=tool["type"], function=SglFunction(**tool["function"]))
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    known_tool_names = {
        str(tool.get("function", {}).get("name") or "")
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    known_tool_names.discard("")
    parser = FunctionCallParser(sgl_tools, parser_name)

    try:
        if not parser.has_tool_call(response_text):
            return [], False, "no_standard_tool_call"
    except Exception as exc:
        logger.debug("standard tool-call intent parser failed", exc_info=True)
        return [], False, f"tool_intent_parser_error:{type(exc).__name__}"

    previous_forward_unknown = os.environ.get("SGLANG_FORWARD_UNKNOWN_TOOLS")
    os.environ["SGLANG_FORWARD_UNKNOWN_TOOLS"] = "1"
    try:
        try:
            normal_text, calls = parser.parse_non_stream(response_text)
        except Exception as exc:
            logger.debug("standard tool-call parser failed", exc_info=True)
            return [], False, f"malformed_standard_tool_call:{type(exc).__name__}"
    finally:
        if previous_forward_unknown is None:
            os.environ.pop("SGLANG_FORWARD_UNKNOWN_TOOLS", None)
        else:
            os.environ["SGLANG_FORWARD_UNKNOWN_TOOLS"] = previous_forward_unknown

    if not calls:
        return [], False, "malformed_standard_tool_call_no_calls"

    actions = []
    unknown_names = []
    content = strip_chat_boundary_tokens(str(normal_text or ""))
    for call in calls:
        name = str(getattr(call, "name", "") or "")
        parameters = getattr(call, "parameters", {}) or {}
        if isinstance(parameters, str):
            parsed_parameters = _parse_json_object(parameters)
            parameters = parsed_parameters if parsed_parameters is not None else {}
        if not isinstance(parameters, dict):
            parameters = {}
        if not name:
            return [], False, "malformed_standard_tool_call_empty_name"
        actions.append(
            {
                "type": "tool_call",
                "name": name,
                "arguments": parameters,
                "content": content,
            }
        )
        if name not in known_tool_names:
            unknown_names.append(name)
    if unknown_names:
        return actions, False, "unknown_standard_tool_call:" + ",".join(unknown_names)
    mode = f"standard_tool_call:{parser_name}" if len(actions) == 1 else f"standard_tool_calls:{parser_name}:n={len(actions)}"
    return actions, True, mode


def parse_standard_tool_call(response_text: str, tools: list[dict[str, Any]], parser_name: str = "qwen") -> tuple[Any, bool, str]:
    actions, valid, mode = parse_standard_tool_calls(response_text, tools, parser_name)
    return (actions[0] if actions else ""), valid, mode


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
    timeout_s = float(cfg_path(args, "timeouts.policy_s", 60.0))

    async def direct_post_model() -> dict:
        import httpx

        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), trust_env=False) as client:
            for attempt in range(3):
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:
                    last_exc = exc
                    if attempt >= 2:
                        raise
                    logger.warning(
                        "%s policy generate request failed, retrying attempt=%d url=%s error=%s",
                        spec.name,
                        attempt + 1,
                        url,
                        exc,
                    )
                    await asyncio.sleep(0.5)
        assert last_exc is not None
        raise last_exc

    # The env-server policy gateway serves requests from HTTP worker threads.
    # Using Slime's distributed post helper there can wait inside Ray without
    # actually issuing SGLang /generate requests. Direct HTTP keeps the gateway
    # independent from Ray's internal async dispatch while still targeting the
    # same SGLang router URL.
    output = await asyncio.wait_for(
        direct_post_model(),
        timeout=timeout_s,
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
    value = arg(args, spec.env_url_arg, None)
    if value is None or not str(value).strip():
        cli_name = spec.env_url_arg.replace("_", "-")
        raise RuntimeError(f"{spec.name} rollout requires --{cli_name}; URL env vars are not propagated to rollout actors.")
    return str(value).rstrip("/")


async def post_env(args: Any, spec: AgentEnvSpec, endpoint: str, payload: dict, max_retries: int = 60) -> dict:
    timeout_s = float(cfg_path(args, "timeouts.env_request_s", 30.0))
    from slime.utils import http_utils

    if getattr(http_utils, "_http_client", None) is None:
        import httpx

        url = f"{env_server_url(args, spec)}{endpoint}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), trust_env=False) as client:
            last_exc: Exception | None = None
            for attempt in range(max(1, int(max_retries))):
                try:
                    response = await client.post(url, json=payload or {})
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:
                    last_exc = exc
                    if attempt + 1 >= max(1, int(max_retries)):
                        raise
                    await asyncio.sleep(1)
            assert last_exc is not None
            raise last_exc
    return await asyncio.wait_for(
        post(f"{env_server_url(args, spec)}{endpoint}", payload, max_retries=max_retries),
        timeout=timeout_s,
    )


def lease_request_id(sample: Sample) -> str:
    # Stable across HTTP retries for this in-memory rollout call only.
    return f"sample-{sample.index}-group-{sample.group_index}-obj-{id(sample)}"


def outcome_reward(args: Any, spec: AgentEnvSpec, success: bool, score: float) -> float:
    reward = float(cfg_path(args, "reward.outcome", 10.0))
    source = str(cfg_path(args, "reward.source", spec.default_reward_source) or "").strip().lower()
    if source == "score":
        return float(score) * reward
    if source in {"success", "won"}:
        return reward if success else 0.0
    raise ValueError(f"Unsupported reward.source={source!r}; expected score, success, or won")


def format_reward(args: Any, valid: bool) -> float:
    if valid:
        return float(cfg_path(args, "reward.format.valid", 0.0))
    return float(cfg_path(args, "reward.format.invalid", -0.1))


def record_format_check(sample_metadata: dict[str, Any], *, turn: int, valid: bool, parse_mode: str) -> None:
    sample_metadata.setdefault("format_checks", []).append(
        {
            "turn": int(turn),
            "valid": bool(valid),
            "parse_mode": str(parse_mode),
        }
    )
    if not valid:
        sample_metadata["format_errors"] = int(sample_metadata.get("format_errors", 0) or 0) + 1


def format_reward_adjustment(args: Any, sample: Sample) -> float:
    sample_metadata = sample.metadata or {}
    checks = sample_metadata.get("format_checks")
    if isinstance(checks, list) and checks:
        invalid_count = sum(1 for item in checks if isinstance(item, dict) and not bool(item.get("valid", False)))
        valid_count = sum(1 for item in checks if isinstance(item, dict) and bool(item.get("valid", False)))
    else:
        invalid_count = int(sample_metadata.get("format_errors", 0) or 0)
        turn_count = int(sample_metadata.get("turn_count", 0) or 0)
        if turn_count <= 0:
            turn_count = len(sample_metadata.get("action_parse_modes") or [])
        valid_count = max(0, turn_count - invalid_count)
    return valid_count * format_reward(args, True) + invalid_count * format_reward(args, False)


def truncated_reward_adjustment(args: Any, sample: Sample) -> float:
    if sample.status != Sample.Status.TRUNCATED:
        return 0.0
    return float(cfg_path(args, "reward.truncated", 0.0))


def _is_hard_discard_sample(sample: Sample) -> bool:
    sample_metadata = metadata(sample)
    if sample.status == Sample.Status.ABORTED:
        return True
    if bool(getattr(sample, "remove_sample", False)) or bool(sample_metadata.get("discard_sample", False)):
        return True
    off_policy_mask = getattr(sample, "off_policy_loss_mask", None)
    off_policy_mask_sum = sum(int(value) for value in off_policy_mask) if off_policy_mask is not None else 0
    if sample.loss_mask is not None and sum(int(value) for value in sample.loss_mask) <= 0 and off_policy_mask_sum <= 0:
        return True
    return False


def _padding_sample_index(rollout_id: int, offset: int) -> int:
    return -((int(rollout_id) + 1) * 10_000_000 + offset + 1)


def _duplicate_valid_sample_for_padding(sample: Sample, *, new_index: int, group_index: int | None) -> Sample:
    duplicated = copy.deepcopy(sample)
    duplicated.index = new_index
    duplicated.rollout_id = None
    duplicated.group_index = group_index
    duplicated.remove_sample = False
    duplicated_metadata = metadata(duplicated)
    duplicated_metadata["glm_padding_duplicate"] = True
    duplicated_metadata["padded_from_sample_index"] = sample.index
    duplicated_metadata["padded_from_group_index"] = sample.group_index
    return duplicated


def _glm_style_pad_group(
    args: Any,
    group: list[Sample],
    *,
    rollout_id: int,
    padding_offset: int,
) -> tuple[list[Sample] | None, int, dict[str, float]]:
    group_size = int(arg(args, "n_samples_per_prompt", len(group)) or len(group))
    min_valid_fraction = float(_runtime_env(args, "AGENT_ENV_GLM_PADDING_MIN_VALID_FRACTION", "0.5"))
    valid = [sample for sample in group if not _is_hard_discard_sample(sample)]
    invalid_count = len(group) - len(valid)
    metrics = {
        "agent_env/glm_padding/group_seen": 1.0,
        "agent_env/glm_padding/invalid_samples": float(invalid_count),
    }

    if len(valid) >= group_size:
        metrics["agent_env/glm_padding/group_kept_full"] = 1.0
        return valid[:group_size], padding_offset, metrics

    if len(valid) <= group_size * min_valid_fraction:
        metrics["agent_env/glm_padding/group_dropped"] = 1.0
        metrics["agent_env/glm_padding/dropped_samples"] = float(len(group))
        return None, padding_offset, metrics

    padded = list(valid)
    group_index = valid[0].group_index if valid else (group[0].group_index if group else None)
    duplicate_i = 0
    while len(padded) < group_size:
        source = valid[duplicate_i % len(valid)]
        padded.append(
            _duplicate_valid_sample_for_padding(
                source,
                new_index=_padding_sample_index(rollout_id, padding_offset),
                group_index=group_index,
            )
        )
        padding_offset += 1
        duplicate_i += 1

    metrics["agent_env/glm_padding/group_padded"] = 1.0
    metrics["agent_env/glm_padding/padded_samples"] = float(group_size - len(valid))
    return padded, padding_offset, metrics


def glm_style_group_padding_filter_keep(args: Any, group: list[Sample]) -> tuple[bool, dict[str, float]]:
    """Return whether a completed group has enough valid samples for padding repair.

    This is an infra/padding pre-filter, not reward dynamic sampling. It keeps
    full-async collection from accepting a group that the later sample-filter
    hook must reject because too many samples were discarded by rollout errors.
    """
    group_size = int(arg(args, "n_samples_per_prompt", len(group)) or len(group))
    min_valid_fraction = float(_runtime_env(args, "AGENT_ENV_GLM_PADDING_MIN_VALID_FRACTION", "0.5"))
    valid_count = sum(1 for sample in group if not _is_hard_discard_sample(sample))
    invalid_count = len(group) - valid_count
    metrics = {
        "agent_env/glm_padding/prefilter_group_seen": 1.0,
        "agent_env/glm_padding/prefilter_invalid_samples": float(invalid_count),
    }
    keep = valid_count > group_size * min_valid_fraction
    if not keep:
        metrics["agent_env/glm_padding/prefilter_group_dropped"] = 1.0
        metrics["agent_env/glm_padding/prefilter_dropped_samples"] = float(len(group))
    return keep, metrics


def _add_metrics(metrics: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        metrics[key] = metrics.get(key, 0.0) + float(value)


def _group_sort_key(group: list[Sample]) -> int:
    for sample in group:
        if sample.index is not None and sample.index >= 0:
            return int(sample.index)
    for sample in group:
        if sample.index is not None:
            return int(sample.index)
    return 0


def glm_style_pad_groups_filter(args: Any, data: list[list[Sample]]) -> None:
    """Repair kept GRPO groups through Slime's stock rollout sample-filter hook."""
    from examples.agent_env.luffy import apply_luffy_teacher_sample, luffy_enabled

    inject_luffy = luffy_enabled(args)
    padding_offset = 0
    padded_groups = 0
    padded_samples = 0
    luffy_groups = 0
    for idx, group in enumerate(data):
        repaired, padding_offset, metrics = _glm_style_pad_group(
            args,
            group,
            rollout_id=_group_sort_key(group),
            padding_offset=padding_offset,
        )
        if repaired is None:
            raise RuntimeError(
                "GLM-style sample padding received a group with too few valid samples after dynamic filtering. "
                "Ensure dynamic-sampling-filter-path drops groups at or below "
                "AGENT_ENV_GLM_PADDING_MIN_VALID_FRACTION before rollout-sample-filter-path runs."
            )
        if inject_luffy:
            repaired = apply_luffy_teacher_sample(args, repaired)
            luffy_groups += 1
        data[idx] = repaired
        padded_groups += int(metrics.get("agent_env/glm_padding/group_padded", 0.0))
        padded_samples += int(metrics.get("agent_env/glm_padding/padded_samples", 0.0))
    if padded_groups or padded_samples:
        logger.info(
            "agent-env GLM-style sample filter padded groups=%d samples=%d",
            padded_groups,
            padded_samples,
        )
    if luffy_groups:
        logger.info("agent-env LUFFY injected teacher samples for groups=%d", luffy_groups)
