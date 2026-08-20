from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.appworld.reward_evidence import render_execution_evidence

logger = logging.getLogger(__name__)
_MISSING_REASONING_WARNED: set[str] = set()
_APPWORLD_OFFICIAL_PROMPT_MARKERS = (
    "I am your supervisor, and you are an AI Assistant whose job is to complete my day-to-day tasks fully autonomously.",
    "python REPL environment",
    "Let's start with the task",
)
_KNOWN_AGENT_ENVS = ("alfworld", "webshop", "tau2", "appworld", "openclaw")

TracePartSpec = bool | int


def _is_strip_spec(value: TracePartSpec) -> bool:
    return isinstance(value, bool) and value


def _truncate_limit(value: TracePartSpec) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _keeps_part(value: TracePartSpec) -> bool:
    return not _is_strip_spec(value)


def _middle_truncate_text(text: Any, max_chars: int, *, label: str = "text") -> str:
    value = str(text or "")
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    marker = f"\n...[truncated {omitted} chars from middle of {label}]...\n"
    if max_chars <= len(marker) + 8:
        return value[:max_chars]
    budget = max_chars - len(marker)
    omitted = len(value) - budget
    marker = f"\n...[truncated {omitted} chars from middle of {label}]...\n"
    budget = max_chars - len(marker)
    head = budget // 2
    tail = budget - head
    return value[:head] + marker + value[-tail:]


def _compress_part_text(text: Any, spec: TracePartSpec, *, label: str) -> str:
    value = str(text or "").strip()
    limit = _truncate_limit(spec)
    if limit is None:
        return value
    return _middle_truncate_text(value, limit, label=label).strip()


@dataclass(frozen=True)
class TraceCompressionOptions:
    strip_reasoning: TracePartSpec = True
    strip_tool_call: TracePartSpec = False
    strip_tool_response: TracePartSpec = False
    strip_assistant_response: TracePartSpec = True
    strip_system_prompt: TracePartSpec = True


def render_answer_for_reward(
    sample: Sample,
    *,
    final_answer: Any = "",
    answer_mode: str = "trace",
    options: TraceCompressionOptions | None = None,
    env_name: str | None = None,
    check_reasoning_presence: bool = False,
) -> str:
    options = options or TraceCompressionOptions()
    if check_reasoning_presence:
        _check_sample_reasoning_contract(sample, final_answer, options, context="reward_answer")
    mode = str(answer_mode or "trace").strip().lower()
    final_text = compress_trace_text(final_answer, options=options, strip_assistant_response=False)
    if mode == "final":
        return final_text
    if mode != "trace":
        raise ValueError(f"Unsupported reward answer_mode={mode!r}; expected 'final' or 'trace'")
    trace_text = render_trace_for_reward(sample, options=options, env_name=env_name)
    return _combine_final_and_trace(final_text, trace_text)


def render_trace_for_reward(
    sample: Sample,
    *,
    options: TraceCompressionOptions | None = None,
    env_name: str | None = None,
) -> str:
    options = options or TraceCompressionOptions()
    sample_metadata = _sample_payload(sample)
    resolved_env_name = _trace_env_name(sample_metadata, explicit_env_name=env_name)

    def finish(rendered: str) -> str:
        return _finalize_reward_trace_text(
            rendered,
            sample_metadata=sample_metadata,
            env_name=resolved_env_name,
        )

    explicit_trace = _first_metadata_value(sample_metadata, ("reward_trace", "judge_trace", "ropd_trace"))
    if explicit_trace not in (None, "", []):
        return finish(compress_trace_text(explicit_trace, options=options))
    turns = sample_metadata.get("turns")
    if isinstance(turns, list):
        rendered = _render_turns(
            turns,
            options=options,
            env_name=resolved_env_name,
            initial_observation=_initial_observation_from_sample_metadata(
                sample_metadata,
                options=options,
                env_name=resolved_env_name,
            ),
        )
        if rendered:
            return finish(rendered)
    segments = sample_metadata.get("token_segments")
    if isinstance(segments, list):
        rendered = _render_token_segments(
            segments,
            sample_metadata=sample_metadata,
            env_name=resolved_env_name,
            options=options,
        )
        if rendered:
            return finish(rendered)
    messages = sample_metadata.get("messages")
    if isinstance(messages, list):
        task_prompt = _task_prompt_from_sample_metadata(sample_metadata)
        appworld_start = (
            _appworld_episode_start_from_messages(messages, task_prompt)
            if resolved_env_name == "appworld"
            else None
        )
        rendered = _render_messages(
            messages,
            options=options,
            final_observation=_final_observation(sample_metadata),
            initial_observation=("Task:\n" + task_prompt) if appworld_start is not None and task_prompt else "",
            start_index=appworld_start,
        )
        if rendered:
            return finish(rendered)
    trace_value = _first_metadata_value(
        sample_metadata,
        ("trace", "trajectory", "rollout_trace", "student_trace"),
    )
    if trace_value not in (None, "", []):
        return finish(compress_trace_text(trace_value, options=options))
    sample_index = getattr(sample, "index", sample_metadata.get("sample_index", "unknown"))
    raise ValueError(
        "reward trace is missing structured trace fields for sample "
        f"{sample_index}; expected one of reward_trace/judge_trace/ropd_trace, turns, "
        "messages, token_segments, trace/trajectory/rollout_trace/student_trace"
    )


def render_teacher_trace_for_reward(
    trace: Any,
    *,
    options: TraceCompressionOptions | None = None,
    env_name: str | None = None,
    context_metadata: dict[str, Any] | None = None,
    check_reasoning_presence: bool = False,
    reasoning_context: str = "teacher_trace",
) -> str:
    """Render teacher trajectories through the same reward trace contract.

    Teacher data may come from structured rollout records or legacy text dumps.
    Do not let raw assistant prose/code-fence transcripts silently become the
    judge input when the student path is rendered as tool-call/tool-response
    trace.
    """

    options = options or TraceCompressionOptions()
    context_payload = dict(context_metadata or {})
    resolved_env_name = _trace_env_name(context_payload, explicit_env_name=env_name)

    def finish(rendered: str) -> str:
        return _finalize_reward_trace_text(
            rendered,
            sample_metadata=context_payload,
            env_name=resolved_env_name,
        )

    if check_reasoning_presence:
        _check_reasoning_contract(trace, options, context=reasoning_context)
    if trace in (None, "", []):
        return ""
    if isinstance(trace, Sample):
        return render_trace_for_reward(trace, options=options, env_name=env_name)
    structured = _render_structured_teacher_trace(
        trace,
        options=options,
        env_name=env_name,
        context_metadata=context_payload,
    )
    if structured:
        return finish(structured)
    text = _message_text(_apply_structured_reasoning_policy(trace, options.strip_reasoning))
    legacy = _render_legacy_teacher_text(text, options=options)
    if legacy:
        return finish(legacy)
    return finish(compress_trace_text(trace, options=options))


def compress_trace_text(
    text: Any,
    *,
    options: TraceCompressionOptions | None = None,
    strip_assistant_response: bool | None = None,
    check_reasoning_presence: bool = False,
    reasoning_context: str = "trace",
) -> str:
    options = options or TraceCompressionOptions()
    if check_reasoning_presence:
        _check_reasoning_contract(text, options, context=reasoning_context)
    raw_value = _apply_structured_reasoning_policy(text, options.strip_reasoning)
    value = _strip_chat_boundary_tokens(_message_text(raw_value))
    if _is_strip_spec(options.strip_system_prompt):
        value = _strip_system_prompt_text(value)
    elif (limit := _truncate_limit(options.strip_system_prompt)) is not None:
        value = _compress_system_prompt_text(value, limit)
    if _is_strip_spec(options.strip_reasoning):
        value = _strip_reasoning_text(value)
    elif (limit := _truncate_limit(options.strip_reasoning)) is not None:
        value = _compress_reasoning_text(value, limit)
    should_strip_assistant = options.strip_assistant_response if strip_assistant_response is None else strip_assistant_response
    if _is_strip_spec(should_strip_assistant):
        value = _strip_assistant_response_text(value)
    elif (limit := _truncate_limit(should_strip_assistant)) is not None:
        value = _compress_assistant_response_text(value, limit)
    if _is_strip_spec(options.strip_tool_call):
        value = _strip_tool_call_text(value)
    elif (limit := _truncate_limit(options.strip_tool_call)) is not None:
        value = _compress_tool_call_text(value, limit)
    if _is_strip_spec(options.strip_tool_response):
        value = _strip_tool_response_text(value)
    elif (limit := _truncate_limit(options.strip_tool_response)) is not None:
        value = _compress_tool_response_text(value, limit)
    return value.strip()


def _render_structured_teacher_trace(
    trace: Any,
    *,
    options: TraceCompressionOptions,
    env_name: str | None = None,
    context_metadata: dict[str, Any] | None = None,
) -> str:
    if isinstance(trace, dict):
        payload = trace
    elif isinstance(trace, (list, tuple)):
        normalized = list(trace)
        role_count = sum(1 for item in normalized if isinstance(item, dict) and item.get("role"))
        payload = {"messages": normalized} if role_count else {"turns": normalized}
    else:
        return ""
    if context_metadata:
        payload = {**context_metadata, **payload}
    try:
        return render_trace_for_reward(Sample(prompt="", metadata=payload), options=options, env_name=env_name)
    except ValueError:
        if _trace_env_name(payload, explicit_env_name=env_name) == "appworld":
            raise
        return ""


def _render_legacy_teacher_text(text: str, *, options: TraceCompressionOptions) -> str:
    value = _strip_chat_boundary_tokens(str(text or ""))
    if _is_strip_spec(options.strip_system_prompt):
        value = _strip_system_prompt_text(value)
    elif (limit := _truncate_limit(options.strip_system_prompt)) is not None:
        value = _compress_system_prompt_text(value, limit)
    if _is_strip_spec(options.strip_reasoning):
        value = _strip_reasoning_text(value)
    elif (limit := _truncate_limit(options.strip_reasoning)) is not None:
        value = _compress_reasoning_text(value, limit)
    if not value.strip():
        return ""
    if _looks_like_legacy_turn_transcript(value):
        rendered = _render_legacy_turn_transcript(value, options=options)
        if rendered:
            return rendered
    if _looks_like_reward_trace(value):
        return compress_trace_text(value, options=options)
    return ""


def _looks_like_legacy_turn_transcript(text: str) -> bool:
    return bool(re.search(r"(?im)^\s*Turn\s+\d+\s*:", str(text or "")))


def _looks_like_reward_trace(text: str) -> bool:
    return bool(
        re.search(
            r"(?im)^\s*(Initial observation|Step\s+\d+|Tool call|Tool response|Action|Observation(?: after action)?)\s*:",
            str(text or ""),
        )
    )


def _render_legacy_turn_transcript(text: str, *, options: TraceCompressionOptions) -> str:
    matches = list(re.finditer(r"(?im)^\s*Turn\s+\d+\s*:\s*", text))
    if not matches:
        return ""
    lines: list[str] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        initial = compress_trace_text(
            prefix,
            options=options,
            strip_assistant_response=False,
        )
        if initial:
            lines.append("Initial observation:\n" + initial)
    step = 1
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        pairs = _legacy_action_observation_pairs(body)
        if not pairs:
            continue
        for action, observation in pairs:
            action_text = _normalize_legacy_tool_call(action)
            parts = [f"Step {step}:"]
            if action_text and _keeps_part(options.strip_tool_call):
                action_text = _compress_part_text(
                    action_text,
                    options.strip_tool_call,
                    label="tool call",
                )
                if action_text:
                    parts.extend(["Tool call:", action_text])
            if observation and _keeps_part(options.strip_tool_response):
                observation = _compress_part_text(
                    observation,
                    options.strip_tool_response,
                    label="tool response",
                )
                if observation:
                    parts.extend(["Tool response:", observation])
            lines.append("\n".join(parts))
            step += 1
    return "\n\n".join(lines).strip()


def _legacy_action_observation_pairs(text: str) -> list[tuple[str, str]]:
    pairs = _explicit_action_observation_pairs(text)
    if pairs:
        return pairs
    return _code_fence_action_observation_pairs(text)


def _explicit_action_observation_pairs(text: str) -> list[tuple[str, str]]:
    chunks = re.split(r"(?im)^\s*Action\s*:\s*", str(text or ""))
    pairs: list[tuple[str, str]] = []
    for chunk in chunks[1:]:
        match = re.search(r"(?im)^\s*Observation(?: after action)?\s*:\s*", chunk)
        if not match:
            action = chunk.strip()
            observation = ""
        else:
            action = chunk[: match.start()].strip()
            observation = chunk[match.end() :].strip()
        if action:
            pairs.append((action, observation))
    return pairs


def _code_fence_action_observation_pairs(text: str) -> list[tuple[str, str]]:
    blocks = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n(.*?)```", str(text or ""), flags=re.S)
        if match.group(1).strip()
    ]
    pairs: list[tuple[str, str]] = []
    cursor = 0
    while cursor < len(blocks):
        action = blocks[cursor].strip()
        observation = blocks[cursor + 1].strip() if cursor + 1 < len(blocks) else ""
        if action:
            pairs.append((action, observation))
        cursor += 2
    return pairs


def _normalize_legacy_tool_call(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if re.match(r"^[A-Za-z_][\w.-]*\s*\(", value, flags=re.S):
        return value
    if "\n" in value or "apis." in value or value.startswith(("import ", "from ")):
        return "execute(" + json.dumps({"code": value}, ensure_ascii=False, sort_keys=True) + ")"
    return value


def _render_turns(
    turns: list[Any],
    *,
    options: TraceCompressionOptions,
    env_name: str = "",
    initial_observation: str = "",
) -> str:
    lines: list[str] = []
    if initial_observation:
        lines.append("Initial observation:\n" + initial_observation)
    for idx, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            text = compress_trace_text(turn, options=options)
            if text:
                lines.append(f"Step {idx}:\n{text}")
            continue
        parts = [f"Step {idx}:"]
        assistant_message = turn.get("assistant_message")
        response_text = (
            turn.get("parser_text")
            or turn.get("response_text")
            or (_message_content(assistant_message) if isinstance(assistant_message, dict) else "")
        )
        if response_text and _keeps_part(options.strip_assistant_response):
            response_text = compress_trace_text(
                response_text,
                options=options,
                strip_assistant_response=False,
            )
            response_text = _compress_part_text(
                response_text,
                options.strip_assistant_response,
                label="assistant response",
            )
            if response_text:
                parts.append(f"Assistant response:\n{response_text}")
        env_action_text = _env_specific_tool_call_text(
            turn,
            env_name=env_name,
            max_chars=_truncate_limit(options.strip_tool_call),
        )
        action_text = env_action_text if env_action_text is not None else _action_text(turn.get("action"))
        if action_text and _keeps_part(options.strip_tool_call):
            if env_action_text is None:
                action_text = compress_trace_text(
                    action_text,
                    options=options,
                    strip_assistant_response=False,
                )
                action_text = _compress_part_text(action_text, options.strip_tool_call, label="tool call")
            if action_text:
                parts.append(f"Tool call:\n{action_text}")
        observation = _env_specific_tool_response_text(
            turn,
            env_name=env_name,
            observation=_turn_observation(turn),
        )
        if observation and _keeps_part(options.strip_tool_response):
            observation = _compress_part_text(
                observation,
                options.strip_tool_response,
                label="tool response",
            )
            if observation:
                parts.append(f"Tool response:\n{observation}")
        if len(parts) > 1:
            lines.append("\n".join(parts))
    return "\n\n".join(lines).strip()


def _env_specific_tool_call_text(
    turn: dict[str, Any],
    *,
    env_name: str,
    max_chars: int | None,
) -> str | None:
    if env_name != "appworld":
        return None
    action_name = _action_name(turn.get("action"))
    if action_name not in {"execute", "python", "python_exec", "finish", "final_response", "submit"}:
        return None
    env_step = turn.get("env_step")
    info = env_step.get("info") if isinstance(env_step, dict) else None
    evidence = info.get("reward_evidence") if isinstance(info, dict) else None
    if evidence is None:
        raise ValueError(
            f"AppWorld structured reward trace step is missing execution evidence for action={action_name!r}"
        )
    return render_execution_evidence(
        evidence,
        max_chars=max_chars,
        observation=_turn_observation(turn),
    )


def _env_specific_tool_response_text(
    turn: dict[str, Any],
    *,
    env_name: str,
    observation: str,
) -> str:
    if env_name != "appworld":
        return observation
    env_step = turn.get("env_step")
    info = env_step.get("info") if isinstance(env_step, dict) else None
    if isinstance(info, dict) and info.get("reward_evidence") is not None:
        return ""
    return observation


def _initial_observation_from_sample_metadata(
    sample_metadata: dict[str, Any],
    *,
    options: TraceCompressionOptions,
    env_name: str,
) -> str:
    task_prompt = _task_prompt_from_sample_metadata(sample_metadata)
    if env_name == "appworld" and task_prompt and _metadata_contains_appworld_official_prompt(sample_metadata):
        return "Task:\n" + task_prompt

    explicit = _first_metadata_value(sample_metadata, ("reward_initial_observation", "initial_observation"))
    if explicit not in (None, "", []):
        return _canonical_initial_observation_text(
            explicit,
            sample_metadata=sample_metadata,
            env_name=env_name,
            options=options,
        )

    if task_prompt:
        return "Task:\n" + task_prompt

    initial_from_messages = _initial_observation_from_messages(sample_metadata.get("messages"), options=options)
    return _canonical_initial_observation_text(
        initial_from_messages,
        sample_metadata=sample_metadata,
        env_name=env_name,
        options=options,
        require_task_for_appworld=False,
    )


def _canonical_initial_observation_text(
    text: Any,
    *,
    sample_metadata: dict[str, Any],
    env_name: str,
    options: TraceCompressionOptions,
    require_task_for_appworld: bool = True,
) -> str:
    if text in (None, "", []):
        return ""
    value = _env_specific_initial_observation_text(
        text,
        sample_metadata=sample_metadata,
        env_name=env_name,
        require_task_for_appworld=require_task_for_appworld,
    )
    return compress_trace_text(
        value,
        options=options,
        strip_assistant_response=False,
    )


def _env_specific_initial_observation_text(
    text: Any,
    *,
    sample_metadata: dict[str, Any],
    env_name: str,
    require_task_for_appworld: bool,
) -> str:
    # Source adapters may accept different trace shapes, but environment
    # semantics live behind this explicit env-name dispatch. Generic compression
    # must stay unaware of AppWorld/WebShop/Tau2-specific prompt conventions.
    if env_name == "appworld":
        return _appworld_initial_observation_text(
            text,
            sample_metadata=sample_metadata,
            require_task=require_task_for_appworld,
        )
    return str(text or "")


def _appworld_initial_observation_text(
    text: Any,
    *,
    sample_metadata: dict[str, Any],
    require_task: bool,
) -> str:
    if not _contains_appworld_official_prompt(text):
        return str(text or "")
    task_prompt = _task_prompt_from_sample_metadata(sample_metadata)
    if task_prompt:
        return "Task:\n" + task_prompt
    if require_task:
        raise ValueError("AppWorld official prompt trace cannot be rendered for reward without task_prompt metadata")
    return str(text or "")


def _task_prompt_from_sample_metadata(sample_metadata: dict[str, Any]) -> str:
    keys = ("query", "task_prompt", "instruction", "question", "task_question", "instruction_text")
    for key in keys:
        value = sample_metadata.get(key)
        if value not in (None, "", []):
            return str(value).strip()
    for nested_key in ("env_metadata", "requested_task", "source_row"):
        nested = sample_metadata.get(nested_key)
        if isinstance(nested, dict):
            for key in keys:
                value = nested.get(key)
                if value not in (None, "", []):
                    return str(value).strip()
    turns = sample_metadata.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            env_step = turn.get("env_step")
            if not isinstance(env_step, dict):
                continue
            info = env_step.get("info")
            if isinstance(info, dict):
                for key in keys:
                    value = info.get(key)
                    if value not in (None, "", []):
                        return str(value).strip()
    return ""


def _initial_observation_from_messages(messages: Any, *, options: TraceCompressionOptions) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "assistant":
            break
        if role in {"system", "developer"}:
            if _keeps_part(options.strip_system_prompt):
                text = _message_content(message)
                if text:
                    text = _compress_part_text(
                        text,
                        options.strip_system_prompt,
                        label=f"{role} prompt",
                    )
                    if text:
                        parts.append(f"{role.title()} prompt:\n{text}")
            continue
        if role in {"user", "tool"}:
            text = _message_content(message)
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def _metadata_contains_appworld_official_prompt(sample_metadata: dict[str, Any]) -> bool:
    for key in ("reward_initial_observation", "initial_observation", "messages", "prompt", "sample_prompt"):
        value = sample_metadata.get(key)
        if value not in (None, "", []) and _contains_appworld_official_prompt(value):
            return True
    return False


def _contains_appworld_official_prompt(value: Any) -> bool:
    text = _normalize_search_text(_message_text(value))
    return all(_normalize_search_text(marker) in text for marker in _APPWORLD_OFFICIAL_PROMPT_MARKERS)


def _appworld_episode_start_from_messages(messages: list[Any], task_prompt: str) -> int | None:
    normalized = [item for item in messages if isinstance(item, dict)]
    if not normalized or not _contains_appworld_official_prompt(normalized):
        return None
    task_prompt = str(task_prompt or "").strip()
    if not task_prompt:
        raise ValueError("AppWorld official prompt trace cannot be rendered for reward without task_prompt metadata")
    task_message_index = _appworld_task_message_index(normalized, task_prompt)
    if task_message_index is None:
        raise ValueError(
            "AppWorld official prompt trace did not contain the task_prompt metadata in any user message; "
            "refusing to render few-shot prompt demo as reward trajectory"
        )
    return task_message_index + 1


def _appworld_task_message_index(messages: list[dict[str, Any]], task_prompt: str) -> int | None:
    needle = _normalize_search_text(task_prompt)
    if not needle:
        return None
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if str(message.get("role") or "") != "user":
            continue
        content = _normalize_search_text(_message_content(message))
        if needle in content:
            return idx
    return None


def _normalize_search_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _finalize_reward_trace_text(text: str, *, sample_metadata: dict[str, Any], env_name: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    return _env_specific_reward_trace_text(value, sample_metadata=sample_metadata, env_name=env_name)


def _env_specific_reward_trace_text(text: str, *, sample_metadata: dict[str, Any], env_name: str) -> str:
    if env_name == "appworld":
        return _finalize_appworld_reward_trace_text(text, sample_metadata=sample_metadata)
    return text


def _finalize_appworld_reward_trace_text(text: str, *, sample_metadata: dict[str, Any]) -> str:
    value = str(text or "").strip()
    if not _contains_appworld_official_prompt(value):
        return value
    task_prompt = _task_prompt_from_sample_metadata(sample_metadata)
    if not task_prompt:
        raise ValueError(
            "AppWorld official prompt trace cannot be rendered for reward without task_prompt metadata"
        )
    replacement = "Initial observation:\nTask:\n" + task_prompt
    replaced, count = re.subn(
        r"(?is)(^|\n)Initial observation\s*:\s*.*?(?=\n\s*Step\s+\d+\s*:|\Z)",
        lambda match: match.group(1) + replacement,
        value,
        count=1,
    )
    if count:
        return replaced.strip()
    first_step = re.search(r"(?im)^\s*Step\s+\d+\s*:", value)
    if first_step:
        return (replacement + "\n\n" + value[first_step.start() :].strip()).strip()
    return replacement


def _render_messages(
    messages: list[Any],
    *,
    options: TraceCompressionOptions,
    final_observation: str = "",
    initial_observation: str = "",
    start_index: int | None = None,
) -> str:
    normalized = [item for item in messages if isinstance(item, dict)]
    if not normalized:
        return ""
    lines: list[str] = []
    cursor = max(0, int(start_index)) if start_index is not None else 0
    if initial_observation:
        lines.append("Initial observation:\n" + initial_observation.strip())
        while cursor < len(normalized) and str(normalized[cursor].get("role") or "") != "assistant":
            cursor += 1
    else:
        initial_parts: list[str] = []
        while cursor < len(normalized):
            message = normalized[cursor]
            role = str(message.get("role") or "")
            if role == "assistant":
                break
            if role in {"system", "developer"}:
                if _keeps_part(options.strip_system_prompt):
                    text = _message_content(message)
                    if text:
                        text = _compress_part_text(
                            text,
                            options.strip_system_prompt,
                            label=f"{role} prompt",
                        )
                        if text:
                            initial_parts.append(f"{role.title()} prompt:\n{text}")
                cursor += 1
                continue
            if role in {"user", "tool"}:
                text = _message_content(message)
                if text:
                    initial_parts.append(text)
            cursor += 1
        if initial_parts:
            lines.append("Initial observation:\n" + "\n\n".join(initial_parts).strip())
    step = 1
    while cursor < len(normalized):
        message = normalized[cursor]
        if str(message.get("role") or "") != "assistant":
            cursor += 1
            continue
        observation, next_cursor = _tool_response_from_messages(normalized, cursor + 1)
        if not observation and final_observation and next_cursor >= len(normalized):
            observation = final_observation
        parts = [f"Step {step}:"]
        assistant_response = _message_content(message)
        if assistant_response and _keeps_part(options.strip_assistant_response):
            assistant_response = compress_trace_text(
                assistant_response,
                options=options,
                strip_assistant_response=False,
            )
            assistant_response = _compress_part_text(
                assistant_response,
                options.strip_assistant_response,
                label="assistant response",
            )
            if assistant_response:
                parts.append(f"Assistant response:\n{assistant_response}")
        action = _assistant_action_text(message)
        if action and _keeps_part(options.strip_tool_call):
            action = compress_trace_text(action, options=options, strip_assistant_response=False)
            action = _compress_part_text(action, options.strip_tool_call, label="tool call")
            if action:
                parts.append(f"Tool call:\n{action}")
        if observation and _keeps_part(options.strip_tool_response):
            observation = _compress_part_text(
                observation,
                options.strip_tool_response,
                label="tool response",
            )
            if observation:
                parts.append(f"Tool response:\n{observation}")
        if len(parts) > 1:
            lines.append("\n".join(parts))
            step += 1
        cursor = max(next_cursor, cursor + 1)
    return "\n\n".join(lines).strip()


def _render_token_segments(
    segments: list[Any],
    *,
    sample_metadata: dict[str, Any],
    env_name: str,
    options: TraceCompressionOptions,
) -> str:
    normalized = [item for item in segments if isinstance(item, dict)]
    if not normalized:
        return ""
    lines: list[str] = []
    step = 1
    pending_action = ""
    for segment in normalized:
        kind = str(segment.get("kind") or "")
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        if kind == "initial_prompt" and not lines:
            initial_text = _canonical_initial_observation_text(
                text,
                sample_metadata=sample_metadata,
                env_name=env_name,
                options=options,
            )
            if initial_text:
                lines.append("Initial observation:\n" + initial_text)
        elif kind == "assistant":
            pending_action = _extract_text_action(
                compress_trace_text(
                    text,
                    options=options,
                    strip_assistant_response=False,
                )
            )
        elif kind == "environment" and pending_action:
            parts = [f"Step {step}:"]
            if _keeps_part(options.strip_assistant_response):
                parts.append(f"Assistant response:\n{pending_action}")
            if _keeps_part(options.strip_tool_call):
                action_text = _compress_part_text(pending_action, options.strip_tool_call, label="tool call")
                if action_text:
                    parts.append(f"Tool call:\n{action_text}")
            if _keeps_part(options.strip_tool_response):
                observation = compress_trace_text(
                    text,
                    options=TraceCompressionOptions(
                        strip_reasoning=options.strip_reasoning,
                        strip_tool_response=False,
                        strip_assistant_response=False,
                        strip_system_prompt=options.strip_system_prompt,
                    ),
                    strip_assistant_response=False,
                )
                observation = _compress_part_text(
                    observation,
                    options.strip_tool_response,
                    label="tool response",
                )
                if observation:
                    parts.append(f"Tool response:\n{observation}")
            lines.append("\n".join(parts))
            pending_action = ""
            step += 1
    return "\n\n".join(lines).strip()


def _assistant_action_text(message: dict[str, Any]) -> str:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        rendered = [_tool_call_text(call) for call in tool_calls]
        return "\n".join(item for item in rendered if item.strip())
    return _extract_text_action(_message_content(message))


def _tool_call_text(call: Any) -> str:
    if not isinstance(call, dict):
        return _message_text(call)
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or call.get("name") or "").strip()
    arguments = function.get("arguments") if "arguments" in function else call.get("arguments")
    if isinstance(arguments, str):
        arguments_text = arguments.strip()
    else:
        arguments_text = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str)
    return f"{name}({arguments_text})" if name else arguments_text


def _action_text(action: Any) -> str:
    if action in (None, "", []):
        return ""
    if isinstance(action, dict):
        if action.get("type") == "tool_call":
            name = str(action.get("name") or "").strip()
            arguments = action.get("arguments")
            if isinstance(arguments, str):
                arguments_text = arguments.strip()
            else:
                arguments_text = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str)
            return f"{name}({arguments_text})" if name else arguments_text
        if action.get("type") == "assistant_message":
            return str(action.get("content") or "").strip()
    return _message_text(action).strip()


def _action_name(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("name") or action.get("tool") or "").strip().lower()


def _turn_observation(turn: dict[str, Any]) -> str:
    env_step = turn.get("env_step")
    if isinstance(env_step, dict) and env_step.get("observation") not in (None, "", []):
        return str(env_step.get("observation") or "").strip()
    if turn.get("observation") not in (None, "", []):
        return str(turn.get("observation") or "").strip()
    return ""


def _tool_response_from_messages(messages: list[dict[str, Any]], start: int) -> tuple[str, int]:
    parts: list[str] = []
    cursor = start
    while cursor < len(messages):
        message = messages[cursor]
        role = str(message.get("role") or "")
        if role == "assistant":
            break
        if role in {"user", "tool"}:
            text = _message_content(message)
            if text:
                parts.append(text)
        cursor += 1
    return "\n\n".join(parts).strip(), cursor


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if content in (None, "", []):
        return ""
    return _message_text(content).strip()


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _first_metadata_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", []):
            return value
    return None


def _trace_env_name(metadata: dict[str, Any], *, explicit_env_name: str | None = None) -> str:
    explicit = _normalize_env_name(explicit_env_name)
    if explicit:
        return explicit
    for key in ("env_name", "agent_env_name", "environment", "task_env"):
        value = _normalize_env_name(metadata.get(key))
        if value:
            return value
    for key in _KNOWN_AGENT_ENVS:
        if isinstance(metadata.get(key), dict):
            return key
    for nested_key in ("env_metadata", "requested_task", "source_row"):
        nested = metadata.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in ("env_name", "agent_env_name", "environment", "task_env"):
            value = _normalize_env_name(nested.get(key))
            if value:
                return value
    return ""


def _normalize_env_name(value: Any) -> str:
    text = re.sub(r"[^a-z0-9_]+", "", str(value or "").strip().lower())
    aliases = {
        "alfworld": "alfworld",
        "webshop": "webshop",
        "tau2": "tau2",
        "tau": "tau2",
        "appworld": "appworld",
        "openclaw": "openclaw",
    }
    return aliases.get(text, text if text in _KNOWN_AGENT_ENVS else "")


def _sample_payload(sample: Any) -> dict[str, Any]:
    """Return the trace-bearing sample payload.

    Slime `Sample` objects store env details in `metadata`, while rollout dumps
    are flattened JSON records. Keep `metadata` authoritative, but let the same
    renderer inspect dump-shaped records too.
    """

    payload: dict[str, Any] = {}
    top_level: dict[str, Any] = {}
    if isinstance(sample, dict):
        top_level = sample
    else:
        top_level = {
            key: getattr(sample, key)
            for key in (
                "reward_trace",
                "judge_trace",
                "ropd_trace",
                "turns",
                "messages",
                "token_segments",
                "trace",
                "trajectory",
                "rollout_trace",
                "student_trace",
                "env_name",
                "agent_env_name",
                "environment",
                "task_env",
                "env_evaluate",
                "env_metadata",
            )
            if hasattr(sample, key)
        }
    payload.update({key: value for key, value in top_level.items() if value not in (None, "", [])})
    metadata = top_level.get("metadata") if isinstance(top_level.get("metadata"), dict) else getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        payload.update({key: value for key, value in metadata.items() if value not in (None, "", [])})
    return payload


def _final_observation(metadata: dict[str, Any]) -> str:
    episode_result = metadata.get("episode_result")
    if isinstance(episode_result, dict):
        return str(episode_result.get("observation") or "").strip()
    return ""


def _combine_final_and_trace(final_answer: str, trace: str) -> str:
    final_answer = str(final_answer or "").strip()
    trace = str(trace or "").strip()
    if final_answer and trace and final_answer != trace:
        return f"[Final Answer]\n{final_answer}\n\n[Trace]\n{trace}"
    return final_answer or trace


def _extract_text_action(text: str) -> str:
    value = _strip_reasoning_text(_strip_chat_boundary_tokens(str(text or "")))
    match = re.search(r"<action>\s*(.*?)\s*</action>", value, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return value.strip()


def _strip_reasoning_text(text: Any) -> str:
    # Native thinking is a structured field. These patterns are only a text
    # sanitizer for old persisted traces that already flattened markers into
    # plain text; they are not used to decide whether reasoning exists.
    value = str(text or "")
    value = re.sub(r"<think\b[^>]*>.*?</think>", "", value, flags=re.I | re.S)
    value = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", value, flags=re.I | re.S)
    value = re.sub(r"<think\b[^>]*>.*?(?=<action\b|$)", "", value, flags=re.I | re.S)
    value = re.sub(r"<\|begin_of_thought\|>.*?(?=<action\b|$)", "", value, flags=re.I | re.S)
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[1]
    if "<|end_of_thought|>" in value:
        value = value.rsplit("<|end_of_thought|>", 1)[1]
    return value.strip()


def _strip_chat_boundary_tokens(text: str) -> str:
    value = str(text or "")
    for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "</s>"):
        value = value.replace(token, "")
    value = re.sub(r"(?m)^(system|assistant|user|tool)\s*$", "", value)
    return value.strip()


def _strip_system_prompt_text(text: str) -> str:
    value = re.sub(r"<\|im_start\|>system\n.*?<\|im_end\|>\n?", "", str(text or ""), flags=re.S)
    value = re.sub(r"<\|im_start\|>developer\n.*?<\|im_end\|>\n?", "", value, flags=re.S)
    return _strip_label_blocks(
        value,
        strip_labels={"SYSTEM", "SYSTEM_PROMPT", "DEVELOPER", "DEVELOPER_PROMPT"},
        resume_labels={
            "USER",
            "ASSISTANT",
            "ACTION",
            "ACTIONS",
            "TOOL",
            "TOOL_CALL",
            "TOOL_CALLS",
            "TOOL_RESPONSE",
            "TOOL_RESULT",
            "OBSERVATION",
            "OBSERVATION_AFTER_ACTION",
            "STEP",
            "INITIAL_OBSERVATION",
        },
    )


def _compress_system_prompt_text(text: str, max_chars: int) -> str:
    return _compress_label_blocks(
        str(text or ""),
        strip_labels={"SYSTEM", "SYSTEM_PROMPT", "DEVELOPER", "DEVELOPER_PROMPT"},
        resume_labels={
            "USER",
            "ASSISTANT",
            "ACTION",
            "ACTIONS",
            "TOOL",
            "TOOL_CALL",
            "TOOL_CALLS",
            "TOOL_RESPONSE",
            "TOOL_RESULT",
            "OBSERVATION",
            "OBSERVATION_AFTER_ACTION",
            "STEP",
            "INITIAL_OBSERVATION",
        },
        max_chars=max_chars,
        label="system prompt",
    )


def _strip_assistant_response_text(text: str) -> str:
    return _strip_label_blocks(
        str(text or ""),
        strip_labels={"ASSISTANT", "ASSISTANT_RESPONSE", "RESPONSE", "TEACHER_RESPONSE"},
        resume_labels={
            "ACTION",
            "ACTIONS",
            "TOOL_CALL",
            "TOOL_CALLS",
            "TOOL_RESPONSE",
            "TOOL_RESULT",
            "TOOL_RESULTS",
            "OBSERVATION",
            "OBSERVATION_AFTER_ACTION",
            "STEP",
            "INITIAL_OBSERVATION",
        },
        resume_prefixes=("<action",),
    )


def _compress_assistant_response_text(text: str, max_chars: int) -> str:
    return _compress_label_blocks(
        str(text or ""),
        strip_labels={"ASSISTANT", "ASSISTANT_RESPONSE", "RESPONSE", "TEACHER_RESPONSE"},
        resume_labels={
            "ACTION",
            "ACTIONS",
            "TOOL_CALL",
            "TOOL_CALLS",
            "TOOL_RESPONSE",
            "TOOL_RESULT",
            "TOOL_RESULTS",
            "OBSERVATION",
            "OBSERVATION_AFTER_ACTION",
            "STEP",
            "INITIAL_OBSERVATION",
        },
        resume_prefixes=("<action",),
        max_chars=max_chars,
        label="assistant response",
    )


def _strip_tool_call_text(text: str) -> str:
    return _strip_label_blocks(
        str(text or ""),
        strip_labels={"ACTION", "ACTIONS", "TOOL_CALL", "TOOL_CALLS"},
        resume_labels={
            "USER",
            "ASSISTANT",
            "TOOL",
            "TOOL_RESPONSE",
            "TOOL_RESULT",
            "TOOL_RESULTS",
            "OBSERVATION",
            "OBSERVATION_AFTER_ACTION",
            "STEP",
            "INITIAL_OBSERVATION",
        },
    )


def _compress_tool_call_text(text: str, max_chars: int) -> str:
    return _compress_label_blocks(
        str(text or ""),
        strip_labels={"ACTION", "ACTIONS", "TOOL_CALL", "TOOL_CALLS"},
        resume_labels={
            "USER",
            "ASSISTANT",
            "TOOL",
            "TOOL_RESPONSE",
            "TOOL_RESULT",
            "TOOL_RESULTS",
            "OBSERVATION",
            "OBSERVATION_AFTER_ACTION",
            "STEP",
            "INITIAL_OBSERVATION",
        },
        resume_prefixes=("<tool_response",),
        max_chars=max_chars,
        label="tool call",
    )


def _strip_tool_response_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"<\|im_start\|>tool\n.*?<\|im_end\|>\n?", "", value, flags=re.S)
    value = re.sub(r"<\|im_start\|>user\n\s*<tool_response>.*?</tool_response>\s*<\|im_end\|>\n?", "", value, flags=re.S)
    value = re.sub(r"<tool_response>.*?</tool_response>", "", value, flags=re.S)
    return _strip_label_blocks(
        value,
        strip_labels={"TOOL", "TOOL_RESULT", "TOOL_RESULTS", "TOOL_RESPONSE", "TOOL_RESPONSES", "OBSERVATION", "OBSERVATION_AFTER_ACTION"},
        resume_labels={"USER", "ASSISTANT", "ACTION", "ACTIONS", "TOOL_CALL", "TOOL_CALLS", "STEP", "INITIAL_OBSERVATION"},
    )


def _compress_tool_response_text(text: str, max_chars: int) -> str:
    value = str(text or "")
    value = re.sub(
        r"(<\|im_start\|>tool\n)(.*?)(<\|im_end\|>\n?)",
        lambda match: match.group(1)
        + _middle_truncate_text(match.group(2), max_chars, label="tool response")
        + match.group(3),
        value,
        flags=re.S,
    )
    value = re.sub(
        r"(<tool_response>)(.*?)(</tool_response>)",
        lambda match: match.group(1)
        + _middle_truncate_text(match.group(2), max_chars, label="tool response")
        + match.group(3),
        value,
        flags=re.S,
    )
    return _compress_label_blocks(
        value,
        strip_labels={"TOOL", "TOOL_RESULT", "TOOL_RESULTS", "TOOL_RESPONSE", "TOOL_RESPONSES", "OBSERVATION", "OBSERVATION_AFTER_ACTION"},
        resume_labels={"USER", "ASSISTANT", "ACTION", "ACTIONS", "TOOL_CALL", "TOOL_CALLS", "STEP", "INITIAL_OBSERVATION"},
        max_chars=max_chars,
        label="tool response",
    )


def _compress_reasoning_text(text: str, max_chars: int) -> str:
    value = str(text or "")
    value = re.sub(
        r"(<think\b[^>]*>)(.*?)(</think>)",
        lambda match: match.group(1)
        + _middle_truncate_text(match.group(2), max_chars, label="reasoning")
        + match.group(3),
        value,
        flags=re.I | re.S,
    )
    value = re.sub(
        r"(<\|begin_of_thought\|>)(.*?)(<\|end_of_thought\|>)",
        lambda match: match.group(1)
        + _middle_truncate_text(match.group(2), max_chars, label="reasoning")
        + match.group(3),
        value,
        flags=re.I | re.S,
    )
    return value.strip()


def _strip_label_blocks(
    text: str,
    *,
    strip_labels: set[str],
    resume_labels: set[str],
    resume_prefixes: tuple[str, ...] = (),
) -> str:
    lines: list[str] = []
    skipping = False
    for line in str(text or "").splitlines():
        label = _line_label(line)
        if label in strip_labels:
            skipping = True
            continue
        if skipping:
            stripped = line.strip().lower()
            if label in resume_labels or any(stripped.startswith(prefix) for prefix in resume_prefixes):
                skipping = False
            else:
                continue
        lines.append(line)
    return "\n".join(lines).strip()


def _compress_label_blocks(
    text: str,
    *,
    strip_labels: set[str],
    resume_labels: set[str],
    max_chars: int,
    label: str,
    resume_prefixes: tuple[str, ...] = (),
) -> str:
    lines = str(text or "").splitlines()
    output: list[str] = []
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        line_label, block_header, inline_body = _line_label_header_and_body(line)
        if line_label not in strip_labels:
            output.append(line)
            cursor += 1
            continue
        block_lines: list[str] = []
        if inline_body.strip():
            block_lines.append(inline_body)
        cursor += 1
        while cursor < len(lines):
            next_line = lines[cursor]
            next_label = _line_label(next_line)
            stripped = next_line.strip().lower()
            if next_label in resume_labels or any(stripped.startswith(prefix) for prefix in resume_prefixes):
                break
            block_lines.append(next_line)
            cursor += 1
        output.append(block_header)
        block = "\n".join(block_lines).strip()
        if block:
            output.append(_middle_truncate_text(block, max_chars, label=label))
    return "\n".join(output).strip()


def _line_label(line: str) -> str:
    label, _, _ = _line_label_header_and_body(line)
    return label


def _line_label_header_and_body(line: str) -> tuple[str, str, str]:
    stripped = line.strip()
    if not stripped:
        return "", line, ""
    if re.match(r"step\s+\d+\s*:?\s*$", stripped, flags=re.I):
        return "STEP", line, ""
    match = re.match(r"(?P<header>\s*\[?(?P<label>[A-Za-z_ ]+)\]?\s*(?:after action)?\s*:\s*)(?P<body>.*)$", line)
    if not match:
        return "", line, ""
    label = re.sub(r"[^A-Z0-9]+", "_", match.group("label").strip().upper()).strip("_")
    return label, match.group("header").rstrip(), match.group("body")


def _check_sample_reasoning_contract(
    sample: Sample,
    final_answer: Any,
    options: TraceCompressionOptions,
    *,
    context: str,
) -> None:
    values: list[Any] = [final_answer, getattr(sample, "response", "")]
    sample_metadata = _sample_payload(sample)
    for key in (
        "reward_trace",
        "judge_trace",
        "ropd_trace",
        "trace",
        "trajectory",
        "rollout_trace",
        "student_trace",
        "messages",
        "turns",
        "token_segments",
    ):
        value = sample_metadata.get(key)
        if value not in (None, "", []):
            values.append(value)
    has_reasoning = any(_has_reasoning_value(value) for value in values)
    if has_reasoning:
        return
    _handle_missing_reasoning(options, context=context)


def _check_reasoning_contract(text: Any, options: TraceCompressionOptions, *, context: str) -> None:
    if _has_reasoning_value(text):
        return
    _handle_missing_reasoning(options, context=context)


def _handle_missing_reasoning(options: TraceCompressionOptions, *, context: str) -> None:
    if not _is_strip_spec(options.strip_reasoning):
        raise ValueError(
            f"{context} was configured to preserve or truncate reasoning, but no reasoning/thinking field was present"
        )
    if context not in _MISSING_REASONING_WARNED:
        _MISSING_REASONING_WARNED.add(context)
        logger.warning(
            "%s has no reasoning/thinking field; strip_reasoning=true so reward trace compression continues",
            context,
        )


def _has_reasoning_value(value: Any) -> bool:
    if value in (None, "", []):
        return False
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _REASONING_FIELD_NAMES and str(item or "").strip():
                return True
            if _has_reasoning_value(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_has_reasoning_value(item) for item in value)
    return False


_REASONING_FIELD_NAMES = {
    "reasoning",
    "reasoning_content",
    "reasoning_text",
    "thinking",
    "thinking_content",
    "thought",
    "thoughts",
}


def _apply_structured_reasoning_policy(value: Any, spec: TracePartSpec) -> Any:
    if _is_strip_spec(spec):
        return _drop_structured_reasoning(value)
    limit = _truncate_limit(spec)
    if limit is not None:
        return _truncate_structured_reasoning(value, limit)
    return value


def _drop_structured_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_structured_reasoning(item)
            for key, item in value.items()
            if str(key).strip().lower() not in _REASONING_FIELD_NAMES
        }
    if isinstance(value, list):
        return [_drop_structured_reasoning(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_drop_structured_reasoning(item) for item in value)
    return value


def _truncate_structured_reasoning(value: Any, max_chars: int) -> Any:
    if isinstance(value, dict):
        output: dict[Any, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _REASONING_FIELD_NAMES:
                output[key] = _middle_truncate_text(item, max_chars, label="reasoning")
            else:
                output[key] = _truncate_structured_reasoning(item, max_chars)
        return output
    if isinstance(value, list):
        return [_truncate_structured_reasoning(item, max_chars) for item in value]
    if isinstance(value, tuple):
        return tuple(_truncate_structured_reasoning(item, max_chars) for item in value)
    return value
