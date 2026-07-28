from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)
_MISSING_REASONING_WARNED: set[str] = set()


@dataclass(frozen=True)
class TraceCompressionOptions:
    strip_reasoning: bool = True
    strip_tool_response: bool = False
    strip_assistant_response: bool = True
    strip_system_prompt: bool = True


def render_answer_for_reward(
    sample: Sample,
    *,
    final_answer: Any = "",
    answer_mode: str = "trace",
    options: TraceCompressionOptions | None = None,
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
    trace_text = render_trace_for_reward(sample, options=options)
    return _combine_final_and_trace(final_text, trace_text)


def render_trace_for_reward(sample: Sample, *, options: TraceCompressionOptions | None = None) -> str:
    options = options or TraceCompressionOptions()
    sample_metadata = _sample_payload(sample)
    explicit_trace = _first_metadata_value(sample_metadata, ("reward_trace", "judge_trace", "ropd_trace"))
    if explicit_trace not in (None, "", []):
        return compress_trace_text(explicit_trace, options=options)
    turns = sample_metadata.get("turns")
    if isinstance(turns, list):
        rendered = _render_turns(
            turns,
            options=options,
            initial_observation=_initial_observation_from_messages(sample_metadata.get("messages"), options=options),
        )
        if rendered:
            return rendered
    messages = sample_metadata.get("messages")
    if isinstance(messages, list):
        rendered = _render_messages(
            messages,
            options=options,
            final_observation=_final_observation(sample_metadata),
        )
        if rendered:
            return rendered
    segments = sample_metadata.get("token_segments")
    if isinstance(segments, list):
        rendered = _render_token_segments(segments, options=options)
        if rendered:
            return rendered
    trace_value = _first_metadata_value(
        sample_metadata,
        ("trace", "trajectory", "rollout_trace", "student_trace"),
    )
    if trace_value not in (None, "", []):
        return compress_trace_text(trace_value, options=options)
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
    if check_reasoning_presence:
        _check_reasoning_contract(trace, options, context=reasoning_context)
    if trace in (None, "", []):
        return ""
    if isinstance(trace, Sample):
        return render_trace_for_reward(trace, options=options)
    structured = _render_structured_teacher_trace(trace, options=options)
    if structured:
        return structured
    text = _message_text(_drop_structured_reasoning(trace) if options.strip_reasoning else trace)
    legacy = _render_legacy_teacher_text(text, options=options)
    if legacy:
        return legacy
    return compress_trace_text(trace, options=options)


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
    raw_value = _drop_structured_reasoning(text) if options.strip_reasoning else text
    value = _strip_chat_boundary_tokens(_message_text(raw_value))
    if options.strip_system_prompt:
        value = _strip_system_prompt_text(value)
    if options.strip_reasoning:
        value = _strip_reasoning_text(value)
    should_strip_assistant = options.strip_assistant_response if strip_assistant_response is None else strip_assistant_response
    if should_strip_assistant:
        value = _strip_assistant_response_text(value)
    if options.strip_tool_response:
        value = _strip_tool_response_text(value)
    return value.strip()


def _render_structured_teacher_trace(trace: Any, *, options: TraceCompressionOptions) -> str:
    if isinstance(trace, dict):
        payload = trace
    elif isinstance(trace, (list, tuple)):
        normalized = list(trace)
        role_count = sum(1 for item in normalized if isinstance(item, dict) and item.get("role"))
        payload = {"messages": normalized} if role_count else {"turns": normalized}
    else:
        return ""
    try:
        return render_trace_for_reward(Sample(prompt="", metadata=payload), options=options)
    except ValueError:
        return ""


def _render_legacy_teacher_text(text: str, *, options: TraceCompressionOptions) -> str:
    value = _strip_chat_boundary_tokens(str(text or ""))
    if options.strip_system_prompt:
        value = _strip_system_prompt_text(value)
    if options.strip_reasoning:
        value = _strip_reasoning_text(value)
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
            if not action_text:
                continue
            parts = [f"Step {step}:", "Tool call:", action_text]
            if observation and not options.strip_tool_response:
                parts.extend(["Tool response:", observation.strip()])
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
        if response_text and not options.strip_assistant_response:
            response_text = compress_trace_text(
                response_text,
                options=options,
                strip_assistant_response=False,
            )
            if response_text:
                parts.append(f"Assistant response:\n{response_text}")
        action_text = _action_text(turn.get("action"))
        if action_text:
            action_text = compress_trace_text(
                action_text,
                options=options,
                strip_assistant_response=False,
            )
            if action_text:
                parts.append(f"Tool call:\n{action_text}")
        observation = _turn_observation(turn)
        if observation and not options.strip_tool_response:
            parts.append(f"Tool response:\n{observation}")
        if len(parts) > 1:
            lines.append("\n".join(parts))
    return "\n\n".join(lines).strip()


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
            if not options.strip_system_prompt:
                text = _message_content(message)
                if text:
                    parts.append(f"{role.title()} prompt:\n{text}")
            continue
        if role in {"user", "tool"}:
            text = _message_content(message)
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def _render_messages(
    messages: list[Any],
    *,
    options: TraceCompressionOptions,
    final_observation: str = "",
) -> str:
    normalized = [item for item in messages if isinstance(item, dict)]
    if not normalized:
        return ""
    lines: list[str] = []
    cursor = 0
    initial_parts: list[str] = []
    while cursor < len(normalized):
        message = normalized[cursor]
        role = str(message.get("role") or "")
        if role == "assistant":
            break
        if role in {"system", "developer"}:
            if not options.strip_system_prompt:
                text = _message_content(message)
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
        if assistant_response and not options.strip_assistant_response:
            assistant_response = compress_trace_text(
                assistant_response,
                options=options,
                strip_assistant_response=False,
            )
            if assistant_response:
                parts.append(f"Assistant response:\n{assistant_response}")
        action = _assistant_action_text(message)
        if action:
            action = compress_trace_text(action, options=options, strip_assistant_response=False)
            if action:
                parts.append(f"Tool call:\n{action}")
        if observation and not options.strip_tool_response:
            parts.append(f"Tool response:\n{observation}")
        if len(parts) > 1:
            lines.append("\n".join(parts))
            step += 1
        cursor = max(next_cursor, cursor + 1)
    return "\n\n".join(lines).strip()


def _render_token_segments(segments: list[Any], *, options: TraceCompressionOptions) -> str:
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
            initial_text = compress_trace_text(
                text,
                options=options,
                strip_assistant_response=False,
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
            if not options.strip_assistant_response:
                parts.append(f"Assistant response:\n{pending_action}")
            parts.append(f"Tool call:\n{pending_action}")
            if not options.strip_tool_response:
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


def _line_label(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if re.match(r"step\s+\d+\s*:?\s*$", stripped, flags=re.I):
        return "STEP"
    match = re.match(r"\[?([A-Za-z_ ]+)\]?\s*(?:after action)?\s*:\s*", stripped)
    if not match:
        return ""
    return re.sub(r"[^A-Z0-9]+", "_", match.group(1).strip().upper()).strip("_")


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
    if not options.strip_reasoning:
        raise ValueError(
            f"{context} was configured with strip_reasoning=false, but no reasoning/thinking field was present"
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
