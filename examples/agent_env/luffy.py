from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.rollout import arg, metadata, tokenizer
from examples.agent_env.rewards.config import reward_cfg_path
from examples.agent_env.rewards.extractors import bool_value, float_value
from examples.agent_env.teacher_data import teacher_trace_text, teacher_trace_value
from examples.agent_env.trace_rendering import compact_trace_part


@dataclass(frozen=True)
class TeacherSegment:
    text: str
    trainable: bool
    kind: str
    source_chars: int | None = None
    truncated: bool = False


def luffy_enabled(args: Any) -> bool:
    return bool_value(_cfg(args, "enable", False), False) and _mode(args) == "token_loss"


def apply_luffy_teacher_sample(args: Any, group: list[Sample]) -> list[Sample]:
    if not luffy_enabled(args):
        return group
    if not group:
        return group
    anchor_pos, anchor = _anchor_sample(group)
    policy_format = _policy_format(args)
    if policy_format == "appworld_structured_turns":
        teacher_payload = teacher_trace_value(
            args,
            anchor,
            config_prefix="luffy",
            env_var="AGENT_ENV_LUFFY_TEACHER_INDEX_PATH",
            key_config_name="teacher_trace_keys",
        )
        if teacher_payload in (None, "", []):
            raise RuntimeError(
                f"LUFFY token loss is enabled but no structured teacher trace matched sample index={anchor.index}"
            )
        segments = _segments_from_appworld_structured_turns(args, teacher_payload)
    else:
        teacher_text = teacher_trace_text(
            args,
            anchor,
            config_prefix="luffy",
            env_var="AGENT_ENV_LUFFY_TEACHER_INDEX_PATH",
            key_config_name="teacher_trace_keys",
        )
        if not teacher_text:
            raise RuntimeError(f"LUFFY token loss is enabled but no teacher trace matched sample index={anchor.index}")
        segments = _teacher_segments(args, anchor, teacher_text)
    if not segments or not any(segment.trainable for segment in segments):
        raise RuntimeError(f"LUFFY teacher trace produced no trainable policy segments for sample index={anchor.index}")

    teacher_sample = _build_teacher_sample(args, anchor, segments)
    if _insertion_mode(args) == "replace_anchor":
        repaired = list(group)
        repaired[anchor_pos] = teacher_sample
        return repaired
    return [*group, teacher_sample]


def _anchor_sample(group: list[Sample]) -> tuple[int, Sample]:
    for idx, sample in enumerate(group):
        if getattr(sample, "remove_sample", False):
            continue
        if sample.status == Sample.Status.ABORTED:
            continue
        return idx, sample
    raise RuntimeError("LUFFY cannot attach a teacher sample to a group without an active student sample")


def _cfg(args: Any, name: str, default: Any = None) -> Any:
    return reward_cfg_path(args, f"luffy.{name}", default)


def _float_cfg(args: Any, name: str, default: float) -> float:
    return float_value(_cfg(args, name, default), default)


def _mode(args: Any) -> str:
    value = str(_cfg(args, "mode", "off") or "off").strip().lower()
    valid = {"off", "token_loss"}
    if value not in valid:
        raise ValueError(f"Unsupported reward.luffy.mode={value!r}; expected one of {sorted(valid)}")
    return value


def _policy_format(args: Any) -> str:
    value = str(_cfg(args, "policy_format", "auto") or "auto").strip().lower()
    valid = {
        "auto",
        "text_action_xml",
        "appworld_markdown_code",
        "appworld_code_tag",
        "appworld_structured_turns",
        "raw",
    }
    if value not in valid:
        raise ValueError(f"Unsupported reward.luffy.policy_format={value!r}; expected one of {sorted(valid)}")
    return value


def _insertion_mode(args: Any) -> str:
    value = str(_cfg(args, "insertion_mode", "replace_anchor") or "replace_anchor").strip().lower()
    valid = {"replace_anchor", "append"}
    if value not in valid:
        raise ValueError(f"Unsupported reward.luffy.insertion_mode={value!r}; expected one of {sorted(valid)}")
    return value


def _teacher_segments(args: Any, sample: Sample, text: str) -> list[TeacherSegment]:
    policy_format = _policy_format(args)
    if policy_format in {"auto", "appworld_markdown_code", "appworld_code_tag"}:
        segments = _segments_from_appworld_markdown(text, policy_format=policy_format)
        if segments:
            return segments

    segments = _segments_from_tool_trace(text, policy_format=policy_format)
    if segments:
        return segments

    if policy_format == "raw":
        return [TeacherSegment(text=text.strip(), trainable=True, kind="raw_teacher_response")]
    return []


def _segments_from_appworld_markdown(text: str, *, policy_format: str) -> list[TeacherSegment]:
    blocks = list(re.finditer(r"```\s*(?:python|py)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL))
    if len(blocks) < 1:
        return []
    segments: list[TeacherSegment] = []
    for block_index, match in enumerate(blocks):
        code = match.group(1).strip()
        if not code:
            continue
        # AppWorld teacher_response alternates policy code blocks and execution
        # output blocks. Only policy code blocks are supervised.
        if block_index % 2 == 0:
            segments.append(
                TeacherSegment(
                    text=_render_appworld_code(code, policy_format=policy_format),
                    trainable=True,
                    kind="assistant_code",
                )
            )
        else:
            segments.append(
                TeacherSegment(
                    text=f"\nTool response:\n{code}\n",
                    trainable=False,
                    kind="tool_response",
                )
            )
    return segments


def _segments_from_appworld_structured_turns(args: Any, payload: Any) -> list[TeacherSegment]:
    if not isinstance(payload, dict):
        raise ValueError("LUFFY appworld_structured_turns requires a structured teacher payload")
    env_name = str(payload.get("env_name") or "").strip().lower()
    if env_name != "appworld":
        raise ValueError(f"LUFFY appworld_structured_turns requires env_name='appworld', got {env_name!r}")
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("LUFFY AppWorld teacher payload has no structured turns")

    response_limit = _cfg(args, "tool_response_max_chars", 2048)
    if isinstance(response_limit, bool):
        raise ValueError("reward.luffy.tool_response_max_chars must be a positive integer")
    try:
        response_limit = int(response_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("reward.luffy.tool_response_max_chars must be a positive integer") from exc
    if response_limit <= 0:
        raise ValueError("reward.luffy.tool_response_max_chars must be a positive integer")

    segments: list[TeacherSegment] = []
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"LUFFY AppWorld teacher turn {turn_index} is not a mapping")
        if turn.get("format_valid") is False:
            raise ValueError(f"LUFFY AppWorld teacher turn {turn_index} is format-invalid")
        action = turn.get("action")
        if not isinstance(action, dict) or str(action.get("name") or "").strip().lower() != "execute":
            raise ValueError(f"LUFFY AppWorld teacher turn {turn_index} is missing an execute action")
        arguments = action.get("arguments")
        code = str(arguments.get("code") or "").strip() if isinstance(arguments, dict) else ""
        if not code:
            raise ValueError(f"LUFFY AppWorld teacher turn {turn_index} has no executable code")
        rendered_code = _render_appworld_code(code, policy_format="appworld_markdown_code")
        segments.append(
            TeacherSegment(
                text=rendered_code,
                trainable=True,
                kind="assistant_code",
                source_chars=len(code),
            )
        )

        env_step = turn.get("env_step")
        observation = str(env_step.get("observation") or "").strip() if isinstance(env_step, dict) else ""
        if not observation:
            continue
        compacted = compact_trace_part(observation, response_limit, label="AppWorld tool response")
        segments.append(
            TeacherSegment(
                text=f"\nTool response:\n{compacted}\n",
                trainable=False,
                kind="tool_response",
                source_chars=len(observation),
                truncated=len(compacted) < len(observation),
            )
        )
    return segments


def _segments_from_tool_trace(text: str, *, policy_format: str) -> list[TeacherSegment]:
    pattern = re.compile(
        r"(?:^|\n)Step\s+\d+\s*:\s*\nTool call:\s*(?P<call>.*?)\nTool response:\s*(?P<response>.*?)(?=\n\s*Step\s+\d+\s*:\s*\nTool call:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    segments: list[TeacherSegment] = []
    for match in pattern.finditer(text):
        call = match.group("call").strip()
        response = match.group("response").strip()
        if call:
            segments.append(TeacherSegment(text=_render_policy_call(call, policy_format), trainable=True, kind="tool_call"))
        if response:
            segments.append(TeacherSegment(text=f"\nTool response:\n{response}\n", trainable=False, kind="tool_response"))
    return segments


def _render_policy_call(call: str, policy_format: str) -> str:
    if policy_format == "appworld_markdown_code":
        return _render_appworld_code(_extract_appworld_code(call), policy_format=policy_format)
    if policy_format == "appworld_code_tag":
        return _render_appworld_code(_extract_appworld_code(call), policy_format=policy_format)
    if policy_format == "text_action_xml" or (policy_format == "auto" and "<action" not in call.lower()):
        return call if re.search(r"<action\b", call, flags=re.IGNORECASE) else f"<action>{call}</action>"
    return call


def _render_appworld_code(code: str, *, policy_format: str) -> str:
    if policy_format == "appworld_code_tag":
        return f"<code>\n{code.strip()}\n</code>"
    return f"```python\n{code.strip()}\n```"


def _extract_appworld_code(call: str) -> str:
    value = call.strip()
    match = re.match(r"execute\(\s*\{\s*[\"']code[\"']\s*:\s*[\"'](?P<code>.*)[\"']\s*\}\s*\)\s*$", value, flags=re.DOTALL)
    if match:
        code = match.group("code")
        return code.replace(r"\"", '"').replace(r"\n", "\n")
    match = re.match(r"execute\((?P<code>.*)\)\s*$", value, flags=re.DOTALL)
    if match:
        return match.group("code").strip()
    return value


def _encode(tok: Any, text: str) -> list[int]:
    if not text:
        return []
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _prompt_token_prefix(anchor: Sample) -> list[int]:
    prompt_len = len(anchor.tokens) - int(anchor.response_length or 0)
    if prompt_len <= 0:
        raise RuntimeError(f"LUFFY anchor sample has no prompt token prefix: sample index={anchor.index}")
    return list(anchor.tokens[:prompt_len])


def _teacher_reward(args: Any, anchor: Sample) -> float:
    source = str(_cfg(args, "teacher_reward_source", "constant") or "constant").strip().lower()
    if source == "constant":
        return _float_cfg(args, "teacher_reward", float_value(reward_cfg_path(args, "outcome", 1.0), 1.0))
    if source != "ropd_teacher_score":
        raise ValueError(
            f"Unsupported reward.luffy.teacher_reward_source={source!r}; "
            "expected 'constant' or 'ropd_teacher_score'"
        )

    sample_metadata = metadata(anchor)
    raw = sample_metadata.get("judge_raw")
    if raw is None:
        rm_reward = sample_metadata.get("rm_reward")
        if isinstance(rm_reward, dict):
            raw = rm_reward.get("raw")
    if not isinstance(raw, dict):
        raise RuntimeError(
            "reward.luffy.teacher_reward_source=ropd_teacher_score requires ROPD judge raw metadata "
            f"before teacher injection: sample index={anchor.index}"
        )
    teacher_scores = raw.get("teacher_scores") or raw.get("ropd_teacher_scores")
    maximum_score = raw.get("maximum_score")
    if not isinstance(teacher_scores, list) or not teacher_scores:
        raise RuntimeError(f"LUFFY requires ROPD teacher_scores in judge raw metadata: sample index={anchor.index}")
    try:
        max_score = float(maximum_score)
        teacher_score = float(teacher_scores[int(_cfg(args, "teacher_score_index", 0) or 0)])
    except (TypeError, ValueError, IndexError) as exc:
        raise RuntimeError(f"Invalid ROPD teacher score metadata for LUFFY: sample index={anchor.index}") from exc
    if max_score <= 0:
        return 0.0
    bounded = max(0.0, min(1.0, teacher_score / max_score))
    weight = _float_cfg(args, "task_success_weight", float(reward_cfg_path(args, "outcome", 10.0) or 10.0))
    return bounded * weight


def _build_teacher_sample(args: Any, anchor: Sample, segments: list[TeacherSegment]) -> Sample:
    tok = tokenizer(args)
    prompt_tokens = _prompt_token_prefix(anchor)
    tokens = list(prompt_tokens)
    response_parts: list[str] = []
    loss_mask: list[int] = []
    off_policy_mask: list[int] = []
    token_segments: list[dict[str, Any]] = []
    for segment in segments:
        ids = _encode(tok, segment.text)
        if not ids:
            continue
        tokens.extend(ids)
        response_parts.append(segment.text)
        loss_mask.extend([0] * len(ids))
        off_policy_mask.extend([1 if segment.trainable else 0] * len(ids))
        token_segments.append(
            {
                "kind": f"luffy_{segment.kind}",
                "role": "assistant" if segment.trainable else "environment",
                "token_count": len(ids),
                "loss_mask_sum": 0,
                "off_policy_loss_mask_sum": len(ids) if segment.trainable else 0,
                "text": segment.text,
                "source_chars": segment.source_chars,
                "truncated": segment.truncated,
            }
        )

    response_length = len(loss_mask)
    if response_length <= 0 or sum(off_policy_mask) <= 0:
        raise RuntimeError(f"LUFFY teacher sample has no trainable response tokens for sample index={anchor.index}")

    max_context = arg(args, "rollout_max_context_len", None)
    if max_context is not None and len(tokens) > int(max_context):
        raise RuntimeError(
            f"LUFFY teacher sample has {len(tokens)} tokens, exceeding rollout_max_context_len={max_context}. "
            "Clean or compress the teacher trace explicitly instead of silently truncating it."
        )

    anchor_metadata = metadata(anchor)
    teacher_reward = _teacher_reward(args, anchor)
    if anchor.rollout_id is None:
        if anchor.index is None:
            raise RuntimeError("LUFFY anchor sample must have index or rollout_id")
        anchor.rollout_id = int(anchor.index)
    teacher = Sample(
        group_index=anchor.group_index,
        index=-abs((int(anchor.index or 0) + 1) * 10_000_003),
        rollout_id=anchor.rollout_id,
        prompt=copy.deepcopy(anchor.prompt),
        tokens=tokens,
        response="".join(response_parts),
        response_length=response_length,
        reward=teacher_reward,
        loss_mask=loss_mask,
        off_policy_loss_mask=off_policy_mask,
        rollout_log_probs=[0.0] * response_length if anchor.rollout_log_probs is not None else None,
        remove_sample=False,
        status=Sample.Status.COMPLETED,
        metadata={
            "off_policy_sample": True,
            "off_policy_method": "luffy",
            "luffy_anchor_sample_index": anchor.index,
            "luffy_anchor_group_index": anchor.group_index,
            "luffy_teacher_reward": teacher_reward,
            "luffy_teacher_token_count": response_length,
            "luffy_teacher_loss_token_count": sum(off_policy_mask),
            "luffy_teacher_source_chars": sum(segment.source_chars or len(segment.text) for segment in segments),
            "luffy_teacher_rendered_chars": sum(len(segment.text) for segment in segments),
            "luffy_teacher_compacted_segment_count": sum(int(segment.truncated) for segment in segments),
            "rm_reward": {
                "score": teacher_reward,
                "components": {"luffy_teacher": teacher_reward},
                "returns_total": True,
                "raw": {
                    "source": "teacher",
                    "anchor_judge_raw": anchor_metadata.get("judge_raw"),
                },
            },
            "raw_reward": teacher_reward,
            "token_segments": token_segments,
        },
    )
    teacher._validate_response_metadata_lengths()
    return teacher
