from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from slime.utils.types import Sample


SYSTEM_PROMPT = (
    "You are a reward judge for text-action agent trajectories. "
    "Score each sample independently from the trajectory summary and environment name. "
    "Reward valid actions that make progress toward the task. Penalize invalid actions, "
    "repeated loops, format errors, and trajectories that do not solve or approach the task. "
    "Follow the requested JSON schema exactly and use scores in [-1, 1]. "
    "Do not include reasoning, markdown, code fences, or any text outside the JSON object."
)

MAX_PROMPT_CHARS = 800
MAX_RESPONSE_CHARS = 1200
MAX_TURNS = 8
MAX_TURN_TEXT_CHARS = 400
MAX_OBSERVATION_CHARS = 400
MAX_ACTION_CHARS = 160
MAX_ACTIONS = 40


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


def _read_secret(path: str) -> str:
    secret_path = str(path or "").strip()
    if not secret_path:
        return ""
    try:
        return Path(secret_path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _judge_mode(args: Any) -> str:
    mode = _runtime_env(args, "AGENT_ENV_JUDGE_MODE", "none").strip().lower()
    if mode not in {"none", "aux"}:
        raise ValueError("AGENT_ENV_JUDGE_MODE must be one of: none, aux")
    return mode


def _truncate(value: Any, max_chars: int, *, tail: bool = False) -> str:
    if value is None:
        return ""
    text = str(value)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 16:
        return text[-max_chars:] if tail else text[:max_chars]
    if tail:
        return "... " + text[-(max_chars - 4) :]
    head = max(1, (max_chars - 5) // 2)
    tail_len = max(1, max_chars - 5 - head)
    return text[:head] + " ... " + text[-tail_len:]


def _compact_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return _truncate(value, max_chars)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return _truncate(text, max_chars)


def _compact_actions(actions: Any) -> list[Any]:
    if not isinstance(actions, list):
        return []
    selected = actions
    if len(actions) > MAX_ACTIONS:
        selected = actions[:1] + [{"omitted_middle_actions": len(actions) - MAX_ACTIONS}] + actions[-(MAX_ACTIONS - 2) :]
    return [_compact_value(action, MAX_ACTION_CHARS) for action in selected]


def _select_turns(turns: list[Any]) -> list[Any]:
    if len(turns) <= MAX_TURNS:
        return turns
    return turns[:1] + [{"omitted_middle_turns": len(turns) - MAX_TURNS}] + turns[-(MAX_TURNS - 2) :]


def _compact_turn(turn: Any) -> dict[str, Any]:
    if not isinstance(turn, dict):
        return {"turn": _compact_value(turn, MAX_TURN_TEXT_CHARS)}
    parser_text = turn.get("parser_text")
    response_text = parser_text if parser_text not in (None, "") else turn.get("response_text")
    compact = {
        "turn": turn.get("turn"),
        "finish_type": turn.get("finish_type"),
        "response_token_count": turn.get("response_token_count"),
        "parse_mode": turn.get("parse_mode"),
        "format_valid": turn.get("format_valid"),
        "action": _compact_value(turn.get("action"), MAX_ACTION_CHARS),
        "response": _truncate(response_text, MAX_TURN_TEXT_CHARS),
        "observation": _truncate(turn.get("observation"), MAX_OBSERVATION_CHARS, tail=True),
        "done": turn.get("done"),
        "score": turn.get("score"),
        "success": turn.get("success"),
        "discard_reason": turn.get("discard_reason"),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [])}


def _sample_payload(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    turns = metadata.get("turns") if isinstance(metadata.get("turns"), list) else []
    return {
        "index": sample.index,
        "group_index": sample.group_index,
        "status": str(getattr(sample.status, "value", sample.status)),
        "remove_sample": bool(getattr(sample, "remove_sample", False)),
        "prompt": _truncate(sample.prompt, MAX_PROMPT_CHARS),
        "response": _truncate(sample.response, MAX_RESPONSE_CHARS, tail=True),
        "actions": _compact_actions(metadata.get("actions") or []),
        "turns": [
            turn if isinstance(turn, dict) and "omitted_middle_turns" in turn else _compact_turn(turn)
            for turn in _select_turns(turns)
        ],
        "env_score": metadata.get("env_score"),
        "env_success": metadata.get("env_success"),
        "env_reward": metadata.get("env_reward"),
        "turn_count": metadata.get("turn_count", len(metadata.get("actions") or [])),
        "format_errors": metadata.get("format_errors", 0),
        "max_response_tokens_hits": metadata.get("max_response_tokens_hits", 0),
        "truncated_reason": metadata.get("truncated_reason"),
        "discard_reason": metadata.get("discard_reason"),
    }


def _chat_completions_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_json_payload(text: str) -> Any:
    content = str(text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = min([idx for idx in (content.find("{"), content.find("[")) if idx >= 0], default=-1)
        end = max(content.rfind("}"), content.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _coerce_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("scores", "rewards", "results", "judgments", "judgements"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("judge response must be a list or a dict containing scores")


def _parse_scores(payload: Any, expected: int) -> tuple[list[float], list[Any]]:
    items = _coerce_items(payload)
    if len(items) != expected:
        raise ValueError(f"judge returned {len(items)} scores for {expected} samples")
    scores: list[float] = []
    for item in items:
        value = item.get("score") if isinstance(item, dict) else item
        if value is None and isinstance(item, dict):
            value = item.get("reward")
        scores.append(float(value))
    return scores, items


def _parse_single_score(payload: Any) -> tuple[float, Any]:
    if isinstance(payload, dict):
        if "score" in payload:
            return float(payload["score"]), payload
        if "reward" in payload:
            return float(payload["reward"]), payload
        for key in ("scores", "rewards", "results", "judgments", "judgements"):
            items = payload.get(key)
            if isinstance(items, list) and len(items) == 1:
                score, item = _parse_single_score(items[0])
                return score, item
    if isinstance(payload, list) and len(payload) == 1:
        return _parse_single_score(payload[0])
    return float(payload), payload


def _judge_prompt(args: Any, samples: list[Sample], *, single: bool) -> str:
    if single:
        payload = {
            "env_name": getattr(args, "env_name", None),
            "sample": _sample_payload(samples[0]),
        }
        return (
            'Score this trajectory. Output exactly one minified JSON object: '
            '{"score":0.0,"reason":"short reason"}\n\n'
            + json.dumps(payload, ensure_ascii=False)
        )
    payload = {
        "env_name": getattr(args, "env_name", None),
        "samples": [_sample_payload(sample) for sample in samples],
    }
    return (
        "Score this group of trajectories. Output exactly one minified JSON object: "
        '{"scores":[{"score":0.0,"reason":"short reason"}]} with one score object per sample '
        "in the same order.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


async def _call_aux_judge(args: Any, samples: list[Sample], *, single: bool = False) -> tuple[list[float], list[Any]]:
    import aiohttp

    judge_mode = _judge_mode(args)
    if judge_mode == "none":
        return [0.0] * len(samples), [{"skipped": "judge_disabled"} for _ in samples]

    base_url = _runtime_env(args, "AUX_ENDPOINT_BASE_URL", "").strip()
    model = _runtime_env(args, "AUX_ENDPOINT_MODEL", "").strip()
    if not base_url or not model:
        raise RuntimeError(
            "AGENT_ENV_JUDGE_MODE=aux requires AUX_ENDPOINT_BASE_URL and AUX_ENDPOINT_MODEL"
        )

    api_key = _runtime_env(args, "AUX_ENDPOINT_API_KEY", "").strip()
    api_key_path = _runtime_env(args, "AUX_ENDPOINT_API_KEY_PATH", "").strip()
    api_key = api_key or _read_secret(api_key_path)
    timeout_s = float(_runtime_env(args, "AUX_ENDPOINT_TIMEOUT_S", "120") or 120)
    max_tokens = int(_runtime_env(args, "AUX_ENDPOINT_MAX_TOKENS", "1024") or 1024)
    temperature = float(_runtime_env(args, "AUX_ENDPOINT_TEMPERATURE", "0.0") or 0.0)
    top_p = float(_runtime_env(args, "AUX_ENDPOINT_TOP_P", "1.0") or 1.0)
    provider = _runtime_env(args, "AUX_ENDPOINT_PROVIDER", "").strip().lower()
    enable_thinking = _bool_value(_runtime_env(args, "AUX_ENDPOINT_ENABLE_THINKING", ""), False)
    separate_reasoning = _bool_value(_runtime_env(args, "AUX_ENDPOINT_SEPARATE_REASONING", ""), True)
    reasoning_effort = _runtime_env(args, "AUX_ENDPOINT_REASONING_EFFORT", "").strip()

    user_prompt = _judge_prompt(args, samples, single=single)
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if provider in {"sglang", "vllm", "aux", "local"}:
        request_body["separate_reasoning"] = separate_reasoning
        request_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    elif provider == "deepseek":
        request_body["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
        if enable_thinking and reasoning_effort:
            request_body["reasoning_effort"] = reasoning_effort
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(_chat_completions_url(base_url), json=request_body, headers=headers) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            f"HTTP {response.status} {response.reason}: "
                            f"{_truncate(body, 800)}; payload_chars={len(user_prompt)}"
                        )
                    data = await response.json()
            content = data["choices"][0]["message"]["content"]
            try:
                payload = _extract_json_payload(content)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"judge returned non-JSON content: {_truncate(content, 800)}") from exc
            if single:
                score, item = _parse_single_score(payload)
                return [score], [item]
            return _parse_scores(payload, len(samples))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 2:
                break
            await asyncio.sleep(min(2**attempt, 30))
    raise RuntimeError(f"agent-env group judge failed after 3 attempts: {last_error!r}")


def _record_judgment(sample: Sample, score: float, item: Any) -> None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    metadata["judge_score"] = float(score)
    metadata["judge_raw"] = item
    if isinstance(item, dict) and "reason" in item:
        metadata["judge_reason"] = item["reason"]


async def group_reward(args: Any, samples: Sample | list[Sample], **_: Any) -> float | list[float]:
    """Slime custom RM entrypoint for both single-sample and group calls.

    The returned value is the judge score only. Environment score/reward and
    format/truncation adjustments are combined later in custom reward
    post-processing.
    """
    if isinstance(samples, Sample):
        scores, items = await _call_aux_judge(args, [samples], single=True)
        _record_judgment(samples, scores[0], items[0])
        return scores[0]
    if not isinstance(samples, list):
        raise TypeError("examples.agent_env.group_rm.group_reward expects a Sample or list[Sample]")
    scores, items = await _call_aux_judge(args, samples, single=False)
    for sample, score, item in zip(samples, scores, items, strict=True):
        _record_judgment(sample, score, item)
    return scores
