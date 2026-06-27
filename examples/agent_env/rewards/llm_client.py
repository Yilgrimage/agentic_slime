from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .config import reward_cfg_path
from .extractors import bool_value, read_secret, runtime_env, truncate

DEFAULT_SYSTEM_PROMPT = (
    "You are a strict reward judge. Follow the requested JSON schema exactly. "
    "Do not include markdown, code fences, or text outside the JSON object."
)


def judge_mode(args: Any) -> str:
    mode = runtime_env(args, "AGENT_ENV_JUDGE_MODE", "").strip().lower()
    if not mode:
        mode = str(
            reward_cfg_path(args, "judge_mode", "")
            or reward_cfg_path(args, "judge.mode", "")
            or "none"
        ).strip().lower()
    if mode not in {"none", "aux"}:
        raise ValueError("AGENT_ENV_JUDGE_MODE must be one of: none, aux")
    return mode


def chat_completions_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/api/v3"):
        return f"{base}/chat/completions"
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def extract_json_payload(text: str) -> Any:
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


def coerce_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("scores", "rewards", "results", "judgments", "judgements"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("judge response must be a list or a dict containing scores")


def parse_scores(payload: Any, expected: int) -> tuple[list[float], list[Any]]:
    items = coerce_items(payload)
    if len(items) != expected:
        raise ValueError(f"judge returned {len(items)} scores for {expected} samples")
    scores: list[float] = []
    for item in items:
        value = item.get("score") if isinstance(item, dict) else item
        if value is None and isinstance(item, dict):
            value = item.get("reward")
        scores.append(float(value))
    return scores, items


def parse_single_score(payload: Any) -> tuple[float, Any]:
    if isinstance(payload, dict):
        if "score" in payload:
            return float(payload["score"]), payload
        if "reward" in payload:
            return float(payload["reward"]), payload
        for key in ("scores", "rewards", "results", "judgments", "judgements"):
            items = payload.get(key)
            if isinstance(items, list) and len(items) == 1:
                score, item = parse_single_score(items[0])
                return score, item
    if isinstance(payload, list) and len(payload) == 1:
        return parse_single_score(payload[0])
    return float(payload), payload


async def call_json_judge(args: Any, user_prompt: str, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Any:
    import aiohttp

    base_url = runtime_env(args, "AUX_ENDPOINT_BASE_URL", "").strip()
    model = runtime_env(args, "AUX_ENDPOINT_MODEL", "").strip()
    if not base_url or not model:
        raise RuntimeError("AGENT_ENV_JUDGE_MODE=aux requires AUX_ENDPOINT_BASE_URL and AUX_ENDPOINT_MODEL")

    api_key = runtime_env(args, "AUX_ENDPOINT_API_KEY", "").strip()
    api_key_path = runtime_env(args, "AUX_ENDPOINT_API_KEY_PATH", "").strip()
    api_key = api_key or read_secret(api_key_path)
    timeout_s = float(runtime_env(args, "AUX_ENDPOINT_TIMEOUT_S", "120") or 120)
    max_tokens = int(runtime_env(args, "AUX_ENDPOINT_MAX_TOKENS", "1024") or 1024)
    temperature = float(runtime_env(args, "AUX_ENDPOINT_TEMPERATURE", "0.0") or 0.0)
    top_p = float(runtime_env(args, "AUX_ENDPOINT_TOP_P", "1.0") or 1.0)
    provider = runtime_env(args, "AUX_ENDPOINT_PROVIDER", "").strip().lower()
    enable_thinking = bool_value(runtime_env(args, "AUX_ENDPOINT_ENABLE_THINKING", ""), False)
    separate_reasoning = bool_value(runtime_env(args, "AUX_ENDPOINT_SEPARATE_REASONING", ""), True)
    reasoning_effort = runtime_env(args, "AUX_ENDPOINT_REASONING_EFFORT", "").strip()

    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
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
                async with session.post(chat_completions_url(base_url), json=request_body, headers=headers) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            f"HTTP {response.status} {response.reason}: "
                            f"{truncate(body, 800)}; payload_chars={len(user_prompt)}"
                        )
                    data = await response.json()
            content = data["choices"][0]["message"]["content"]
            try:
                return extract_json_payload(content)
            except Exception as exc:
                raise ValueError(f"judge returned non-JSON content: {truncate(content, 800)}") from exc
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
            await asyncio.sleep(min(2**attempt, 30))
    raise RuntimeError(f"agent-env judge failed after 3 attempts: {last_error!r}")
