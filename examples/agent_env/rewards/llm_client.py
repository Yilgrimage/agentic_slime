from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from .config import reward_cfg_path
from .extractors import bool_value, read_secret, runtime_env, truncate

DEFAULT_SYSTEM_PROMPT = (
    "You are a strict reward judge. Follow the requested JSON schema exactly. "
    "Do not include markdown, code fences, or text outside the JSON object."
)

_ENDPOINT_POOL_CACHE: dict[str, tuple[float, int, list[dict[str, str]]]] = {}
_ENDPOINT_POOL_NEXT: dict[str, int] = {}


def judge_mode(args: Any) -> str:
    mode = str(
        reward_cfg_path(args, "judge_mode", "")
        or reward_cfg_path(args, "judge.mode", "")
        or "none"
    ).strip().lower()
    if mode not in {"none", "aux"}:
        raise ValueError("reward.judge_mode must be one of: none, aux")
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


async def call_json_judge(
    args: Any,
    user_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    api_key: str | None = None,
    api_key_path: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
    response_format: str | dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    max_attempts: int | None = None,
    rate_limit_backoff_s: float | None = None,
    retry_backoff_s: float | None = None,
    endpoint_pool_path: str | None = None,
    payload_validator: Callable[[Any], None] | None = None,
) -> Any:
    payload, _metadata = await call_json_judge_with_metadata(
        args,
        user_prompt,
        system_prompt=system_prompt,
        api_key=api_key,
        api_key_path=api_key_path,
        provider=provider,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        response_format=response_format,
        extra_body=extra_body,
        max_attempts=max_attempts,
        rate_limit_backoff_s=rate_limit_backoff_s,
        retry_backoff_s=retry_backoff_s,
        endpoint_pool_path=endpoint_pool_path,
        payload_validator=payload_validator,
    )
    return payload


async def call_json_judge_with_metadata(
    args: Any,
    user_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    api_key: str | None = None,
    api_key_path: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
    response_format: str | dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    max_attempts: int | None = None,
    rate_limit_backoff_s: float | None = None,
    retry_backoff_s: float | None = None,
    endpoint_pool_path: str | None = None,
    payload_validator: Callable[[Any], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    import aiohttp

    default_base_url = (base_url or runtime_env(args, "AUX_ENDPOINT_BASE_URL", "")).strip()
    default_model = (model or runtime_env(args, "AUX_ENDPOINT_MODEL", "")).strip()
    default_api_key = (api_key or runtime_env(args, "AUX_ENDPOINT_API_KEY", "")).strip()
    default_api_key_path = (api_key_path or runtime_env(args, "AUX_ENDPOINT_API_KEY_PATH", "")).strip()
    pool_path = str(endpoint_pool_path or runtime_env(args, "AUX_ENDPOINT_POOL_PATH", "")).strip()
    if not pool_path and (not default_base_url or not default_model):
        raise RuntimeError("reward.judge_mode=aux requires AUX_ENDPOINT_BASE_URL/AUX_ENDPOINT_MODEL or AUX_ENDPOINT_POOL_PATH")

    timeout_s = float(timeout_s if timeout_s is not None else (runtime_env(args, "AUX_ENDPOINT_TIMEOUT_S", "120") or 120))
    max_tokens = int(max_tokens if max_tokens is not None else (runtime_env(args, "AUX_ENDPOINT_MAX_TOKENS", "1024") or 1024))
    temperature = float(runtime_env(args, "AUX_ENDPOINT_TEMPERATURE", "0.0") or 0.0)
    top_p = float(runtime_env(args, "AUX_ENDPOINT_TOP_P", "1.0") or 1.0)
    default_provider = (provider or runtime_env(args, "AUX_ENDPOINT_PROVIDER", "")).strip().lower()
    enable_thinking = bool_value(runtime_env(args, "AUX_ENDPOINT_ENABLE_THINKING", ""), False)
    separate_reasoning = bool_value(runtime_env(args, "AUX_ENDPOINT_SEPARATE_REASONING", ""), True)
    reasoning_effort = runtime_env(args, "AUX_ENDPOINT_REASONING_EFFORT", "").strip()

    last_error: Exception | None = None
    attempts = max(1, int(max_attempts or 3))
    for attempt in range(attempts):
        started = time.monotonic()
        endpoint = _select_endpoint(pool_path) if pool_path else {}
        request_base_url = str(endpoint.get("base_url") or default_base_url).strip()
        request_model = str(endpoint.get("model") or default_model).strip()
        request_provider = str(endpoint.get("provider") or default_provider).strip().lower()
        request_api_key = str(endpoint.get("api_key") or default_api_key).strip()
        request_api_key_path = str(endpoint.get("api_key_path") or default_api_key_path).strip()
        request_api_key = request_api_key or read_secret(request_api_key_path)
        if not request_base_url or not request_model:
            raise RuntimeError("reward.judge_mode=aux selected endpoint is missing base_url or model")

        request_body = {
            "model": request_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if response_format:
            request_body["response_format"] = _normalize_response_format(response_format)
        if extra_body:
            request_body.update(extra_body)
        if request_provider in {"sglang", "vllm", "aux", "local"}:
            request_body["separate_reasoning"] = separate_reasoning
            request_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        elif request_provider == "deepseek":
            request_body["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
            if enable_thinking and reasoning_effort:
                request_body["reasoning_effort"] = reasoning_effort

        headers = {"Content-Type": "application/json"}
        if request_api_key:
            headers["Authorization"] = f"Bearer {request_api_key}"

        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(chat_completions_url(request_base_url), json=request_body, headers=headers) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            f"HTTP {response.status} {response.reason}: "
                            f"{truncate(body, 800)}; payload_chars={len(user_prompt)}"
                        )
                    data = await response.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text") or item.get("content") or ""))
                    else:
                        parts.append(str(getattr(item, "text", "") or getattr(item, "content", "") or ""))
                content = "\n".join(part for part in parts if part)
            if not str(content).strip():
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                raise ValueError(
                    "judge returned empty content "
                    f"finish_reason={choice.get('finish_reason')!r} "
                    f"reasoning_chars={len(str(reasoning or ''))}"
                )
            try:
                payload = extract_json_payload(content)
            except Exception as exc:
                raise ValueError(f"judge returned non-JSON content: {truncate(content, 800)}") from exc
            if payload_validator is not None:
                payload_validator(payload)
            usage = data.get("usage") if isinstance(data, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            metadata: dict[str, Any] = {
                "provider": request_provider or "openai_compatible",
                "model": request_model,
                "attempt": attempt + 1,
                "latency_s": time.monotonic() - started,
                "prompt_chars": len(user_prompt),
                "max_tokens": float(max_tokens),
            }
            if endpoint:
                metadata["endpoint_name"] = endpoint.get("name") or ""
                metadata["endpoint_pool_size"] = float(endpoint.get("_pool_size") or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if value is not None:
                    try:
                        metadata[key] = float(value)
                    except (TypeError, ValueError):
                        metadata[key] = value
            return payload, metadata
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            if _is_rate_limit_error(exc):
                await asyncio.sleep(float(rate_limit_backoff_s if rate_limit_backoff_s is not None else min(2**attempt, 30)))
            else:
                await asyncio.sleep(float(retry_backoff_s if retry_backoff_s is not None else min(2**attempt, 30)))
    raise RuntimeError(f"agent-env JSON reward request failed after {attempts} attempts: {last_error!r}")


def _select_endpoint(pool_path: str) -> dict[str, str]:
    endpoints = _load_endpoint_pool(pool_path)
    cache_key = str(Path(pool_path).expanduser())
    index = _ENDPOINT_POOL_NEXT.get(cache_key, 0)
    _ENDPOINT_POOL_NEXT[cache_key] = index + 1
    endpoint = dict(endpoints[index % len(endpoints)])
    endpoint["_pool_size"] = str(len(endpoints))
    return endpoint


def _load_endpoint_pool(pool_path: str) -> list[dict[str, str]]:
    path = Path(os.path.expandvars(str(pool_path).strip())).expanduser()
    stat = path.stat()
    cache_key = str(path)
    cached = _ENDPOINT_POOL_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty endpoint pool: {path}")
    parsed: Any | None = None
    if text[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    if parsed is not None:
        if isinstance(parsed, dict):
            raw_items = parsed.get("endpoints") or parsed.get("upstreams")
        else:
            raw_items = parsed
    else:
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not isinstance(raw_items, list):
        raise ValueError(f"endpoint pool must be a JSON list, JSONL, or object with endpoints/upstreams: {path}")

    endpoints = [_normalize_endpoint(item, path, idx) for idx, item in enumerate(raw_items)]
    if not endpoints:
        raise ValueError(f"endpoint pool has no endpoints: {path}")
    _ENDPOINT_POOL_CACHE[cache_key] = (stat.st_mtime, stat.st_size, endpoints)
    return endpoints


def _normalize_endpoint(item: Any, pool_path: Path, idx: int) -> dict[str, str]:
    if not isinstance(item, dict):
        raise ValueError(f"endpoint pool item #{idx} must be an object")
    endpoint = {
        "name": str(item.get("name") or item.get("label") or f"{pool_path.name}:{idx}").strip(),
        "provider": str(item.get("provider") or "openai").strip().lower(),
        "base_url": str(item.get("base_url") or item.get("url") or item.get("endpoint") or "").strip(),
        "model": str(item.get("model") or "").strip(),
        "api_key": str(item.get("api_key") or "").strip(),
        "api_key_path": str(item.get("api_key_path") or item.get("key_path") or "").strip(),
    }
    if not endpoint["base_url"] or not endpoint["model"]:
        raise ValueError(f"endpoint pool item #{idx} must define base_url and model")
    if endpoint["api_key_path"]:
        key_path = Path(os.path.expandvars(endpoint["api_key_path"])).expanduser()
        if not key_path.is_absolute():
            key_path = pool_path.parent / key_path
        if not key_path.is_file():
            raise FileNotFoundError(f"endpoint pool item #{idx} api_key_path does not exist: {key_path}")
        endpoint["api_key_path"] = str(key_path)
    return endpoint


def _normalize_response_format(value: str | dict[str, Any]) -> dict[str, Any] | str:
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if text == "json_object":
        return {"type": "json_object"}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return parsed if isinstance(parsed, dict) else text


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "429",
            "ratelimit",
            "rate limit",
            "too many requests",
            "toomanyrequests",
            "endpointtpmexceeded",
            "tpm",
        )
    )
