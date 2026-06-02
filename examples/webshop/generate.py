from __future__ import annotations

import copy
import asyncio
import logging
import os
import re
from typing import Any

from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """You are an expert shopping agent in WebShop.
At each turn, read the current webpage observation and available actions, then respond in exactly this format:
<think>
Briefly reason about the shopping instruction, current page, and best next action.
</think>
<action>one valid action</action>

Actions must use WebShop syntax:
- search[query words]
- click[visible option or button text]

The action text should exactly match one available action when possible."""


def _arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _cfg(args: Any, name: str, legacy_name: str | None = None, default: Any = None) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    if legacy_name is not None:
        return getattr(args, legacy_name, default)
    return default


def _tokenizer(args: Any):
    from slime.rollout.sglang_rollout import GenerateState

    return GenerateState(args).tokenizer


def _metadata(sample: Sample) -> dict:
    if sample.metadata is None:
        sample.metadata = {}
    return sample.metadata


def _task_index(args: Any, sample: Sample) -> int:
    metadata = _metadata(sample)
    if "task_index" in metadata:
        return int(metadata["task_index"])
    if sample.group_index is not None:
        return int(sample.group_index)
    if sample.index is not None:
        return int(sample.index)
    return 0


def _first(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _available_actions(info: dict) -> list[str]:
    available = info.get("available_actions") if info else None
    if isinstance(available, list):
        return [str(action) for action in available]
    if not isinstance(available, dict):
        return []
    actions: list[str] = []
    if available.get("has_search_bar"):
        actions.append("search[query words]")
    for item in available.get("clickables") or []:
        actions.append(f"click[{item}]")
    return actions


def _format_actions(actions: list[str]) -> str:
    if not actions:
        return ""
    return "\nAvailable actions:\n" + "\n".join(f"- {action}" for action in actions) + "\n"


def _obs_block(args: Any, observation: str, info: dict) -> str:
    text = f"Observation:\n{observation.strip()}\n"
    if _cfg(args, "include_available_actions", None, True):
        text += _format_actions(_available_actions(info))
    return text


def _initial_prompt(args: Any, sample: Sample, observation: str, info: dict) -> str:
    base = sample.prompt.strip() if isinstance(sample.prompt, str) and sample.prompt.strip() else DEFAULT_PROMPT
    available = _format_actions(_available_actions(info)).strip()
    if "{observation}" in base or "{available_actions}" in base:
        return base.format(observation=observation.strip(), available_actions=available)
    return f"{base}\n\n{_obs_block(args, observation, info)}Response:"


def _append(
    sample: Sample,
    text: str,
    token_ids: list[int] | None,
    mask_value: int,
    tokenizer: Any,
    log_probs: list[float] | None = None,
) -> None:
    if not text and not token_ids:
        return
    ids = list(token_ids) if token_ids is not None else tokenizer.encode(text, add_special_tokens=False)
    sample.tokens.extend(ids)
    sample.response += text
    sample.response_length += len(ids)
    _metadata(sample).setdefault("token_rewards", []).extend([0.0] * len(ids))
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask.extend([mask_value] * len(ids))
    if sample.rollout_log_probs is not None:
        if log_probs is None:
            log_probs = [0.0] * len(ids)
        sample.rollout_log_probs.extend(log_probs)
        assert len(sample.rollout_log_probs) == sample.response_length


def _add_last_token_reward(sample: Sample, value: float) -> None:
    if value == 0:
        return
    token_rewards = _metadata(sample).setdefault("token_rewards", [])
    if token_rewards:
        token_rewards[-1] += float(value)


def _ensure_rollout_shapes(args: Any, sample: Sample) -> None:
    response_length = int(sample.response_length or 0)
    metadata = _metadata(sample)
    token_rewards = list(metadata.get("token_rewards") or [])
    if len(token_rewards) < response_length:
        token_rewards.extend([0.0] * (response_length - len(token_rewards)))
    metadata["token_rewards"] = token_rewards[:response_length]

    if sample.loss_mask is None:
        sample.loss_mask = [0] * response_length
    elif len(sample.loss_mask) < response_length:
        sample.loss_mask.extend([0] * (response_length - len(sample.loss_mask)))
    elif len(sample.loss_mask) > response_length:
        sample.loss_mask = sample.loss_mask[:response_length]

    if _cfg(args, "return_logprob", None, True):
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = [0.0] * response_length
        elif len(sample.rollout_log_probs) < response_length:
            sample.rollout_log_probs.extend([0.0] * (response_length - len(sample.rollout_log_probs)))
        elif len(sample.rollout_log_probs) > response_length:
            sample.rollout_log_probs = sample.rollout_log_probs[:response_length]


def _parse_action(response_text: str) -> tuple[str, bool, str]:
    text = response_text.replace(chr(96) * 3, "").strip()
    action_match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if action_match:
        action = action_match.group(1).strip().strip(chr(34)).strip(chr(39))
        if action:
            return action, True, "action_tag"

    unterminated_action = re.search(r"<action>\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if unterminated_action:
        action = unterminated_action.group(1).strip().splitlines()[0].strip().strip(chr(34)).strip(chr(39))
        if action:
            return action, False, "unterminated_action_tag"

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    for line in text.splitlines() or [text]:
        line = line.strip().strip(chr(34)).strip(chr(39))
        if ":" in line and line.split(":", 1)[0].strip().lower() in {"action", "act"}:
            line = line.split(":", 1)[1].strip()
        line = line.lstrip("-*0123456789. ").strip()
        if line:
            return line, False, "legacy"
    return "look", False, "fallback"


def _decode_tokens(tokenizer: Any, token_ids: list[int], fallback_text: str) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    return fallback_text


def _response_context(
    args: Any,
    response_text: str,
    response_token_ids: list[int],
    response_log_probs: list[float],
    tokenizer: Any,
    action: str,
    format_valid: bool,
) -> tuple[str, list[int], list[float]]:
    if format_valid:
        text = re.sub(r"<think>.*?</think>\s*", "", response_text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        if not text:
            text = f"<action>{action}</action>"
        ids = tokenizer.encode(text, add_special_tokens=False)
        return text, ids, [0.0] * len(ids)

    keep = max(0, int(_cfg(args, "format_error_context_tokens", None, 20)))
    if keep == 0:
        return "", [], []
    ids = response_token_ids[-keep:]
    log_probs = response_log_probs[-len(ids) :] if response_log_probs else [0.0] * len(ids)
    text = _decode_tokens(tokenizer, ids, response_text[-200:])
    return text, ids, log_probs


def _choose_action(args: Any, action: str, actions: list[str], sample: Sample) -> str:
    if not _cfg(args, "restrict_to_available", None, False) or not actions:
        return action
    norm = {cmd.lower(): cmd for cmd in actions}
    if action.lower() in norm:
        return norm[action.lower()]
    _metadata(sample).setdefault("invalid_actions", []).append(action)
    fallback = _cfg(args, "invalid_action_fallback", None, "model")
    if fallback == "first_available":
        return actions[0]
    return action


def _turn_params(args: Any, sampling_params: dict, remaining: int | None) -> dict:
    params = copy.deepcopy(sampling_params)
    max_tokens = int(_cfg(args, "action_max_tokens", None, 128))
    if remaining is not None:
        max_tokens = max(0, min(max_tokens, remaining))
    params["max_new_tokens"] = max_tokens
    stop = _cfg(args, "generation_stop", None, None)
    if stop is not None:
        params["stop"] = stop
        params["no_stop_trim"] = False
    params.setdefault("skip_special_tokens", True)
    return params


async def _call_policy(args: Any, sample: Sample, sampling_params: dict) -> tuple[str, list[int], list[float], str]:
    from slime.rollout.sglang_rollout import get_model_url

    url = get_model_url(args, "actor", "/generate")
    headers = None
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": sample.session_id}
    payload = {"input_ids": sample.tokens, "sampling_params": sampling_params, "return_logprob": True}
    output = await asyncio.wait_for(
        post(url, payload, headers=headers),
        timeout=float(_cfg(args, "policy_timeout_s", None, 60.0)),
    )
    text = output.get("text", "")
    meta = output.get("meta_info", {})
    token_logprobs = meta.get("output_token_logprobs") or []
    finish_type = meta.get("finish_reason", {}).get("type", "stop")
    return text, [item[1] for item in token_logprobs], [item[0] for item in token_logprobs], finish_type


def _env_server_url(args: Any) -> str:
    return str(
        _arg(args, "env_server_url", None) or os.environ.get("WEBSHOP_ENV_SERVER_URL", "http://127.0.0.1:18180")
    ).rstrip("/")


async def _post_env(args: Any, endpoint: str, payload: dict, max_retries: int = 60) -> dict:
    timeout_s = float(_cfg(args, "env_request_timeout_s", None, 30.0))
    return await asyncio.wait_for(
        post(f"{_env_server_url(args)}{endpoint}", payload, max_retries=max_retries),
        timeout=timeout_s,
    )


def _lease_request_id(sample: Sample) -> str:
    # This only needs to be stable across HTTP retries for the same in-memory
    # rollout call. Do not use sample.session_id alone: routing/session keys can
    # be shared by grouped samples, which would incorrectly reuse one env lease.
    return f"sample-{sample.index}-group-{sample.group_index}-obj-{id(sample)}"


async def _allocate_env(args: Any, sample: Sample) -> dict:
    split = _metadata(sample).get("split") or _cfg(args, "env_split", None, "train")
    task_index = _task_index(args, sample)
    payload = {
        "split": split,
        "task_key": f"{split}:{task_index}",
        "request_id": _lease_request_id(sample),
    }
    return await _post_env(args, "/allocate", payload)


async def _reset_env(args: Any, sample: Sample, lease_id: str) -> dict:
    task_index = _task_index(args, sample)
    payload = {
        "lease_id": lease_id,
        "split": _metadata(sample).get("split") or _cfg(args, "env_split", None, "train"),
        "task_index": task_index,
        "seed": _metadata(sample).get("seed", sample.group_index if sample.group_index is not None else sample.index),
    }
    return await _post_env(args, "/reset", payload)


async def _step_env(args: Any, lease_id: str, action: str) -> dict:
    return await _post_env(args, "/step", {"lease_id": lease_id, "action": action})


async def _evaluate_env(args: Any, lease_id: str | None) -> dict:
    if not lease_id:
        return {}
    return await _post_env(args, "/evaluate", {"lease_id": lease_id}, max_retries=3)


async def _close_env(args: Any, lease_id: str | None) -> None:
    if not lease_id:
        return
    try:
        await _post_env(args, "/close", {"lease_id": lease_id}, max_retries=3)
    except Exception:
        logger.debug("Failed to close WebShop server lease %s", lease_id, exc_info=True)


def _success(info: dict, score: float) -> bool:
    if info and "done" in info:
        return bool(info.get("done")) and score > 0
    return score > 0


def _reward(args: Any, success: bool, score: float) -> float:
    outcome_reward = float(_cfg(args, "outcome_reward", None, 10.0))
    if _cfg(args, "reward_source", None, "score") == "score":
        return float(score) * outcome_reward
    return outcome_reward if success else 0.0


def _format_reward(args: Any, valid: bool) -> float:
    if valid:
        return float(_arg(args, "format_reward", 0.0))
    return float(_arg(args, "format_penalty", 0.0))


def _remaining(args: Any, sample: Sample) -> int | None:
    max_len = _arg(args, "rollout_max_context_len")
    if max_len is None:
        return None
    return int(max_len) - len(sample.tokens)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    assert not _arg(args, "partial_rollout", False), "WebShop rollout does not support partial rollout yet."

    tokenizer = _tokenizer(args)
    actions: list[str] = []
    final_score = 0.0
    success = False
    task_index = _task_index(args, sample)
    lease_id = None
    metadata = _metadata(sample)
    split = metadata.get("split") or _cfg(args, "env_split", None, "train")

    try:
        lease = await _allocate_env(args, sample)
        lease_id = lease["lease_id"]
        reset = await _reset_env(args, sample, lease_id)
        observation = str(reset.get("observation", ""))
        info = reset.get("info") or {}
        split = reset.get("split") or split

        prompt = _initial_prompt(args, sample, observation, info)
        sample.prompt = prompt
        sample.tokens = list(tokenizer.encode(prompt, add_special_tokens=False))
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = [] if _cfg(args, "return_logprob", None, True) else None
        metadata["token_rewards"] = []
        metadata["actions"] = actions
        metadata["format_errors"] = 0

        for _turn in range(int(_cfg(args, "max_turns", None, 20))):
            remaining = _remaining(args, sample)
            if remaining is not None and remaining <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            turn_params = _turn_params(args, sampling_params, remaining)
            if turn_params["max_new_tokens"] <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            response_text, response_token_ids, response_log_probs, finish_type = await _call_policy(args, sample, turn_params)
            if not response_token_ids and response_text:
                response_token_ids = tokenizer.encode(response_text, add_special_tokens=False)
                response_log_probs = [0.0] * len(response_token_ids)

            action, format_valid, parse_mode = _parse_action(response_text)
            context_text, context_token_ids, context_log_probs = _response_context(
                args,
                response_text,
                response_token_ids,
                response_log_probs,
                tokenizer,
                action,
                format_valid,
            )
            _append(sample, context_text, context_token_ids, 1, tokenizer, context_log_probs)
            _add_last_token_reward(sample, _format_reward(args, format_valid))
            metadata.setdefault("action_parse_modes", []).append(parse_mode)
            if not format_valid:
                metadata["format_errors"] = int(metadata.get("format_errors", 0)) + 1

            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break
            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                break

            action = _choose_action(args, action, _available_actions(info), sample)
            actions.append(action)

            if not sample.response.endswith("\n"):
                _append(sample, "\n", None, 0, tokenizer)

            step = await _step_env(args, lease_id, action)
            observation = str(step.get("observation", ""))
            final_score = float(step.get("score", 0.0) or 0.0)
            done = bool(step.get("done", False))
            info = step.get("info") or {}
            success = _success(info, final_score)
            env_text = _obs_block(args, observation, info)
            if not done:
                env_text += "Response:"
            _append(sample, env_text, None, 0, tokenizer)

            if done:
                sample.status = Sample.Status.COMPLETED
                break
        else:
            sample.status = Sample.Status.TRUNCATED

        env_reward = _reward(args, success, final_score)
        _add_last_token_reward(sample, env_reward)
        metadata.update(
            {
                "turn_count": len(actions),
                "format_ok": int(metadata.get("format_errors", 0)) == 0,
                "env_score": final_score,
                "env_success": success,
                "env_reward": env_reward,
                "webshop": {
                    "task_index": task_index,
                    "split": split,
                    "server_url": _env_server_url(args),
                    "lease_id": lease_id,
                },
            }
        )

        if sample.status == Sample.Status.ABORTED:
            sample.reward = 0.0
        elif _arg(args, "use_opd", False) and _arg(args, "opd_type") == "sglang":
            sample.reward = None
        else:
            sample.reward = float(sum(metadata.get("token_rewards", [])))
        _ensure_rollout_shapes(args, sample)
        return sample
    except Exception as exc:
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        _ensure_rollout_shapes(args, sample)
        _metadata(sample).setdefault("error", repr(exc))
        logger.exception("WebShop rollout failed")
        return sample
    finally:
        if lease_id is not None:
            try:
                eval_payload = await _evaluate_env(args, lease_id)
                if eval_payload:
                    _metadata(sample).setdefault("env_evaluate", eval_payload)
            except Exception:
                    logger.debug("Failed to evaluate WebShop lease %s before close", lease_id, exc_info=True)
        await _close_env(args, lease_id)
