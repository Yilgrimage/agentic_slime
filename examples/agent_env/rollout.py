from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


InfoFn = Callable[[dict], list[str]]
TextFn = Callable[[Any, str, dict], str]
PromptFn = Callable[[Any, Sample, str, dict], str]
ChooseFn = Callable[[Any, Any, list[str], Sample], Any]
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
    arg_legacy: dict[str, str] | None = None
    default_max_turns: int = 20
    default_action_max_tokens: int = 128
    default_reward_source: str = "score"

    def legacy(self, name: str) -> str | None:
        return (self.arg_legacy or {}).get(name)


def arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def cfg(args: Any, name: str, legacy_name: str | None = None, default: Any = None) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    if legacy_name is not None:
        return getattr(args, legacy_name, default)
    return default


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


def task_index(sample: Sample) -> int:
    sample_metadata = metadata(sample)
    if "task_index" in sample_metadata:
        return int(sample_metadata["task_index"])
    if sample.group_index is not None:
        return int(sample.group_index)
    if sample.index is not None:
        return int(sample.index)
    return 0


def tokenizer(args: Any):
    from slime.rollout.sglang_rollout import GenerateState

    return GenerateState(args).tokenizer


def _strip_outer_code_fences(text: str) -> str:
    text = text.strip()
    fence = chr(96) * 3
    if not text.startswith(fence) or not text.endswith(fence):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text.strip(fence).strip()
    return "\n".join(lines[1:-1]).strip()


def _strip_chat_special_tokens(text: str) -> str:
    # Qwen chat delimiters can appear as literal text when rollout prompts are raw strings.
    text = re.sub(r"<\|im_start\|>\s*(?:system|user|assistant)?", "", text)
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    return text.strip()


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


def _parse_tool_call(response_text: str) -> tuple[dict[str, Any] | None, bool, str]:
    response_text = _strip_chat_special_tokens(response_text)
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


def parse_action(response_text: str) -> tuple[Any, bool, str]:
    response_text = _strip_chat_special_tokens(response_text)
    tool_action, tool_valid, tool_mode = _parse_tool_call(response_text)
    if tool_action is not None or tool_mode != "no_tool_call":
        return tool_action or "", tool_valid, tool_mode

    text = _strip_outer_code_fences(response_text)
    action_match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if action_match:
        action = action_match.group(1).strip().strip(chr(34)).strip(chr(39))
        if action:
            return action, True, "action_tag"
        return "", False, "empty_action_tag"

    unterminated_action = re.search(r"<action>\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if unterminated_action:
        action_lines = unterminated_action.group(1).strip().splitlines()
        if not action_lines:
            return "", False, "empty_unterminated_action_tag"
        action = action_lines[0].strip().strip(chr(34)).strip(chr(39))
        if action:
            return action, False, "unterminated_action_tag"
        return "", False, "empty_unterminated_action_tag"

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    for line in text.splitlines() or [text]:
        line = line.strip().strip(chr(34)).strip(chr(39))
        if ":" in line and line.split(":", 1)[0].strip().lower() in {"action", "act"}:
            line = line.split(":", 1)[1].strip()
        line = line.lstrip("-*0123456789. ").strip()
        if line:
            return line, False, "legacy"
    return "look", False, "fallback"


def action_text(action: Any) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        return json.dumps(action, ensure_ascii=False, sort_keys=True)
    return str(action)


def append_tokens(
    sample: Sample,
    text: str,
    token_ids: list[int] | None,
    mask_value: int,
    tok: Any,
    log_probs: list[float] | None = None,
) -> None:
    if not text and not token_ids:
        return
    ids = list(token_ids) if token_ids is not None else tok.encode(text, add_special_tokens=False)
    sample.tokens.extend(ids)
    sample.response += text
    sample.response_length += len(ids)
    metadata(sample).setdefault("token_rewards", []).extend([0.0] * len(ids))
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask.extend([mask_value] * len(ids))
    if sample.rollout_log_probs is not None:
        if log_probs is None:
            log_probs = [0.0] * len(ids)
        sample.rollout_log_probs.extend(log_probs)
        assert len(sample.rollout_log_probs) == sample.response_length


def add_last_token_reward(sample: Sample, value: float) -> bool:
    if value == 0:
        return True
    token_rewards = metadata(sample).setdefault("token_rewards", [])
    if token_rewards:
        token_rewards[-1] += float(value)
        return True
    return False


def add_reward(sample: Sample, value: float) -> None:
    if add_last_token_reward(sample, value):
        return
    sample_metadata = metadata(sample)
    sample_metadata["unassigned_token_reward"] = float(sample_metadata.get("unassigned_token_reward", 0.0)) + float(value)


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

    if cfg(args, "return_logprob", spec.legacy("return_logprob"), True):
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = [0.0] * response_length
        elif len(sample.rollout_log_probs) > response_length:
            sample.rollout_log_probs = sample.rollout_log_probs[-response_length:] if response_length > 0 else []
        elif len(sample.rollout_log_probs) < response_length:
            sample.rollout_log_probs.extend([0.0] * (response_length - len(sample.rollout_log_probs)))


def _decode_tokens(tok: Any, token_ids: list[int], fallback_text: str) -> str:
    if hasattr(tok, "decode"):
        return tok.decode(token_ids, skip_special_tokens=False)
    return fallback_text


def _aligned_log_probs(token_ids: list[int], source_ids: list[int], source_log_probs: list[float]) -> list[float]:
    if not token_ids:
        return []
    if source_log_probs and len(source_ids) >= len(token_ids) and source_ids[-len(token_ids) :] == token_ids:
        return source_log_probs[-len(token_ids) :]
    return [0.0] * len(token_ids)


def response_context(
    args: Any,
    spec: AgentEnvSpec,
    response_text: str,
    response_token_ids: list[int],
    response_log_probs: list[float],
    tok: Any,
    action: Any,
    format_valid: bool,
) -> tuple[str, list[int], list[float]]:
    response_text = _strip_chat_special_tokens(response_text)
    if format_valid:
        if cfg(args, "keep_think_in_context", spec.legacy("keep_think_in_context"), True):
            text = response_text.strip()
        else:
            text = re.sub(r"<think>.*?</think>\s*", "", response_text, flags=re.IGNORECASE | re.DOTALL).strip()
            text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        if not text:
            text = action_text(action)
        ids = tok.encode(text, add_special_tokens=False)
        return text, ids, _aligned_log_probs(ids, response_token_ids, response_log_probs)

    keep = max(0, int(cfg(args, "format_error_context_tokens", spec.legacy("format_error_context_tokens"), 20)))
    if keep == 0:
        return "", [], []
    ids = response_token_ids[-keep:]
    log_probs = response_log_probs[-len(ids) :] if response_log_probs else [0.0] * len(ids)
    text = _strip_chat_special_tokens(_decode_tokens(tok, ids, response_text[-200:]))
    if not text:
        return "", [], []
    ids = tok.encode(text, add_special_tokens=False)
    log_probs = _aligned_log_probs(ids, response_token_ids, response_log_probs)
    return text, ids, log_probs


def turn_params(args: Any, spec: AgentEnvSpec, sampling_params: dict, remaining: int | None) -> dict:
    params = copy.deepcopy(sampling_params)
    max_tokens = int(cfg(args, "action_max_tokens", spec.legacy("action_max_tokens"), spec.default_action_max_tokens))
    if remaining is not None:
        max_tokens = max(0, min(max_tokens, remaining))
    params["max_new_tokens"] = max_tokens
    stop = cfg(args, "generation_stop", spec.legacy("generation_stop"), None)
    if stop is not None:
        params["stop"] = stop
        params["no_stop_trim"] = False
    params.setdefault("skip_special_tokens", True)
    return params


async def call_policy(args: Any, spec: AgentEnvSpec, sample: Sample, sampling_params: dict) -> tuple[str, list[int], list[float], str]:
    from slime.rollout.sglang_rollout import get_model_url

    url = get_model_url(args, "actor", "/generate")
    headers = None
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": sample.session_id}
    payload = {"input_ids": sample.tokens, "sampling_params": sampling_params, "return_logprob": True}
    output = await asyncio.wait_for(
        post(url, payload, headers=headers),
        timeout=float(cfg(args, "policy_timeout_s", spec.legacy("policy_timeout_s"), 60.0)),
    )
    text = output.get("text", "")
    meta = output.get("meta_info", {})
    token_logprobs = meta.get("output_token_logprobs") or []
    finish_type = meta.get("finish_reason", {}).get("type", "stop")
    return text, [item[1] for item in token_logprobs], [item[0] for item in token_logprobs], finish_type


def env_server_url(args: Any, spec: AgentEnvSpec) -> str:
    return str(os.environ.get(spec.env_url_envvar) or arg(args, spec.env_url_arg, None) or spec.default_env_url).rstrip("/")


async def post_env(args: Any, spec: AgentEnvSpec, endpoint: str, payload: dict, max_retries: int = 60) -> dict:
    timeout_s = float(cfg(args, "env_request_timeout_s", spec.legacy("env_request_timeout_s"), 30.0))
    return await asyncio.wait_for(
        post(f"{env_server_url(args, spec)}{endpoint}", payload, max_retries=max_retries),
        timeout=timeout_s,
    )


def lease_request_id(sample: Sample) -> str:
    # Stable across HTTP retries for this in-memory rollout call only.
    return f"sample-{sample.index}-group-{sample.group_index}-obj-{id(sample)}"


async def allocate_env(args: Any, spec: AgentEnvSpec, sample: Sample) -> dict:
    split = metadata(sample).get("split") or cfg(args, "env_split", spec.legacy("env_split"), spec.default_split)
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
        {"split": split, "task_key": f"{split}:{index}", "request_id": lease_request_id(sample)},
    )


async def reset_env(args: Any, spec: AgentEnvSpec, sample: Sample, lease_id: str, extra_payload: dict | None = None) -> dict:
    index = task_index(sample)
    payload = {
        "lease_id": lease_id,
        "split": metadata(sample).get("split") or cfg(args, "env_split", spec.legacy("env_split"), spec.default_split),
        "task_index": index,
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
    reward = float(cfg(args, "outcome_reward", spec.legacy("outcome_reward"), 10.0))
    source = cfg(args, "reward_source", spec.legacy("reward_source"), spec.default_reward_source)
    if source == "score":
        return float(score) * reward
    return reward if success else 0.0


def format_reward(args: Any, valid: bool) -> float:
    if valid:
        return float(arg(args, "format_reward", 0.0))
    return float(arg(args, "format_penalty", -0.1))


def remaining_context(args: Any, sample: Sample) -> int | None:
    max_len = arg(args, "rollout_max_context_len")
    if max_len is None:
        return None
    return int(max_len) - len(sample.tokens)


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
    split = sample_metadata.get("split") or cfg(args, "env_split", spec.legacy("env_split"), spec.default_split)

    try:
        lease = await allocate_env(args, spec, sample)
        lease_id = lease["lease_id"]
        reset = await reset_env(args, spec, sample, lease_id, reset_payload)
        observation = str(reset.get("observation", ""))
        info = reset.get("info") or {}
        split = reset.get("split") or split

        prompt = spec.initial_prompt(args, sample, observation, info)
        sample.prompt = prompt
        sample.tokens = list(tok.encode(prompt, add_special_tokens=False))
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = [] if cfg(args, "return_logprob", spec.legacy("return_logprob"), True) else None
        sample_metadata["token_rewards"] = []
        sample_metadata["actions"] = actions
        sample_metadata["format_errors"] = 0

        max_turns = int(cfg(args, "max_turns", spec.legacy("max_turns"), spec.default_max_turns))
        for _turn in range(max_turns):
            remaining = remaining_context(args, sample)
            if remaining is not None and remaining <= 0:
                sample.status = Sample.Status.TRUNCATED
                sample_metadata["truncated_reason"] = "context_limit"
                break

            params = turn_params(args, spec, sampling_params, remaining)
            if params["max_new_tokens"] <= 0:
                sample.status = Sample.Status.TRUNCATED
                sample_metadata["truncated_reason"] = "context_limit"
                break

            response_text, response_token_ids, response_log_probs, finish_type = await call_policy(args, spec, sample, params)
            if not response_token_ids and response_text:
                response_token_ids = tok.encode(response_text, add_special_tokens=False)
                response_log_probs = [0.0] * len(response_token_ids)

            action, format_valid, parse_mode = parse_action(response_text)
            context_text, context_token_ids, context_log_probs = response_context(
                args,
                spec,
                response_text,
                response_token_ids,
                response_log_probs,
                tok,
                action,
                format_valid,
            )
            append_tokens(sample, context_text, context_token_ids, 1, tok, context_log_probs)
            add_reward(sample, format_reward(args, format_valid))
            sample_metadata.setdefault("action_parse_modes", []).append(parse_mode)
            if not format_valid:
                sample_metadata["format_errors"] = int(sample_metadata.get("format_errors", 0)) + 1

            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break
            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                sample_metadata["truncated_reason"] = "action_max_tokens"
                break

            action = spec.choose_action(args, action, spec.info_actions(info), sample)
            actions.append(action)

            if not sample.response.endswith("\n"):
                append_tokens(sample, "\n", None, 0, tok)

            step = await step_env(args, spec, lease_id, action)
            observation = str(step.get("observation", ""))
            final_score = float(step.get("score", 0.0) or 0.0)
            done = bool(step.get("done", False))
            info = step.get("info") or {}
            success = spec.success(info, final_score)
            env_text = spec.observation_text(args, observation, info)
            if not done:
                env_text += "Response:"
            append_tokens(sample, env_text, None, 0, tok)

            if done:
                sample.status = Sample.Status.COMPLETED
                break
        else:
            sample.status = Sample.Status.TRUNCATED
            sample_metadata["truncated_reason"] = "max_turns"

        env_reward = outcome_reward(args, spec, success, final_score)
        add_reward(sample, env_reward)
        env_meta = spec.env_metadata(reset, index, split, lease_id)
        env_meta.setdefault("server_url", env_server_url(args, spec))
        sample_metadata.update(
            {
                "turn_count": len(actions),
                "format_ok": int(sample_metadata.get("format_errors", 0)) == 0,
                "env_score": final_score,
                "env_success": success,
                "env_reward": env_reward,
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
        ensure_rollout_shapes(args, sample, spec)
        return sample
    except Exception as exc:
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        ensure_rollout_shapes(args, sample, spec)
        metadata(sample).setdefault("error", repr(exc))
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
