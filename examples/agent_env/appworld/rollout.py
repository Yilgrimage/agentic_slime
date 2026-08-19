from __future__ import annotations

import re
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.episode import generate_server_episode_rollout
from examples.agent_env.metrics import log_eval_rollout_data_for_env, log_rollout_data_for_env
from examples.agent_env.prompting import require_prompt
from examples.agent_env.rollout import (
    AgentEnvSpec,
    cfg_path,
    parse_tool_call_action,
)


def _available_actions(info: dict) -> list[str]:
    tools = info.get("tools") if info else None
    return [str(tool) for tool in tools] if isinstance(tools, list) else ["execute", "finish"]


def _format_tools(actions: list[str]) -> str:
    return "\nAvailable AppWorld actions:\n" + "\n".join(f"- {action}" for action in actions) + "\n"


def _observation_text(args: Any, observation: str, info: dict) -> str:
    text = f"Observation:\n{observation.strip()}\n"
    if cfg_path(args, "observation.include_actions", False):
        text += _format_tools(_available_actions(info))
    return text


def _initial_prompt(args: Any, sample: Sample, observation: str, info: dict) -> str:
    base = require_prompt(sample.prompt, env_name="AppWorld")
    tools = _format_tools(_available_actions(info)).strip()
    if "{observation}" in base or "{available_tools}" in base:
        return base.format(observation=observation.strip(), available_tools=tools)
    return f"{base}\n\n{_observation_text(args, observation, info)}"


def _parse_code_block(response_text: str) -> tuple[Any, bool, str]:
    code_match = re.search(r"<code>\s*(.*?)\s*</code>", response_text, flags=re.IGNORECASE | re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        if code:
            return {"type": "tool_call", "name": "execute", "arguments": {"code": code}}, True, "code_tag"
        return "", False, "empty_code_tag"

    unterminated = re.search(r"<code>\s*(.*)", response_text, flags=re.IGNORECASE | re.DOTALL)
    if unterminated:
        code = unterminated.group(1).strip()
        if code:
            return {"type": "tool_call", "name": "execute", "arguments": {"code": code}}, False, "unterminated_code_tag"
        return "", False, "empty_unterminated_code_tag"

    fence_match = re.search(r"```\s*(?:python|py)?\s*\n(.*?)```", response_text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        code = fence_match.group(1).strip()
        if code:
            return {"type": "tool_call", "name": "execute", "arguments": {"code": code}}, True, "python_code_fence"
        return "", False, "empty_python_code_fence"

    action, valid, mode = parse_tool_call_action(response_text)
    if isinstance(action, dict):
        name = str(action.get("name") or "")
        if name in {"execute", "python", "python_exec", "finish"}:
            return action, valid, mode

    return "", False, "no_code_block"


def _choose_action(args: Any, action: Any, actions: list[str], sample: Sample) -> Any:
    if isinstance(action, dict):
        name = str(action.get("name") or "")
        if name in {"python", "python_exec"}:
            action = dict(action)
            action["name"] = "execute"
        return action
    text = str(action)
    if cfg_path(args, "action.legacy_text_as_code", False):
        return {"type": "tool_call", "name": "execute", "arguments": {"code": text}}
    return {"type": "tool_call", "name": "format_error", "arguments": {"response": text[:500]}}


def _success(info: dict, score: float) -> bool:
    if info and "success" in info:
        return bool(info["success"])
    return score >= 1.0


def _env_metadata(reset: dict, task_index: int, split: str, lease_id: str | None) -> dict:
    info = reset.get("info") or {}
    return {
        "task_index": task_index,
        "task_id": info.get("task_id"),
        "dataset_name": info.get("dataset_name"),
        "split": split,
        "lease_id": lease_id,
    }


APPWORLD_SPEC = AgentEnvSpec(
    name="appworld",
    env_url_arg="env_server_url",
    default_split="train",
    info_actions=_available_actions,
    observation_text=_observation_text,
    initial_prompt=_initial_prompt,
    choose_action=_choose_action,
    success=_success,
    env_metadata=_env_metadata,
    default_max_turns=40,
    default_response_max_tokens=1024,
    default_reward_source="score",
    default_interaction_mode="text_action",
    parse_action_fn=_parse_code_block,
)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    return await generate_server_episode_rollout(
        args,
        sample,
        sampling_params,
        spec=APPWORLD_SPEC,
        evaluation=evaluation,
    )


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return log_rollout_data_for_env("appworld", rollout_id, args, samples, rollout_extra_metrics, rollout_time)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_rollout_data_for_env("appworld", rollout_id, args, data, extra_metrics)
