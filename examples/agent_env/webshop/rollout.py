from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from examples.agent_env.metrics import log_eval_rollout_data_for_env, log_rollout_data_for_env
from examples.agent_env.rollout import AgentEnvSpec, cfg, generate_agent_rollout, metadata

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


def _observation_text(args: Any, observation: str, info: dict) -> str:
    text = f"Observation:\n{observation.strip()}\n"
    if cfg(args, "include_available_actions", None, True):
        text += _format_actions(_available_actions(info))
    return text


def _initial_prompt(args: Any, sample: Sample, observation: str, info: dict) -> str:
    base = sample.prompt.strip() if isinstance(sample.prompt, str) and sample.prompt.strip() else DEFAULT_PROMPT
    available = _format_actions(_available_actions(info)).strip()
    if "{observation}" in base or "{available_actions}" in base:
        return base.format(observation=observation.strip(), available_actions=available)
    return f"{base}\n\n{_observation_text(args, observation, info)}Response:"


def _choose_action(args: Any, action: str, actions: list[str], sample: Sample) -> str:
    if not cfg(args, "restrict_to_available", None, False) or not actions:
        return action
    norm = {cmd.lower(): cmd for cmd in actions}
    if action.lower() in norm:
        return norm[action.lower()]
    metadata(sample).setdefault("invalid_actions", []).append(action)
    fallback = cfg(args, "invalid_action_fallback", None, "model")
    if fallback == "first_available":
        return actions[0]
    return action


def _success(info: dict, score: float) -> bool:
    if info and "done" in info:
        return bool(info.get("done")) and score > 0
    return score > 0


def _env_metadata(reset: dict, task_index: int, split: str, lease_id: str | None) -> dict:
    return {"task_index": task_index, "split": split, "lease_id": lease_id}


WEBSHOP_SPEC = AgentEnvSpec(
    name="webshop",
    default_env_url="http://127.0.0.1:18180",
    env_url_arg="env_server_url",
    env_url_envvar="WEBSHOP_ENV_SERVER_URL",
    default_split="train",
    info_actions=_available_actions,
    observation_text=_observation_text,
    initial_prompt=_initial_prompt,
    choose_action=_choose_action,
    success=_success,
    env_metadata=_env_metadata,
    default_max_turns=20,
    default_action_max_tokens=128,
    default_reward_source="score",
)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    return await generate_agent_rollout(args, sample, sampling_params, spec=WEBSHOP_SPEC)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return log_rollout_data_for_env("webshop", rollout_id, args, samples, rollout_extra_metrics, rollout_time)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_rollout_data_for_env("webshop", rollout_id, args, data, extra_metrics)
