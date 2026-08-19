from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from examples.agent_env.episode import generate_server_episode_rollout
from examples.agent_env.metrics import log_eval_rollout_data_for_env, log_rollout_data_for_env
from examples.agent_env.prompting import require_prompt
from examples.agent_env.rollout import AgentEnvSpec, cfg_path, metadata


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
    if cfg_path(args, "observation.include_actions", True):
        text += _format_actions(_available_actions(info))
    return text


def _initial_prompt(args: Any, sample: Sample, observation: str, info: dict) -> str:
    base = require_prompt(sample.prompt, env_name="WebShop")
    available = _format_actions(_available_actions(info)).strip()
    if "{observation}" in base or "{available_actions}" in base:
        return base.format(observation=observation.strip(), available_actions=available)
    return f"{base}\n\n{_observation_text(args, observation, info)}"


def _choose_action(args: Any, action: str, actions: list[str], sample: Sample) -> str:
    if not cfg_path(args, "action.restrict_to_available", False) or not actions:
        return action
    norm = {cmd.lower(): cmd for cmd in actions}
    if action.lower() in norm:
        return norm[action.lower()]
    metadata(sample).setdefault("invalid_actions", []).append(action)
    fallback = cfg_path(args, "action.invalid_fallback", "model")
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
    env_url_arg="env_server_url",
    default_split="train",
    info_actions=_available_actions,
    observation_text=_observation_text,
    initial_prompt=_initial_prompt,
    choose_action=_choose_action,
    success=_success,
    env_metadata=_env_metadata,
    default_max_turns=15,
    default_response_max_tokens=512,
    default_reward_source="score",
    default_interaction_mode="text_action",
)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    return await generate_server_episode_rollout(
        args,
        sample,
        sampling_params,
        spec=WEBSHOP_SPEC,
        evaluation=evaluation,
    )


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return log_rollout_data_for_env("webshop", rollout_id, args, samples, rollout_extra_metrics, rollout_time)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_rollout_data_for_env("webshop", rollout_id, args, data, extra_metrics)
