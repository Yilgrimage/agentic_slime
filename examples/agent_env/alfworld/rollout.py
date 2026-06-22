from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from examples.agent_env.metrics import log_eval_rollout_data_for_env, log_rollout_data_for_env
from examples.agent_env.rollout import AgentEnvSpec, cfg_path, first, generate_agent_rollout, metadata

DEFAULT_PROMPT = """You are an expert household task agent in ALFWorld.
At each turn, read the current observation and valid actions, then respond in exactly this format:
<action>one valid action</action>

The action text must exactly match one of the valid actions when possible."""


def _admissible(info: dict) -> list[str]:
    commands = first(info.get("admissible_commands") if info else None, [])
    return [str(cmd) for cmd in list(commands or [])]


def _format_actions(commands: list[str]) -> str:
    if not commands:
        return ""
    return "\nValid actions:\n" + "\n".join(f"- {cmd}" for cmd in commands) + "\n"


def _observation_text(args: Any, observation: str, info: dict) -> str:
    text = f"Observation:\n{observation.strip()}\n"
    if cfg_path(args, "observation.include_actions", True):
        text += _format_actions(_admissible(info))
    return text


def _initial_prompt(args: Any, sample: Sample, observation: str, info: dict) -> str:
    base = sample.prompt.strip() if isinstance(sample.prompt, str) and sample.prompt.strip() else DEFAULT_PROMPT
    admissible = _format_actions(_admissible(info)).strip()
    if "{observation}" in base or "{admissible_actions}" in base:
        return base.format(observation=observation.strip(), admissible_actions=admissible)
    return f"{base}\n\n{_observation_text(args, observation, info)}"


def _choose_action(args: Any, action: str, commands: list[str], sample: Sample) -> str:
    if not cfg_path(args, "action.restrict_to_available", False) or not commands:
        return action
    norm = {cmd.lower(): cmd for cmd in commands}
    if action.lower() in norm:
        return norm[action.lower()]
    metadata(sample).setdefault("invalid_actions", []).append(action)
    fallback = cfg_path(args, "action.invalid_fallback", "model")
    if fallback == "first_admissible":
        return commands[0]
    if fallback == "look" and "look" in norm:
        return norm["look"]
    return action


def _success(info: dict, score: float) -> bool:
    won = first(info.get("won") if info else None, None)
    if won is not None:
        return bool(won)
    return score > 0


def _env_metadata(reset: dict, task_index: int, split: str, lease_id: str | None) -> dict:
    return {
        "task_index": task_index,
        "split": split,
        "game_file": reset.get("game_file"),
        "lease_id": lease_id,
    }


ALFWORLD_SPEC = AgentEnvSpec(
    name="alfworld",
    default_env_url="http://127.0.0.1:18080",
    env_url_arg="alfworld_env_server_url",
    env_url_envvar="ALFWORLD_ENV_SERVER_URL",
    default_split="train",
    info_actions=_admissible,
    observation_text=_observation_text,
    initial_prompt=_initial_prompt,
    choose_action=_choose_action,
    success=_success,
    env_metadata=_env_metadata,
    default_max_turns=30,
    default_response_max_tokens=512,
    default_reward_source="won",
    default_interaction_mode="text_action",
)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    reset_payload = {
        "direct_game_file": cfg_path(args, "alfworld.direct_game_file", True),
        "skip_to_task": cfg_path(args, "alfworld.skip_to_task", False),
        "num_tasks": cfg_path(args, "alfworld.num_tasks", None),
    }
    return await generate_agent_rollout(args, sample, sampling_params, spec=ALFWORLD_SPEC, reset_payload=reset_payload)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return log_rollout_data_for_env("alfworld", rollout_id, args, samples, rollout_extra_metrics, rollout_time)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_rollout_data_for_env("alfworld", rollout_id, args, data, extra_metrics)
