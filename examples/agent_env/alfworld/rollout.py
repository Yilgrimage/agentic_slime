from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from examples.agent_env.metrics import log_eval_rollout_data_for_env, log_rollout_data_for_env
from examples.agent_env.rollout import AgentEnvSpec, cfg, first, generate_agent_rollout, metadata

DEFAULT_PROMPT = """You are an expert household task agent in ALFWorld.
At each turn, read the current observation and valid actions, then respond in exactly this format:
<think>
Briefly reason about the goal, the current state, and the best next action.
</think>
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
    if cfg(args, "include_admissible_actions", "alfworld_include_admissible_actions", True):
        text += _format_actions(_admissible(info))
    return text


def _initial_prompt(args: Any, sample: Sample, observation: str, info: dict) -> str:
    base = sample.prompt.strip() if isinstance(sample.prompt, str) and sample.prompt.strip() else DEFAULT_PROMPT
    admissible = _format_actions(_admissible(info)).strip()
    if "{observation}" in base or "{admissible_actions}" in base:
        return base.format(observation=observation.strip(), admissible_actions=admissible)
    return f"{base}\n\n{_observation_text(args, observation, info)}Response:"


def _choose_action(args: Any, action: str, commands: list[str], sample: Sample) -> str:
    if not cfg(args, "restrict_to_admissible", "alfworld_restrict_to_admissible", False) or not commands:
        return action
    norm = {cmd.lower(): cmd for cmd in commands}
    if action.lower() in norm:
        return norm[action.lower()]
    metadata(sample).setdefault("invalid_actions", []).append(action)
    fallback = cfg(args, "invalid_action_fallback", "alfworld_invalid_action_fallback", "model")
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
    arg_legacy={
        "env_split": "alfworld_split",
        "return_logprob": "alfworld_return_logprob",
        "max_turns": "alfworld_max_turns",
        "action_max_tokens": "alfworld_action_max_tokens",
        "generation_stop": "alfworld_stop",
        "format_error_context_tokens": "alfworld_format_error_context_tokens",
        "keep_think_in_context": "alfworld_keep_think_in_context",
        "env_request_timeout_s": "alfworld_env_request_timeout_s",
        "policy_timeout_s": "alfworld_policy_timeout_s",
        "outcome_reward": "alfworld_outcome_reward",
        "reward_source": "alfworld_reward_source",
    },
    default_max_turns=30,
    default_action_max_tokens=512,
    default_reward_source="won",
)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    reset_payload = {
        "direct_game_file": getattr(args, "alfworld_direct_game_file", True),
        "skip_to_task": getattr(args, "alfworld_skip_to_task", False),
        "num_tasks": getattr(args, "alfworld_num_tasks", None),
    }
    return await generate_agent_rollout(args, sample, sampling_params, spec=ALFWORLD_SPEC, reset_payload=reset_payload)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return log_rollout_data_for_env("alfworld", rollout_id, args, samples, rollout_extra_metrics, rollout_time)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_rollout_data_for_env("alfworld", rollout_id, args, data, extra_metrics)
