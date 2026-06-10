from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from examples.agent_env.metrics import log_eval_rollout_data_for_env, log_rollout_data_for_env
from examples.agent_env.rollout import AgentEnvSpec, cfg_path, generate_agent_rollout

DEFAULT_PROMPT = """You are an expert task-solving assistant in tau2.
Use the available tools to inspect information and mutate the environment according to the domain policy.
Send a normal assistant message when you need to talk to the user. Make a tool call when you need to inspect or update the environment.
Do not send a natural-language message and make a tool call in the same turn."""


def _available_actions(info: dict) -> list[str]:
    tools = info.get("tools") if info else None
    actions = []
    if isinstance(tools, list):
        actions.extend(str(tool) for tool in tools)
    return actions


def _format_tools(actions: list[str]) -> str:
    if not actions:
        return ""
    return "\nAvailable tool names:\n" + "\n".join(f"- {action}" for action in actions) + "\n"


def _observation_text(args: Any, observation: str, info: dict) -> str:
    text = f"Observation:\n{observation.strip()}\n"
    if cfg_path(args, "observation.include_actions", True):
        text += _format_tools(_available_actions(info))
    return text


def _initial_prompt(args: Any, sample: Sample, observation: str, info: dict) -> str:
    base = sample.prompt.strip() if isinstance(sample.prompt, str) and sample.prompt.strip() else DEFAULT_PROMPT
    tools = _format_tools(_available_actions(info)).strip()
    if "{observation}" in base or "{available_tools}" in base:
        return base.format(observation=observation.strip(), available_tools=tools)
    return f"{base}\n\n{_observation_text(args, observation, info)}"


def _choose_action(args: Any, action: Any, actions: list[str], sample: Sample) -> Any:
    if isinstance(action, dict):
        name = str(action.get("name") or "")
        if not cfg_path(args, "action.restrict_to_available", False) or name in actions or not actions:
            return action
        return {"type": "assistant_message", "content": f"Unable to use unavailable tool: {name}"}
    text = str(action)
    return {"type": "assistant_message", "content": text}


def _success(info: dict, score: float) -> bool:
    if info and "success" in info:
        return bool(info["success"])
    return score >= 1.0


def _env_metadata(reset: dict, task_index: int, split: str, lease_id: str | None) -> dict:
    info = reset.get("info") or {}
    return {
        "task_index": task_index,
        "task_id": info.get("task_id"),
        "domain": info.get("domain"),
        "task_set": info.get("task_set"),
        "split": split,
        "lease_id": lease_id,
    }


TAU2_SPEC = AgentEnvSpec(
    name="tau2",
    default_env_url="http://127.0.0.1:18182",
    env_url_arg="env_server_url",
    env_url_envvar="TAU2_ENV_SERVER_URL",
    default_split="train",
    info_actions=_available_actions,
    observation_text=_observation_text,
    initial_prompt=_initial_prompt,
    choose_action=_choose_action,
    success=_success,
    env_metadata=_env_metadata,
    default_max_turns=20,
    default_response_max_tokens=512,
    default_reward_source="score",
    default_interaction_mode="tool_call",
    allow_assistant_message=True,
)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    return await generate_agent_rollout(args, sample, sampling_params, spec=TAU2_SPEC)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return log_rollout_data_for_env("tau2", rollout_id, args, samples, rollout_extra_metrics, rollout_time)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_rollout_data_for_env("tau2", rollout_id, args, data, extra_metrics)
