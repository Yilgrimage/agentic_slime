from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from examples.agent_env.episode import generate_server_episode_rollout
from examples.agent_env.metrics import log_eval_rollout_data_for_env, log_rollout_data_for_env
from examples.agent_env.rollout import AgentEnvSpec, cfg_path

DEFAULT_PROMPT = """You are an OpenClaw agent.
Use available tools to inspect state, take actions, and complete the task. When the task is complete, call the finish tool if it is available.
Do not send a natural-language message and make a tool call in the same turn."""


def _available_actions(info: dict) -> list[str]:
    tools = info.get("tools") if info else None
    if isinstance(tools, list):
        return [str(tool) for tool in tools]
    schemas = info.get("tool_schemas") if info else None
    actions = []
    for schema in schemas if isinstance(schemas, list) else []:
        fn = schema.get("function") if isinstance(schema, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            actions.append(str(fn["name"]))
    return actions


def _format_tools(actions: list[str]) -> str:
    if not actions:
        return ""
    return "\nAvailable OpenClaw tool names:\n" + "\n".join(f"- {action}" for action in actions) + "\n"


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
        return {"type": "assistant_message", "content": f"Unable to use unavailable OpenClaw tool: {name}"}
    return {"type": "assistant_message", "content": str(action)}


def _success(info: dict, score: float) -> bool:
    if info and "success" in info:
        return bool(info["success"])
    return score >= 1.0


def _env_metadata(reset: dict, task_index: int, split: str, lease_id: str | None) -> dict:
    info = reset.get("info") or {}
    return {
        "task_index": task_index,
        "task_id": info.get("task_id"),
        "openclaw": info.get("openclaw"),
        "openclaw_session_id": info.get("session_id"),
        "split": split,
        "lease_id": lease_id,
    }


OPENCLAW_SPEC = AgentEnvSpec(
    name="openclaw",
    env_url_arg="env_server_url",
    default_split="train",
    info_actions=_available_actions,
    observation_text=_observation_text,
    initial_prompt=_initial_prompt,
    choose_action=_choose_action,
    success=_success,
    env_metadata=_env_metadata,
    default_max_turns=20,
    default_response_max_tokens=1024,
    default_reward_source="score",
    default_interaction_mode="tool_call",
    allow_assistant_message=True,
)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    return await generate_server_episode_rollout(args, sample, sampling_params, spec=OPENCLAW_SPEC)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return log_rollout_data_for_env("openclaw", rollout_id, args, samples, rollout_extra_metrics, rollout_time)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_rollout_data_for_env("openclaw", rollout_id, args, data, extra_metrics)
