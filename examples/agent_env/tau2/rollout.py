from __future__ import annotations

import json
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.metrics import log_eval_rollout_data_for_env, log_rollout_data_for_env
from examples.agent_env.rollout import AgentEnvSpec, cfg, generate_agent_rollout

DEFAULT_PROMPT = """You are an expert task-solving assistant in tau2.
At each turn, inspect the task, policy, and tool results, then respond in exactly this format:
<think>
Briefly reason about the state and the next tool call.
</think>
<tool_call>
<name>tool_name</name>
<arguments>{"argument_name": "argument_value"}</arguments>
</tool_call>

Use domain tools when you need information or need to mutate the environment. When the task is complete, call:
<tool_call>
<name>respond</name>
<arguments>{"message": "concise final response"}</arguments>
</tool_call>"""


def _available_actions(info: dict) -> list[str]:
    tools = info.get("tools") if info else None
    finish_tools = info.get("finish_tools") if info else None
    actions = []
    if isinstance(tools, list):
        actions.extend(str(tool) for tool in tools)
    if isinstance(finish_tools, list):
        actions.extend(str(tool) for tool in finish_tools)
    return actions


def _format_tools(actions: list[str]) -> str:
    if not actions:
        return ""
    return "\nAvailable tool names:\n" + "\n".join(f"- {action}" for action in actions) + "\n"


def _observation_text(args: Any, observation: str, info: dict) -> str:
    text = f"Observation:\n{observation.strip()}\n"
    if cfg(args, "include_available_actions", None, True):
        text += _format_tools(_available_actions(info))
    return text


def _initial_prompt(args: Any, sample: Sample, observation: str, info: dict) -> str:
    base = sample.prompt.strip() if isinstance(sample.prompt, str) and sample.prompt.strip() else DEFAULT_PROMPT
    tools = _format_tools(_available_actions(info)).strip()
    if "{observation}" in base or "{available_tools}" in base:
        return base.format(observation=observation.strip(), available_tools=tools)
    return f"{base}\n\n{_observation_text(args, observation, info)}Response:"


def _choose_action(args: Any, action: Any, actions: list[str], sample: Sample) -> Any:
    if isinstance(action, dict):
        name = str(action.get("name") or "")
        if not cfg(args, "restrict_to_available", None, False) or name in actions or not actions:
            return action
        return {"type": "tool_call", "name": "respond", "arguments": {"message": f"Unable to use unavailable tool: {name}"}}
    text = str(action)
    if not cfg(args, "legacy_text_as_respond", None, True):
        return {"type": "tool_call", "name": text, "arguments": {}}
    return {"type": "tool_call", "name": "respond", "arguments": {"message": text}}


def _success(info: dict, score: float) -> bool:
    if info and "success" in info:
        return bool(info["success"])
    return score > 0


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
    default_action_max_tokens=512,
    default_reward_source="score",
)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    return await generate_agent_rollout(args, sample, sampling_params, spec=TAU2_SPEC)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return log_rollout_data_for_env("tau2", rollout_id, args, samples, rollout_extra_metrics, rollout_time)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_rollout_data_for_env("tau2", rollout_id, args, data, extra_metrics)


def tool_call(name: str, arguments: dict[str, Any]) -> str:
    return "<tool_call>\n<name>{}</name>\n<arguments>{}</arguments>\n</tool_call>".format(
        name,
        json.dumps(arguments, ensure_ascii=False),
    )
