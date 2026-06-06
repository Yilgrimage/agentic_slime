from __future__ import annotations

import json
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.metrics import log_eval_rollout_data_for_env, log_rollout_data_for_env
from examples.agent_env.rollout import AgentEnvSpec, cfg, generate_agent_rollout

DEFAULT_PROMPT = """You are an expert AppWorld agent.
At each turn, reason briefly, then call one tool in exactly this format:
<think>
Briefly reason about the task, available apps, and next code snippet.
</think>
<tool_call>
<name>execute</name>
<arguments>{"code": "print(dir(apis.spotify))"}</arguments>
</tool_call>

Use `execute` to run Python code against the AppWorld `apis` object. Inspect available app APIs with `dir(apis.<app>)`, then call the APIs needed to solve the task.
Important rules:
- Use the existing `apis` object only. Do not import `apis`, instantiate hidden classes, or access the real file system.
- Do not guess usernames, passwords, emails, dates, contacts, or payment cards. Personal information and app credentials are available through the supervisor app APIs.
- If an app API says you are unauthorized, inspect `apis.supervisor` for the relevant account information before trying to log in.
- For temporal requests, compute exact time boundaries. Use the phone app or task state for current date/time when needed.
- For paginated APIs, inspect all pages before deciding.
- When the task asks for an answer, finish with just the entity or number, not a full sentence.
When the task is complete, either execute `apis.supervisor.complete_task(...)` yourself or call:
<tool_call>
<name>finish</name>
<arguments>{"answer": "concise answer if the task asks a question"}</arguments>
</tool_call>"""


def _available_actions(info: dict) -> list[str]:
    tools = info.get("tools") if info else None
    return [str(tool) for tool in tools] if isinstance(tools, list) else ["execute", "finish"]


def _format_tools(actions: list[str]) -> str:
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
        if name in {"python", "python_exec"}:
            action = dict(action)
            action["name"] = "execute"
        return action
    text = str(action)
    if cfg(args, "legacy_text_as_code", None, False):
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
    default_env_url="http://127.0.0.1:18183",
    env_url_arg="env_server_url",
    env_url_envvar="APPWORLD_ENV_SERVER_URL",
    default_split="train",
    info_actions=_available_actions,
    observation_text=_observation_text,
    initial_prompt=_initial_prompt,
    choose_action=_choose_action,
    success=_success,
    env_metadata=_env_metadata,
    default_max_turns=40,
    default_action_max_tokens=1024,
    default_reward_source="score",
)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    return await generate_agent_rollout(args, sample, sampling_params, spec=APPWORLD_SPEC)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return log_rollout_data_for_env("appworld", rollout_id, args, samples, rollout_extra_metrics, rollout_time)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_rollout_data_for_env("appworld", rollout_id, args, data, extra_metrics)


def tool_call(name: str, arguments: dict[str, Any]) -> str:
    return "<tool_call>\n<name>{}</name>\n<arguments>{}</arguments>\n</tool_call>".format(
        name,
        json.dumps(arguments, ensure_ascii=False),
    )
