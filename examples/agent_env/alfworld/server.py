from __future__ import annotations

import argparse
import ast
import atexit
import logging
import os
import shutil
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from examples.agent_env.alfworld.task_ids import normalize_alfworld_task_id
from examples.agent_env.env_episode import (
    call_policy_chat,
    choose_text_action,
    finish_reason_is_length,
    parse_text_action,
    policy_context_limit_reached,
)
from examples.agent_env.prompting import require_prompt
from examples.agent_env.server import serve_process_pool

logger = logging.getLogger(__name__)
_FAST_DOWNWARD_PATCHED = False

def _first(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _deep_get(data: dict, *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [item.strip().strip("'\"") for item in text[1:-1].split(",") if item.strip()]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text.strip("'\"")


def _safe_load_config(path: str) -> dict:
    with Path(path).expanduser().open(encoding="utf-8") as f:
        if yaml is not None:
            return yaml.safe_load(f) or {}
        data = {}
        pending_list_key = None
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line:
                continue
            if line[:1].isspace():
                stripped = line.strip()
                if pending_list_key and stripped.startswith("- "):
                    data[pending_list_key].append(_parse_scalar(stripped[2:]))
                    continue
                raise RuntimeError(f"PyYAML is required to load nested ALFWorld configs; unsupported indentation at {path}:{line_no}")
            if ":" not in line:
                raise RuntimeError(f"Invalid flat YAML line at {path}:{line_no}: {raw_line.rstrip()}")
            key, value = line.split(":", 1)
            key = key.strip()
            if value.strip():
                data[key] = _parse_scalar(value)
                pending_list_key = None
            else:
                data[key] = []
                pending_list_key = key
        return data


def _server_config(raw: dict) -> dict:
    env_server = raw.get("env_server") if isinstance(raw.get("env_server"), dict) else {}
    if "pool_size" not in env_server:
        raise ValueError("Missing env_server.pool_size in ALFWorld env_config.yaml")
    return {
        "pool_size": int(env_server["pool_size"]),
        "acquire_timeout_s": float(env_server.get("acquire_timeout_s", 30.0)),
        "lease_ttl_s": float(env_server.get("lease_ttl_s", 1800.0)),
        "idempotency_ttl_s": float(env_server.get("idempotency_ttl_s", 300.0)),
        "reuse_workers": bool(env_server.get("reuse_workers", True)),
        "reset_on_release": bool(env_server.get("reset_on_release", False)),
        "worker_start_timeout_s": float(env_server.get("worker_start_timeout_s", 120.0)),
        "worker_request_timeout_s": float(env_server.get("worker_request_timeout_s", 120.0)),
        "prewarm_splits": list(env_server.get("prewarm_splits", ["train"])),
        "honor_direct_game_file": bool(env_server.get("honor_direct_game_file", True)),
        "require_task_id": bool(env_server.get("require_task_id", True)),
    }


def _runtime_config(raw: dict) -> dict:
    return {
        "max_turns": int(raw.get("max_turns", _deep_get(raw, "alfworld", "max_turns", 30))),
        "timeouts": raw.get("timeouts") if isinstance(raw.get("timeouts"), dict) else {},
        "interaction": raw.get("interaction") if isinstance(raw.get("interaction"), dict) else {},
        "observation": raw.get("observation") if isinstance(raw.get("observation"), dict) else {},
        "action": raw.get("action") if isinstance(raw.get("action"), dict) else {},
    }


def _default_alfworld_config(raw: dict) -> dict:
    alfworld = raw.get("alfworld") if isinstance(raw.get("alfworld"), dict) else {}
    data_dir = os.path.expandvars(
        str(alfworld.get("data_dir") or os.environ.get("AGENT_ENV_DATA_DIR") or os.environ.get("ALFWORLD_DATA", ""))
    ).rstrip("/")
    max_steps = int(raw.get("max_turns", alfworld.get("max_turns", 50)))
    return {
        "dataset": {
            "data_path": alfworld.get("data_path") or f"{data_dir}/json_2.1.1/train",
            "eval_id_data_path": alfworld.get("eval_id_data_path") or f"{data_dir}/json_2.1.1/valid_seen",
            "eval_ood_data_path": alfworld.get("eval_ood_data_path") or f"{data_dir}/json_2.1.1/valid_unseen",
            "num_train_games": int(alfworld.get("num_train_games", -1)),
            "num_eval_games": int(alfworld.get("num_eval_games", -1)),
        },
        "env": {
            "type": alfworld.get("env_type") or "AlfredTWEnv",
            "domain_randomization": bool(alfworld.get("domain_randomization", False)),
            "task_types": list(alfworld.get("task_types", [1, 2, 3, 4, 5, 6])),
            "expert_type": alfworld.get("expert_type") or "handcoded",
            "goal_desc_human_anns_prob": float(alfworld.get("goal_desc_human_anns_prob", 0.0)),
        },
        "general": {"training_method": alfworld.get("training_method") or "dqn"},
        "rl": {"training": {"max_nb_steps_per_episode": max_steps}},
        "dagger": {"training": {"max_nb_steps_per_episode": max_steps}},
        "logic": {
            "domain": alfworld.get("domain_path") or f"{data_dir}/logic/alfred.pddl",
            "grammar": alfworld.get("grammar_path") or f"{data_dir}/logic/alfred.twl2",
        },
    }


def _load_configs(path: str, overrides: dict | None = None) -> tuple[dict, dict, dict]:
    raw = _safe_load_config(path)
    server_config = _server_config(raw)
    config_path = _deep_get(raw, "alfworld", "config_path")
    if config_path:
        config = _safe_load_config(config_path)
    elif "dataset" in raw and "env" in raw:
        config = raw
    else:
        config = _default_alfworld_config(raw)
    return _deep_update(config, overrides or {}), server_config, _runtime_config(raw)


def _select_game_file(game_files: list[str], task_index: int) -> str:
    return game_files[int(task_index) % len(game_files)]


def _select_game_file_by_task_id(game_files: list[str], task_id: str) -> tuple[int, str]:
    normalized = str(task_id).strip()
    for idx, game_file in enumerate(game_files):
        if normalize_alfworld_task_id(game_file) == normalized:
            return idx, game_file
    raise KeyError(f"ALFWorld task_id not found in split game files: {task_id}")


def _task_id_for_game_file(game_file: str | None) -> str | None:
    return normalize_alfworld_task_id(game_file) if game_file else None


def _alfworld_backend_split(split: str) -> str:
    return {
        "valid_seen": "eval_in_distribution",
        "valid_unseen": "eval_out_of_distribution",
        "eval_seen": "eval_in_distribution",
        "eval_unseen": "eval_out_of_distribution",
    }.get(split, split)


def _configure_fast_downward_lib(downward_lib: Any, interface: Any) -> Any:
    downward_lib.load_sas.argtypes = [interface.c_char_p]
    downward_lib.load_sas.restype = None

    downward_lib.load_sas_replan.argtypes = [interface.c_char_p]
    downward_lib.load_sas_replan.restype = None

    downward_lib.cleanup.argtypes = []
    downward_lib.cleanup.restype = None

    downward_lib.get_applicable_operators_count.argtypes = []
    downward_lib.get_applicable_operators_count.restype = int
    downward_lib.get_applicable_operators.argtypes = [interface.POINTER(interface.Operator)]
    downward_lib.get_applicable_operators.restype = None

    downward_lib.get_state_size.argtypes = []
    downward_lib.get_state_size.restype = int
    downward_lib.get_state.argtypes = [interface.POINTER(interface.Atom)]
    downward_lib.get_state.restype = None

    downward_lib.apply_operator.argtypes = [interface.c_int, interface.POINTER(interface.Atom)]
    downward_lib.apply_operator.restype = int

    downward_lib.check_goal.argtypes = []
    downward_lib.check_goal.restype = bool

    downward_lib.solve.argtypes = [interface.c_bool]
    downward_lib.solve.restype = bool

    downward_lib.solve_sas.argtypes = [interface.c_char_p, interface.c_bool]
    downward_lib.solve_sas.restype = bool

    downward_lib.replan.argtypes = [interface.c_bool]
    downward_lib.replan.restype = bool

    downward_lib.get_last_plan_length.argtypes = []
    downward_lib.get_last_plan_length.restype = int

    downward_lib.get_last_plan.argtypes = [interface.POINTER(interface.Operator)]
    downward_lib.get_last_plan.restype = None

    downward_lib.check_solution.argtypes = [interface.c_int, interface.POINTER(interface.Operator)]
    downward_lib.check_solution.restype = bool
    return downward_lib


def _patch_fast_downward_loader() -> None:
    global _FAST_DOWNWARD_PATCHED
    if _FAST_DOWNWARD_PATCHED:
        return

    import fast_downward
    import fast_downward.interface as interface

    source_lib = Path(str(interface.DOWNWARD_LIB_PATH))
    if not source_lib.is_file():
        raise RuntimeError(f"Cannot find Fast Downward library: {source_lib}")

    runtime_root = Path(os.environ.get("LOCAL_RUNTIME_DIR") or "/tmp/server-ops-runtime")
    lib_dir = runtime_root / "alfworld" / "fast_downward" / str(os.getpid())
    lib_dir.mkdir(parents=True, exist_ok=True)
    stable_lib = lib_dir / "libdownward.so"
    if not stable_lib.exists() or stable_lib.stat().st_size != source_lib.stat().st_size:
        tmp_lib = stable_lib.with_name(f"{stable_lib.name}.tmp")
        shutil.copyfile(source_lib, tmp_lib)
        os.replace(tmp_lib, stable_lib)

    cached_lib: Any | None = None

    def load_stable_lib() -> Any:
        nonlocal cached_lib
        if cached_lib is None:
            cached_lib = _configure_fast_downward_lib(interface.cdll.LoadLibrary(str(stable_lib)), interface)
        return cached_lib

    def cleanup_stable_lib() -> None:
        shutil.rmtree(lib_dir, ignore_errors=True)

    fast_downward.load_lib = load_stable_lib
    interface.load_lib = load_stable_lib
    atexit.register(cleanup_stable_lib)
    _FAST_DOWNWARD_PATCHED = True
    logger.info("Patched Fast Downward lib loader to reuse %s", stable_lib)


class ALFWorldBackend:
    def __init__(self, worker_id: str, split: str, config: dict[str, Any]) -> None:
        _patch_fast_downward_loader()
        self.worker_id = worker_id
        self.split = split
        self.config = config["alfworld_config"]
        self.env_type = config.get("env_type") or self.config.get("env", {}).get("type", "AlfredTWEnv")
        self.default_direct_game_file = bool(config.get("direct_game_file", True))
        self.honor_direct_game_file = bool(config.get("honor_direct_game_file", True))
        self.require_task_id = bool(config.get("require_task_id", True))
        self.wrapper: Any | None = None
        self.env: Any | None = None
        self.base_game_files: list[str] = []
        self.loaded_split: str | None = None
        self.registered_game_file: str | None = None
        self.game_file: str | None = None
        self.reset_count = 0
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.success = False
        self.last_info: dict[str, Any] = {}
        self.task_index: int | None = None
        self.requested_task_id: str | None = None

    @property
    def task_id(self) -> str | None:
        return _task_id_for_game_file(self.game_file)

    @property
    def runtime(self) -> dict[str, Any]:
        return self.config.get("_agent_env_runtime", {}) if isinstance(self.config.get("_agent_env_runtime"), dict) else {}

    def _admissible(self, info: dict[str, Any]) -> list[str]:
        commands = _first(info.get("admissible_commands") if info else None, [])
        return [str(command) for command in list(commands or [])]

    def _format_actions(self, commands: list[str]) -> str:
        if not commands:
            return ""
        return "\nValid actions:\n" + "\n".join(f"- {command}" for command in commands) + "\n"

    def _observation_text(self, observation: str, info: dict[str, Any]) -> str:
        text = f"Observation:\n{str(observation).strip()}\n"
        observation_cfg = self.runtime.get("observation") if isinstance(self.runtime.get("observation"), dict) else {}
        if bool(observation_cfg.get("include_actions", True)):
            text += self._format_actions(self._admissible(info))
        return text

    def _initial_messages(self, prompt: str, observation: str, info: dict[str, Any]) -> list[dict[str, str]]:
        system_prompt = require_prompt(prompt, env_name="ALFWorld", source="run_episode.prompt")
        user_prompt = self._observation_text(observation, info).strip()
        if not user_prompt:
            raise ValueError("ALFWorld initial user prompt is empty after reset")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _load_wrapper(self, split: str) -> dict[str, Any]:
        import sys

        alfworld_lib = os.environ.get("ALFWORLD_LIB")
        if alfworld_lib and alfworld_lib not in sys.path:
            sys.path.insert(0, alfworld_lib)
        from alfworld.agents.environment import get_environment

        self._close_env()
        env_cls = get_environment(self.env_type)
        backend_split = _alfworld_backend_split(split)
        self.wrapper = env_cls(self.config, train_eval=backend_split)
        self.base_game_files = list(getattr(self.wrapper, "game_files", None) or [])
        if not self.base_game_files:
            raise RuntimeError(
                f"ALFWorld split={split} backend_split={backend_split} has no games. "
                "Check data_path and game.tw-pddl files."
            )
        self.split = split
        self.loaded_split = split
        self.registered_game_file = None
        self.game_file = None
        return {"num_tasks": len(self.base_game_files)}

    def start(self) -> dict[str, Any]:
        return self._load_wrapper(self.split)

    def _ensure_split(self, split: str) -> None:
        if self.wrapper is None or self.loaded_split != split:
            self._load_wrapper(split)

    def _close_env(self) -> None:
        if self.env is not None and hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                logger.debug("Failed to close ALFWorld TextWorld env", exc_info=True)
        self.env = None

    def _ensure_env_for(self, game_file: str | None) -> None:
        assert self.wrapper is not None
        if game_file is None:
            if self.env is None or self.registered_game_file is not None:
                self._close_env()
                self.wrapper.game_files = self.base_game_files
                self.env = self.wrapper.init_env(batch_size=1)
                self.registered_game_file = None
            return
        if self.env is None or self.registered_game_file != game_file:
            self._close_env()
            # TextWorld captures the registered game list at init_env time.
            # Exact task-index reset therefore needs singleton re-registration.
            self.wrapper.game_files = [game_file]
            self.env = self.wrapper.init_env(batch_size=1)
            self.registered_game_file = game_file

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        split = str(payload.get("split") or self.split)
        self._ensure_split(split)
        self.task_index = int(payload.get("task_index") or 0)
        seed = payload.get("seed", self.task_index)
        direct_game_file = bool(payload.get("direct_game_file", self.default_direct_game_file))
        skip_to_task = bool(payload.get("skip_to_task", False))
        num_tasks = payload.get("num_tasks")
        requested_task_id = str(payload.get("task_id") or "").strip()
        self.requested_task_id = requested_task_id or None
        if self.require_task_id and not requested_task_id:
            raise ValueError("ALFWorld prompt data must provide metadata.task_id")
        if requested_task_id and not (direct_game_file and self.honor_direct_game_file):
            raise ValueError("ALFWorld prompt data provided task_id, but direct game-file selection is disabled")

        if direct_game_file and self.honor_direct_game_file:
            if requested_task_id:
                self.task_index, self.game_file = _select_game_file_by_task_id(self.base_game_files, requested_task_id)
            else:
                self.game_file = _select_game_file(self.base_game_files, self.task_index)
            self._ensure_env_for(self.game_file)
        else:
            self.game_file = None
            self._ensure_env_for(None)

        assert self.env is not None
        if seed is not None and hasattr(self.env, "seed"):
            try:
                self.env.seed(int(seed))
            except Exception:
                logger.debug("ALFWorld env did not accept seed=%s", seed, exc_info=True)
        if skip_to_task and self.task_index > 0 and not direct_game_file:
            skip_count = self.task_index % int(num_tasks) if num_tasks else self.task_index
            for _ in range(skip_count):
                self.env.reset()
        obs, info = self.env.reset()
        self.reset_count += 1
        self.final_score = 0.0
        self.done = False
        self.success = False
        self.last_info = info or {}
        selected_task_id = self.task_id
        if requested_task_id and selected_task_id != requested_task_id:
            raise ValueError(f"ALFWorld selected task_id mismatch: requested={requested_task_id} selected={selected_task_id}")
        self.last_info.setdefault("task_id", selected_task_id)
        if requested_task_id:
            self.last_info.setdefault("requested_task_id", requested_task_id)
        return {
            "observation": str(_first(obs, "")),
            "info": self.last_info,
            "split": self.split,
            "game_file": self.game_file,
            "task_id": selected_task_id,
            "requested_task_id": requested_task_id or None,
            "task_index": self.task_index,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def run_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        if not policy:
            raise ValueError("ALFWorld run_episode requires policy endpoint")
        reset = self.reset(payload)
        observation = str(reset.get("observation", ""))
        info = reset.get("info") if isinstance(reset.get("info"), dict) else {}
        messages = self._initial_messages(str(payload.get("prompt") or ""), observation, info)
        runtime = self.runtime
        action_cfg = runtime.get("action") if isinstance(runtime.get("action"), dict) else {}
        interaction_cfg = runtime.get("interaction") if isinstance(runtime.get("interaction"), dict) else {}
        tag = str(
            (
                interaction_cfg.get("text_action")
                if isinstance(interaction_cfg.get("text_action"), dict)
                else {}
            ).get("tag", "action")
        )
        metadata: dict[str, Any] = {
            "actions": [],
            "action_parse_modes": [],
            "format_checks": [],
            "format_errors": 0,
            "policy_usage": [],
            "turn_count": 0,
        }
        include_trace = bool(payload.get("include_trace", False))
        include_messages = bool(payload.get("include_messages", False)) or include_trace
        if include_messages:
            metadata["messages"] = messages
        if include_trace:
            metadata["turns"] = []

        max_turns = int(payload.get("max_turns") or runtime.get("max_turns") or 30)
        sampling_params = payload.get("sampling_params") if isinstance(payload.get("sampling_params"), dict) else {}
        max_tokens = int(payload.get("max_response_tokens") or 512)
        timeout_s = float((payload.get("timeouts") or {}).get("policy_s") or (runtime.get("timeouts") or {}).get("policy_s") or 300)
        final_score = 0.0
        success = False
        status = "truncated"
        truncated_reason = "max_turns"
        last_step: dict[str, Any] = reset

        for turn in range(max_turns):
            turn_trace: dict[str, Any] = {"turn": turn} if include_trace else {}
            reply = call_policy_chat(
                policy=policy,
                messages=messages,
                sampling_params=sampling_params,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            assistant_message = reply.message
            messages.append(assistant_message)
            metadata["policy_usage"].append(reply.usage)
            if policy_context_limit_reached(reply):
                metadata["context_limit_hits"] = int(metadata.get("context_limit_hits", 0) or 0) + 1
                truncated_reason = "context_limit_after_observation"
                if include_trace:
                    turn_trace.update(
                        {
                            "assistant_message": assistant_message,
                            "format_valid": False,
                            "parse_mode": "context_limit",
                            "finish_reason": reply.finish_reason,
                            "truncated_reason": truncated_reason,
                        }
                    )
                    metadata["turns"].append(turn_trace)
                break
            if finish_reason_is_length(reply):
                metadata["max_response_tokens_hits"] = int(metadata.get("max_response_tokens_hits", 0) or 0) + 1
            action, valid, parse_mode = parse_text_action(str(assistant_message.get("content") or ""), tag=tag)
            metadata["action_parse_modes"].append(parse_mode)
            metadata["format_checks"].append({"turn": turn, "valid": bool(valid), "parse_mode": parse_mode})
            if not valid:
                metadata["format_errors"] = int(metadata.get("format_errors", 0) or 0) + 1
            action = choose_text_action(
                action,
                self._admissible(info),
                restrict_to_available=bool(action_cfg.get("restrict_to_available", False)),
                invalid_fallback=str(action_cfg.get("invalid_fallback") or "model"),
                metadata=metadata,
            )
            metadata["actions"].append(action)
            step = self.step({"action": action})
            last_step = step
            observation = str(step.get("observation", ""))
            info = step.get("info") if isinstance(step.get("info"), dict) else {}
            final_score = float(step.get("score", 0.0) or 0.0)
            done = bool(step.get("done", False))
            success = bool(_first(info.get("won") if info else None, final_score > 0))
            if include_trace:
                turn_trace.update(
                    {
                        "assistant_message": assistant_message,
                        "action": action,
                        "format_valid": bool(valid),
                        "parse_mode": parse_mode,
                        "finish_reason": reply.finish_reason,
                        "env_step": step,
                    }
                )
                metadata["turns"].append(turn_trace)
            if done:
                status = "completed"
                truncated_reason = ""
                break
            messages.append({"role": "user", "content": self._observation_text(observation, info)})

        metadata["turn_count"] = len(metadata["actions"])
        metadata["format_ok"] = int(metadata.get("format_errors", 0) or 0) == 0
        if include_messages:
            metadata["messages"] = messages
        if truncated_reason:
            metadata["truncated_reason"] = truncated_reason
        return {
            "status": status,
            "observation": observation,
            "score": final_score,
            "done": status == "completed",
            "success": success,
            "info": last_step.get("info") if isinstance(last_step.get("info"), dict) else {},
            "game_file": self.game_file,
            "task_id": self.task_id,
            "requested_task_id": self.requested_task_id,
            "task_index": self.task_index,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
            "metadata": metadata,
        }

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.env is not None
        action = str(payload.get("action") or "look")
        obs, scores, dones, info = self.env.step([action])
        self.step_count += 1
        self.final_score = float(_first(scores, 0.0) or 0.0)
        self.done = bool(_first(dones, False))
        won = _first(info.get("won") if info else None, None)
        self.success = bool(won) if won is not None else self.final_score > 0
        self.last_info = info or {}
        self.last_info.setdefault("task_id", self.task_id)
        if self.requested_task_id:
            self.last_info.setdefault("requested_task_id", self.requested_task_id)
        return {
            "observation": str(_first(obs, "")),
            "score": self.final_score,
            "done": self.done,
            "success": self.success,
            "info": self.last_info,
            "game_file": self.game_file,
            "task_id": self.task_id,
            "requested_task_id": self.requested_task_id,
            "task_index": self.task_index,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": float(self.final_score),
            "success": bool(self.success),
            "done": bool(self.done),
            "info": self.last_info,
            "game_file": self.game_file,
            "task_id": self.task_id,
            "requested_task_id": self.requested_task_id,
            "task_index": self.task_index,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        if bool(payload.get("reset_on_release", False)) and self.env is not None:
            self.env.reset()
            self.reset_count += 1
        return {"reset_count": self.reset_count, "step_count": self.step_count}

    def close(self) -> dict[str, Any]:
        self._close_env()
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a process-isolated ALFWorld environment server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-type", default=None)
    parser.add_argument("--no-direct-game-file", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    alfworld_config, server_config, runtime_config = _load_configs(args.config)
    alfworld_config["_agent_env_runtime"] = runtime_config
    env_config = {
        "alfworld_config": alfworld_config,
        "env_type": args.env_type,
        "direct_game_file": not args.no_direct_game_file,
        "honor_direct_game_file": server_config.get("honor_direct_game_file", True),
        "require_task_id": server_config.get("require_task_id", True),
    }
    serve_process_pool(
        host=args.host,
        port=args.port,
        backend_cls=ALFWorldBackend,
        env_config=env_config,
        server_config=server_config,
        env_name="alfworld",
    )


if __name__ == "__main__":
    main()
