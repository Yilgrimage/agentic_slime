from __future__ import annotations

import argparse
import ast
import logging
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from examples.agent_env.server import serve_process_pool

logger = logging.getLogger(__name__)


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
    return {
        "pool_size": int(raw.get("alfworld_server_pool_size", 8)),
        "acquire_timeout_s": float(raw.get("alfworld_server_acquire_timeout_s", 30.0)),
        "lease_ttl_s": float(raw.get("alfworld_server_lease_ttl_s", raw.get("alfworld_server_session_ttl_s", 1800.0))),
        "idempotency_ttl_s": float(raw.get("alfworld_server_idempotency_ttl_s", 300.0)),
        "reuse_workers": bool(raw.get("alfworld_server_reuse_envs", True)),
        "reset_on_release": bool(raw.get("alfworld_server_reset_on_release", False)),
        "worker_start_timeout_s": float(raw.get("alfworld_server_worker_start_timeout_s", 120.0)),
        "worker_request_timeout_s": float(raw.get("alfworld_server_worker_request_timeout_s", 120.0)),
        "prewarm_splits": list(raw.get("alfworld_server_prewarm_splits", ["train"])),
        "honor_direct_game_file": bool(raw.get("alfworld_server_honor_direct_game_file", True)),
    }


def _default_alfworld_config(raw: dict) -> dict:
    data_dir = str(raw.get("alfworld_data_dir") or "$ALFWORLD_DATA").rstrip("/")
    max_steps = int(raw.get("max_turns", raw.get("alfworld_max_turns", 50)))
    return {
        "dataset": {
            "data_path": raw.get("alfworld_data_path") or f"{data_dir}/json_2.1.1/train",
            "eval_id_data_path": raw.get("alfworld_eval_id_data_path") or f"{data_dir}/json_2.1.1/valid_seen",
            "eval_ood_data_path": raw.get("alfworld_eval_ood_data_path") or f"{data_dir}/json_2.1.1/valid_unseen",
            "num_train_games": int(raw.get("alfworld_num_train_games", -1)),
            "num_eval_games": int(raw.get("alfworld_num_eval_games", -1)),
        },
        "env": {
            "type": raw.get("alfworld_env_type") or "AlfredTWEnv",
            "domain_randomization": bool(raw.get("alfworld_domain_randomization", False)),
            "task_types": list(raw.get("alfworld_task_types", [1, 2, 3, 4, 5, 6])),
            "expert_type": raw.get("alfworld_expert_type") or "handcoded",
            "goal_desc_human_anns_prob": float(raw.get("alfworld_goal_desc_human_anns_prob", 0.0)),
        },
        "general": {"training_method": raw.get("alfworld_training_method") or "dqn"},
        "rl": {"training": {"max_nb_steps_per_episode": max_steps}},
        "dagger": {"training": {"max_nb_steps_per_episode": max_steps}},
        "logic": {
            "domain": raw.get("alfworld_domain_path") or f"{data_dir}/logic/alfred.pddl",
            "grammar": raw.get("alfworld_grammar_path") or f"{data_dir}/logic/alfred.twl2",
        },
    }


def _load_configs(path: str, overrides: dict | None = None) -> tuple[dict, dict]:
    raw = _safe_load_config(path)
    server_config = _server_config(raw)
    if raw.get("alfworld_config_path"):
        config = _safe_load_config(raw["alfworld_config_path"])
    elif "dataset" in raw and "env" in raw:
        config = raw
    else:
        config = _default_alfworld_config(raw)
    return _deep_update(config, overrides or {}), server_config


def _select_game_file(game_files: list[str], task_index: int) -> str:
    return game_files[int(task_index) % len(game_files)]


def _alfworld_backend_split(split: str) -> str:
    return {
        "valid_seen": "eval_in_distribution",
        "valid_unseen": "eval_out_of_distribution",
        "eval_seen": "eval_in_distribution",
        "eval_unseen": "eval_out_of_distribution",
    }.get(split, split)


class ALFWorldBackend:
    def __init__(self, worker_id: str, split: str, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.split = split
        self.config = config["alfworld_config"]
        self.env_type = config.get("env_type") or self.config.get("env", {}).get("type", "AlfredTWEnv")
        self.default_direct_game_file = bool(config.get("direct_game_file", True))
        self.honor_direct_game_file = bool(config.get("honor_direct_game_file", True))
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

        if direct_game_file and self.honor_direct_game_file:
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
        return {
            "observation": str(_first(obs, "")),
            "info": self.last_info,
            "split": self.split,
            "game_file": self.game_file,
            "task_index": self.task_index,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
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
        return {
            "observation": str(_first(obs, "")),
            "score": self.final_score,
            "done": self.done,
            "success": self.success,
            "info": self.last_info,
            "game_file": self.game_file,
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
    alfworld_config, server_config = _load_configs(args.config)
    env_config = {
        "alfworld_config": alfworld_config,
        "env_type": args.env_type,
        "direct_game_file": not args.no_direct_game_file,
        "honor_direct_game_file": server_config.get("honor_direct_game_file", True),
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
