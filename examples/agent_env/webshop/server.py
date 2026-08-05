from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

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

def _load_text_env_class(webshop_lib: str | None):
    if not webshop_lib:
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

        return WebAgentTextEnv

    module_path = Path(webshop_lib) / "web_agent_site" / "envs" / "web_agent_text_env.py"
    spec = importlib.util.spec_from_file_location("webshop_text_env", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load WebShop text env from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WebAgentTextEnv


def _install_text_env_import_stubs() -> None:
    import sys
    import types

    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch_stub = types.ModuleType("torch")
            torch_stub.load = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("torch is not installed in the WebShop env; image observations are disabled")
            )
            torch_stub.zeros = lambda *args, **kwargs: [0.0] * (int(args[0]) if args else 0)
            torch_stub.set_default_tensor_type = lambda *args, **kwargs: None
            torch_stub.FloatTensor = list
            sys.modules["torch"] = torch_stub

    if "pyserini.encode" not in sys.modules:
        encode_stub = types.ModuleType("pyserini.encode")

        class _UnusedEncoder:
            def __init__(self, *args, **kwargs) -> None:
                raise RuntimeError("Pyserini dense/impact encoders are disabled for WebShop text env")

        for name in [
            "QueryEncoder",
            "TokFreqQueryEncoder",
            "UniCoilQueryEncoder",
            "CachedDataQueryEncoder",
            "SpladeQueryEncoder",
        ]:
            setattr(encode_stub, name, _UnusedEncoder)
        sys.modules["pyserini.encode"] = encode_stub

    if "pyserini.search.faiss" not in sys.modules:
        faiss_stub = types.ModuleType("pyserini.search.faiss")

        class _UnusedDenseSearch:
            def __init__(self, *args, **kwargs) -> None:
                raise RuntimeError("Pyserini dense/faiss search is disabled for WebShop text env")

        for name in [
            "DenseSearchResult",
            "PRFDenseSearchResult",
            "FaissSearcher",
            "BinaryDenseSearcher",
            "QueryEncoder",
            "DprQueryEncoder",
            "BprQueryEncoder",
            "DkrrDprQueryEncoder",
            "TctColBertQueryEncoder",
            "AnceQueryEncoder",
            "AutoQueryEncoder",
            "AnceEncoder",
            "DenseVectorAveragePrf",
            "DenseVectorRocchioPrf",
            "DenseVectorAncePrf",
        ]:
            setattr(faiss_stub, name, _UnusedDenseSearch)
        sys.modules["pyserini.search.faiss"] = faiss_stub


def _deep_get(raw: dict, section: str, key: str, default: Any = None) -> Any:
    value = raw.get(key)
    if value is not None:
        return value
    nested = raw.get(section)
    if isinstance(nested, dict):
        return nested.get(key, default)
    return default


def _server_config(raw: dict) -> dict:
    pool_size = _deep_get(raw, "env_server", "pool_size", None)
    if pool_size is None:
        raise ValueError("Missing env_server.pool_size in WebShop env_config.yaml")
    return {
        "pool_size": int(pool_size),
        "acquire_timeout_s": float(_deep_get(raw, "env_server", "acquire_timeout_s", 600.0)),
        "lease_ttl_s": float(_deep_get(raw, "env_server", "lease_ttl_s", 1800.0)),
        "idempotency_ttl_s": float(_deep_get(raw, "env_server", "idempotency_ttl_s", 300.0)),
        "worker_start_timeout_s": float(_deep_get(raw, "env_server", "worker_start_timeout_s", 300.0)),
        "worker_request_timeout_s": float(_deep_get(raw, "env_server", "worker_request_timeout_s", 180.0)),
        "prewarm_splits": list(_deep_get(raw, "env_server", "prewarm_splits", ["train"])),
        "reuse_workers": True,
        "reset_on_release": False,
    }


def _environment_config(raw: dict) -> dict:
    default_data_dir = os.environ.get("AGENT_ENV_DATA_DIR") or os.environ.get("WEBSHOP_DATA", "")
    data_dir = os.path.expandvars(str(_deep_get(raw, "webshop", "data_dir", default_data_dir)))
    product_file = _deep_get(raw, "webshop", "product_file", None)
    attr_file = _deep_get(raw, "webshop", "attr_file", None)
    num_products = _deep_get(raw, "webshop", "num_products", None)
    missing = [
        name
        for name, value in (
            ("webshop.product_file", product_file),
            ("webshop.attr_file", attr_file),
            ("webshop.num_products", num_products),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required WebShop env_config.yaml fields: {', '.join(missing)}")
    return {
        "env_id": _deep_get(raw, "webshop", "env_id", "WebAgentTextEnv-v0"),
        "observation_mode": _deep_get(raw, "webshop", "observation_mode", "text"),
        "data_dir": data_dir,
        "product_file": os.path.expandvars(str(product_file)) if product_file else None,
        "attr_file": os.path.expandvars(str(attr_file)) if attr_file else None,
        "num_products": num_products,
        "human_goals": _deep_get(raw, "webshop", "human_goals", True),
        "_agent_env_runtime": {
            "max_turns": int(raw.get("max_turns", _deep_get(raw, "webshop", "max_turns", 20))),
            "timeouts": raw.get("timeouts") if isinstance(raw.get("timeouts"), dict) else {},
            "interaction": raw.get("interaction") if isinstance(raw.get("interaction"), dict) else {},
            "observation": raw.get("observation") if isinstance(raw.get("observation"), dict) else {},
            "action": raw.get("action") if isinstance(raw.get("action"), dict) else {},
        },
    }


def _load_config(path: str) -> tuple[dict, dict]:
    with Path(path).expanduser().open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _environment_config(raw), _server_config(raw)


def _available_actions(env: Any, info: dict | None = None) -> list[str]:
    info = info or {}
    value = info.get("available_actions")
    if value is None and hasattr(env, "get_available_actions"):
        value = env.get_available_actions()
    if value is None and hasattr(env, "available_actions"):
        value = getattr(env, "available_actions")

    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        actions: list[str] = []
        if value.get("has_search_bar"):
            actions.append("search[query words]")
        for item in value.get("clickables") or []:
            actions.append(f"click[{item}]")
        return actions
    return []


def _reset_env(env: Any, task_index: int):
    try:
        result = env.reset(session=task_index)
    except TypeError:
        result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _step_env(env: Any, action: str) -> tuple[Any, float, bool, dict]:
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward or 0.0), bool(terminated or truncated), info or {}
    obs, reward, done, info = result
    return obs, float(reward or 0.0), bool(done), info or {}


def _instruction_from_observation(observation: Any) -> str:
    text = str(observation or "")
    if "[SEP]" in text:
        sep_parts = [part.strip() for part in re.split(r"\s*\[SEP\]\s*", text) if part.strip()]
        for idx, part in enumerate(sep_parts):
            lowered = part.lower()
            if lowered in {"instruction", "instruction:"} and idx + 1 < len(sep_parts):
                return re.sub(r"\s+", " ", sep_parts[idx + 1]).strip()
            if lowered.startswith("instruction:"):
                value = part.split(":", 1)[1].strip()
                if value:
                    return re.sub(r"\s+", " ", value).strip()
                if idx + 1 < len(sep_parts):
                    return re.sub(r"\s+", " ", sep_parts[idx + 1]).strip()
    match = re.search(r"Instruction:\s*(.*?)(?:\n\s*\[|$)", text, flags=re.S)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


class WebShopBackend:
    def __init__(self, worker_id: str, split: str, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.split = split
        self.config = config
        self.env: Any | None = None
        self.num_tasks = 0
        self.task_index = 0
        self.reset_count = 0
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.last_info: dict[str, Any] = {}

    @property
    def runtime(self) -> dict[str, Any]:
        value = self.config.get("_agent_env_runtime")
        return value if isinstance(value, dict) else {}

    def _format_actions(self, actions: list[str]) -> str:
        if not actions:
            return ""
        return "\nAvailable actions:\n" + "\n".join(f"- {action}" for action in actions) + "\n"

    def _observation_text(self, observation: str, info: dict[str, Any]) -> str:
        text = f"Observation:\n{str(observation).strip()}\n"
        observation_cfg = self.runtime.get("observation") if isinstance(self.runtime.get("observation"), dict) else {}
        if bool(observation_cfg.get("include_actions", True)):
            text += self._format_actions(_available_actions(self.env, info) if self.env is not None else [])
        return text

    def _initial_messages(self, prompt: str, observation: str, info: dict[str, Any]) -> list[dict[str, str]]:
        system_prompt = require_prompt(prompt, env_name="WebShop", source="run_episode.prompt")
        user_prompt = self._observation_text(observation, info).strip()
        if not user_prompt:
            raise ValueError("WebShop initial user prompt is empty after reset")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def start(self) -> dict[str, Any]:
        import sys

        webshop_lib = os.environ.get("WEBSHOP_LIB")
        if webshop_lib and webshop_lib not in sys.path:
            sys.path.insert(0, webshop_lib)
        data_dir = self.config.get("data_dir")
        if data_dir:
            os.environ.setdefault("WEBSHOP_DATA", data_dir)

        _install_text_env_import_stubs()
        WebAgentTextEnv = _load_text_env_class(webshop_lib)
        product_file = self.config.get("product_file")
        attr_file = self.config.get("attr_file")
        if attr_file:
            import web_agent_site.engine.engine as engine
            import web_agent_site.utils as utils

            engine.DEFAULT_ATTR_PATH = str(attr_file)
            utils.DEFAULT_ATTR_PATH = str(attr_file)
        if product_file:
            import web_agent_site.utils as utils

            utils.DEFAULT_FILE_PATH = str(product_file)

        kwargs = {
            "observation_mode": self.config.get("observation_mode", "text"),
            "num_products": self.config["num_products"],
            "human_goals": self.config.get("human_goals", True),
        }
        if product_file:
            kwargs["file_path"] = str(product_file)
        if self.config.get("env_id", "WebAgentTextEnv-v0") != "WebAgentTextEnv-v0":
            raise ValueError(f"Unsupported WebShop env_id={self.config.get('env_id')}")
        self.env = WebAgentTextEnv(**kwargs)
        self.num_tasks = len(getattr(getattr(self.env, "server", None), "goals", []) or [])
        return {"num_tasks": self.num_tasks}

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.env is not None
        self.split = str(payload.get("split") or self.split)
        self.task_index = int(payload.get("task_index") or 0)
        requested_task_id = str(payload.get("task_id") or "").strip()
        if requested_task_id:
            expected_task_id = f"webshop:{self.split}:{self.task_index}"
            if requested_task_id != expected_task_id:
                raise ValueError(f"WebShop task_id mismatch: expected {expected_task_id}, got {requested_task_id}")
        obs = _reset_env(self.env, self.task_index)
        self.reset_count += 1
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        instruction = _instruction_from_observation(obs)
        self.last_info = {
            "available_actions": _available_actions(self.env),
            "task_id": f"webshop:{self.split}:{self.task_index}",
            "split": self.split,
        }
        if instruction:
            self.last_info["instruction"] = instruction
            self.last_info["task_prompt"] = instruction
        return {
            "observation": str(obs),
            "info": self.last_info,
            "split": self.split,
            "task_index": self.task_index,
            "num_tasks": self.num_tasks,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.env is not None
        action = str(payload.get("action") or "")
        previous_info = dict(self.last_info)
        obs, reward, done, info = _step_env(self.env, action)
        self.step_count += 1
        self.final_score = float(reward)
        self.done = bool(done)
        self.last_info = dict(info or {})
        for key in ("task_id", "split", "instruction", "task_prompt"):
            value = previous_info.get(key)
            if value not in (None, "", []):
                self.last_info.setdefault(key, value)
        self.last_info.setdefault("available_actions", _available_actions(self.env, self.last_info))
        self.last_info["done"] = self.done
        return {
            "observation": str(obs),
            "score": self.final_score,
            "done": self.done,
            "success": self.final_score > 0,
            "info": self.last_info,
            "task_index": self.task_index,
            "num_tasks": self.num_tasks,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def run_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        if not policy:
            raise ValueError("WebShop run_episode requires policy endpoint")
        reset = self.reset(payload)
        observation = str(reset.get("observation", ""))
        info = reset.get("info") if isinstance(reset.get("info"), dict) else {}
        messages = self._initial_messages(str(payload.get("prompt") or ""), observation, info)
        runtime = self.runtime
        action_cfg = runtime.get("action") if isinstance(runtime.get("action"), dict) else {}
        interaction_cfg = runtime.get("interaction") if isinstance(runtime.get("interaction"), dict) else {}
        text_action_cfg = interaction_cfg.get("text_action") if isinstance(interaction_cfg.get("text_action"), dict) else {}
        tag = str(text_action_cfg.get("tag", "action"))
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

        max_turns = int(payload.get("max_turns") or runtime.get("max_turns") or 20)
        sampling_params = payload.get("sampling_params") if isinstance(payload.get("sampling_params"), dict) else {}
        max_tokens = int(payload.get("max_response_tokens") or 512)
        timeout_s = float((payload.get("timeouts") or {}).get("policy_s") or (runtime.get("timeouts") or {}).get("policy_s") or 120)
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
                _available_actions(self.env, info) if self.env is not None else [],
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
            success = final_score > 0
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
            "task_id": info.get("task_id") or f"webshop:{self.split}:{self.task_index}",
            "split": self.split,
            "instruction": info.get("instruction"),
            "task_prompt": info.get("task_prompt"),
            "task_index": self.task_index,
            "num_tasks": self.num_tasks,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
            "metadata": metadata,
        }

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": self.final_score,
            "success": self.final_score > 0,
            "done": self.done,
            "info": self.last_info,
            "task_index": self.task_index,
            "num_tasks": self.num_tasks,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"reset_count": self.reset_count, "step_count": self.step_count}

    def close(self) -> dict[str, Any]:
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()
        self.env = None
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a process-isolated WebShop environment server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18180)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    env_config, server_config = _load_config(args.config)
    serve_process_pool(
        host=args.host,
        port=args.port,
        backend_cls=WebShopBackend,
        env_config=env_config,
        server_config=server_config,
        env_name="webshop",
    )


if __name__ == "__main__":
    main()
