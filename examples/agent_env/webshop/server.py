from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import yaml

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
    return {
        "pool_size": int(_deep_get(raw, "env_server", "pool_size", 8)),
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
    data_dir = str(_deep_get(raw, "webshop", "data_dir", os.environ.get("WEBSHOP_DATA", "")))
    return {
        "env_id": _deep_get(raw, "webshop", "env_id", "WebAgentTextEnv-v0"),
        "observation_mode": _deep_get(raw, "webshop", "observation_mode", "text"),
        "data_dir": data_dir,
        "product_file": _deep_get(raw, "webshop", "product_file", None),
        "attr_file": _deep_get(raw, "webshop", "attr_file", None),
        "num_products": _deep_get(raw, "webshop", "num_products", 1000),
        "human_goals": _deep_get(raw, "webshop", "human_goals", True),
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


class WebShopBackend:
    def __init__(self, worker_id: str, split: str, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.split = split
        self.config = config
        self.env: Any | None = None
        self.num_goals = 0
        self.task_index = 0
        self.reset_count = 0
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.last_info: dict[str, Any] = {}

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
            "num_products": self.config.get("num_products", 1000),
            "human_goals": self.config.get("human_goals", True),
        }
        if product_file:
            kwargs["file_path"] = str(product_file)
        if self.config.get("env_id", "WebAgentTextEnv-v0") != "WebAgentTextEnv-v0":
            raise ValueError(f"Unsupported WebShop env_id={self.config.get('env_id')}")
        self.env = WebAgentTextEnv(**kwargs)
        self.num_goals = len(getattr(getattr(self.env, "server", None), "goals", []) or [])
        return {"num_goals": self.num_goals}

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.env is not None
        self.split = str(payload.get("split") or self.split)
        self.task_index = int(payload.get("task_index") or 0)
        obs = _reset_env(self.env, self.task_index)
        self.reset_count += 1
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.last_info = {"available_actions": _available_actions(self.env)}
        return {
            "observation": str(obs),
            "info": self.last_info,
            "split": self.split,
            "task_index": self.task_index,
            "num_goals": self.num_goals,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.env is not None
        action = str(payload.get("action") or "")
        obs, reward, done, info = _step_env(self.env, action)
        self.step_count += 1
        self.final_score = float(reward)
        self.done = bool(done)
        self.last_info = dict(info or {})
        self.last_info.setdefault("available_actions", _available_actions(self.env, self.last_info))
        self.last_info["done"] = self.done
        return {
            "observation": str(obs),
            "score": self.final_score,
            "done": self.done,
            "success": self.final_score > 0,
            "info": self.last_info,
            "task_index": self.task_index,
            "num_goals": self.num_goals,
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": self.final_score,
            "success": self.final_score > 0,
            "done": self.done,
            "info": self.last_info,
            "task_index": self.task_index,
            "num_goals": self.num_goals,
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
