from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request

import yaml

from examples.agent_env.prompting import require_prompt
from examples.agent_env.server import serve_process_pool

logger = logging.getLogger(__name__)


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
        raise ValueError("Missing env_server.pool_size in OpenClaw env_config.yaml")
    return {
        "pool_size": int(pool_size),
        "acquire_timeout_s": float(_deep_get(raw, "env_server", "acquire_timeout_s", 600.0)),
        "lease_ttl_s": float(_deep_get(raw, "env_server", "lease_ttl_s", 1800.0)),
        "idempotency_ttl_s": float(_deep_get(raw, "env_server", "idempotency_ttl_s", 300.0)),
        "worker_start_timeout_s": float(_deep_get(raw, "env_server", "worker_start_timeout_s", 300.0)),
        "worker_request_timeout_s": float(_deep_get(raw, "env_server", "worker_request_timeout_s", 300.0)),
        "prewarm_splits": list(_deep_get(raw, "env_server", "prewarm_splits", ["train"])),
        "reuse_workers": bool(_deep_get(raw, "env_server", "reuse_workers", True)),
        "reset_on_release": bool(_deep_get(raw, "env_server", "reset_on_release", True)),
        "shared_pool": bool(_deep_get(raw, "env_server", "shared_pool", True)),
    }


def _expand_text(value: Any) -> str:
    return os.path.expandvars(str(value or "")).strip()


def _agent_env_data_dir() -> Path | None:
    value = os.environ.get("AGENT_ENV_DATA_DIR", "").strip()
    if value:
        return Path(os.path.expandvars(value)).expanduser()
    local_root = os.environ.get("LOCAL_RUNTIME_DIR", "").strip()
    if local_root:
        return Path(os.path.expandvars(local_root)).expanduser() / "data" / "openclaw"
    return None


def _resolve_runtime_path(value: Any, config_dir: Path | None = None) -> Path:
    path = Path(_expand_text(value)).expanduser()
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    data_dir = _agent_env_data_dir()
    if data_dir is not None:
        candidates.append(data_dir / path)
    if config_dir is not None:
        candidates.append(config_dir / path)
    candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    value = json.loads(line, strict=False)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("tasks"), list):
        return [item for item in value["tasks"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported OpenClaw task file format: {path}")


def _missing_env_placeholder(value: str) -> bool:
    text = value.strip()
    return text.startswith("${") and text.endswith("}")


def _headers(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("headers") or {}
    if not isinstance(raw, dict):
        raise ValueError("openclaw.headers must be a mapping")
    headers = {str(key): _expand_text(value) for key, value in raw.items()}
    api_key_env = str(config.get("api_key_env") or "").strip()
    if api_key_env and os.environ.get(api_key_env):
        headers.setdefault("Authorization", f"Bearer {os.environ[api_key_env]}")
    return headers


def _environment_config(raw: dict, config_dir: Path | None = None) -> dict:
    cfg = dict(raw.get("openclaw") or {})
    backend = str(cfg.get("backend") or "http_harness").strip().lower()
    if backend != "http_harness":
        raise ValueError(f"Unsupported openclaw.backend={backend!r}; expected http_harness")

    tasks = cfg.get("tasks") or raw.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError("openclaw.tasks must be a list")
    task_file = _expand_text(cfg.get("task_file") or raw.get("task_file"))
    if task_file:
        tasks = [*tasks, *_load_json_or_jsonl(_resolve_runtime_path(task_file, config_dir))]
    if not tasks:
        raise ValueError("openclaw.tasks or openclaw.task_file is required")

    base_url = _expand_text(cfg.get("base_url") or os.environ.get("OPENCLAW_HARNESS_URL", ""))
    if not base_url or _missing_env_placeholder(base_url):
        raise ValueError("openclaw.base_url is required. Set OPENCLAW_HARNESS_URL or openclaw.base_url.")

    policy = str(cfg.get("policy") or "")
    policy_file = _expand_text(cfg.get("policy_file") or "")
    if policy_file:
        policy = _resolve_runtime_path(policy_file, config_dir).read_text(encoding="utf-8")
    if not policy.strip():
        raise ValueError("openclaw.policy or openclaw.policy_file is required")

    return {
        "name": str(cfg.get("name") or "openclaw_harness"),
        "backend": backend,
        "base_url": base_url,
        "episode_path": str(cfg.get("episode_path") or "/run_episode"),
        "close_path": str(cfg.get("close_path") or ""),
        "request_timeout_s": float(cfg.get("request_timeout_s", 120.0)),
        "headers": _headers(cfg),
        "policy": policy,
        "tasks": tasks,
        "max_turns": int(_deep_get(raw, "openclaw", "max_turns", raw.get("max_turns", 20))),
    }


def _load_config(path: str) -> tuple[dict, dict]:
    config_path = Path(path).expanduser()
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _environment_config(raw, config_path.parent), _server_config(raw)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "success", "done"}
    return bool(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


class OpenClawHarnessClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = str(config["base_url"]).strip()
        self.timeout_s = float(config.get("request_timeout_s", 120.0))
        self.headers = dict(config.get("headers") or {})
        self.opener = request.build_opener(request.ProxyHandler({}))

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = _join_url(self.base_url, path)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with self.opener.open(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenClaw harness POST {url} failed: HTTP {exc.code}: {body[:1000]}") from exc
        if not body.strip():
            return {}
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError(f"OpenClaw harness {url} returned non-object JSON: {type(value).__name__}")
        return value


class OpenClawBackend:
    def __init__(self, worker_id: str, split: str, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.split = split
        self.config = config
        self.client: OpenClawHarnessClient | None = None
        self.tasks = list(config.get("tasks") or [])
        self.task: dict[str, Any] = {}
        self.task_index = 0
        self.session_id = ""
        self.reset_count = 0
        self.step_count = 0
        self.final_score = 0.0
        self.done = False
        self.success = False
        self.last_info: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        self.client = OpenClawHarnessClient(self.config)
        return {"num_tasks": len(self.tasks), "openclaw": self.config["name"]}

    def _base_info(self) -> dict[str, Any]:
        return {
            "task_id": self.task.get("id") or self.task.get("task_id") or self.task_index,
            "split": self.split,
            "openclaw": self.config["name"],
            "session_id": self.session_id,
            "success": self.success,
            "done": self.done,
        }

    def run_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("OpenClaw harness client is not started")
        self.split = str(payload.get("split") or self.split)
        payload_task = payload.get("task") if isinstance(payload.get("task"), dict) else None
        payload_task_id = str((payload_task or {}).get("id") or (payload_task or {}).get("task_id") or "").strip()
        requested_task_id = str(payload.get("task_id") or payload_task_id).strip()
        if payload.get("task_id") not in (None, "", []) and payload_task_id and str(payload.get("task_id")).strip() != payload_task_id:
            raise ValueError(f"OpenClaw task_id mismatch between metadata and task payload: {payload.get('task_id')} != {payload_task_id}")
        if payload_task is not None:
            self.task = dict(payload_task)
            self.task_index = int(payload.get("task_index") or 0)
        elif requested_task_id:
            matches = [
                index
                for index, task in enumerate(self.tasks)
                if str(task.get("id") or task.get("task_id") or "").strip() == requested_task_id
            ]
            if not matches:
                raise KeyError(f"OpenClaw task_id from prompt data is not available: {requested_task_id}")
            self.task_index = matches[0]
            self.task = dict(self.tasks[self.task_index])
        else:
            self.task_index = int(payload.get("task_index") or 0) % max(1, len(self.tasks))
            self.task = dict(self.tasks[self.task_index])
        self.session_id = f"{self.worker_id}-{uuid.uuid4().hex[:12]}"
        self.reset_count += 1
        self.final_score = 0.0
        self.done = False
        self.success = False
        request_payload = {
            "session_id": self.session_id,
            "worker_id": self.worker_id,
            "split": self.split,
            "task_index": self.task_index,
            "task": self.task,
            "max_turns": int(payload.get("max_turns") or self.config.get("max_turns", 20)),
            "policy": payload.get("policy") or {},
            "sampling_params": payload.get("sampling_params") or {},
            "policy_instruction": require_prompt(payload.get("prompt"), env_name="OpenClaw", source="run_episode.prompt"),
            "metadata": {
                key: value
                for key, value in payload.items()
                if key not in {"policy", "sampling_params", "task", "max_turns"}
            },
        }
        result = self.client.post(str(self.config["episode_path"]), request_payload)
        info = self._base_info()
        raw_info = result.get("info")
        if isinstance(raw_info, dict):
            info.update(raw_info)
        self.final_score = _as_float(result.get("score", result.get("reward")), 0.0)
        self.done = _as_bool(result.get("done", True))
        self.success = _as_bool(result.get("success", self.final_score > 0))
        info.update({"done": self.done, "success": self.success, "score": self.final_score})
        self.last_info = info
        return {
            "status": result.get("status") or ("completed" if self.done else "truncated"),
            "observation": str(result.get("observation") or result.get("message") or ""),
            "score": self.final_score,
            "done": self.done,
            "success": self.success,
            "info": info,
            "task_index": self.task_index,
            "num_tasks": len(self.tasks),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.client is not None and self.session_id and self.config.get("close_path"):
            try:
                self.client.post(str(self.config["close_path"]), {"session_id": self.session_id, "release": True})
            except Exception:
                logger.debug("Failed to release OpenClaw harness session %s", self.session_id, exc_info=True)
        return {"reset_count": self.reset_count, "step_count": self.step_count}

    def close(self) -> dict[str, Any]:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a process-isolated OpenClaw episode server adapter.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18185)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    env_config, server_config = _load_config(args.config)
    serve_process_pool(
        host=args.host,
        port=args.port,
        backend_cls=OpenClawBackend,
        env_config=env_config,
        server_config=server_config,
        env_name="openclaw",
    )


if __name__ == "__main__":
    main()
