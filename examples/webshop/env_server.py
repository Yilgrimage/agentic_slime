from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

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
    }


def _environment_config(raw: dict) -> dict:
    return {
        "env_id": _deep_get(raw, "webshop", "env_id", "WebAgentTextEnv-v0"),
        "observation_mode": _deep_get(raw, "webshop", "observation_mode", "text"),
        "data_dir": str(_deep_get(raw, "webshop", "data_dir", os.environ.get("WEBSHOP_DATA", ""))),
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


def _worker_loop(worker_id: str, split: str, config: dict, conn: Any) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    try:
        import sys

        webshop_lib = os.environ.get("WEBSHOP_LIB")
        if webshop_lib and webshop_lib not in sys.path:
            sys.path.insert(0, webshop_lib)
        data_dir = config.get("data_dir")
        if data_dir:
            os.environ.setdefault("WEBSHOP_DATA", data_dir)

        _install_text_env_import_stubs()
        WebAgentTextEnv = _load_text_env_class(webshop_lib)

        kwargs = {
            "observation_mode": config.get("observation_mode", "text"),
            "num_products": config.get("num_products", 1000),
            "human_goals": config.get("human_goals", True),
        }
        if config.get("env_id", "WebAgentTextEnv-v0") != "WebAgentTextEnv-v0":
            raise ValueError(f"Unsupported WebShop env_id={config.get('env_id')}")
        env = WebAgentTextEnv(**kwargs)
        task_index = 0
        reset_count = 0
        step_count = 0
        final_score = 0.0
        done = False
        last_info: dict = {}
        conn.send({"ok": True, "event": "ready", "worker_id": worker_id, "split": split})

        while True:
            request = conn.recv()
            cmd = request.get("cmd")
            payload = request.get("payload") or {}
            try:
                if cmd == "reset":
                    task_index = int(payload.get("task_index") or 0)
                    obs = _reset_env(env, task_index)
                    reset_count += 1
                    step_count = 0
                    final_score = 0.0
                    done = False
                    last_info = {"available_actions": _available_actions(env)}
                    conn.send(
                        {
                            "ok": True,
                            "observation": str(obs),
                            "info": last_info,
                            "split": split,
                            "task_index": task_index,
                            "reset_count": reset_count,
                            "step_count": step_count,
                        }
                    )
                elif cmd == "step":
                    action = str(payload.get("action") or "")
                    obs, reward, done, info = _step_env(env, action)
                    step_count += 1
                    final_score = float(reward)
                    last_info = dict(info or {})
                    last_info.setdefault("available_actions", _available_actions(env, last_info))
                    last_info["done"] = done
                    conn.send(
                        {
                            "ok": True,
                            "observation": str(obs),
                            "score": final_score,
                            "done": done,
                            "success": final_score > 0,
                            "info": last_info,
                            "task_index": task_index,
                            "reset_count": reset_count,
                            "step_count": step_count,
                        }
                    )
                elif cmd == "evaluate":
                    conn.send(
                        {
                            "ok": True,
                            "score": final_score,
                            "success": final_score > 0,
                            "done": done,
                            "info": last_info,
                            "task_index": task_index,
                            "reset_count": reset_count,
                            "step_count": step_count,
                        }
                    )
                elif cmd == "close":
                    if hasattr(env, "close"):
                        env.close()
                    conn.send({"ok": True})
                    break
                else:
                    raise ValueError(f"Unknown worker command: {cmd}")
            except Exception as exc:
                conn.send({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})
    except Exception as exc:
        try:
            conn.send({"ok": False, "event": "startup_failed", "error": repr(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


@dataclass
class Worker:
    worker_id: str
    split: str
    process: mp.Process
    conn: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Lease:
    lease_id: str
    worker: Worker
    split: str
    request_id: str | None = None
    task_key: str | None = None
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    final_score: float = 0.0
    done: bool = False
    success: bool = False
    last_info: dict = field(default_factory=dict)


class CapacityError(Exception):
    pass


class LeaseStore:
    def __init__(self, env_config: dict, server_config: dict) -> None:
        self.env_config = env_config
        self.server_config = server_config
        self.acquire_timeout_s = float(server_config["acquire_timeout_s"])
        self.lease_ttl_s = float(server_config["lease_ttl_s"])
        self.idempotency_ttl_s = float(server_config["idempotency_ttl_s"])
        self.worker_start_timeout_s = float(server_config["worker_start_timeout_s"])
        self.worker_request_timeout_s = float(server_config["worker_request_timeout_s"])
        self.mp_ctx = mp.get_context("spawn")
        self.lock = threading.Lock()
        self.available: dict[str, queue.Queue[Worker]] = {}
        self.created: dict[str, list[Worker]] = {}
        self.leases: dict[str, Lease] = {}
        self.idempotency: dict[tuple[str, str], tuple[str, float]] = {}
        for split in server_config["prewarm_splits"]:
            self._ensure_pool(split)

    def _spawn_worker(self, split: str, index: int | str) -> Worker:
        parent_conn, child_conn = self.mp_ctx.Pipe()
        worker_id = f"{split}-{index}"
        process = self.mp_ctx.Process(
            target=_worker_loop,
            args=(worker_id, split, self.env_config, child_conn),
            name=f"webshop-env-{worker_id}",
            daemon=True,
        )
        process.start()
        child_conn.close()
        return Worker(worker_id=worker_id, split=split, process=process, conn=parent_conn)

    def _wait_ready(self, worker: Worker) -> None:
        if not worker.conn.poll(self.worker_start_timeout_s):
            worker.process.terminate()
            worker.process.join(timeout=5)
            raise RuntimeError(f"WebShop worker {worker.worker_id} did not become ready in {self.worker_start_timeout_s}s")
        ready = worker.conn.recv()
        if not ready.get("ok"):
            worker.process.join(timeout=5)
            raise RuntimeError(f"WebShop worker {worker.worker_id} failed during startup: {ready}")
        logger.info("Started WebShop worker %s pid=%s split=%s", worker.worker_id, worker.process.pid, worker.split)

    def _ensure_pool(self, split: str) -> None:
        if split in self.available:
            return
        pool_size = int(self.server_config["pool_size"])
        q: queue.Queue[Worker] = queue.Queue(maxsize=pool_size)
        logger.info("Prewarming %d process-isolated WebShop workers for split=%s", pool_size, split)
        workers = [self._spawn_worker(split, i) for i in range(pool_size)]
        for worker in workers:
            self._wait_ready(worker)
            q.put(worker)
        self.available[split] = q
        self.created[split] = workers

    def _worker_request(self, worker: Worker, cmd: str, payload: dict | None = None) -> dict:
        with worker.lock:
            if not worker.process.is_alive():
                raise RuntimeError(f"WebShop worker {worker.worker_id} pid={worker.process.pid} is not alive")
            worker.conn.send({"cmd": cmd, "payload": payload or {}})
            if not worker.conn.poll(self.worker_request_timeout_s):
                raise TimeoutError(f"WebShop worker {worker.worker_id} timed out on {cmd}")
            result = worker.conn.recv()
            if not result.get("ok"):
                raise RuntimeError(f"WebShop worker {worker.worker_id} failed on {cmd}: {result}")
            return result

    def _reap_locked(self) -> list[Lease]:
        now = time.time()
        for key, (_, ts) in list(self.idempotency.items()):
            if now - ts > self.idempotency_ttl_s:
                self.idempotency.pop(key, None)
        expired = []
        for lease_id, lease in list(self.leases.items()):
            if now - lease.last_used_at > self.lease_ttl_s:
                expired.append(self.leases.pop(lease_id))
        return expired

    def _release(self, lease: Lease) -> None:
        self.available[lease.worker.split].put(lease.worker)

    def _release_expired(self, expired: list[Lease]) -> None:
        for lease in expired:
            logger.warning("Reaping expired WebShop lease %s", lease.lease_id)
            self._release(lease)

    def _acquire(self, split: str) -> Worker:
        self._ensure_pool(split)
        try:
            return self.available[split].get(timeout=self.acquire_timeout_s)
        except queue.Empty as exc:
            raise CapacityError(f"No WebShop worker available for split={split} within {self.acquire_timeout_s}s") from exc

    def allocate(self, payload: dict) -> dict:
        split = str(payload.get("split") or "train")
        task_key = str(payload.get("task_key") or split)
        request_id = payload.get("request_id")
        with self.lock:
            expired = self._reap_locked()
            if request_id:
                cached = self.idempotency.get((task_key, str(request_id)))
                if cached and cached[0] in self.leases:
                    lease = self.leases[cached[0]]
                    lease.last_used_at = time.time()
                    self._release_expired(expired)
                    return {"ok": True, "lease_id": lease.lease_id, "worker_id": lease.worker.worker_id, "reused": True}
        self._release_expired(expired)
        worker = self._acquire(split)
        lease_id = f"lease-{uuid.uuid4().hex[:16]}"
        lease = Lease(lease_id=lease_id, worker=worker, split=split, request_id=str(request_id) if request_id else None, task_key=task_key)
        with self.lock:
            self.leases[lease_id] = lease
            if request_id:
                self.idempotency[(task_key, str(request_id))] = (lease_id, time.time())
        return {"ok": True, "lease_id": lease_id, "worker_id": worker.worker_id, "reused": False, "split": split}

    def _get_lease(self, lease_id: str) -> Lease:
        with self.lock:
            expired = self._reap_locked()
            lease = self.leases.get(lease_id)
            if lease is not None:
                lease.last_used_at = time.time()
        self._release_expired(expired)
        if lease is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        return lease

    def reset(self, payload: dict) -> dict:
        lease = self._get_lease(str(payload.get("lease_id") or payload.get("session_id")))
        task_index = int(payload.get("task_index") or 0)
        result = self._worker_request(lease.worker, "reset", {"task_index": task_index, "seed": payload.get("seed", task_index)})
        lease.final_score = 0.0
        lease.done = False
        lease.success = False
        lease.last_info = result.get("info") or {}
        return {"ok": True, "lease_id": lease.lease_id, "worker_id": lease.worker.worker_id, **result}

    def step(self, payload: dict) -> dict:
        lease = self._get_lease(str(payload.get("lease_id") or payload.get("session_id")))
        result = self._worker_request(lease.worker, "step", {"action": str(payload.get("action") or "")})
        lease.final_score = float(result.get("score", 0.0) or 0.0)
        lease.done = bool(result.get("done", False))
        lease.success = bool(result.get("success", False))
        lease.last_info = result.get("info") or {}
        return {"ok": True, "lease_id": lease.lease_id, "worker_id": lease.worker.worker_id, **result}

    def evaluate(self, payload: dict) -> dict:
        lease = self._get_lease(str(payload.get("lease_id") or payload.get("session_id")))
        result = self._worker_request(lease.worker, "evaluate")
        return {"ok": True, "lease_id": lease.lease_id, "worker_id": lease.worker.worker_id, **result}

    def close(self, payload: dict) -> dict:
        lease_id = str(payload.get("lease_id") or payload.get("session_id") or "")
        if not lease_id:
            return {"ok": True, "found": False}
        with self.lock:
            lease = self.leases.pop(lease_id, None)
        if lease is None:
            return {"ok": True, "found": False}
        self._release(lease)
        return {"ok": True, "found": True}

    def health(self) -> dict:
        return {
            "ok": True,
            "leases": len(self.leases),
            "pool_size": int(self.server_config["pool_size"]),
            "splits": list(self.available),
        }

    def shutdown(self) -> None:
        for workers in self.created.values():
            for worker in workers:
                try:
                    self._worker_request(worker, "close")
                except Exception:
                    pass
                worker.process.join(timeout=5)
                if worker.process.is_alive():
                    worker.process.terminate()


class Handler(BaseHTTPRequestHandler):
    store: LeaseStore

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, self.store.health())
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/allocate":
                result = self.store.allocate(payload)
            elif self.path == "/reset":
                result = self.store.reset(payload)
            elif self.path == "/step":
                result = self.store.step(payload)
            elif self.path == "/evaluate":
                result = self.store.evaluate(payload)
            elif self.path == "/close":
                result = self.store.close(payload)
            else:
                self._send(404, {"ok": False, "error": "not found"})
                return
            self._send(200, result)
        except CapacityError as exc:
            self._send(503, {"ok": False, "error": repr(exc)})
        except Exception as exc:
            logger.exception("Request failed: %s", self.path)
            self._send(500, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a process-isolated WebShop environment server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18180)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    env_config, server_config = _load_config(args.config)
    store = LeaseStore(env_config, server_config)
    Handler.store = store
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("WebShop env server listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        store.shutdown()


if __name__ == "__main__":
    main()
