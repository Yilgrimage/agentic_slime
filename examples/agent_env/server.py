from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class AgentThreadingHTTPServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


class EnvBackend(Protocol):
    def start(self) -> dict[str, Any]: ...

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def step(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def release(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> dict[str, Any]: ...


class CapacityError(Exception):
    pass


@dataclass
class Worker:
    worker_id: str
    split: str
    process: mp.Process
    conn: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    reset_count: int = 0
    step_count: int = 0
    ready: dict[str, Any] = field(default_factory=dict)


@dataclass
class Lease:
    lease_id: str
    worker: Worker
    split: str
    pooled: bool
    request_id: str | None = None
    task_key: str | None = None
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    reset_at: float | None = None
    final_score: float = 0.0
    done: bool = False
    success: bool = False
    last_info: dict[str, Any] = field(default_factory=dict)
    task_index: int | None = None


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _format_error(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _backend_worker_loop(
    backend_cls: type[EnvBackend],
    worker_id: str,
    split: str,
    env_config: dict[str, Any],
    conn: Any,
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    backend: EnvBackend | None = None
    try:
        backend = backend_cls(worker_id, split, env_config)  # type: ignore[call-arg]
        ready = backend.start()
        ready.update({"ok": True, "event": "ready", "worker_id": worker_id, "split": split})
        conn.send(ready)

        while True:
            request = conn.recv()
            cmd = request.get("cmd")
            payload = request.get("payload") or {}
            try:
                if cmd == "reset":
                    result = backend.reset(payload)
                elif cmd == "step":
                    result = backend.step(payload)
                elif cmd == "evaluate":
                    result = backend.evaluate(payload)
                elif cmd == "release":
                    result = backend.release(payload)
                elif cmd == "close":
                    result = backend.close()
                    conn.send({"ok": True, **(result or {})})
                    break
                else:
                    raise ValueError(f"Unknown worker command: {cmd}")
                conn.send({"ok": True, **(result or {})})
            except Exception as exc:
                conn.send({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})
    except Exception as exc:
        try:
            conn.send({"ok": False, "event": "startup_failed", "error": repr(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                logger.debug("Failed to close backend for worker %s", worker_id, exc_info=True)
        try:
            conn.close()
        except Exception:
            pass


class ProcessPoolEnvServer:
    def __init__(
        self,
        *,
        backend_cls: type[EnvBackend],
        env_config: dict[str, Any],
        server_config: dict[str, Any],
        env_name: str,
    ) -> None:
        self.backend_cls = backend_cls
        self.env_config = env_config
        self.server_config = normalize_server_config(server_config)
        self.env_name = env_name
        self.acquire_timeout_s = float(self.server_config["acquire_timeout_s"])
        self.lease_ttl_s = float(self.server_config["lease_ttl_s"])
        self.idempotency_ttl_s = float(self.server_config["idempotency_ttl_s"])
        self.worker_start_timeout_s = float(self.server_config["worker_start_timeout_s"])
        self.worker_request_timeout_s = float(self.server_config["worker_request_timeout_s"])
        self.reuse_workers = bool(self.server_config["reuse_workers"])
        self.reset_on_release = bool(self.server_config["reset_on_release"])
        self.mp_ctx = mp.get_context("spawn")
        self.lock = threading.Lock()
        self.available: dict[str, queue.Queue[Worker]] = {}
        self.created: dict[str, list[Worker]] = {}
        self.leases: dict[str, Lease] = {}
        self.idempotency: dict[tuple[str, str], tuple[str, float]] = {}
        for split in self.server_config["prewarm_splits"]:
            self._ensure_pool(str(split))

    def _spawn_worker(self, split: str, index: int | str) -> Worker:
        parent_conn, child_conn = self.mp_ctx.Pipe()
        worker_id = f"{split}-{index}"
        process = self.mp_ctx.Process(
            target=_backend_worker_loop,
            args=(self.backend_cls, worker_id, split, self.env_config, child_conn),
            name=f"{self.env_name}-env-{worker_id}",
            daemon=True,
        )
        process.start()
        child_conn.close()
        return Worker(worker_id=worker_id, split=split, process=process, conn=parent_conn)

    def _wait_ready(self, worker: Worker) -> None:
        if not worker.conn.poll(self.worker_start_timeout_s):
            worker.process.terminate()
            worker.process.join(timeout=5)
            raise RuntimeError(
                f"{self.env_name} worker {worker.worker_id} did not become ready in {self.worker_start_timeout_s}s"
            )
        ready = worker.conn.recv()
        if not ready.get("ok"):
            worker.process.join(timeout=5)
            raise RuntimeError(f"{self.env_name} worker {worker.worker_id} failed during startup: {ready}")
        worker.ready = dict(ready)
        logger.info("Started %s worker %s pid=%s split=%s", self.env_name, worker.worker_id, worker.process.pid, worker.split)

    def _start_worker(self, split: str, index: int | str) -> Worker:
        worker = self._spawn_worker(split, index)
        self._wait_ready(worker)
        return worker

    def _ensure_pool(self, split: str) -> None:
        if split in self.available:
            return
        pool_size = int(self.server_config["pool_size"])
        workers = [self._spawn_worker(split, i) for i in range(pool_size)]
        q: queue.Queue[Worker] = queue.Queue(maxsize=pool_size)
        logger.info("Prewarming %d process-isolated %s workers for split=%s", pool_size, self.env_name, split)
        for worker in workers:
            self._wait_ready(worker)
            q.put(worker)
        self.available[split] = q
        self.created[split] = workers

    def _worker_request(self, worker: Worker, cmd: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with worker.lock:
            if not worker.process.is_alive():
                raise RuntimeError(f"{self.env_name} worker {worker.worker_id} pid={worker.process.pid} is not alive")
            worker.conn.send({"cmd": cmd, "payload": payload or {}})
            if not worker.conn.poll(self.worker_request_timeout_s):
                raise TimeoutError(f"{self.env_name} worker {worker.worker_id} timed out on {cmd}")
            result = worker.conn.recv()
            if not result.get("ok"):
                raise RuntimeError(f"{self.env_name} worker {worker.worker_id} failed on {cmd}: {result}")
            worker.reset_count = int(result.get("reset_count", worker.reset_count) or worker.reset_count)
            worker.step_count = int(result.get("step_count", worker.step_count) or worker.step_count)
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

    def _release_worker(self, lease: Lease) -> None:
        if self.reset_on_release:
            self._worker_request(lease.worker, "release", {"reset_on_release": True})
        if lease.pooled:
            self.available[lease.worker.split].put(lease.worker)
        else:
            self._worker_request(lease.worker, "close")
            lease.worker.process.join(timeout=5)
            if lease.worker.process.is_alive():
                lease.worker.process.terminate()

    def _release_expired(self, expired: list[Lease]) -> None:
        for lease in expired:
            logger.warning("Reaping expired %s lease %s", self.env_name, lease.lease_id)
            self._release_worker(lease)

    def _acquire(self, split: str) -> tuple[Worker, bool]:
        if not self.reuse_workers:
            return self._start_worker(split, f"dedicated-{uuid.uuid4().hex[:8]}"), False
        self._ensure_pool(split)
        try:
            return self.available[split].get(timeout=self.acquire_timeout_s), True
        except queue.Empty as exc:
            raise CapacityError(f"No {self.env_name} worker available for split={split} within {self.acquire_timeout_s}s") from exc

    def allocate(self, payload: dict[str, Any]) -> dict[str, Any]:
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
        worker, pooled = self._acquire(split)
        lease_id = f"lease-{uuid.uuid4().hex[:16]}"
        lease = Lease(
            lease_id=lease_id,
            worker=worker,
            split=split,
            pooled=pooled,
            request_id=str(request_id) if request_id else None,
            task_key=task_key,
        )
        with self.lock:
            self.leases[lease_id] = lease
            if request_id:
                self.idempotency[(task_key, str(request_id))] = (lease_id, time.time())
        return {"ok": True, "lease_id": lease_id, "session_id": lease_id, "worker_id": worker.worker_id, "reused": False, "split": split}

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

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(payload.get("lease_id") or payload.get("session_id") or "")
        if not lease_id:
            alloc = self.allocate(payload)
            lease_id = str(alloc["lease_id"])
        lease = self._get_lease(lease_id)
        result = self._worker_request(lease.worker, "reset", payload)
        lease.split = str(result.get("split", lease.split))
        lease.reset_at = time.time()
        lease.last_used_at = time.time()
        lease.final_score = 0.0
        lease.done = False
        lease.success = False
        lease.last_info = result.get("info") or {}
        if result.get("task_index") is not None:
            lease.task_index = int(result["task_index"])
        return {"ok": True, "lease_id": lease.lease_id, "session_id": lease.lease_id, "worker_id": lease.worker.worker_id, **result}

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease = self._get_lease(str(payload.get("lease_id") or payload.get("session_id")))
        result = self._worker_request(lease.worker, "step", payload)
        lease.final_score = float(result.get("score", 0.0) or 0.0)
        lease.done = bool(result.get("done", False))
        lease.success = bool(result.get("success", False))
        lease.last_info = result.get("info") or {}
        if result.get("task_index") is not None:
            lease.task_index = int(result["task_index"])
        return {"ok": True, "lease_id": lease.lease_id, "session_id": lease.lease_id, "worker_id": lease.worker.worker_id, **result}

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease = self._get_lease(str(payload.get("lease_id") or payload.get("session_id")))
        result = self._worker_request(lease.worker, "evaluate", payload)
        lease.final_score = float(result.get("score", lease.final_score) or 0.0)
        lease.success = bool(result.get("success", lease.success))
        lease.done = bool(result.get("done", lease.done))
        lease.last_info = result.get("info") or lease.last_info
        return {"ok": True, "lease_id": lease.lease_id, "session_id": lease.lease_id, "worker_id": lease.worker.worker_id, **result}

    def heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease = self._get_lease(str(payload.get("lease_id") or payload.get("session_id")))
        lease.last_used_at = time.time()
        return {"ok": True, "lease_id": lease.lease_id, "session_id": lease.lease_id}

    def close(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(payload.get("lease_id") or payload.get("session_id") or "")
        if not lease_id:
            return {"ok": True, "found": False}
        with self.lock:
            lease = self.leases.pop(lease_id, None)
        if lease is None:
            return {"ok": True, "found": False}
        self._release_worker(lease)
        return {"ok": True, "found": True, "lease_id": lease.lease_id, "session_id": lease.lease_id, "worker_id": lease.worker.worker_id}

    def health(self) -> dict[str, Any]:
        ready_values = [worker.ready for workers in self.created.values() for worker in workers]
        payload: dict[str, Any] = {
            "ok": True,
            "leases": len(self.leases),
            "active_leases": len(self.leases),
            "pool_size": int(self.server_config["pool_size"]),
            "splits": list(self.available),
            "lease_ttl_s": self.lease_ttl_s,
            "worker_request_timeout_s": self.worker_request_timeout_s,
        }
        for key in ("num_goals", "num_games"):
            value = next((ready.get(key) for ready in ready_values if ready.get(key) is not None), None)
            if value is not None:
                payload[key] = value
        return payload

    def status(self) -> dict[str, Any]:
        with self.lock:
            active_leases = len(self.leases)
            leases = [
                {
                    "lease_id": lease_id,
                    "worker_id": lease.worker.worker_id,
                    "split": lease.split,
                    "task_index": lease.task_index,
                    "age_s": round(time.time() - lease.created_at, 3),
                    "idle_s": round(time.time() - lease.last_used_at, 3),
                    "reset": lease.reset_at is not None,
                    "done": lease.done,
                    "pooled": lease.pooled,
                }
                for lease_id, lease in self.leases.items()
            ]
        pools = {}
        for split, workers in self.created.items():
            pools[split] = {
                "pool_size": len(workers),
                "available": self.available[split].qsize() if split in self.available else 0,
                "resets": sum(w.reset_count for w in workers),
                "steps": sum(w.step_count for w in workers),
                "alive": sum(1 for w in workers if w.process.is_alive()),
                "pids": [w.process.pid for w in workers],
            }
        return {"ok": True, **self.health(), "active_leases": active_leases, "leases": leases, "pools": pools}

    def shutdown(self) -> None:
        for workers in self.created.values():
            for worker in workers:
                try:
                    self._worker_request(worker, "close")
                except Exception:
                    pass
                if worker.process.is_alive():
                    worker.process.terminate()
                worker.process.join(timeout=5)


def normalize_server_config(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "pool_size": int(raw.get("pool_size", 8)),
        "acquire_timeout_s": float(raw.get("acquire_timeout_s", 600.0)),
        "lease_ttl_s": float(raw.get("lease_ttl_s", 1800.0)),
        "idempotency_ttl_s": float(raw.get("idempotency_ttl_s", 300.0)),
        "worker_start_timeout_s": float(raw.get("worker_start_timeout_s", 300.0)),
        "worker_request_timeout_s": float(raw.get("worker_request_timeout_s", 180.0)),
        "prewarm_splits": list(raw.get("prewarm_splits", ["train"])),
        "reuse_workers": bool(raw.get("reuse_workers", True)),
        "reset_on_release": bool(raw.get("reset_on_release", False)),
    }


class EnvRequestHandler(BaseHTTPRequestHandler):
    store: ProcessPoolEnvServer

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path in ("/health", "/healthz"):
            _json_response(self, 200, self.store.health())
            return
        if self.path == "/status":
            _json_response(self, 200, self.store.status())
            return
        _json_response(self, 404, {"ok": False, "error": "not found"})

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
            elif self.path == "/heartbeat":
                result = self.store.heartbeat(payload)
            elif self.path == "/close":
                result = self.store.close(payload)
            else:
                _json_response(self, 404, {"ok": False, "error": "not found"})
                return
            _json_response(self, 200, result)
        except CapacityError as exc:
            _json_response(self, 503, {"ok": False, "error": _format_error(exc)})
        except KeyError as exc:
            _json_response(self, 404, {"ok": False, "error": _format_error(exc)})
        except Exception as exc:
            logger.exception("env server request failed path=%s", self.path)
            _json_response(self, 500, {"ok": False, "error": _format_error(exc), "traceback": traceback.format_exc()})

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)


def serve_process_pool(
    *,
    host: str,
    port: int,
    backend_cls: type[EnvBackend],
    env_config: dict[str, Any],
    server_config: dict[str, Any],
    env_name: str,
) -> None:
    EnvRequestHandler.store = ProcessPoolEnvServer(
        backend_cls=backend_cls,
        env_config=env_config,
        server_config=server_config,
        env_name=env_name,
    )
    logger.info("%s env server listening on %s:%s with server_config=%s", env_name, host, port, server_config)
    AgentThreadingHTTPServer((host, port), EnvRequestHandler).serve_forever()
