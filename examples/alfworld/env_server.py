from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import queue
import os
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


def _server_config(raw: dict) -> dict:
    return {
        "pool_size": int(raw.get("alfworld_server_pool_size", 8)),
        "acquire_timeout_s": float(raw.get("alfworld_server_acquire_timeout_s", 30.0)),
        "lease_ttl_s": float(
            raw.get("alfworld_server_lease_ttl_s", raw.get("alfworld_server_session_ttl_s", 1800.0))
        ),
        "idempotency_ttl_s": float(raw.get("alfworld_server_idempotency_ttl_s", 300.0)),
        "reuse_envs": bool(raw.get("alfworld_server_reuse_envs", True)),
        "reset_on_release": bool(raw.get("alfworld_server_reset_on_release", False)),
        "reset_parallelism": int(raw.get("alfworld_server_reset_parallelism", 1)),
        "worker_start_timeout_s": float(raw.get("alfworld_server_worker_start_timeout_s", 120.0)),
        "worker_request_timeout_s": float(raw.get("alfworld_server_worker_request_timeout_s", 120.0)),
        "prewarm_splits": list(raw.get("alfworld_server_prewarm_splits", ["train"])),
        "honor_direct_game_file": bool(raw.get("alfworld_server_honor_direct_game_file", True)),
    }


def _default_alfworld_config(raw: dict) -> dict:
    data_dir = str(raw.get("alfworld_data_dir") or "$ALFWORLD_DATA").rstrip("/")
    max_steps = int(raw.get("alfworld_max_turns", 50))
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
    with Path(path).expanduser().open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    server_config = _server_config(raw)
    if raw.get("alfworld_config_path"):
        with Path(raw["alfworld_config_path"]).expanduser().open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    elif "dataset" in raw and "env" in raw:
        config = raw
    else:
        config = _default_alfworld_config(raw)
    return _deep_update(config, overrides or {}), server_config


@dataclass
class EnvWorker:
    worker_id: str
    split: str
    process: mp.Process
    conn: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    game_file: str | None = None
    ready: bool = False
    num_games: int = 0
    reset_count: int = 0
    step_count: int = 0


@dataclass
class LeaseState:
    lease_id: str
    worker: EnvWorker
    split: str
    pooled: bool
    task_index: int | None = None
    game_file: str | None = None
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    reset_at: float | None = None
    request_id: str | None = None
    task_key: str | None = None
    final_score: float = 0.0
    done: bool = False
    success: bool = False
    last_info: dict = field(default_factory=dict)


class CapacityError(Exception):
    pass


def _select_game_file(game_files: list[str], task_index: int) -> str:
    return game_files[int(task_index) % len(game_files)]


def _alfworld_worker_loop(
    worker_id: str,
    split: str,
    config: dict,
    env_type: str,
    honor_direct_game_file: bool,
    conn: Any,
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    try:
        import sys

        alfworld_lib = os.environ.get("ALFWORLD_LIB")
        if alfworld_lib and alfworld_lib not in sys.path:
            sys.path.insert(0, alfworld_lib)
        from alfworld.agents.environment import get_environment

        env_cls = get_environment(env_type)
        wrapper = env_cls(config, train_eval=split)
        game_files = list(getattr(wrapper, "game_files", None) or [])
        if not game_files:
            raise RuntimeError(f"ALFWorld split={split} has no games. Check data_path and game.tw-pddl files.")
        env = wrapper.init_env(batch_size=1)
        game_file: str | None = None
        reset_count = 0
        step_count = 0
        final_score = 0.0
        done = False
        success = False
        last_info: dict = {}
        task_index: int | None = None
        conn.send({"ok": True, "event": "ready", "worker_id": worker_id, "split": split, "num_games": len(game_files)})

        while True:
            request = conn.recv()
            cmd = request.get("cmd")
            payload = request.get("payload") or {}
            try:
                if cmd == "reset":
                    task_index = int(payload.get("task_index") or 0)
                    seed = payload.get("seed", task_index)
                    direct_game_file = bool(payload.get("direct_game_file", True))
                    skip_to_task = bool(payload.get("skip_to_task", False))
                    num_tasks = payload.get("num_tasks")
                    if direct_game_file and honor_direct_game_file:
                        game_file = _select_game_file(game_files, task_index)
                        if hasattr(env, "gamefiles"):
                            env.gamefiles = [game_file]
                    if seed is not None and hasattr(env, "seed"):
                        try:
                            env.seed(int(seed))
                        except Exception:
                            logging.getLogger(__name__).debug("ALFWorld env did not accept seed=%s", seed, exc_info=True)
                    if skip_to_task and task_index > 0 and not direct_game_file:
                        skip_count = task_index % int(num_tasks) if num_tasks else task_index
                        for _ in range(skip_count):
                            env.reset()
                    obs, info = env.reset()
                    reset_count += 1
                    final_score = 0.0
                    done = False
                    success = False
                    last_info = info or {}
                    conn.send(
                        {
                            "ok": True,
                            "observation": str(_first(obs, "")),
                            "info": last_info,
                            "split": split,
                            "game_file": game_file,
                            "task_index": task_index,
                            "reset_count": reset_count,
                            "step_count": step_count,
                        }
                    )
                elif cmd == "step":
                    action = str(payload.get("action") or "look")
                    obs, scores, dones, info = env.step([action])
                    step_count += 1
                    final_score = float(_first(scores, 0.0) or 0.0)
                    done = bool(_first(dones, False))
                    won = _first(info.get("won") if info else None, None)
                    success = bool(won) if won is not None else final_score > 0
                    last_info = info or {}
                    conn.send(
                        {
                            "ok": True,
                            "observation": str(_first(obs, "")),
                            "score": final_score,
                            "done": done,
                            "success": success,
                            "info": last_info,
                            "game_file": game_file,
                            "task_index": task_index,
                            "reset_count": reset_count,
                            "step_count": step_count,
                        }
                    )
                elif cmd == "evaluate":
                    conn.send(
                        {
                            "ok": True,
                            "score": float(final_score),
                            "success": bool(success),
                            "done": bool(done),
                            "info": last_info,
                            "game_file": game_file,
                            "task_index": task_index,
                            "reset_count": reset_count,
                            "step_count": step_count,
                        }
                    )
                elif cmd == "release":
                    if bool(payload.get("reset_on_release", False)):
                        env.reset()
                        reset_count += 1
                    conn.send({"ok": True, "reset_count": reset_count, "step_count": step_count})
                elif cmd == "close":
                    if hasattr(env, "close"):
                        env.close()
                    conn.send({"ok": True})
                    break
                else:
                    raise ValueError(f"Unknown ALFWorld worker command: {cmd}")
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


class AlfworldLeaseStore:
    def __init__(
        self,
        config: dict,
        server_config: dict,
        env_type: str | None = None,
        direct_game_file: bool = True,
    ) -> None:
        self.config = config
        self.server_config = server_config
        self.env_type = env_type or config.get("env", {}).get("type", "AlfredTWEnv")
        self.direct_game_file = direct_game_file
        self.reuse_envs = bool(server_config["reuse_envs"])
        self.honor_direct_game_file = bool(server_config["honor_direct_game_file"])
        self.acquire_timeout_s = float(server_config["acquire_timeout_s"])
        self.lease_ttl_s = float(server_config["lease_ttl_s"])
        self.idempotency_ttl_s = float(server_config["idempotency_ttl_s"])
        self.reset_on_release = bool(server_config["reset_on_release"])
        self.worker_start_timeout_s = float(server_config["worker_start_timeout_s"])
        self.worker_request_timeout_s = float(server_config["worker_request_timeout_s"])
        self.mp_ctx = mp.get_context("spawn")
        self.leases: dict[str, LeaseState] = {}
        self.idempotency: dict[tuple[str, str], tuple[str, float]] = {}
        self.lock = threading.Lock()
        self.available_workers: dict[str, queue.Queue[EnvWorker]] = {}
        self.created_workers: dict[str, list[EnvWorker]] = {}

        if self.reuse_envs:
            for split in server_config["prewarm_splits"]:
                self._ensure_pool(split)

    def _spawn_worker(self, split: str, index: int | str) -> EnvWorker:
        parent_conn, child_conn = self.mp_ctx.Pipe()
        worker_id = f"{split}-{index}"
        process = self.mp_ctx.Process(
            target=_alfworld_worker_loop,
            args=(worker_id, split, self.config, self.env_type, self.honor_direct_game_file, child_conn),
            name=f"alfworld-env-{worker_id}",
            daemon=True,
        )
        process.start()
        child_conn.close()
        return EnvWorker(worker_id=worker_id, split=split, process=process, conn=parent_conn)

    def _wait_worker_ready(self, worker: EnvWorker) -> EnvWorker:
        if not worker.conn.poll(self.worker_start_timeout_s):
            worker.process.terminate()
            worker.process.join(timeout=5)
            raise RuntimeError(f"ALFWorld worker {worker.worker_id} did not become ready in {self.worker_start_timeout_s}s")
        ready = worker.conn.recv()
        if not ready.get("ok"):
            worker.process.join(timeout=5)
            raise RuntimeError(f"ALFWorld worker {worker.worker_id} failed during startup: {ready}")
        worker.ready = True
        worker.num_games = int(ready.get("num_games") or 0)
        logger.info(
            "Started ALFWorld worker process %s pid=%s split=%s games=%s",
            worker.worker_id,
            worker.process.pid,
            worker.split,
            worker.num_games,
        )
        return worker

    def _start_worker(self, split: str, index: int | str) -> EnvWorker:
        worker = self._spawn_worker(split, index)
        self._wait_worker_ready(worker)
        return worker

    def _worker_request(self, worker: EnvWorker, cmd: str, payload: dict | None = None) -> dict:
        with worker.lock:
            if not worker.process.is_alive():
                raise RuntimeError(f"ALFWorld worker {worker.worker_id} pid={worker.process.pid} is not alive")
            worker.conn.send({"cmd": cmd, "payload": payload or {}})
            if not worker.conn.poll(self.worker_request_timeout_s):
                raise TimeoutError(
                    f"ALFWorld worker {worker.worker_id} timed out on {cmd} after {self.worker_request_timeout_s}s"
                )
            result = worker.conn.recv()
            if not result.get("ok"):
                raise RuntimeError(f"ALFWorld worker {worker.worker_id} failed on {cmd}: {result}")
            worker.reset_count = int(result.get("reset_count", worker.reset_count) or worker.reset_count)
            worker.step_count = int(result.get("step_count", worker.step_count) or worker.step_count)
            worker.game_file = result.get("game_file", worker.game_file)
            return result

    def _ensure_pool(self, split: str) -> None:
        if split in self.available_workers:
            return
        pool_size = int(self.server_config["pool_size"])
        q: queue.Queue[EnvWorker] = queue.Queue(maxsize=pool_size)
        logger.info("Prewarming %d process-isolated ALFWorld env workers for split=%s", pool_size, split)
        workers = [self._spawn_worker(split, i) for i in range(pool_size)]
        for worker in workers:
            self._wait_worker_ready(worker)
            q.put(worker)
        self.available_workers[split] = q
        self.created_workers[split] = workers
        logger.info("Prewarmed %d ALFWorld env workers for split=%s", pool_size, split)

    def _reap_expired_locked(self) -> list[LeaseState]:
        now = time.time()
        expired: list[LeaseState] = []
        for key, (_, ts) in list(self.idempotency.items()):
            if now - ts > self.idempotency_ttl_s:
                self.idempotency.pop(key, None)
        for lease_id, lease in list(self.leases.items()):
            if now - lease.last_used_at > self.lease_ttl_s:
                expired.append(self.leases.pop(lease_id))
        return expired

    def _release_worker(self, worker: EnvWorker, pooled: bool) -> None:
        if self.reset_on_release:
            self._worker_request(worker, "release", {"reset_on_release": True})
        if pooled:
            self.available_workers[worker.split].put(worker)
            return
        try:
            self._worker_request(worker, "close")
        finally:
            worker.process.join(timeout=5)
            if worker.process.is_alive():
                worker.process.terminate()

    def _release_expired(self, expired: list[LeaseState]) -> None:
        for lease in expired:
            logger.warning("Reaping expired ALFWorld lease %s", lease.lease_id)
            self._release_worker(lease.worker, lease.pooled)

    def _acquire_worker(self, split: str) -> tuple[EnvWorker, bool]:
        if not self.reuse_envs:
            return self._start_worker(split, f"dedicated-{uuid.uuid4().hex[:8]}"), False

        self._ensure_pool(split)
        try:
            return self.available_workers[split].get(timeout=self.acquire_timeout_s), True
        except queue.Empty as exc:
            raise CapacityError(f"No ALFWorld env worker available for split={split} within {self.acquire_timeout_s}s") from exc

    def allocate(self, payload: dict) -> dict:
        split = payload.get("split") or "train"
        task_key = str(payload.get("task_key") or split)
        request_id = payload.get("request_id")

        with self.lock:
            expired = self._reap_expired_locked()
            if request_id:
                idem_key = (task_key, str(request_id))
                cached = self.idempotency.get(idem_key)
                if cached is not None:
                    lease_id, _ = cached
                    if lease_id in self.leases:
                        lease = self.leases[lease_id]
                        lease.last_used_at = time.time()
                        self._release_expired(expired)
                        return {
                            "ok": True,
                            "lease_id": lease_id,
                            "session_id": lease_id,
                            "worker_id": lease.worker.worker_id,
                            "reused": True,
                            "split": lease.split,
                        }

        self._release_expired(expired)
        worker, pooled = self._acquire_worker(split)
        lease_id = f"lease-{uuid.uuid4().hex[:16]}"
        lease = LeaseState(
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
        return {
            "ok": True,
            "lease_id": lease_id,
            "session_id": lease_id,
            "worker_id": worker.worker_id,
            "reused": False,
            "split": split,
        }

    def _get_lease(self, lease_id: str) -> LeaseState:
        with self.lock:
            expired = self._reap_expired_locked()
            lease = self.leases.get(lease_id)
            if lease is not None:
                lease.last_used_at = time.time()
        self._release_expired(expired)
        if lease is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        return lease

    def reset(self, payload: dict) -> dict:
        lease_id = payload.get("lease_id") or payload.get("session_id")
        legacy_allocate = False
        if not lease_id:
            alloc = self.allocate(
                {
                    "split": payload.get("split") or "train",
                    "task_key": payload.get("task_key"),
                    "request_id": payload.get("request_id"),
                }
            )
            lease_id = alloc["lease_id"]
            legacy_allocate = True

        lease = self._get_lease(str(lease_id))
        split = payload.get("split") or lease.split
        task_index = int(payload.get("task_index") or 0)
        seed = payload.get("seed", task_index)
        direct_game_file = bool(payload.get("direct_game_file", self.direct_game_file))
        skip_to_task = bool(payload.get("skip_to_task", False))
        num_tasks = payload.get("num_tasks")

        try:
            result = self._worker_request(
                lease.worker,
                "reset",
                {
                    "task_index": task_index,
                    "seed": seed,
                    "direct_game_file": direct_game_file,
                    "skip_to_task": skip_to_task,
                    "num_tasks": num_tasks,
                },
            )
            lease.split = split
            lease.task_index = task_index
            lease.game_file = result.get("game_file")
            lease.reset_at = time.time()
            lease.last_used_at = time.time()
            lease.final_score = 0.0
            lease.done = False
            lease.success = False
            lease.last_info = result.get("info") or {}
            return {
                "ok": True,
                "lease_id": lease.lease_id,
                "session_id": lease.lease_id,
                "worker_id": lease.worker.worker_id,
                "observation": str(result.get("observation", "")),
                "info": lease.last_info,
                "split": split,
                "game_file": lease.game_file,
                "legacy_allocated": legacy_allocate,
            }
        except Exception:
            if legacy_allocate:
                self.close({"lease_id": lease.lease_id})
            raise

    def step(self, payload: dict) -> dict:
        lease_id = payload.get("lease_id") or payload.get("session_id")
        if not lease_id:
            raise ValueError("lease_id is required")
        action = str(payload.get("action") or "look")
        lease = self._get_lease(str(lease_id))
        result = self._worker_request(lease.worker, "step", {"action": action})
        score = float(result.get("score", 0.0) or 0.0)
        done = bool(result.get("done", False))
        success = bool(result.get("success", False))
        lease.final_score = score
        lease.done = done
        lease.success = success
        lease.last_info = result.get("info") or {}
        lease.game_file = result.get("game_file", lease.game_file)
        lease.task_index = result.get("task_index", lease.task_index)
        lease.last_used_at = time.time()
        return {
            "ok": True,
            "observation": str(result.get("observation", "")),
            "score": score,
            "done": done,
            "success": success,
            "info": lease.last_info,
            "worker_id": lease.worker.worker_id,
            "lease_id": lease.lease_id,
            "session_id": lease.lease_id,
        }

    def evaluate(self, payload: dict) -> dict:
        lease_id = payload.get("lease_id") or payload.get("session_id")
        if not lease_id:
            raise ValueError("lease_id is required")
        lease = self._get_lease(str(lease_id))
        result = self._worker_request(lease.worker, "evaluate")
        lease.final_score = float(result.get("score", lease.final_score) or 0.0)
        lease.success = bool(result.get("success", lease.success))
        lease.done = bool(result.get("done", lease.done))
        lease.last_info = result.get("info") or lease.last_info
        lease.game_file = result.get("game_file", lease.game_file)
        lease.task_index = result.get("task_index", lease.task_index)
        return {
            "ok": True,
            "lease_id": lease.lease_id,
            "session_id": lease.lease_id,
            "score": float(lease.final_score),
            "success": bool(lease.success),
            "done": bool(lease.done),
            "info": lease.last_info,
            "game_file": lease.game_file,
            "task_index": lease.task_index,
            "worker_id": lease.worker.worker_id,
        }

    def heartbeat(self, payload: dict) -> dict:
        lease_id = payload.get("lease_id") or payload.get("session_id")
        if not lease_id:
            raise ValueError("lease_id is required")
        lease = self._get_lease(str(lease_id))
        lease.last_used_at = time.time()
        return {"ok": True, "lease_id": lease.lease_id, "session_id": lease.lease_id}

    def close(self, payload: dict) -> dict:
        lease_id = payload.get("lease_id") or payload.get("session_id")
        if not lease_id:
            return {"ok": True, "found": False}
        with self.lock:
            lease = self.leases.pop(str(lease_id), None)
        if lease is None:
            return {"ok": True, "found": False, "missing": True}
        self._release_worker(lease.worker, lease.pooled)
        return {
            "ok": True,
            "found": True,
            "lease_id": lease.lease_id,
            "session_id": lease.lease_id,
            "worker_id": lease.worker.worker_id,
        }

    def stats(self) -> dict:
        with self.lock:
            expired = self._reap_expired_locked()
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
            idempotency_entries = len(self.idempotency)
        self._release_expired(expired)
        splits = {}
        for split, workers in self.created_workers.items():
            splits[split] = {
                "pool_size": len(workers),
                "available": self.available_workers[split].qsize(),
                "resets": sum(w.reset_count for w in workers),
                "steps": sum(w.step_count for w in workers),
                "alive": sum(1 for w in workers if w.process.is_alive()),
                "pids": [w.process.pid for w in workers],
            }
        return {
            "ok": True,
            "backend": "process",
            "reuse_envs": self.reuse_envs,
            "active_leases": active_leases,
            "active_sessions": active_leases,
            "idempotency_entries": idempotency_entries,
            "lease_ttl_s": self.lease_ttl_s,
            "worker_request_timeout_s": self.worker_request_timeout_s,
            "splits": splits,
            "leases": leases,
        }


class Handler(BaseHTTPRequestHandler):
    store: AlfworldLeaseStore

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path in {"/health", "/healthz"}:
            self._send(200, {"ok": True})
        elif self.path in {"/stats", "/status"}:
            self._send(200, self.store.stats())
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/allocate":
                self._send(200, self.store.allocate(payload))
            elif self.path == "/reset":
                self._send(200, self.store.reset(payload))
            elif self.path == "/step":
                self._send(200, self.store.step(payload))
            elif self.path == "/evaluate":
                self._send(200, self.store.evaluate(payload))
            elif self.path == "/heartbeat":
                self._send(200, self.store.heartbeat(payload))
            elif self.path == "/close":
                self._send(200, self.store.close(payload))
            else:
                self._send(404, {"ok": False, "error": "not found"})
        except CapacityError as exc:
            logger.warning("ALFWorld env server capacity error: %s", exc)
            self._send(429, {"ok": False, "error": str(exc), "code": "CAPACITY_EXHAUSTED"})
        except Exception as exc:
            logger.exception("ALFWorld env server request failed")
            self._send(500, {"ok": False, "error": repr(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-type", default=None)
    parser.add_argument("--no-direct-game-file", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    alfworld_config, server_config = _load_configs(args.config)
    Handler.store = AlfworldLeaseStore(
        alfworld_config,
        server_config,
        env_type=args.env_type,
        direct_game_file=not args.no_direct_game_file,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info(
        "ALFWorld env server listening on %s:%s with server_config=%s",
        args.host,
        args.port,
        server_config,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
