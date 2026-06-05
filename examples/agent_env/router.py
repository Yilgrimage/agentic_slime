from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger(__name__)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class AgentThreadingHTTPServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


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


def _payload_status(payload: dict[str, Any], default: int) -> int:
    status = payload.get("status_code")
    return status if isinstance(status, int) else default


@dataclass(frozen=True)
class WorkerRef:
    idx: int
    url: str


class EnvRouter:
    """Routes generic agent-env lease traffic across multiple env pool servers.

    The router deliberately knows nothing about ALFWorld, WebShop, rewards, observations,
    or action formats. It only owns worker selection at allocation time and sticky
    forwarding for the lifetime of a lease.
    """

    def __init__(
        self,
        worker_urls: list[str],
        *,
        request_timeout_s: float = 600.0,
        allocate_timeout_s: float | None = None,
        retry_unreachable: bool = True,
        retry_capacity: bool = True,
    ) -> None:
        if not worker_urls:
            raise ValueError("At least one worker URL is required")
        self.workers = [WorkerRef(i, url.rstrip("/")) for i, url in enumerate(worker_urls)]
        self.request_timeout_s = float(request_timeout_s)
        self.allocate_timeout_s = float(allocate_timeout_s if allocate_timeout_s is not None else request_timeout_s)
        self.retry_unreachable = bool(retry_unreachable)
        self.retry_capacity = bool(retry_capacity)

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    @staticmethod
    def encode_lease(worker_idx: int, worker_lease_id: str) -> str:
        return f"{worker_idx}:{worker_lease_id}"

    @staticmethod
    def decode_lease(global_lease_id: str) -> tuple[int, str]:
        sep = global_lease_id.index(":")
        return int(global_lease_id[:sep]), global_lease_id[sep + 1 :]

    def select_worker_idx(self, task_key: str) -> int:
        digest = hashlib.sha1(task_key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % self.num_workers

    def iter_candidates(self, primary_idx: int) -> list[WorkerRef]:
        return [self.workers[(primary_idx + offset) % self.num_workers] for offset in range(self.num_workers)]

    def worker(self, idx: int) -> WorkerRef:
        return self.workers[idx]

    def request(
        self,
        worker: WorkerRef,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        *,
        method: str = "POST",
        timeout_s: float | None = None,
    ) -> tuple[dict[str, Any], int]:
        url = f"{worker.url}{endpoint}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with _OPENER.open(req, timeout=timeout_s or self.request_timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}, int(resp.status)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8")
            try:
                payload = json.loads(raw) if raw else {}
            except Exception:
                payload = {"ok": False, "error": raw}
            payload.setdefault("status_code", exc.code)
            return payload, int(exc.code)

    @staticmethod
    def should_retry_worker(payload: dict[str, Any], status: int, retry_capacity: bool) -> bool:
        if status in (502, 503, 504):
            if status == 503:
                return retry_capacity
            return True
        if payload.get("ok", True):
            return False
        error = str(payload.get("error", "")).lower()
        capacity_markers = ("capacity", "pool exhausted", "no ", "worker available")
        return retry_capacity and any(marker in error for marker in capacity_markers)

    def allocate(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        task_key = str(payload.get("task_key") or payload.get("request_id") or time.time_ns())
        primary_idx = self.select_worker_idx(task_key)
        upstream_errors: list[dict[str, Any]] = []

        for worker in self.iter_candidates(primary_idx):
            try:
                result, status = self.request(worker, "/allocate", payload, timeout_s=self.allocate_timeout_s)
            except Exception as exc:
                logger.warning(
                    "env worker unreachable on allocate task_key=%s worker_idx=%s url=%s error=%s",
                    task_key,
                    worker.idx,
                    worker.url,
                    _format_error(exc),
                )
                upstream_errors.append({"worker_idx": worker.idx, "worker_url": worker.url, "error": _format_error(exc)})
                if not self.retry_unreachable:
                    break
                continue

            if result.get("ok") and result.get("lease_id"):
                worker_lease_id = str(result["lease_id"])
                logger.debug(
                    "allocated task_key=%s primary_worker_idx=%s worker_idx=%s worker_url=%s lease_id=%s",
                    task_key,
                    primary_idx,
                    worker.idx,
                    worker.url,
                    worker_lease_id,
                )
                result["lease_id"] = self.encode_lease(worker.idx, worker_lease_id)
                result["session_id"] = result["lease_id"]
                result["worker_idx"] = worker.idx
                result["worker_url"] = worker.url
                result["worker_lease_id"] = worker_lease_id
                result["primary_worker_idx"] = primary_idx
                return result, _payload_status(result, status)

            upstream_errors.append(
                {
                    "worker_idx": worker.idx,
                    "worker_url": worker.url,
                    "status_code": status,
                    "payload": result,
                }
            )
            if not self.should_retry_worker(result, status, self.retry_capacity):
                return result, _payload_status(result, status)

        return (
            {
                "ok": False,
                "error": "No env worker could allocate a lease",
                "task_key": task_key,
                "primary_worker_idx": primary_idx,
                "upstream_errors": upstream_errors,
            },
            503,
        )

    def lease_proxy(self, endpoint: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        global_lease_id = str(payload.get("lease_id") or payload.get("session_id") or "")
        if not global_lease_id:
            return {"ok": False, "error": "lease_id is required"}, 400
        try:
            worker_idx, worker_lease_id = self.decode_lease(global_lease_id)
            worker = self.worker(worker_idx)
        except Exception as exc:
            return {"ok": False, "error": f"Invalid lease_id format: {_format_error(exc)}"}, 400

        forwarded = dict(payload)
        forwarded["lease_id"] = worker_lease_id
        forwarded["session_id"] = worker_lease_id
        try:
            result, status = self.request(worker, endpoint, forwarded)
        except Exception as exc:
            return (
                {
                    "ok": False,
                    "error": f"Env worker unreachable: {_format_error(exc)}",
                    "worker_idx": worker.idx,
                    "worker_url": worker.url,
                    "lease_id": global_lease_id,
                },
                502,
            )

        if result.get("ok", False):
            result["lease_id"] = global_lease_id
            result["session_id"] = global_lease_id
            result["worker_idx"] = worker.idx
            result["worker_url"] = worker.url
        return result, _payload_status(result, status)

    def health(self) -> dict[str, Any]:
        return {"ok": True, "num_workers": self.num_workers, "workers": [w.url for w in self.workers]}

    def status(self) -> dict[str, Any]:
        workers: list[dict[str, Any]] = []
        for worker in self.workers:
            try:
                data, status = self.request(worker, "/health", None, method="GET", timeout_s=10.0)
                workers.append({"worker_idx": worker.idx, "worker_url": worker.url, "status_code": status, **data})
            except Exception as exc:
                workers.append({"worker_idx": worker.idx, "worker_url": worker.url, "ok": False, "error": _format_error(exc)})
        return {"ok": True, "num_workers": self.num_workers, "workers": workers}


class Handler(BaseHTTPRequestHandler):
    router: EnvRouter

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path in ("/health", "/healthz"):
            _json_response(self, 200, self.router.health())
            return
        if self.path == "/status":
            _json_response(self, 200, self.router.status())
            return
        _json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/allocate":
                result, status = self.router.allocate(payload)
            elif self.path in ("/reset", "/step", "/evaluate", "/close", "/heartbeat"):
                result, status = self.router.lease_proxy(self.path, payload)
            else:
                _json_response(self, 404, {"ok": False, "error": "not found"})
                return
            _json_response(self, status, result)
        except Exception as exc:
            logger.exception("router request failed path=%s", self.path)
            _json_response(self, 500, {"ok": False, "error": _format_error(exc), "traceback": traceback.format_exc()})

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic multi-worker router for agentic environment pool servers.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--workers", required=True, help="Comma-separated worker URLs.")
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--allocate-timeout-s", type=float, default=None)
    parser.add_argument("--no-retry-unreachable", action="store_true")
    parser.add_argument("--no-retry-capacity", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    worker_urls = [url.strip() for url in args.workers.split(",") if url.strip()]
    Handler.router = EnvRouter(
        worker_urls,
        request_timeout_s=args.request_timeout_s,
        allocate_timeout_s=args.allocate_timeout_s,
        retry_unreachable=not args.no_retry_unreachable,
        retry_capacity=not args.no_retry_capacity,
    )
    logger.info("agent env router listening on http://%s:%s workers=%s", args.host, args.port, worker_urls)
    AgentThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
