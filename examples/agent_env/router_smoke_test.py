from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from examples.agent_env.router import EnvRouter


class FakeWorker:
    def __init__(self, name: str, fail_allocate: bool = False) -> None:
        self.name = name
        self.fail_allocate = fail_allocate
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.leases: set[str] = set()

    def handle(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        self.requests.append((path, dict(payload)))
        if path == "/allocate":
            if self.fail_allocate:
                return {"ok": False, "error": "pool exhausted", "status_code": 503}, 503
            lease_id = f"{self.name}-lease-{len(self.leases)}"
            self.leases.add(lease_id)
            return {"ok": True, "lease_id": lease_id, "worker_id": self.name}, 200
        lease_id = payload.get("lease_id")
        if lease_id not in self.leases:
            return {"ok": False, "error": f"unknown lease {lease_id}"}, 404
        if path == "/reset":
            return {"ok": True, "lease_id": lease_id, "observation": f"reset:{self.name}"}, 200
        if path == "/step":
            return {"ok": True, "lease_id": lease_id, "observation": f"step:{self.name}", "score": 0.0}, 200
        if path == "/close":
            self.leases.discard(str(lease_id))
            return {"ok": True}, 200
        return {"ok": False, "error": "not found"}, 404


def start_fake_worker(worker: FakeWorker) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length") or 0)
            return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "worker": worker.name})
                return
            self._send(404, {"ok": False})

        def do_POST(self) -> None:
            payload, status = worker.handle(self.path, self._read())
            self._send(status, payload)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def main() -> None:
    w0 = FakeWorker("w0", fail_allocate=True)
    w1 = FakeWorker("w1")
    s0, u0 = start_fake_worker(w0)
    s1, u1 = start_fake_worker(w1)
    try:
        router = EnvRouter([u0, u1], request_timeout_s=5, allocate_timeout_s=5)
        task_key = next(f"train:{i}" for i in range(1000) if router.select_worker_idx(f"train:{i}") == 0)
        allocation, status = router.allocate({"task_key": task_key, "request_id": "req-0"})
        assert status == 200, allocation
        assert allocation["ok"] is True
        assert allocation["worker_idx"] == 1
        assert allocation["worker_lease_id"].startswith("w1-lease")

        reset, status = router.lease_proxy("/reset", {"lease_id": allocation["lease_id"]})
        assert status == 200, reset
        assert reset["observation"] == "reset:w1"
        assert reset["lease_id"] == allocation["lease_id"]

        step, status = router.lease_proxy("/step", {"lease_id": allocation["lease_id"], "action": "look"})
        assert status == 200, step
        assert step["observation"] == "step:w1"
        assert w0.requests and w0.requests[0][0] == "/allocate"
        assert all(path == "/allocate" for path, _ in w0.requests)
        print("agent env router smoke test passed")
    finally:
        s0.shutdown()
        s1.shutdown()


if __name__ == "__main__":
    main()
