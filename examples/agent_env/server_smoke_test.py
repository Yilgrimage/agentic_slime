from __future__ import annotations

import json
import threading
import time
import urllib.request

from examples.agent_env.server import AgentThreadingHTTPServer, EnvRequestHandler, ProcessPoolEnvServer

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

class FakeBackend:
    def __init__(self, worker_id: str, split: str, config: dict) -> None:
        self.worker_id = worker_id
        self.split = split
        self.config = config
        self.reset_count = 0
        self.step_count = 0
        self.score = 0.0
        self.done = False

    def start(self) -> dict:
        return {"num_games": int(self.config.get("num_games", 3))}

    def reset(self, payload: dict) -> dict:
        self.reset_count += 1
        self.step_count = 0
        self.score = 0.0
        self.done = False
        return {
            "observation": f"reset:{payload.get('task_index')}",
            "info": {"admissible_commands": [["look"]]},
            "split": str(payload.get("split") or self.split),
            "task_index": int(payload.get("task_index") or 0),
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def step(self, payload: dict) -> dict:
        self.step_count += 1
        self.score = 1.0
        self.done = True
        return {
            "observation": f"step:{payload.get('action')}",
            "score": self.score,
            "done": self.done,
            "success": True,
            "info": {"won": [True]},
            "reset_count": self.reset_count,
            "step_count": self.step_count,
        }

    def evaluate(self, payload: dict) -> dict:
        return {"score": self.score, "done": self.done, "success": self.score > 0}

    def release(self, payload: dict) -> dict:
        return {"reset_count": self.reset_count, "step_count": self.step_count}

    def close(self) -> dict:
        return {}


def _post(base_url: str, endpoint: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{base_url}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with _OPENER.open(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    store = ProcessPoolEnvServer(
        backend_cls=FakeBackend,
        env_config={"num_games": 7},
        server_config={"pool_size": 1, "prewarm_splits": ["train"], "worker_start_timeout_s": 30},
        env_name="fake",
    )
    EnvRequestHandler.store = store
    server = AgentThreadingHTTPServer(("127.0.0.1", 0), EnvRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        health = json.loads(_OPENER.open(f"{base_url}/health", timeout=10).read().decode())
        assert health["ok"] is True
        assert health["num_games"] == 7
        lease = _post(base_url, "/allocate", {"split": "train", "task_key": "train:0", "request_id": "r0"})
        assert lease["ok"] is True
        lease_id = lease["lease_id"]
        worker_id = lease["worker_id"]
        reset = _post(base_url, "/reset", {"lease_id": lease_id, "split": "train", "task_index": 0})
        assert reset["observation"] == "reset:0"
        step = _post(base_url, "/step", {"lease_id": lease_id, "action": "look"})
        assert step["success"] is True
        eval_payload = _post(base_url, "/evaluate", {"lease_id": lease_id})
        assert eval_payload["score"] == 1.0
        close = _post(base_url, "/close", {"lease_id": lease_id})
        assert close["found"] is True
        eval_lease = _post(base_url, "/allocate", {"split": "valid_seen", "task_key": "valid_seen:1", "request_id": "e0"})
        assert eval_lease["ok"] is True
        assert eval_lease["worker_id"] == worker_id
        eval_reset = _post(base_url, "/reset", {"lease_id": eval_lease["lease_id"], "split": "valid_seen", "task_index": 1})
        assert eval_reset["split"] == "valid_seen"
        _post(base_url, "/close", {"lease_id": eval_lease["lease_id"]})
        status = json.loads(_OPENER.open(f"{base_url}/status", timeout=10).read().decode())
        assert status["active_leases"] == 0
        assert list(status["pools"]) == ["__shared__"]
        assert status["pools"]["__shared__"]["pool_size"] == 1
        print("agent env server smoke test passed")
    finally:
        server.shutdown()
        store.shutdown()
        thread.join(timeout=5)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
