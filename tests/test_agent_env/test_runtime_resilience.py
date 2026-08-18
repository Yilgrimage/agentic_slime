import asyncio
import threading
from argparse import Namespace
from unittest.mock import patch

import httpx
import pytest

from slime.utils.types import Sample

from examples.agent_env.env_episode import PolicyCallError
from examples.agent_env.fully_async_rollout import _no_progress_timeout_s
from examples.agent_env.rollout import AgentEnvSpec, call_policy
from examples.agent_env.server import Lease, ProcessPoolEnvServer, RecoverableWorkerError, Worker


def _spec() -> AgentEnvSpec:
    return AgentEnvSpec(
        name="test",
        env_url_arg="test_env_server_url",
        default_split="train",
        info_actions=lambda info: [],
        observation_text=lambda args, observation, info: observation,
        initial_prompt=lambda args, sample, observation, info: observation,
        choose_action=lambda args, tok, actions, sample: actions[0],
        success=lambda info, reward: False,
        env_metadata=lambda info, turn, action, parse_mode: {},
    )


def test_policy_timeout_aborts_the_matching_sglang_request() -> None:
    calls: list[tuple[str, dict]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, *, json, headers=None, timeout=None):
            calls.append((url, json))
            if url.endswith("/generate"):
                raise httpx.ReadTimeout("policy timed out", request=httpx.Request("POST", url))
            return FakeResponse()

    args = Namespace(timeouts={"policy_s": 30}, router_policy="cache_aware")
    sample = Sample(session_id="session-1")

    def model_url(args, role, endpoint):
        return f"http://router:30000{endpoint}"

    with (
        patch("slime.rollout.sglang_rollout.get_model_url", side_effect=model_url),
        patch("httpx.AsyncClient", FakeAsyncClient),
        pytest.raises(httpx.ReadTimeout),
    ):
        asyncio.run(call_policy(args, _spec(), sample, [1, 2], {"max_new_tokens": 4}))

    assert calls[0][0].endswith("/generate")
    assert calls[0][1]["rid"]
    assert calls[1] == ("http://router:30000/abort_request", {"rid": calls[0][1]["rid"]})


def test_recoverable_episode_failure_releases_worker_without_marking_it_dead() -> None:
    server = object.__new__(ProcessPoolEnvServer)
    server.lock = threading.Lock()
    server.worker_episode_timeout_s = 900.0
    worker = Worker(worker_id="train-0", split="train", process=None, conn=None)
    lease = Lease(lease_id="lease-1", worker=worker, split="train", pooled=True)
    server.leases = {lease.lease_id: lease}
    server._get_lease = lambda lease_id: lease
    server._worker_request = lambda *args, **kwargs: (_ for _ in ()).throw(RecoverableWorkerError("timeout"))
    released: list[Lease] = []
    server._release_worker = released.append

    with pytest.raises(RecoverableWorkerError):
        server.run_episode({"lease_id": lease.lease_id})

    assert lease.lease_id not in server.leases
    assert released == [lease]
    assert not worker.dead


def test_worker_protocol_preserves_recoverable_policy_failure() -> None:
    class FakeProcess:
        pid = 123

        def is_alive(self) -> bool:
            return True

    class FakeConnection:
        def send(self, payload) -> None:
            return None

        def poll(self, timeout) -> bool:
            return True

        def recv(self):
            return {"ok": False, "recoverable": True, "error": "PolicyCallError('timeout')"}

    server = object.__new__(ProcessPoolEnvServer)
    server.env_name = "appworld"
    server.worker_request_timeout_s = 900.0
    worker = Worker(worker_id="train-0", split="train", process=FakeProcess(), conn=FakeConnection())

    assert PolicyCallError.recoverable_worker
    with pytest.raises(RecoverableWorkerError):
        server._worker_request(worker, "run_episode", {})
    assert not worker.dead


def test_no_progress_timeout_reuses_existing_env_request_budget() -> None:
    assert _no_progress_timeout_s(Namespace(timeouts={"env_request_s": 900})) == 1800.0
    assert _no_progress_timeout_s(Namespace(timeouts=Namespace(env_request_s=120))) == 300.0
