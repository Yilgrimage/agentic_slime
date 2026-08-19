import asyncio
import threading
from argparse import Namespace
from unittest.mock import patch

import httpx
import pytest

from slime.utils.types import Sample

from examples.agent_env.env_episode import PolicyCallError
from examples.agent_env.episode import generate_server_episode_rollout
from examples.agent_env.fully_async_rollout import _no_progress_timeout_s
from examples.agent_env.metrics import environment_metrics
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


def test_policy_timeout_leaves_request_abort_to_sglang() -> None:
    calls: list[tuple[str, dict, float | None]] = []
    client_timeouts: list[float] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            client_timeouts.append(float(kwargs["timeout"].read))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, *, json, headers=None, timeout=None):
            calls.append((url, json, timeout))
            raise httpx.ReadTimeout("policy timed out", request=httpx.Request("POST", url))

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

    assert len(calls) == 1
    assert calls[0][0] == "http://router:30000/generate"
    assert "rid" not in calls[0][1]
    assert client_timeouts == [27.0]


def test_native_eval_retries_a_fresh_whole_episode_after_infra_failure() -> None:
    candidates: list[Sample] = []

    async def fake_episode_once(args, candidate, sampling_params, *, spec, episode_payload=None):
        candidates.append(candidate)
        candidate.metadata = {}
        if len(candidates) == 1:
            candidate.remove_sample = True
            candidate.metadata.update(
                discard_reason="server_episode_failure",
                error="PolicyCallError('timeout')",
            )
        else:
            candidate.remove_sample = False
        return candidate

    with patch(
        "examples.agent_env.episode._generate_server_episode_rollout_once",
        new=fake_episode_once,
    ):
        result = asyncio.run(
            generate_server_episode_rollout(
                Namespace(),
                Sample(),
                {},
                spec=_spec(),
                evaluation=True,
            )
        )

    assert len(candidates) == 2
    assert candidates[0] is not candidates[1]
    assert result is candidates[1]
    assert result.remove_sample is False
    assert result.metadata["eval_episode_attempts"] == 2
    assert result.metadata["eval_episode_retry_count"] == 1
    assert result.metadata["eval_episode_retry_errors"] == ["PolicyCallError('timeout')"]
    metrics = environment_metrics([result], prefix="eval/test")
    assert metrics["eval/test/eval_episode_retry_rate"] == 1.0
    assert metrics["eval/test/eval_episode_retry_count"] == 1.0


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
