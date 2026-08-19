import asyncio
import sys
from argparse import Namespace
from types import SimpleNamespace

from examples.agent_env.rewards.llm_client import call_json_judge_with_metadata


def test_payload_validation_failure_retries_the_judge_request(monkeypatch):
    responses = [
        {"choices": [{"message": {"content": '{"valid": false}'}}]},
        {"choices": [{"message": {"content": '{"valid": true}'}}]},
    ]
    request_count = 0

    class FakeResponse:
        status = 200
        reason = "OK"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return responses.pop(0)

    class FakeSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            nonlocal request_count
            request_count += 1
            return FakeResponse()

    fake_aiohttp = SimpleNamespace(ClientTimeout=lambda **_kwargs: object(), ClientSession=FakeSession)
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    def validate(payload):
        if payload.get("valid") is not True:
            raise ValueError("invalid domain schema")

    payload, metadata = asyncio.run(
        call_json_judge_with_metadata(
            Namespace(reward={}),
            "judge this",
            base_url="http://judge.invalid/v1",
            model="judge-model",
            max_attempts=2,
            retry_backoff_s=0,
            payload_validator=validate,
        )
    )

    assert payload == {"valid": True}
    assert metadata["attempt"] == 2
    assert request_count == 2
