import pytest

from slime.utils.types import Sample

from examples.agent_env.appworld.reward_evidence import (
    SCHEMA_VERSION,
    build_api_call_evidence,
    build_execution_evidence,
    render_execution_evidence,
)
from examples.agent_env.appworld.server import AppWorldBackend
from examples.agent_env.trace_rendering import (
    TraceCompressionOptions,
    compress_trace_text,
    render_teacher_trace_for_reward,
    render_trace_for_reward,
)


def test_appworld_official_prompt_demo_is_not_rendered_as_reward_trace() -> None:
    task = "Like all the songs played so far in my spotify music player queue, including the current one."
    official_demo = (
        "I am your supervisor, and you are an AI Assistant whose job is to complete my "
        "day-to-day tasks fully autonomously.\n"
        "Use a python REPL environment. Let's start with the task\n"
        "Task: How many playlists do I have in my Spotify playlist library?"
    )
    messages = [
        {"role": "user", "content": official_demo},
        {"role": "assistant", "content": "```python\nprint('demo')\n```"},
        {"role": "user", "content": "Output:\n```\ndemo output\n```"},
        {"role": "user", "content": f"Task: {task}"},
        {"role": "assistant", "content": "```python\nprint(apis.spotify.show_queue())\n```"},
        {"role": "user", "content": "Output:\n```\nqueue item 1\n```"},
    ]
    sample = Sample(prompt="prompt", metadata={"messages": messages, "task_prompt": task, "appworld": {"task_id": "demo"}})

    rendered = render_trace_for_reward(sample, options=TraceCompressionOptions())

    assert task in rendered
    assert "show_queue" in rendered
    assert "How many playlists do I have" not in rendered
    assert "demo output" not in rendered


def test_appworld_strip_is_idempotent_for_clean_teacher_trace() -> None:
    teacher = (
        "Initial observation:\n"
        "Task:\n"
        "Clean task\n\n"
        "Step 1:\n"
        "Tool call:\n"
        "execute({})\n"
        "Tool response:\n"
        "OK"
    )

    assert render_teacher_trace_for_reward(teacher, options=TraceCompressionOptions()) == teacher
    assert compress_trace_text(teacher, options=TraceCompressionOptions()) == teacher


def test_token_segments_take_priority_over_lossy_messages() -> None:
    messages = [
        {"role": "user", "content": "Task: update the calendar."},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Output:\n```\ncalendar updated\n```"},
    ]
    token_segments = [
        {"kind": "initial_prompt", "turn": 0, "token_count": 3, "loss_mask_sum": 0, "text": "Task: update the calendar."},
        {
            "kind": "assistant",
            "turn": 0,
            "token_count": 4,
            "loss_mask_sum": 4,
            "text": "apis.calendar.create_event({'title': 'sync'})",
        },
        {"kind": "environment", "turn": 0, "token_count": 3, "loss_mask_sum": 0, "text": "calendar updated"},
    ]
    sample = Sample(prompt="prompt", metadata={"messages": messages, "token_segments": token_segments})

    rendered = render_trace_for_reward(sample, options=TraceCompressionOptions())

    assert "Tool call:\napis.calendar.create_event" in rendered
    assert "Tool response:\ncalendar updated" in rendered


def test_appworld_token_segments_initial_prompt_uses_task_prompt_only() -> None:
    task = "Like all the songs played so far in my spotify music player queue, including the current one."
    official_demo = (
        "I am your supervisor, and you are an AI Assistant whose job is to complete my "
        "day-to-day tasks fully autonomously.\n"
        "Use a python REPL environment. Let's start with the task\n"
        "Task: How many playlists do I have in my Spotify playlist library?\n"
        f"Task: {task}"
    )
    token_segments = [
        {"kind": "initial_prompt", "turn": 0, "token_count": 100, "loss_mask_sum": 0, "text": official_demo},
        {
            "kind": "assistant",
            "turn": 0,
            "token_count": 4,
            "loss_mask_sum": 4,
            "text": "apis.spotify.show_queue()",
        },
        {"kind": "environment", "turn": 0, "token_count": 3, "loss_mask_sum": 0, "text": "queue item 1"},
    ]
    sample = Sample(prompt="prompt", metadata={"token_segments": token_segments, "task_prompt": task, "appworld": {"task_id": "demo"}})

    rendered = render_trace_for_reward(sample, options=TraceCompressionOptions())

    assert f"Task:\n{task}" in rendered
    assert "show_queue" in rendered
    assert "queue item 1" in rendered
    assert "How many playlists do I have" not in rendered
    assert "python REPL environment" not in rendered


def test_appworld_token_segments_uses_explicit_env_name() -> None:
    token_segments = [
        {
            "kind": "initial_prompt",
            "turn": 0,
            "token_count": 100,
            "loss_mask_sum": 0,
            "text": (
                "I am your supervisor, and you are an AI Assistant whose job is to complete my "
                "day-to-day tasks fully autonomously.\n"
                "Use a python REPL environment. Let's start with the task"
            ),
        }
    ]
    sample = Sample(
        prompt="prompt",
        metadata={
            "token_segments": token_segments,
            "task_prompt": "Update my calendar.",
        },
    )

    rendered = render_trace_for_reward(sample, options=TraceCompressionOptions(), env_name="appworld")

    assert "Task:\nUpdate my calendar." in rendered
    assert "python REPL environment" not in rendered


def test_appworld_token_segments_official_prompt_requires_task_prompt() -> None:
    token_segments = [
        {
            "kind": "initial_prompt",
            "turn": 0,
            "token_count": 100,
            "loss_mask_sum": 0,
            "text": (
                "I am your supervisor, and you are an AI Assistant whose job is to complete my "
                "day-to-day tasks fully autonomously.\n"
                "Use a python REPL environment. Let's start with the task"
            ),
        }
    ]
    sample = Sample(prompt="prompt", metadata={"token_segments": token_segments, "appworld": {"task_id": "demo"}})

    with pytest.raises(ValueError, match="without task_prompt metadata"):
        render_trace_for_reward(sample, options=TraceCompressionOptions())


def test_appworld_sanitizer_is_env_scoped() -> None:
    token_segments = [
        {
            "kind": "initial_prompt",
            "turn": 0,
            "token_count": 100,
            "loss_mask_sum": 0,
            "text": (
                "I am your supervisor, and you are an AI Assistant whose job is to complete my "
                "day-to-day tasks fully autonomously.\n"
                "Use a python REPL environment. Let's start with the task"
            ),
        }
    ]
    sample = Sample(prompt="prompt", metadata={"token_segments": token_segments, "webshop": {"task_id": "demo"}})

    rendered = render_trace_for_reward(sample, options=TraceCompressionOptions())

    assert "python REPL environment" in rendered


def test_appworld_structured_turn_uses_execution_evidence_instead_of_python() -> None:
    evidence = build_execution_evidence(
        observation="Execution successful.",
        api_calls=[
            build_api_call_evidence(
                app_name="spotify",
                api_name="show_queue",
                arguments={"access_token": "secret-token", "limit": 200},
                result={"items": [{"id": index, "name": "song"} for index in range(30)]},
            ),
            build_api_call_evidence(
                app_name="supervisor",
                api_name="complete_task",
                arguments={"answer": "27", "status": "success"},
                result={"message": "Execution successful."},
            ),
        ],
    )
    sample = Sample(
        prompt="prompt",
        metadata={
            "env_name": "appworld",
            "task_prompt": "Count the songs in the queue.",
            "turns": [
                {
                    "action": {
                        "type": "tool_call",
                        "name": "execute",
                        "arguments": {"code": "print(apis.spotify.show_queue(access_token='secret-token'))"},
                    },
                    "env_step": {"observation": "Execution successful.", "info": {"reward_evidence": evidence}},
                }
            ],
        },
    )

    rendered = render_trace_for_reward(
        sample,
        options=TraceCompressionOptions(strip_tool_call=256, strip_tool_response=120),
    )

    assert "AppWorld execution evidence" in rendered
    assert "spotify.show_queue" in rendered
    assert "supervisor.complete_task" in rendered
    assert '"answer":"27"' in rendered
    assert "secret-token" not in rendered
    assert "print(apis.spotify" not in rendered
    assert "Execution successful." not in rendered


def test_appworld_structured_turn_rejects_missing_execution_evidence() -> None:
    sample = Sample(
        prompt="prompt",
        metadata={
            "env_name": "appworld",
            "task_prompt": "Update the calendar.",
            "turns": [
                {
                    "action": {"type": "tool_call", "name": "execute", "arguments": {"code": "pass"}},
                    "env_step": {"observation": "Execution successful.", "info": {}},
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="missing execution evidence"):
        render_trace_for_reward(sample, options=TraceCompressionOptions(strip_tool_call=256))


def test_appworld_structured_teacher_does_not_fall_back_when_evidence_is_missing() -> None:
    teacher = {
        "env_name": "appworld",
        "task_prompt": "Update the calendar.",
        "turns": [
            {
                "action": {"type": "tool_call", "name": "execute", "arguments": {"code": "pass"}},
                "env_step": {"observation": "Execution successful.", "info": {}},
            }
        ],
    }

    with pytest.raises(ValueError, match="missing execution evidence"):
        render_teacher_trace_for_reward(
            teacher,
            options=TraceCompressionOptions(strip_tool_call=256),
            env_name="appworld",
        )


def test_appworld_evidence_budget_preserves_api_identity_and_terminal_answer() -> None:
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "code_execution": "ok",
        "output_kind": "no_stdout",
        "api_calls": [
            {
                "api": f"calendar.lookup_event_{index}",
                "status": "returned",
                "arguments": {"query": "x" * 500},
                "result_summary": {"rows": ["y" * 500]},
            }
            for index in range(8)
        ]
        + [
            {
                "api": "supervisor.complete_task",
                "status": "returned",
                "arguments": {"answer": "final exact answer", "status": "success"},
            }
        ],
    }

    rendered = render_execution_evidence(evidence, max_chars=128)

    for index in range(8):
        assert f"calendar.lookup_event_{index}" in rendered
    assert "supervisor.complete_task" in rendered
    assert "final exact answer" in rendered
    assert "configured_budget_exceeded" in rendered


def test_appworld_evidence_budget_preserves_state_changing_call_arguments() -> None:
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "code_execution": "ok",
        "output_kind": "no_stdout",
        "api_calls": [
            {
                "api": f"venmo.show_sent_payment_requests_{index}",
                "status": "returned",
                "result_summary": {"rows": ["x" * 400]},
            }
            for index in range(8)
        ]
        + [
            {
                "api": "venmo.create_payment_request",
                "status": "returned",
                "arguments": {
                    "amount": 41.0,
                    "description": "Dinner with Colleagues",
                    "user_id": 917,
                },
                "result_summary": {"message": "Payment request created.", "payment_request_id": 6097},
            }
        ],
    }

    rendered = render_execution_evidence(evidence, max_chars=128)

    assert '"state_changing_calls"' in rendered
    assert "venmo.create_payment_request" in rendered
    assert '"amount":41.0' in rendered
    assert "Dinner with Colleagues" in rendered
    assert '"user_id":917' in rendered
    assert "Payment request created." in rendered
    assert '"payment_request_id":6097' in rendered


def test_appworld_evidence_drops_only_generic_execution_confirmation() -> None:
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "code_execution": "ok",
        "output_kind": "no_stdout",
        "api_calls": [
            {
                "api": "venmo.create_payment_request",
                "status": "returned",
                "result_summary": {
                    "message": "Execution successful.",
                    "payment_request_id": 6097,
                },
            }
        ],
    }

    rendered = render_execution_evidence(evidence, max_chars=128)

    assert "Execution successful." not in rendered
    assert '"payment_request_id":6097' in rendered


def test_appworld_evidence_uses_structured_compaction_before_identity_only() -> None:
    evidence = build_execution_evidence(
        observation="Execution successful.",
        api_calls=[
            build_api_call_evidence(
                app_name="spotify",
                api_name="show_queue",
                arguments={"limit": 200, "query": "recent favorites"},
                result={"items": [{"id": index, "name": "song-" + "x" * 60} for index in range(30)]},
            )
        ],
    )

    rendered = render_execution_evidence(evidence, max_chars=512)

    assert len(evidence["api_calls"][0]["result_summary"]["items"]) == 30
    assert '"details_compacted":"structured"' in rendered
    assert '"arguments"' in rendered
    assert '"limit":200' in rendered
    assert '"result_summary"' in rendered
    assert '"item_count"' in rendered
    assert "configured_budget_exceeded" not in rendered


def test_appworld_execution_evidence_carries_safe_stdout_without_raw_tool_response() -> None:
    evidence = build_execution_evidence(
        observation='Collected 57 songs: {"title": "A Love That Never Was", "like_count": 18}',
        api_calls=[
            build_api_call_evidence(
                app_name="spotify",
                api_name="show_song",
                arguments={"song_id": 78},
                result={"id": 78, "title": "A Love That Never Was", "like_count": 18},
            )
        ],
    )
    sample = Sample(
        prompt="prompt",
        metadata={
            "env_name": "appworld",
            "task_prompt": "Find the most-liked song.",
            "turns": [
                {
                    "action": {"type": "tool_call", "name": "execute", "arguments": {"code": "..."}},
                    "env_step": {
                        "observation": 'Collected 57 songs: {"title": "A Love That Never Was", "like_count": 18}',
                        "info": {"reward_evidence": evidence},
                    },
                }
            ],
        },
    )

    rendered = render_trace_for_reward(
        sample,
        options=TraceCompressionOptions(strip_tool_call=512, strip_tool_response=True),
    )

    assert '"output_summary":"Collected 57 songs' in rendered
    assert "A Love That Never Was" in rendered
    assert "Tool response:" not in rendered


def test_appworld_execution_evidence_suppresses_sensitive_api_output() -> None:
    password_evidence = build_execution_evidence(
        observation='[{"account_name":"spotify","password":"secret-password"}]',
        api_calls=[
            build_api_call_evidence(
                app_name="supervisor",
                api_name="show_account_passwords",
                arguments={"account_name": "spotify"},
                result={"password": "secret-password"},
            )
        ],
    )
    login_evidence = build_execution_evidence(
        observation="secret-login-token",
        api_calls=[
            build_api_call_evidence(
                app_name="spotify",
                api_name="login",
                arguments={"password": "secret-password"},
                result="secret-login-token",
            )
        ],
    )

    password_rendered = render_execution_evidence(
        password_evidence,
        max_chars=512,
        observation='[{"account_name":"spotify","password":"secret-password"}]',
    )
    login_rendered = render_execution_evidence(
        login_evidence,
        max_chars=512,
        observation="secret-login-token",
    )

    assert "secret-password" not in password_rendered
    assert "secret-login-token" not in login_rendered
    assert "show_account_passwords" in password_rendered
    assert "spotify.login" in login_rendered


def test_appworld_backend_captures_evaluated_api_calls_without_changing_execution() -> None:
    class FakeRequester:
        def request(self, *args: object, **kwargs: object) -> dict[str, object]:
            assert kwargs["_app_name"] == "calendar"
            assert kwargs["_api_name"] == "create_event"
            return {"event_id": 17, "title": kwargs["title"]}

    class FakeWorld:
        def __init__(self) -> None:
            self.requester = FakeRequester()

        def execute(self, code: str) -> str:
            assert code == "create event"
            result = self.requester.request(
                _app_name="calendar",
                _api_name="create_event",
                title="Weekly sync",
                password="not-for-the-judge",
            )
            assert result == {"event_id": 17, "title": "Weekly sync"}
            return "Execution successful."

    backend = AppWorldBackend.__new__(AppWorldBackend)
    backend.world = FakeWorld()
    backend.task_id = "fake-task"
    backend._active_api_evidence = None
    backend._install_api_evidence_capture()

    observation, info = backend._execute_code("create event")

    assert observation == "Execution successful."
    evidence = info["reward_evidence"]
    assert evidence["output_kind"] == "no_stdout"
    assert evidence["api_calls"] == [
        {
            "api": "calendar.create_event",
            "status": "returned",
            "arguments": {"title": "Weekly sync", "password": "[REDACTED]"},
            "result_summary": {"event_id": 17, "title": "Weekly sync"},
        }
    ]


def test_appworld_finish_evidence_does_not_prevent_episode_completion() -> None:
    class FakeRequester:
        def request(self, *args: object, **kwargs: object) -> dict[str, str]:
            return {"message": "task submitted"}

    class FakeWorld:
        def __init__(self) -> None:
            self.requester = FakeRequester()

        def execute(self, code: str) -> str:
            assert "apis.supervisor.complete_task" in code
            self.requester.request(
                _app_name="supervisor",
                _api_name="complete_task",
                answer="done",
                status="success",
            )
            return "Execution successful."

        def task_completed(self) -> bool:
            return False

    backend = AppWorldBackend.__new__(AppWorldBackend)
    backend.world = FakeWorld()
    backend.task_id = "fake-task"
    backend.task_ids = ["fake-task"]
    backend.task_index = 0
    backend.reset_count = 1
    backend.step_count = 0
    backend.final_score = 0.0
    backend.done = False
    backend.last_info = {}
    backend._active_api_evidence = None
    backend._evaluate = lambda: {"success": True}
    backend._install_api_evidence_capture()

    result = backend.step(
        {"action": {"type": "tool_call", "name": "finish", "arguments": {"answer": "done"}}}
    )

    assert result["done"] is True
    assert result["info"]["reward_evidence"]["api_calls"][0]["api"] == "supervisor.complete_task"
