import pytest

from slime.utils.types import Sample

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
