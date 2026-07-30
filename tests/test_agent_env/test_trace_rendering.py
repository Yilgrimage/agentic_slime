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
    sample = Sample(prompt="prompt", metadata={"messages": messages, "task_prompt": task})

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
