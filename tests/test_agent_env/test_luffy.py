from argparse import Namespace

import pytest

from examples.agent_env.luffy import _segments_from_appworld_structured_turns


def _args(max_chars: int = 128) -> Namespace:
    return Namespace(reward={"luffy": {"tool_response_max_chars": max_chars}})


def test_appworld_luffy_uses_structured_action_and_compacts_observation():
    payload = {
        "env_name": "appworld",
        "turns": [
            {
                "format_valid": True,
                "assistant_message": {
                    "role": "assistant",
                    "content": "prose and a misleading ```python\nprint('wrong')\n``` block",
                },
                "action": {
                    "type": "tool_call",
                    "name": "execute",
                    "arguments": {"code": "print('right')"},
                },
                "env_step": {"observation": "head-" + ("x" * 200) + "-tail"},
            }
        ],
    }

    segments = _segments_from_appworld_structured_turns(_args(), payload)

    assert len(segments) == 2
    assert segments[0].trainable is True
    assert segments[0].text == "```python\nprint('right')\n```"
    assert "wrong" not in segments[0].text
    assert segments[1].trainable is False
    assert "head-" in segments[1].text
    assert "-tail" in segments[1].text
    assert "truncated" in segments[1].text
    assert segments[1].truncated is True
    assert segments[1].source_chars == 210


def test_appworld_luffy_rejects_invalid_structured_turn_instead_of_falling_back():
    payload = {
        "env_name": "appworld",
        "turns": [
            {
                "format_valid": False,
                "assistant_message": {"role": "assistant", "content": "```python\nprint('fallback')\n```"},
                "env_step": {"observation": "Execution successful."},
            }
        ],
    }

    with pytest.raises(ValueError, match="format-invalid"):
        _segments_from_appworld_structured_turns(_args(), payload)


@pytest.mark.parametrize("max_chars", [0, -1, True, "invalid"])
def test_appworld_luffy_requires_explicit_positive_compaction_limit(max_chars):
    payload = {
        "env_name": "appworld",
        "turns": [
            {
                "format_valid": True,
                "action": {"name": "execute", "arguments": {"code": "print('ok')"}},
                "env_step": {"observation": "ok"},
            }
        ],
    }

    with pytest.raises(ValueError, match="positive integer"):
        _segments_from_appworld_structured_turns(_args(max_chars), payload)
