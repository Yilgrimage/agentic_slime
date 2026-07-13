from __future__ import annotations

from typing import Any


def require_prompt(prompt: Any, *, env_name: str, source: str = "sample.prompt") -> str:
    if isinstance(prompt, list):
        if len(prompt) != 1 or not isinstance(prompt[0], dict):
            raise ValueError(
                f"{env_name} prompt from {source} must be a non-empty string or "
                f"a single policy message; got list with {len(prompt)} item(s)."
            )
        content = prompt[0].get("content")
        if not isinstance(content, str):
            raise ValueError(
                f"{env_name} prompt from {source} message content must be a string; "
                f"got {type(content).__name__}."
            )
        prompt = content

    if not isinstance(prompt, str):
        raise ValueError(
            f"{env_name} prompt from {source} must be a non-empty string; "
            f"got {type(prompt).__name__}. Regenerate prompt data with the env prompt-data script."
        )
    text = prompt.strip()
    if not text:
        raise ValueError(
            f"{env_name} prompt from {source} is empty. Regenerate prompt data; "
            "env servers and rollouts must not substitute fallback prompts."
        )
    return text
