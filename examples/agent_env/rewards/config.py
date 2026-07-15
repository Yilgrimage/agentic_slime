from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from examples.agent_env.rollout import cfg_path


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _runtime_vars(args: Any) -> dict[str, str]:
    values = dict(os.environ)
    train_env_vars = getattr(args, "train_env_vars", None) or {}
    if isinstance(train_env_vars, dict):
        values.update({str(key): str(value) for key, value in train_env_vars.items() if value not in (None, "")})
    return values


def _expand_runtime_vars(args: Any, value: Any) -> str:
    text = str(value or "")
    values = _runtime_vars(args)

    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain")
        return values.get(name, match.group(0))

    return re.sub(r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)", replace, text)


def resolve_path(args: Any, value: Any) -> Path:
    text = _expand_runtime_vars(args, value).strip()
    path = Path(text).expanduser()
    if path.is_absolute():
        return path

    custom_config_path = str(getattr(args, "custom_config_path", "") or "").strip()
    if custom_config_path:
        candidate = Path(custom_config_path).expanduser().parent / path
        if candidate.exists():
            return candidate

    return Path.cwd() / path


def reward_config(args: Any) -> dict[str, Any]:
    config = _mapping(cfg_path(args, "reward", {}))
    nested_profile = config.get("config_file") or config.get("profile")
    if nested_profile:
        raise ValueError("Nested reward config_file/profile is not supported; select REWARD_PROFILE in the run profile")
    return config


def reward_cfg_path(args: Any, path: str, default: Any = None) -> Any:
    value: Any = reward_config(args)
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
        if value is None:
            return default
    return value
