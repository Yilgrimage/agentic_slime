from __future__ import annotations

import os
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from examples.agent_env.rollout import cfg_path


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


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


@lru_cache(maxsize=32)
def _load_yaml(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Reward config must be a mapping: {path}")
    return raw


def reward_config(args: Any) -> dict[str, Any]:
    inline = _mapping(cfg_path(args, "reward", {}))
    config_file = inline.get("config_file") or inline.get("profile")
    if not config_file:
        return inline

    external = _load_yaml(str(resolve_path(args, config_file)))
    inline_override = {key: value for key, value in inline.items() if key not in {"config_file", "profile"}}
    return _deep_merge(external, inline_override)


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
