from __future__ import annotations

import os
from pathlib import Path


def _official_react_prompt_path() -> Path:
    configured = os.environ.get("APPWORLD_REACT_PROMPT_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    code_root = Path(__file__).resolve().parents[4]
    return code_root / "appworld" / "experiments" / "prompts" / "react_code_agent" / "instructions.txt"


def load_official_react_prompt() -> str:
    path = _official_react_prompt_path()
    if not path.exists():
        raise FileNotFoundError(
            "AppWorld official react prompt is required but was not found at "
            f"{path}. Set APPWORLD_REACT_PROMPT_PATH or materialize the AppWorld source tree."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"AppWorld official react prompt is empty: {path}")
    return text


DEFAULT_PROMPT = load_official_react_prompt()
