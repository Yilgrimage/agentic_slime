from __future__ import annotations


def normalize_alfworld_task_id(path: str | None) -> str | None:
    if not path:
        return None
    value = str(path).replace("\\", "/").rstrip("/")
    if value.endswith("/game.tw-pddl"):
        value = value[: -len("/game.tw-pddl")]
    for marker in ("/json_2.1.1/", "/json_2.1.2/"):
        if marker in value:
            return value.split(marker, 1)[1]
    return value or None
