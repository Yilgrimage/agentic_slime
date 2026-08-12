from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any


def _sanitize_jsonable(value: Any, seen: set[int] | None = None) -> Any:
    if value is Ellipsis:
        return "<Ellipsis>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return "<recursive>"
    seen.add(obj_id)
    try:
        if isinstance(value, dict):
            return {_sanitize_jsonable(k, seen): _sanitize_jsonable(v, seen) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_sanitize_jsonable(item, seen) for item in value]
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: _sanitize_jsonable(getattr(value, field.name), seen) for field in fields(value)}

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return _sanitize_jsonable(model_dump(), seen)
        dict_fn = getattr(value, "dict", None)
        if callable(dict_fn):
            return _sanitize_jsonable(dict_fn(), seen)
        if hasattr(value, "__dict__"):
            return _sanitize_jsonable(vars(value), seen)
        return value
    finally:
        seen.discard(obj_id)


def _looks_like_ellipsis_encode_error(exc: BaseException) -> bool:
    return "ellipsis" in str(exc).lower()


def install_appworld_patches() -> None:
    """Install narrow AppWorld compatibility patches inside each env worker.

    AppWorld saves API-call logs after every executed code block. Model-generated
    Python can pass the literal `...` into an API call; the API may run, but
    AppWorld's request tracker then fails to JSON-encode the recorded Ellipsis.
    Patch only AppWorld's requester encoder so environment execution is not
    turned into a tool error by a logging failure.
    """

    import appworld.requester as requester

    original = getattr(requester, "jsonable_encoder", None)
    if original is None:
        raise RuntimeError("AppWorld requester has no jsonable_encoder to patch")
    if getattr(original, "_agentic_slime_appworld_patch", False):
        return

    def patched_jsonable_encoder(value: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original(value, *args, **kwargs)
        except Exception as exc:
            if not _looks_like_ellipsis_encode_error(exc):
                raise
            return original(_sanitize_jsonable(value), *args, **kwargs)

    patched_jsonable_encoder._agentic_slime_appworld_patch = True  # type: ignore[attr-defined]
    patched_jsonable_encoder._agentic_slime_original = original  # type: ignore[attr-defined]
    requester.jsonable_encoder = patched_jsonable_encoder
