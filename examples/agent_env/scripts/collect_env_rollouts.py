#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from examples.agent_env.trace_rendering import TraceCompressionOptions, render_trace_for_reward


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            rows.append(item)
    return rows


def _parse_header(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --policy-header {value!r}; expected KEY=VALUE")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --policy-header {value!r}; empty key")
        headers[key] = item
    return headers


def _read_secret(path: str) -> str:
    if not path:
        return ""
    return Path(os.path.expandvars(path)).expanduser().read_text(encoding="utf-8").strip()


def _optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _row_task_index(row_index: int, row: dict[str, Any]) -> int:
    metadata = _metadata(row)
    for value in (metadata.get("task_index"), row.get("task_index"), row_index):
        if value is not None:
            return int(value)
    return row_index


def _row_split(row: dict[str, Any], default: str) -> str:
    metadata = _metadata(row)
    return str(metadata.get("split") or row.get("split") or default)


def _row_task_id(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return str(metadata.get("task_id") or row.get("task_id") or "")


def _row_prompt(row: dict[str, Any]) -> Any:
    if "prompt" not in row:
        return ""
    return row.get("prompt")


def _build_payload(args: argparse.Namespace, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(row)
    split = _row_split(row, args.split)
    task_index = _row_task_index(row_index, row)
    task_id = _row_task_id(row)
    task_key = str(metadata.get("task_key") or row.get("task_key") or f"{split}:{task_id or task_index}")
    request_id = str(metadata.get("request_id") or row.get("request_id") or f"offline-{row_index}-{uuid.uuid4().hex[:12]}")
    sampling_params: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.stop:
        sampling_params["stop"] = args.stop
    payload = {
        "split": split,
        "task_index": task_index,
        "task_key": task_key,
        "request_id": request_id,
        "release_on_done": True,
        "include_trace": bool(args.include_trace),
        "prompt": _row_prompt(row),
        "max_turns": args.max_turns,
        "max_response_tokens": args.max_response_tokens,
        "sampling_params": sampling_params,
        "timeouts": {
            "policy_s": args.policy_timeout_s,
        },
        "policy": {
            "base_url": args.policy_base_url.rstrip("/"),
            "chat_completions_path": args.policy_chat_path,
            "api_key": args.policy_api_key,
            "model": args.policy_model,
            "headers": _parse_header(args.policy_header),
        },
        "offline_rollout": {
            "row_index": row_index,
            "task_id": task_id,
            "metadata": metadata,
        },
    }
    parallel_tool_calls = _optional_bool(args.policy_parallel_tool_calls)
    if parallel_tool_calls is not None:
        payload["policy"]["parallel_tool_calls"] = parallel_tool_calls
    if task_id:
        payload["task_id"] = task_id
    for key in ("task_ref", "domain", "task_set", "data_source", "dataset_name"):
        value = metadata.get(key, row.get(key))
        if value not in (None, "", []):
            payload[key] = value
    return payload


def _limit_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _teacher_trace_sample(record: dict[str, Any]) -> Sample:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    return Sample(prompt=_row_prompt(record.get("input") if isinstance(record.get("input"), dict) else {}), metadata=metadata)


def _teacher_full_trace(record: dict[str, Any], max_chars: int) -> str:
    text = render_trace_for_reward(
        _teacher_trace_sample(record),
        options=TraceCompressionOptions(
            strip_reasoning=False,
            strip_tool_response=False,
            strip_assistant_response=False,
            strip_system_prompt=False,
        ),
    )
    return _limit_text(text, max_chars)


def _record_success(record: dict[str, Any]) -> bool:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    return bool(result.get("success", False))


def _teacher_row(record: dict[str, Any], max_chars: int) -> dict[str, Any]:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    input_row = record.get("input") if isinstance(record.get("input"), dict) else {}
    input_metadata = _metadata(input_row)
    task_id = str(record.get("task_id") or input_metadata.get("task_id") or "")
    full_trace = _teacher_full_trace(record, max_chars)
    row: dict[str, Any] = {
        "task_id": task_id,
        "task_index": record.get("task_index"),
        "split": record.get("split"),
        "teacher_full_trace_text": full_trace,
        "teacher_response": full_trace,
        "teacher_success": _record_success(record),
        "teacher_score": result.get("score"),
        "teacher_status": result.get("status"),
        "teacher_source": "collect_env_rollouts",
        "teacher_request_id": record.get("request_id"),
        "teacher_elapsed_s": record.get("elapsed_s"),
    }
    for key in ("env", "domain", "task_set", "task_ref", "dataset_name"):
        if input_metadata.get(key) not in (None, "", []):
            row[key] = input_metadata[key]
    return row


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return int(response.status), json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            parsed = {"body": body}
        return int(exc.code), parsed


def _run_one(args: argparse.Namespace, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    url = f"{args.env_server_url.rstrip('/')}/run_episode"
    payload = _build_payload(args, row_index, row)
    started = time.time()
    record: dict[str, Any] = {
        "row_index": row_index,
        "task_index": payload["task_index"],
        "task_id": payload.get("task_id", ""),
        "split": payload["split"],
        "task_key": payload["task_key"],
        "request_id": payload["request_id"],
        "input": row,
    }
    try:
        status, result = _post_json(url, payload, args.request_timeout_s)
        record.update(
            {
                "ok": 200 <= status < 300 and bool(result.get("ok", True)),
                "http_status": status,
                "elapsed_s": time.time() - started,
                "result": result,
            }
        )
    except Exception as exc:  # noqa: BLE001
        record.update(
            {
                "ok": False,
                "http_status": None,
                "elapsed_s": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return record


def _select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(rows))
    if args.only_task_id:
        wanted = set(args.only_task_id)
        indexed = [(idx, row) for idx, row in indexed if _row_task_id(row) in wanted]
    if args.only_task_index:
        wanted_idx = {int(item) for item in args.only_task_index}
        indexed = [(idx, row) for idx, row in indexed if _row_task_index(idx, row) in wanted_idx]
    if args.offset:
        indexed = indexed[args.offset :]
    if args.limit is not None:
        indexed = indexed[: args.limit]
    return indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect offline agent-env rollouts by calling an existing env server/router "
            "/run_episode endpoint with an OpenAI-compatible policy endpoint."
        )
    )
    parser.add_argument("--env-server-url", required=True, help="Env server or router base URL.")
    parser.add_argument("--prompt-data", required=True, help="JSONL rows with prompt and metadata.")
    parser.add_argument("--output-jsonl", required=True, help="Path for rollout result JSONL.")
    parser.add_argument("--policy-base-url", required=True, help="OpenAI-compatible policy base URL.")
    parser.add_argument("--policy-chat-path", default="/v1/chat/completions")
    parser.add_argument("--policy-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--policy-api-key-path", default=os.environ.get("OPENAI_API_KEY_PATH", ""))
    parser.add_argument("--policy-model", default=os.environ.get("OPENAI_MODEL", "agent-env-policy"))
    parser.add_argument("--policy-header", action="append", default=[], help="Extra policy header as KEY=VALUE.")
    parser.add_argument("--policy-parallel-tool-calls", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only-task-id", action="append", default=[])
    parser.add_argument("--only-task-index", action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--policy-timeout-s", type=float, default=120.0)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--teacher-jsonl",
        default="",
        help="Optional compact teacher JSONL path for ROPD/SFT prompt-data merging.",
    )
    parser.add_argument(
        "--teacher-success-only",
        action="store_true",
        help="Only write successful episodes to --teacher-jsonl.",
    )
    parser.add_argument(
        "--teacher-max-chars",
        type=int,
        default=0,
        help="Per-field teacher text truncation limit. 0 keeps full trace text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if not args.policy_api_key and args.policy_api_key_path:
        args.policy_api_key = _read_secret(args.policy_api_key_path)
    rows = _select_rows(_jsonl_rows(Path(args.prompt_data)), args)
    if not rows:
        raise RuntimeError("No prompt rows selected.")

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    teacher_output = Path(args.teacher_jsonl) if args.teacher_jsonl else None
    if teacher_output is not None:
        teacher_output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    completed = 0
    failed = 0
    teachers = 0
    started = time.time()
    with output.open(mode, encoding="utf-8") as f:
        teacher_f = teacher_output.open(mode, encoding="utf-8") if teacher_output is not None else None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                future_to_row = {executor.submit(_run_one, args, idx, row): idx for idx, row in rows}
                for future in concurrent.futures.as_completed(future_to_row):
                    record = future.result()
                    completed += 1
                    failed += int(not bool(record.get("ok", False)))
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    if teacher_f is not None and (not args.teacher_success_only or _record_success(record)):
                        teacher_f.write(json.dumps(_teacher_row(record, args.teacher_max_chars), ensure_ascii=False) + "\n")
                        teacher_f.flush()
                        teachers += 1
                    print(
                        "rollout "
                        f"{completed}/{len(rows)} ok={record.get('ok')} "
                        f"task={record.get('task_id') or record.get('task_index')} "
                        f"elapsed={float(record.get('elapsed_s') or 0.0):.1f}s",
                        flush=True,
                    )
        finally:
            if teacher_f is not None:
                teacher_f.close()
    print(
        f"wrote {completed} rows to {output} failed={failed} "
        f"teacher_rows={teachers} total_elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
