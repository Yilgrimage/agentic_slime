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

from examples.agent_env.appworld.reward_evidence import (
    SCHEMA_VERSION,
    has_execution_evidence,
    parse_execution_evidence_trace,
)
from examples.agent_env.trace_rendering import (
    TraceCompressionOptions,
    render_teacher_trace_for_reward,
)


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


def _build_payload(
    args: argparse.Namespace,
    row_index: int,
    row: dict[str, Any],
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    metadata = _metadata(row)
    split = _row_split(row, args.split)
    task_index = _row_task_index(row_index, row)
    task_id = _row_task_id(row)
    task_key = str(metadata.get("task_key") or row.get("task_key") or f"{split}:{task_id or task_index}")
    request_id = str(
        request_id
        or metadata.get("request_id")
        or row.get("request_id")
        or f"offline-{row_index}-{uuid.uuid4().hex[:12]}"
    )
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
    for key in (
        "task_ref",
        "domain",
        "task_set",
        "data_source",
        "dataset_name",
        "query",
        "task_prompt",
        "instruction",
        "question",
        "task_question",
        "instruction_text",
    ):
        value = metadata.get(key, row.get(key))
        if value not in (None, "", []):
            payload[key] = value
    return payload


def _result_metadata(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    metadata = result.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _teacher_turns(record: dict[str, Any], *, training_view: bool) -> list[dict[str, Any]]:
    turns = _result_metadata(record).get("turns")
    if not isinstance(turns, list):
        raise ValueError("teacher rollout is missing metadata.turns for teacher materialization")
    normalized = [turn for turn in turns if isinstance(turn, dict)]
    if not training_view:
        return normalized
    return [turn for turn in normalized if _teacher_turn_is_valid(turn)]


def _teacher_turn_is_valid(turn: dict[str, Any]) -> bool:
    if turn.get("format_valid") is False:
        return False
    action = turn.get("action")
    action_name = str(action.get("name") or "").strip().lower() if isinstance(action, dict) else ""
    return action_name not in {"format_error", "invalid_format"}


def _teacher_trace_sample(record: dict[str, Any], *, training_view: bool = True) -> Sample:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    input_row = record.get("input") if isinstance(record.get("input"), dict) else {}
    input_metadata = _metadata(input_row)
    result_metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    metadata = {**input_metadata, **result_metadata}
    if training_view:
        metadata["turns"] = _teacher_turns(record, training_view=True)
    for key in ("env", "env_name", "environment", "task_id", "task_prompt", "instruction", "query"):
        value = input_row.get(key)
        if key not in metadata and value not in (None, "", []):
            metadata[key] = value
    return Sample(prompt=_row_prompt(input_row), metadata=metadata)


def _teacher_reward_trace_payload(record: dict[str, Any]) -> dict[str, Any]:
    sample = _teacher_trace_sample(record, training_view=True)
    sample_metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    payload: dict[str, Any] = {
        "env_name": str(sample_metadata.get("env_name") or sample_metadata.get("environment") or "").strip(),
        "turns": _teacher_turns(record, training_view=True),
    }
    for key in ("task_id", "task_prompt", "instruction", "query"):
        value = sample_metadata.get(key)
        if value not in (None, "", []):
            payload[key] = value
    return payload


def _raw_policy_session(record: dict[str, Any], *, training_view: bool) -> str:
    parts: list[str] = []
    for turn in _teacher_turns(record, training_view=training_view):
        assistant = turn.get("assistant_message")
        content = assistant.get("content") if isinstance(assistant, dict) else ""
        if content not in (None, "", []):
            parts.append(str(content).strip())
        env_step = turn.get("env_step")
        observation = env_step.get("observation") if isinstance(env_step, dict) else ""
        if observation not in (None, "", []):
            parts.append("Output:\n```\n" + str(observation).strip() + "\n```")
    text = "\n\n".join(part for part in parts if part).strip()
    if not text:
        view = "training" if training_view else "raw"
        raise ValueError(f"teacher rollout produced an empty {view} policy session")
    return text


def _teacher_traces(record: dict[str, Any], max_chars: int) -> tuple[str, str, dict[str, Any], str, dict[str, Any]]:
    reward_payload = _teacher_reward_trace_payload(record)
    raw_text = _raw_policy_session(record, training_view=False)
    policy_text = _raw_policy_session(record, training_view=True)
    tool_trace = render_teacher_trace_for_reward(
        reward_payload,
        options=TraceCompressionOptions(
            strip_reasoning=True,
            strip_tool_response=False,
            strip_assistant_response=True,
            strip_system_prompt=True,
        ),
        env_name=str(reward_payload.get("env_name") or ""),
        check_reasoning_presence=False,
    )
    env_name = str(reward_payload.get("env_name") or "").strip().lower()
    if env_name == "appworld":
        parse_execution_evidence_trace(tool_trace)
    raw_truncated = max_chars > 0 and len(raw_text) > max_chars
    policy_truncated = max_chars > 0 and len(policy_text) > max_chars
    tool_truncated = max_chars > 0 and len(tool_trace) > max_chars
    if raw_truncated or policy_truncated or tool_truncated:
        raise ValueError(
            "teacher trace exceeds --teacher-max-chars; refusing to truncate policy or reward evidence "
            f"(raw={len(raw_text)}, policy={len(policy_text)}, reward={len(tool_trace)}, limit={max_chars})"
        )
    all_turns = _teacher_turns(record, training_view=False)
    valid_turns = _teacher_turns(record, training_view=True)
    metadata = {
        "teacher_raw_trace_chars_original": len(raw_text),
        "teacher_policy_trace_chars_original": len(policy_text),
        "teacher_tool_trace_chars_original": len(tool_trace),
        "teacher_raw_trace_truncated": bool(raw_truncated),
        "teacher_policy_trace_truncated": bool(policy_truncated),
        "teacher_tool_trace_truncated": bool(tool_truncated),
        "teacher_turn_count_original": len(all_turns),
        "teacher_turn_count": len(valid_turns),
        "teacher_removed_format_error_turns": len(all_turns) - len(valid_turns),
    }
    if max_chars > 0:
        metadata["teacher_trace_max_chars"] = int(max_chars)
    return raw_text, policy_text, reward_payload, tool_trace, metadata


def _record_success(record: dict[str, Any]) -> bool:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    return bool(result.get("success", False))


def _teacher_row(record: dict[str, Any], max_chars: int) -> dict[str, Any]:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    input_row = record.get("input") if isinstance(record.get("input"), dict) else {}
    input_metadata = _metadata(input_row)
    result_info = result.get("info") if isinstance(result.get("info"), dict) else {}
    task_id = str(record.get("task_id") or input_metadata.get("task_id") or "")
    raw_trace, policy_trace, reward_payload, tool_trace, trace_metadata = _teacher_traces(record, max_chars)
    row: dict[str, Any] = {
        "task_id": task_id,
        "task_index": record.get("task_index"),
        "split": record.get("split"),
        "teacher_raw_trace_text": raw_trace,
        "teacher_response": policy_trace,
        "teacher_reward_trace_payload": reward_payload,
        "teacher_full_trace_text": tool_trace,
        "teacher_trace": tool_trace,
        "teacher_tool_trace": tool_trace,
        "teacher_success": _record_success(record),
        "teacher_score": result.get("score"),
        "teacher_status": result.get("status"),
        "teacher_source": "collect_env_rollouts",
        "teacher_request_id": record.get("request_id"),
        "teacher_elapsed_s": record.get("elapsed_s"),
    }
    if has_execution_evidence(tool_trace):
        row["teacher_reward_evidence_schema"] = SCHEMA_VERSION
    row.update(trace_metadata)
    for key in (
        "env",
        "domain",
        "task_set",
        "task_ref",
        "dataset_name",
        "query",
        "task_prompt",
        "instruction",
        "question",
        "task_question",
        "instruction_text",
    ):
        value = input_metadata.get(key, input_row.get(key, result_info.get(key)))
        if value not in (None, "", []):
            row[key] = value
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


def _run_one(
    args: argparse.Namespace,
    row_index: int,
    row: dict[str, Any],
    *,
    attempt_index: int = 0,
    request_id: str | None = None,
) -> dict[str, Any]:
    url = f"{args.env_server_url.rstrip('/')}/run_episode"
    payload = _build_payload(args, row_index, row, request_id=request_id)
    started = time.time()
    record: dict[str, Any] = {
        "row_index": row_index,
        "task_index": payload["task_index"],
        "task_id": payload.get("task_id", ""),
        "split": payload["split"],
        "task_key": payload["task_key"],
        "request_id": payload["request_id"],
        "attempt_index": int(attempt_index),
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


def _run_attempts(args: argparse.Namespace, row_index: int, row: dict[str, Any]) -> list[dict[str, Any]]:
    collection_id = f"offline-{row_index}-{uuid.uuid4().hex[:12]}"
    records: list[dict[str, Any]] = []
    for attempt_index in range(int(args.attempts_per_task)):
        record = _run_one(
            args,
            row_index,
            row,
            attempt_index=attempt_index,
            request_id=f"{collection_id}-attempt-{attempt_index + 1}",
        )
        records.append(record)
        if _record_success(record):
            break
    return records


def _teacher_candidate(record: dict[str, Any]) -> bool:
    if not bool(record.get("ok", False)):
        return False
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    if not isinstance(metadata.get("turns"), list) or not metadata["turns"]:
        return False
    try:
        return bool(_teacher_turns(record, training_view=True))
    except ValueError:
        return False


def _teacher_selection_rank(record: dict[str, Any]) -> tuple[bool, float, bool, int, int, float, int]:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    try:
        score = float(result.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return (
        _record_success(record),
        score,
        bool(record.get("ok", False)),
        -int(metadata.get("format_errors") or 0),
        -int(metadata.get("turn_count") or 0),
        -float(record.get("elapsed_s") or 0.0),
        -int(record.get("attempt_index") or 0),
    )


def _select_teacher_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [record for record in records if _teacher_candidate(record)]
    return max(candidates, key=_teacher_selection_rank) if candidates else None


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
    parser.add_argument(
        "--attempts-per-task",
        type=int,
        default=1,
        help="Maximum attempts per task; retry only until strict env success. Every attempt is written to output JSONL.",
    )
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
        help="Optional per-field validation limit; exceeding it fails instead of truncating. 0 keeps full traces.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.attempts_per_task < 1:
        raise ValueError("--attempts-per-task must be positive")
    if args.teacher_jsonl and not args.include_trace:
        raise ValueError("--teacher-jsonl requires --include-trace so teacher materialization is auditable")
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
    attempts = 0
    failed_attempts = 0
    teachers = 0
    started = time.time()
    with output.open(mode, encoding="utf-8") as f:
        teacher_f = teacher_output.open(mode, encoding="utf-8") if teacher_output is not None else None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                future_to_row = {executor.submit(_run_attempts, args, idx, row): idx for idx, row in rows}
                for future in concurrent.futures.as_completed(future_to_row):
                    records = future.result()
                    completed += 1
                    attempts += len(records)
                    failed_attempts += sum(int(not bool(record.get("ok", False))) for record in records)
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                    selected = _select_teacher_record(records)
                    if teacher_f is not None and selected is not None:
                        if not args.teacher_success_only or _record_success(selected):
                            teacher_row = _teacher_row(selected, args.teacher_max_chars)
                            teacher_row["teacher_attempt_count"] = len(records)
                            teacher_row["teacher_selected_attempt"] = int(selected.get("attempt_index") or 0) + 1
                            teacher_row["teacher_selection_rank"] = list(_teacher_selection_rank(selected))
                            teacher_f.write(json.dumps(teacher_row, ensure_ascii=False) + "\n")
                            teacher_f.flush()
                            teachers += 1
                    final_record = records[-1]
                    print(
                        "rollout "
                        f"{completed}/{len(rows)} ok={final_record.get('ok')} "
                        f"task={final_record.get('task_id') or final_record.get('task_index')} "
                        f"attempts={len(records)} success={_record_success(final_record)} "
                        f"elapsed={sum(float(item.get('elapsed_s') or 0.0) for item in records):.1f}s",
                        flush=True,
                    )
        finally:
            if teacher_f is not None:
                teacher_f.close()
    print(
        f"wrote {attempts} attempts for {completed} tasks to {output} failed_attempts={failed_attempts} "
        f"teacher_rows={teachers} total_elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
