#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


TRACE_KEYS = (
    "teacher_tool_trace",
    "teacher_trace",
    "teacher_full_trace_text",
    "reference_trace",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    metadata = _metadata(row)
    for source in (row, metadata):
        for key in keys:
            value = source.get(key)
            if value not in (None, "", []):
                return value
    return None


def _task_id(row: dict[str, Any]) -> str:
    value = _first(row, ("task_id", "id"))
    return str(value).strip() if value not in (None, "") else ""


def _task_index(row: dict[str, Any]) -> str:
    value = _first(row, ("task_index", "prompt_index", "row_index"))
    if value in (None, ""):
        match = re.search(r":(\d+)$", _task_id(row))
        if match:
            return match.group(1)
        return ""
    return str(int(value)) if isinstance(value, (int, float)) else str(value).strip()


def _split(row: dict[str, Any], default: str) -> str:
    value = _first(row, ("split", "task_set", "dataset_split"))
    return str(value or default).strip()


def _trace_text(row: dict[str, Any]) -> str:
    value = _first(row, TRACE_KEYS)
    if value in (None, "", []):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _success(row: dict[str, Any]) -> bool:
    value = _first(row, ("teacher_success", "success", "completed"))
    return bool(value)


def _score(row: dict[str, Any]) -> float | None:
    value = _first(row, ("teacher_score", "score", "env_score"))
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_instruction(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .")


def _price(text: Any) -> float | None:
    match = re.search(r"price lower than\s+(\d+(?:\.\d+)?)\s+dollars", str(text or "").lower())
    return float(match.group(1)) if match else None


def _trace_instruction(trace: str) -> str:
    text = str(trace or "")
    match = re.search(r"(?:Task|Instruction):\s*\[SEP\]\s*(.*?)\s*\[SEP\]", text, flags=re.S)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    if "[SEP]" in text:
        sep_parts = [part.strip() for part in re.split(r"\s*\[SEP\]\s*", text) if part.strip()]
        for idx, part in enumerate(sep_parts):
            lowered = part.lower()
            if lowered in {"instruction", "instruction:"} and idx + 1 < len(sep_parts):
                return re.sub(r"\s+", " ", sep_parts[idx + 1]).strip()
            if lowered.startswith("instruction:"):
                value = part.split(":", 1)[1].strip()
                if value:
                    return re.sub(r"\s+", " ", value).strip()
                if idx + 1 < len(sep_parts):
                    return re.sub(r"\s+", " ", sep_parts[idx + 1]).strip()
    match = re.search(r"Instruction:\s*\n?\s*(.*?)(?:\n\[button\]|\Z)", trace, flags=re.S)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _prompt_rows(path: Path, default_split: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        metadata = _metadata(row)
        normalized.append(
            {
                "task_id": str(metadata.get("task_id") or row.get("task_id") or ""),
                "task_index": str(metadata.get("task_index") or row.get("task_index") or idx),
                "split": str(metadata.get("split") or row.get("split") or default_split),
            }
        )
    return normalized


def _coverage(rows: list[dict[str, Any]], prompt_rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    teacher_task_ids = {_task_id(row) for row in rows if _task_id(row)}
    teacher_task_indices = {_task_index(row) for row in rows if _split(row, split) == split and _task_index(row)}
    prompt = [row for row in prompt_rows if str(row.get("split") or split) == split]
    covered = 0
    missing_examples: list[dict[str, str]] = []
    for row in prompt:
        task_id = str(row.get("task_id") or "")
        task_index = str(row.get("task_index") or "")
        hit = bool((task_id and task_id in teacher_task_ids) or (task_index and task_index in teacher_task_indices))
        covered += int(hit)
        if not hit and len(missing_examples) < 10:
            missing_examples.append({"task_id": task_id, "task_index": task_index})
    return {
        "prompt_rows": len(prompt),
        "covered_prompt_rows": covered,
        "coverage": covered / len(prompt) if prompt else 0.0,
        "missing_examples": missing_examples,
    }


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    rows = _read_jsonl(Path(args.teacher_jsonl))
    errors: list[str] = []
    task_ids = [_task_id(row) for row in rows if _task_id(row)]
    task_indices = [_task_index(row) for row in rows if _task_index(row)]
    traces = [_trace_text(row) for row in rows]
    success_rows = sum(1 for row in rows if _success(row))
    scored_rows = sum(1 for row in rows if _score(row) is not None)
    instruction_mismatches = 0
    price_mismatches = 0
    for row, trace in zip(rows, traces, strict=True):
        expected_instruction = str(
            _first(row, ("teacher_instruction", "instruction", "instruction_text", "query", "task_prompt")) or ""
        ).strip()
        observed_instruction = _trace_instruction(trace)
        if expected_instruction and observed_instruction:
            if _normalize_instruction(expected_instruction) != _normalize_instruction(observed_instruction):
                instruction_mismatches += 1
            if _price(expected_instruction) != _price(observed_instruction):
                price_mismatches += 1

    summary: dict[str, Any] = {
        "teacher_jsonl": str(args.teacher_jsonl),
        "env": args.env,
        "rows": len(rows),
        "unique_task_ids": len(set(task_ids)),
        "unique_task_indices": len(set(task_indices)),
        "duplicate_task_ids": len(task_ids) - len(set(task_ids)),
        "duplicate_task_indices": len(task_indices) - len(set(task_indices)),
        "missing_task_identity_rows": sum(1 for row in rows if not _task_id(row) and not _task_index(row)),
        "missing_trace_rows": sum(1 for value in traces if not value.strip()),
        "success_rows": success_rows,
        "success_rate": success_rows / len(rows) if rows else 0.0,
        "scored_rows": scored_rows,
        "trace_chars_min": min((len(value) for value in traces), default=0),
        "trace_chars_max": max((len(value) for value in traces), default=0),
        "instruction_mismatch_rows": instruction_mismatches,
        "price_mismatch_rows": price_mismatches,
    }
    if args.prompt_data:
        summary.update(_coverage(rows, _prompt_rows(Path(args.prompt_data), args.split), args.split))
    elif args.expected_count > 0:
        summary["coverage"] = len(rows) / args.expected_count
        summary["expected_count"] = args.expected_count

    if not rows:
        errors.append("teacher JSONL is empty")
    if summary["missing_task_identity_rows"]:
        errors.append(f"{summary['missing_task_identity_rows']} rows are missing task_id/task_index")
    if summary["missing_trace_rows"]:
        errors.append(f"{summary['missing_trace_rows']} rows are missing teacher trace text")
    if summary["duplicate_task_ids"]:
        errors.append(f"{summary['duplicate_task_ids']} duplicate task_id entries")
    if args.env == "webshop" and summary["price_mismatch_rows"]:
        errors.append(f"{summary['price_mismatch_rows']} WebShop rows have price/instruction mismatch")
    if args.require_success_only and success_rows != len(rows):
        errors.append(f"{len(rows) - success_rows} rows are not successful")
    if args.min_success_rate and summary["success_rate"] < args.min_success_rate:
        errors.append(f"success_rate {summary['success_rate']:.4f} < {args.min_success_rate:.4f}")
    if args.min_coverage and float(summary.get("coverage", 0.0)) < args.min_coverage:
        errors.append(f"coverage {float(summary.get('coverage', 0.0)):.4f} < {args.min_coverage:.4f}")
    return summary, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate agent-env ROPD teacher JSONL files.")
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--env", default="generic", choices=("generic", "webshop", "alfworld", "appworld", "tau2"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-data", default="", help="Optional prompt-data JSONL for exact coverage checks.")
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    parser.add_argument("--require-success-only", action="store_true")
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, errors = validate(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
