#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/teacher_traces/processed/appworld_deepseek_v4_flash_train_bestofn_teacher_full_trace.jsonl"
DEFAULT_OUTPUT = "data/teacher_traces/processed/appworld_deepseek_v4_flash_train_bestofn_teacher_clean_trace.jsonl"

TURN_HEADER_RE = re.compile(r"(?m)^Turn\s+(\d+)\s*:\s*")
FORMAT_ERROR_PATTERNS = (
    re.compile(r"(?im)^\s*Action:\s*format_error\s*\("),
    re.compile(r"(?i)\bformat_error\s*\("),
    re.compile(r"(?i)Invalid response format"),
    re.compile(r"(?i)Respond with exactly one markdown Python code block"),
)


def _resolve(path: str) -> Path:
    expanded = os.path.expandvars(path)
    value = Path(expanded).expanduser()
    if value.is_absolute():
        return value
    root = Path(os.environ.get("ROOT_DIR") or os.getcwd()).expanduser()
    return root / value


def _split_turns(text: str) -> tuple[str, list[str]]:
    matches = list(TURN_HEADER_RE.finditer(text))
    if not matches:
        return "", [text.strip()] if text.strip() else []
    prefix = text[: matches[0].start()].strip()
    turns: list[str] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        turns.append(text[match.start() : end].strip())
    return prefix, turns


def _is_format_error_turn(turn: str) -> bool:
    return any(pattern.search(turn) for pattern in FORMAT_ERROR_PATTERNS)


def _reindex_turns(prefix: str, turns: list[str]) -> str:
    rendered: list[str] = []
    if prefix:
        rendered.append(prefix)
    for idx, turn in enumerate(turns):
        rendered.append(TURN_HEADER_RE.sub(f"Turn {idx}:", turn, count=1).strip())
    return "\n\n".join(part for part in rendered if part).strip()


def clean_trace(text: Any) -> tuple[str, dict[str, int]]:
    value = str(text or "").strip()
    prefix, turns = _split_turns(value)
    kept = [turn for turn in turns if not _is_format_error_turn(turn)]
    cleaned = _reindex_turns(prefix, kept)
    return cleaned, {
        "original_turn_count": len(turns),
        "cleaned_turn_count": len(kept),
        "removed_format_error_turns": len(turns) - len(kept),
    }


def _trace_text(row: dict[str, Any]) -> str:
    for key in ("teacher_full_trace_text", "teacher_response", "teacher_trace"):
        value = row.get(key)
        if value not in (None, "", []):
            return str(value)
    return ""


def _clean_row(row: dict[str, Any], source_file: str) -> tuple[dict[str, Any], dict[str, int]]:
    raw_trace = _trace_text(row)
    if not raw_trace:
        raise ValueError(f"missing teacher trace for task_id={row.get('task_id')!r}")
    cleaned_trace, stats = clean_trace(raw_trace)
    if not cleaned_trace:
        raise ValueError(f"empty cleaned teacher trace for task_id={row.get('task_id')!r}")
    if any(pattern.search(cleaned_trace) for pattern in FORMAT_ERROR_PATTERNS):
        raise ValueError(f"format-error marker survived cleaning for task_id={row.get('task_id')!r}")

    cleaned = dict(row)
    original_format_errors = int(row.get("teacher_format_errors") or 0)
    cleaned["teacher_full_trace_text"] = cleaned_trace
    cleaned["teacher_response"] = cleaned_trace
    cleaned["teacher_format_errors_original"] = original_format_errors
    cleaned["teacher_format_errors"] = 0
    if "teacher_turn_count" in cleaned:
        cleaned["teacher_turn_count_original"] = cleaned.get("teacher_turn_count")
    cleaned["teacher_turn_count"] = stats["cleaned_turn_count"]
    cleaned["teacher_trace_cleaned"] = True
    cleaned["teacher_trace_cleaning"] = {
        "source_file": source_file,
        "removed_format_error_turns": stats["removed_format_error_turns"],
        "original_turn_count": stats["original_turn_count"],
        "cleaned_turn_count": stats["cleaned_turn_count"],
        "original_teacher_format_errors": original_format_errors,
    }
    return cleaned, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove AppWorld format-error turns from ROPD teacher traces.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default="")
    args = parser.parse_args()

    input_path = _resolve(args.input)
    output_path = _resolve(args.output)
    summary_path = _resolve(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")

    rows_written = 0
    rows_with_removed_turns = 0
    total_removed_turns = 0
    total_original_turns = 0
    total_cleaned_turns = 0
    success_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_no}: expected a JSON object")
            cleaned, stats = _clean_row(row, input_path.name)
            sink.write(json.dumps(cleaned, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1
            success_rows += int(bool(cleaned.get("teacher_success")))
            total_original_turns += stats["original_turn_count"]
            total_cleaned_turns += stats["cleaned_turn_count"]
            total_removed_turns += stats["removed_format_error_turns"]
            rows_with_removed_turns += int(stats["removed_format_error_turns"] > 0)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "rows_written": rows_written,
        "teacher_success_rows": success_rows,
        "rows_with_removed_format_error_turns": rows_with_removed_turns,
        "total_original_turns": total_original_turns,
        "total_cleaned_turns": total_cleaned_turns,
        "total_removed_format_error_turns": total_removed_turns,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
