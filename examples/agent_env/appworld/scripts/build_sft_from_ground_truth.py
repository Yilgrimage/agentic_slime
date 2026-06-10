#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = """I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

You will interact with apps such as spotify, venmo, gmail, phone, simple_note, todoist, splitwise, amazon, and file_system by writing Python code executed in an AppWorld REPL. Each turn, generate one Python code cell. The environment will execute it and return the result, which you can use in later turns.

At every turn, output exactly one complete response in this format:
<think>
Briefly reason about the next code cell.
</think>
<code>
print(apis.api_docs.show_app_descriptions())
</code>

Key instructions:
1. Only use the existing `apis` object and Python standard library. Do not import app packages, instantiate hidden classes, or access the real OS file system.
2. Any file-system task refers to the `file_system` app, not the operating system.
3. Do not guess usernames, passwords, emails, dates, contacts, payment cards, or access tokens. Use `apis.supervisor` and app APIs to obtain them.
4. API documentation is available through `apis.api_docs`. Inspect API docs before calling unfamiliar APIs.
5. For paginated APIs, inspect all relevant pages before deciding.
6. When the task is complete, call `apis.supervisor.complete_task(...)`."""


def _solution_body(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solution":
            statements = [ast.get_source_segment(source, stmt) or "" for stmt in node.body]
            body = "\n".join(stmt.rstrip() for stmt in statements if stmt.strip()).strip()
            if body:
                return body
    raise ValueError(f"Could not find non-empty solution(apis, ...) body in {path}")


def _strip_comments(code: str) -> str:
    kept: list[str] = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _clean_comment(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^#+\s?", "", text)
    if not text:
        return ""
    return text[:1].upper() + text[1:]


def _react_blocks_from_compiled_solution(path: Path) -> list[dict[str, str]]:
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solution":
            blocks: list[dict[str, str]] = []
            cursor = node.lineno + 1
            for stmt in node.body:
                start = stmt.lineno
                end = getattr(stmt, "end_lineno", stmt.lineno)
                comments: list[str] = []
                for line_no in range(cursor, start):
                    if 1 <= line_no <= len(source_lines):
                        comment = _clean_comment(source_lines[line_no - 1])
                        if comment:
                            comments.append(comment)
                raw_code = "\n".join(source_lines[start - 1 : end])
                inline_comments: list[str] = []
                for line in raw_code.splitlines():
                    if line.lstrip().startswith("#"):
                        comment = _clean_comment(line)
                        if comment:
                            inline_comments.append(comment)
                code = _strip_comments(raw_code)
                if code:
                    blocks.append(
                        {
                            "think": " ".join(comments + inline_comments).strip()
                            or "Run the next step of the reference AppWorld solution.",
                            "code": code,
                        }
                    )
                cursor = end + 1
            if blocks:
                return blocks
    raise ValueError(f"Could not find non-empty solution(apis, ...) blocks in {path}")


def _initial_observation(task_id: str, instruction: str, required_apps: list[str] | None) -> str:
    parts = [f"Task id: {task_id}", f"Instruction:\n{instruction.strip()}"]
    if required_apps:
        parts.append("Apps likely needed: " + ", ".join(required_apps))
    parts.append(
        "Execute Python snippets against the AppWorld `apis` object. "
        "When complete, call `apis.supervisor.complete_task(...)`."
    )
    return "\n\n".join(parts)


def _messages(task_id: str, instruction: str, required_apps: list[str] | None, code: str) -> list[dict[str, str]]:
    user = f"{DEFAULT_SYSTEM_PROMPT}\n\nObservation:\n{_initial_observation(task_id, instruction, required_apps)}\nResponse:"
    assistant = (
        "<think>\n"
        "Use the reference AppWorld solution to inspect the required app data and complete the task.\n"
        "</think>\n"
        "<code>\n"
        f"{code.rstrip()}\n"
        "</code>\n"
    )
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _react_messages(
    task_id: str,
    instruction: str,
    required_apps: list[str] | None,
    blocks: list[dict[str, str]],
) -> list[dict[str, str]]:
    user = f"{DEFAULT_SYSTEM_PROMPT}\n\nObservation:\n{_initial_observation(task_id, instruction, required_apps)}\nResponse:"
    messages = [{"role": "user", "content": user}]
    for i, block in enumerate(blocks):
        assistant = (
            "<think>\n"
            f"{block['think'].strip()}\n"
            "</think>\n"
            "<code>\n"
            f"{block['code'].rstrip()}\n"
            "</code>\n"
        )
        messages.append({"role": "assistant", "content": assistant})
        if i != len(blocks) - 1:
            messages.append({"role": "user", "content": "Observation:\nExecution successful.\nResponse:"})
    return messages


def _verify_solution(task_id: str, code: str, max_interactions: int) -> dict[str, Any]:
    from appworld.environment import AppWorld
    from appworld.evaluator import evaluate_task

    experiment_name = f"sft_gt_verify_{uuid.uuid4().hex[:10]}"
    with AppWorld(task_id, experiment_name=experiment_name, max_interactions=max_interactions, raise_on_failure=False) as world:
        output = str(world.execute(code))
    tracker = evaluate_task(task_id, experiment_name=experiment_name, suppress_errors=True, save_report=False)
    success = bool(getattr(tracker, "success", False))
    pass_percentage = float(getattr(tracker, "pass_percentage", 0.0) or 0.0)
    return {
        "success": success,
        "pass_percentage": pass_percentage,
        "pass_count": int(getattr(tracker, "pass_count", 0) or 0),
        "fail_count": int(getattr(tracker, "fail_count", 0) or 0),
        "execute_output": output[:1000],
        "experiment_name": experiment_name,
    }


def _write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".jsonl":
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if output.suffix != ".parquet":
        raise ValueError("Output must end with .parquet or .jsonl")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), output)


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.appworld_root:
        os.environ["APPWORLD_ROOT"] = str(Path(args.appworld_root).expanduser())

    from appworld.task import Task, load_task_ids

    task_ids = list(load_task_ids(args.dataset_name))
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    root = Path(os.environ.get("APPWORLD_ROOT", ".")).expanduser()
    for index, task_id in enumerate(task_ids):
        task = Task.load(task_id)
        compiled_solution = root / "data" / "tasks" / task_id / "ground_truth" / "compiled_solution.py"
        try:
            code = _solution_body(compiled_solution)
            blocks = _react_blocks_from_compiled_solution(compiled_solution)
        except Exception as exc:
            skipped.append({"task_id": task_id, "reason": f"extract_failed: {exc}"})
            continue

        verify: dict[str, Any] | None = None
        if args.verify:
            try:
                verify = _verify_solution(task_id, code, args.max_interactions)
            except Exception as exc:
                skipped.append({"task_id": task_id, "reason": f"verify_failed: {exc}"})
                continue
            if args.success_only and not verify.get("success", False):
                skipped.append({"task_id": task_id, "reason": "verification_not_success", "verify": verify})
                continue

        required_apps = getattr(task.ground_truth, "required_apps", None) or []
        metadata = {
            "env": "appworld",
            "dataset_name": args.dataset_name,
            "task_id": task_id,
            "task_index": index,
            "source": "ground_truth_compiled_solution",
            "style": args.style,
            "required_apps": list(required_apps),
        }
        if verify is not None:
            metadata["verify"] = verify
        rows.append(
            {
                "messages": (
                    _react_messages(task_id, str(task.instruction), list(required_apps), blocks)
                    if args.style == "react"
                    else _messages(task_id, str(task.instruction), list(required_apps), code)
                ),
                "metadata": metadata,
            }
        )

    _write_rows(rows, Path(args.output))
    if args.skipped_output:
        Path(args.skipped_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.skipped_output).open("w", encoding="utf-8") as f:
            json.dump(skipped, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "output": args.output,
                "num_rows": len(rows),
                "num_skipped": len(skipped),
                "verified": args.verify,
                "success_only": args.success_only,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a slime SFT dataset from AppWorld ground-truth solutions.")
    parser.add_argument("--appworld-root", default=os.environ.get("APPWORLD_ROOT", ""))
    parser.add_argument("--dataset-name", default="train", choices=["train", "dev"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--style", choices=["react", "single"], default="react")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verify", action="store_true", help="Execute each solution and evaluate it before writing.")
    parser.add_argument("--success-only", action="store_true", help="Keep only tasks that pass verification.")
    parser.add_argument("--max-interactions", type=int, default=5)
    parser.add_argument("--skipped-output", default="")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
