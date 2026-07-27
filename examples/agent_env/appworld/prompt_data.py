from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from examples.agent_env.appworld.prompt import DEFAULT_PROMPT
from examples.agent_env.prompting import require_prompt


def _env_path(value: Any, envvar: str) -> str:
    text = str(value or "").strip()
    if text in {"", f"${{{envvar}}}"}:
        return os.environ.get(envvar, "")
    return os.path.expandvars(text)


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(os.path.expandvars(path)).expanduser()
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _split_dataset(config: dict[str, Any], split: str) -> str:
    appworld = config.get("appworld") if isinstance(config.get("appworld"), dict) else {}
    if split in {"eval", "validation", "val", "dev"}:
        return str(appworld.get("eval_dataset_name") or "dev")
    if split in {"test", "test_normal", "test_challenge"}:
        return "test_normal" if split == "test" else split
    return str(appworld.get("dataset_name") or split)


def _load_task_ids(config: dict[str, Any], split: str) -> list[str]:
    appworld = config.get("appworld") if isinstance(config.get("appworld"), dict) else {}
    root = _env_path(appworld.get("root") or os.environ.get("APPWORLD_ROOT", ""), "APPWORLD_ROOT")
    if root:
        os.environ["APPWORLD_ROOT"] = root
        os.environ.setdefault("HOME", root)

    from appworld.task import load_task_ids

    ids = load_task_ids(
        dataset_name=_split_dataset(config, split),
        difficulty=appworld.get("difficulty"),
        num_tasks_per_scenario=appworld.get("num_tasks_per_scenario"),
        only_tagged=appworld.get("only_tagged"),
    )
    limit = appworld.get("num_tasks")
    if limit is not None:
        ids = ids[: int(limit)]
    return [str(item) for item in ids]


def _read_task_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            row_split = str(row.get("split") or split)
            if row_split != split:
                continue
            task_id = str(row.get("task_id") or "").strip()
            task_index = row.get("task_index")
            if not task_id and task_index in (None, ""):
                continue
            key = task_id or str(task_index)
            if key in seen:
                continue
            seen.add(key)
            task_row: dict[str, Any] = {
                "task_id": task_id,
                "split": row_split,
            }
            if task_index not in (None, ""):
                task_row["task_index"] = int(task_index)
            dataset_name = row.get("dataset_name")
            if dataset_name not in (None, "", []):
                task_row["dataset_name"] = str(dataset_name)
            rows.append(task_row)
    if not rows:
        raise RuntimeError(f"No AppWorld task rows for split={split} in {path}")
    return rows


def _num_tasks(value: str, available: int) -> int:
    if value.strip().lower() == "all":
        return available
    count = int(value)
    if count < 1:
        raise ValueError("--num-tasks must be positive or all")
    return count


def write_split(path: Path, split: str, num_tasks: int, prompt: str, start_task: int, task_ids: list[str], dataset_name: str) -> None:
    prompt = require_prompt(prompt, env_name="AppWorld", source="prompt_data --prompt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for offset in range(num_tasks):
            task_index = start_task + offset
            task_id = task_ids[task_index % len(task_ids)]
            row = {
                "prompt": prompt,
                "metadata": {
                    "task_index": task_index,
                    "task_id": task_id,
                    "split": split,
                    "dataset_name": dataset_name,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_task_rows(path: Path, split: str, num_tasks: int, prompt: str, start_task: int, task_rows: list[dict[str, Any]], dataset_name: str) -> None:
    prompt = require_prompt(prompt, env_name="AppWorld", source="prompt_data --prompt")
    if start_task >= len(task_rows):
        raise ValueError(f"--start-task={start_task} is outside explicit task rows length={len(task_rows)}")
    selected = task_rows[start_task : start_task + num_tasks]
    if len(selected) < num_tasks:
        raise ValueError(f"Requested {num_tasks} tasks from explicit task rows but only {len(selected)} are available")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for offset, task_row in enumerate(selected):
            task_index = int(task_row.get("task_index", start_task + offset))
            task_id = str(task_row.get("task_id") or "")
            row = {
                "prompt": prompt,
                "metadata": {
                    "task_index": task_index,
                    "task_id": task_id,
                    "split": str(task_row.get("split") or split),
                    "dataset_name": str(task_row.get("dataset_name") or dataset_name),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AppWorld prompt data for Slime rollouts/eval.")
    parser.add_argument("--output", help="Write one jsonl file for --split.")
    parser.add_argument("--output-dir", help="Write one <split>_<num_tasks>.jsonl file per split.")
    parser.add_argument("--num-tasks", default="all")
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--split", default="train", help="Split used with --output.")
    parser.add_argument("--splits", nargs="+", default=("train", "dev"), help="Splits used with --output-dir.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--config", default="examples/agent_env/appworld/env_config.yaml")
    parser.add_argument("--task-id-file", help="JSONL teacher/prompt file containing AppWorld task_id/task_index fields.")
    args = parser.parse_args()

    if bool(args.output) == bool(args.output_dir):
        parser.error("Specify exactly one of --output or --output-dir.")

    config = _load_config(args.config)

    def build(path: Path, split: str) -> None:
        dataset_name = _split_dataset(config, split)
        if args.task_id_file:
            task_rows = _read_task_rows(Path(args.task_id_file), split)
            count = _num_tasks(str(args.num_tasks), len(task_rows))
            write_task_rows(path, split, count, args.prompt, args.start_task, task_rows, dataset_name)
            return
        task_ids = _load_task_ids(config, split)
        if not task_ids:
            raise RuntimeError(f"No AppWorld tasks found for split={split}")
        count = _num_tasks(str(args.num_tasks), len(task_ids))
        write_split(path, split, count, args.prompt, args.start_task, task_ids, dataset_name)

    if args.output:
        build(Path(args.output), args.split)
        return

    output_dir = Path(args.output_dir)
    for split in args.splits:
        dataset_name = _split_dataset(config, split)
        if args.task_id_file:
            task_rows = _read_task_rows(Path(args.task_id_file), split)
            count = _num_tasks(str(args.num_tasks), len(task_rows))
            write_task_rows(output_dir / f"{split}_{count}.jsonl", split, count, args.prompt, args.start_task, task_rows, dataset_name)
            continue
        task_ids = _load_task_ids(config, split)
        count = _num_tasks(str(args.num_tasks), len(task_ids))
        write_split(output_dir / f"{split}_{count}.jsonl", split, count, args.prompt, args.start_task, task_ids, dataset_name)


if __name__ == "__main__":
    main()
