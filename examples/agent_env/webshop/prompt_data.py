import argparse
import json
import re
from pathlib import Path
from typing import Any

from examples.agent_env.prompting import require_prompt
from examples.agent_env.webshop.prompt import DEFAULT_PROMPT


def _read_task_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            row_split = str(row.get("split") or split)
            if row_split != split:
                continue
            task_index = row.get("task_index")
            task_id = row.get("task_id")
            if task_index in (None, "") and task_id not in (None, ""):
                match = re.search(r":(\d+)$", str(task_id))
                if match:
                    task_index = int(match.group(1))
            if task_index in (None, ""):
                continue
            task_index = int(task_index)
            if task_index in seen:
                continue
            seen.add(task_index)
            task_prompt = str(row.get("task_prompt") or row.get("instruction") or row.get("query") or "").strip()
            task_row = {
                "task_index": task_index,
                "task_id": str(task_id or f"webshop:{split}:{task_index}"),
            }
            if task_prompt:
                task_row["task_prompt"] = task_prompt
                task_row["instruction"] = task_prompt
            rows.append(task_row)
    if not rows:
        raise RuntimeError(f"No WebShop task rows for split={split} in {path}")
    return rows


def write_split(
    path: Path,
    split: str,
    num_tasks: int | None,
    prompt: str,
    start_task: int,
    *,
    task_rows: list[dict[str, Any]] | None = None,
) -> None:
    prompt = require_prompt(prompt, env_name="WebShop", source="prompt_data --prompt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if task_rows is None:
            if num_tasks is None:
                raise ValueError("WebShop prompt data requires --num-tasks unless --task-id-file is provided")
            task_rows = [{"task_index": task_index} for task_index in range(start_task, start_task + num_tasks)]
        else:
            task_rows = task_rows[start_task:]
            if num_tasks is not None:
                task_rows = task_rows[:num_tasks]
        for task_row in task_rows:
            task_index = int(task_row["task_index"])
            row = {
                "prompt": prompt,
                "metadata": {
                    "task_index": task_index,
                    "task_id": str(task_row.get("task_id") or f"webshop:{split}:{task_index}"),
                    "split": split,
                },
            }
            for key in ("task_prompt", "instruction"):
                value = task_row.get(key)
                if value not in (None, "", []):
                    row["metadata"][key] = value
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create WebShop prompt data for slime rollouts/eval.")
    parser.add_argument("--output", help="Write one jsonl file for --split. Kept for backward compatibility.")
    parser.add_argument("--output-dir", help="Write one <split>_<num_tasks>.jsonl file per split.")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--split", default="train", help="Split used with --output.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "valid_seen", "valid_unseen"),
        help="Splits used with --output-dir.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--task-id-file", help="JSONL teacher/prompt file containing WebShop task_id/task_index fields.")
    args = parser.parse_args()

    if bool(args.output) == bool(args.output_dir):
        parser.error("Specify exactly one of --output or --output-dir.")

    if args.output:
        task_rows = _read_task_rows(Path(args.task_id_file), args.split) if args.task_id_file else None
        write_split(Path(args.output), args.split, args.num_tasks, args.prompt, args.start_task, task_rows=task_rows)
        return

    output_dir = Path(args.output_dir)
    for split in args.splits:
        task_rows = _read_task_rows(Path(args.task_id_file), split) if args.task_id_file else None
        suffix = args.num_tasks if args.num_tasks is not None else len(task_rows or [])
        write_split(output_dir / f"{split}_{suffix}.jsonl", split, args.num_tasks, args.prompt, args.start_task, task_rows=task_rows)


if __name__ == "__main__":
    main()
