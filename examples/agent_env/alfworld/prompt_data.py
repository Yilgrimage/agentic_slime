import argparse
import json
import os
import sys
from pathlib import Path

from examples.agent_env.alfworld.task_ids import normalize_alfworld_task_id


def _read_task_ids(path: Path) -> list[str]:
    task_ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            raw = row.get("task_id") or row.get("game_file") or row.get("source_member")
            task_id = normalize_alfworld_task_id(str(raw)) if raw not in (None, "", []) else None
            if not task_id or task_id in seen:
                continue
            seen.add(task_id)
            task_ids.append(task_id)
    return task_ids


def _candidate_game_roots(data_dir: Path, split: str) -> list[Path]:
    roots: list[Path] = []
    for version in ("json_2.1.1", "json_2.1.2"):
        root = data_dir / version / split
        if root.exists():
            roots.append(root)
    direct = data_dir / split
    if direct.exists():
        roots.append(direct)
    return roots


def _available_task_ids(data_dir: Path, split: str) -> set[str]:
    task_ids: set[str] = set()
    for root in _candidate_game_roots(data_dir, split):
        for game_file in root.rglob("game.tw-pddl"):
            task_id = normalize_alfworld_task_id(str(game_file))
            if task_id:
                task_ids.add(task_id)
    return task_ids


def _default_data_dir() -> Path | None:
    raw = os.environ.get("AGENT_ENV_DATA_DIR") or os.environ.get("ALFWORLD_DATA")
    return Path(os.path.expandvars(raw)).expanduser() if raw else None


def _filter_available_task_ids(task_ids: list[str], data_dir: Path | None, split: str) -> list[str]:
    if data_dir is None:
        return task_ids
    if not data_dir.exists():
        raise FileNotFoundError(f"ALFWorld data dir does not exist: {data_dir}")
    available = _available_task_ids(data_dir, split)
    if not available:
        raise RuntimeError(f"No ALFWorld game.tw-pddl files found for split={split} under {data_dir}")
    selected = [task_id for task_id in task_ids if task_id in available]
    missing = len(task_ids) - len(selected)
    print(
        f"ALFWorld prompt task filter: split={split} input={len(task_ids)} "
        f"available={len(available)} kept={len(selected)} missing={missing} data_dir={data_dir}",
        file=sys.stderr,
    )
    if not selected:
        raise RuntimeError(f"No task ids from task-id file are available for split={split} under {data_dir}")
    return selected


def write_split(
    path: Path,
    split: str,
    num_tasks: int | None,
    prompt: str,
    start_task: int,
    *,
    task_ids: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if task_ids is not None:
            selected_ids = task_ids[start_task:]
            if num_tasks is not None:
                selected_ids = selected_ids[:num_tasks]
            for task_index, task_id in enumerate(selected_ids, start=start_task):
                row = {
                    "prompt": prompt,
                    "metadata": {
                        "task_index": task_index,
                        "task_id": task_id,
                        "split": split,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return

        if num_tasks is None:
            raise ValueError("--num-tasks is required unless --task-id-file is provided")
        for task_index in range(start_task, start_task + num_tasks):
            row = {
                "prompt": prompt,
                "metadata": {
                    "task_index": task_index,
                    "split": split,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create ALFWorld prompt data for slime rollouts/eval.")
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
    parser.add_argument("--prompt", default="")
    parser.add_argument("--task-id-file", help="JSONL teacher/prompt file containing task_id-compatible fields.")
    parser.add_argument(
        "--alfworld-data-dir",
        default=None,
        help="Filter --task-id-file to tasks with local game.tw-pddl files. Defaults to AGENT_ENV_DATA_DIR when set.",
    )
    args = parser.parse_args()

    if bool(args.output) == bool(args.output_dir):
        parser.error("Specify exactly one of --output or --output-dir.")

    data_dir = Path(os.path.expandvars(args.alfworld_data_dir)).expanduser() if args.alfworld_data_dir else _default_data_dir()
    task_ids = _read_task_ids(Path(args.task_id_file)) if args.task_id_file else None
    if args.num_tasks is None and task_ids is None:
        parser.error("--num-tasks is required unless --task-id-file is provided.")

    if args.output:
        if task_ids is not None:
            task_ids = _filter_available_task_ids(task_ids, data_dir, args.split)
        write_split(Path(args.output), args.split, args.num_tasks, args.prompt, args.start_task, task_ids=task_ids)
        return

    output_dir = Path(args.output_dir)
    for split in args.splits:
        split_task_ids = _filter_available_task_ids(task_ids, data_dir, split) if task_ids is not None else None
        suffix = args.num_tasks if args.num_tasks is not None else len(split_task_ids or [])
        write_split(
            output_dir / f"{split}_{suffix}.jsonl",
            split,
            args.num_tasks,
            args.prompt,
            args.start_task,
            task_ids=split_task_ids,
        )


if __name__ == "__main__":
    main()
