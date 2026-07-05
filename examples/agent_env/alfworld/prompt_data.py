import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from examples.agent_env.alfworld.prompt import DEFAULT_PROMPT
from examples.agent_env.alfworld.task_ids import normalize_alfworld_task_id
from examples.agent_env.prompting import require_prompt


TEACHER_METADATA_KEYS = (
    "teacher_response",
    "teacher_answer",
    "teacher_final_answer",
    "teacher_actions",
    "teacher_success",
    "ropd_rubric",
    "reward_rubric",
    "rubric",
)

TEACHER_SOURCE_KEYS = {
    "source": "teacher_source",
    "source_zip": "teacher_source_zip",
    "source_member": "teacher_source_member",
}


def _row_hash(row: dict[str, Any]) -> str:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _teacher_metadata(row: dict[str, Any], task_id: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "task_id": task_id,
        "teacher_task_id": task_id,
        "teacher_row_hash": _row_hash(row),
    }
    for key in TEACHER_METADATA_KEYS:
        value = row.get(key)
        if value not in (None, "", []):
            metadata[key] = value
    for source_key, metadata_key in TEACHER_SOURCE_KEYS.items():
        value = row.get(source_key)
        if value not in (None, "", []):
            metadata[metadata_key] = value
    return metadata


def _read_task_rows(path: Path) -> list[dict[str, Any]]:
    task_rows: list[dict[str, Any]] = []
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
            task_rows.append(_teacher_metadata(row, task_id))
    return task_rows


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


def _wrapper_game_files(data_dir: Path, split: str, config_path: Path | None) -> list[str]:
    if config_path is None:
        raise ValueError("ALFWorld prompt data requires --config so task order can match ALFWorld wrapper.game_files")
    os.environ["AGENT_ENV_DATA_DIR"] = str(data_dir)
    from alfworld.agents.environment import get_environment

    from examples.agent_env.alfworld.server import _alfworld_backend_split, _load_configs

    alfworld_config, _, _ = _load_configs(str(config_path))
    env_type = alfworld_config.get("env", {}).get("type", "AlfredTWEnv")
    env_cls = get_environment(env_type)
    wrapper = env_cls(alfworld_config, train_eval=_alfworld_backend_split(split))
    game_files = [str(item) for item in list(getattr(wrapper, "game_files", None) or [])]
    if not game_files:
        raise RuntimeError(f"ALFWorld wrapper reported no game files for split={split} config={config_path}")
    return game_files


def _available_task_rows(data_dir: Path, split: str, config_path: Path | None) -> list[dict[str, Any]]:
    task_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for game_file in _wrapper_game_files(data_dir, split, config_path):
        task_id = normalize_alfworld_task_id(str(game_file))
        if task_id and task_id not in seen:
            seen.add(task_id)
            task_rows.append({"task_id": task_id, "game_file": str(game_file)})
    return task_rows


def _default_data_dir() -> Path | None:
    raw = os.environ.get("AGENT_ENV_DATA_DIR") or os.environ.get("ALFWORLD_DATA")
    return Path(os.path.expandvars(raw)).expanduser() if raw else None


def _filter_available_task_rows(task_rows: list[dict[str, Any]], data_dir: Path | None, split: str, config_path: Path | None) -> list[dict[str, Any]]:
    if data_dir is None:
        raise ValueError("ALFWorld prompt data requires AGENT_ENV_DATA_DIR, ALFWORLD_DATA, or --alfworld-data-dir")
    if not data_dir.exists():
        raise FileNotFoundError(f"ALFWorld data dir does not exist: {data_dir}")
    available_rows = _available_task_rows(data_dir, split, config_path)
    if not available_rows:
        raise RuntimeError(f"No ALFWorld game.tw-pddl files found for split={split} under {data_dir}")
    available = {str(row["task_id"]) for row in available_rows}
    selected = [task_row for task_row in task_rows if str(task_row.get("task_id") or "") in available]
    missing = len(task_rows) - len(selected)
    print(
        f"ALFWorld prompt task filter: split={split} input={len(task_rows)} "
        f"available={len(available)} kept={len(selected)} missing={missing} data_dir={data_dir}",
        file=sys.stderr,
    )
    if not selected:
        raise RuntimeError(f"No task ids from task-id file are available for split={split} under {data_dir}")
    return selected


def _local_task_rows(data_dir: Path | None, split: str, config_path: Path | None) -> list[dict[str, Any]]:
    if data_dir is None:
        raise ValueError("ALFWorld prompt data requires AGENT_ENV_DATA_DIR, ALFWORLD_DATA, or --alfworld-data-dir")
    if not data_dir.exists():
        raise FileNotFoundError(f"ALFWorld data dir does not exist: {data_dir}")
    task_rows = _available_task_rows(data_dir, split, config_path)
    if not task_rows:
        raise RuntimeError(f"No ALFWorld game.tw-pddl files found for split={split} under {data_dir}")
    return task_rows


def write_split(
    path: Path,
    split: str,
    num_tasks: int | None,
    prompt: str,
    start_task: int,
    *,
    task_rows: list[dict[str, Any]] | None = None,
) -> None:
    prompt = require_prompt(prompt, env_name="ALFWorld", source="prompt_data --prompt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if task_rows is not None:
            selected_rows = task_rows[start_task:]
            if num_tasks is not None:
                selected_rows = selected_rows[:num_tasks]
            for task_index, task_row in enumerate(selected_rows, start=start_task):
                metadata = dict(task_row)
                metadata.update(
                    {
                        "task_index": task_index,
                        "task_id": task_row["task_id"],
                        "split": split,
                    }
                )
                row = {
                    "prompt": prompt,
                    "metadata": metadata,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return

        raise ValueError("ALFWorld prompt data requires explicit task rows from local data or --task-id-file")


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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--config", help="ALFWorld env_config.yaml; required to preserve wrapper.game_files task order.")
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
    config_path = Path(os.path.expandvars(args.config)).expanduser() if args.config else None
    task_rows = _read_task_rows(Path(args.task_id_file)) if args.task_id_file else None
    if args.output:
        if task_rows is not None:
            task_rows = _filter_available_task_rows(task_rows, data_dir, args.split, config_path)
        else:
            task_rows = _local_task_rows(data_dir, args.split, config_path)
        write_split(Path(args.output), args.split, args.num_tasks, args.prompt, args.start_task, task_rows=task_rows)
        return

    output_dir = Path(args.output_dir)
    for split in args.splits:
        split_task_rows = (
            _filter_available_task_rows(task_rows, data_dir, split, config_path)
            if task_rows is not None
            else _local_task_rows(data_dir, split, config_path)
        )
        suffix = args.num_tasks if args.num_tasks is not None else len(split_task_rows or [])
        write_split(
            output_dir / f"{split}_{suffix}.jsonl",
            split,
            args.num_tasks,
            args.prompt,
            args.start_task,
            task_rows=split_task_rows,
        )


if __name__ == "__main__":
    main()
