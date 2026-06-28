import argparse
import json
import os
from pathlib import Path

import yaml


def _agent_env_data_dir() -> Path | None:
    value = os.environ.get("AGENT_ENV_DATA_DIR", "").strip()
    if value:
        return Path(os.path.expandvars(value)).expanduser()
    local_root = os.environ.get("LOCAL_RUNTIME_DIR", "").strip()
    if local_root:
        return Path(os.path.expandvars(local_root)).expanduser() / "data" / "mcp_server"
    return None


def _resolve_path(value: str, *, config_dir: Path | None = None) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    data_dir = _agent_env_data_dir()
    if data_dir is not None:
        candidates.append(data_dir / path)
    if config_dir is not None:
        candidates.append(config_dir / path)
    candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    value = json.loads(line, strict=False)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("tasks"), list):
        return [item for item in value["tasks"] if isinstance(item, dict)]
    return []


def load_tasks(config_path: Path) -> list[dict]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    mcp_cfg = cfg.get("mcp_server") or {}
    tasks = mcp_cfg.get("tasks") or cfg.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError("mcp_server.tasks must be a list")
    task_file = str(mcp_cfg.get("task_file") or cfg.get("task_file") or "").strip()
    if task_file:
        tasks.extend(_load_json_or_jsonl(_resolve_path(task_file, config_dir=config_path.parent)))
    if not tasks:
        tasks = [{"id": "default", "prompt": "Use MCP tools to solve the task."}]
    return tasks


def reward_metadata(task: dict) -> dict:
    keys = (
        "reference",
        "references",
        "ground_truth",
        "gt",
        "target",
        "expected_answer",
        "gold",
        "label",
        "expected_skills",
        "required_skills",
        "target_skills",
        "gold_skills",
        "expected_tools",
        "required_tools",
        "target_tools",
        "gold_tools",
        "domain",
        "task_set",
        "difficulty",
        "teacher_response",
        "teacher_answer",
        "teacher_final_answer",
        "student_response",
        "student_answer",
        "student_final_answer",
        "rubric",
        "reward_rubric",
        "ropd_rubric",
        "rubric_id",
    )
    return {key: task[key] for key in keys if key in task and task[key] not in (None, "", [])}


def write_split(path: Path, split: str, num_tasks: int, prompt: str, start_task: int, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for offset in range(num_tasks):
            task_index = start_task + offset
            task = tasks[task_index % len(tasks)]
            row = {
                "prompt": prompt or str(task.get("prompt") or task.get("query") or task.get("task_question") or ""),
                "metadata": {
                    "task_index": task_index,
                    "split": split,
                    "task_id": task.get("id") or task.get("task_id", task_index),
                },
            }
            row["metadata"].update(reward_metadata(task))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MCP-server prompt data for slime rollouts/eval.")
    parser.add_argument("--output", help="Write one jsonl file for --split.")
    parser.add_argument("--output-dir", help="Write one <split>_<num_tasks>.jsonl file per split.")
    parser.add_argument("--num-tasks", default="all")
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--split", default="train", help="Split used with --output.")
    parser.add_argument("--splits", nargs="+", default=("train", "eval"), help="Splits used with --output-dir.")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--config", default="examples/agent_env/mcp_server/env_config.yaml")
    args = parser.parse_args()

    if bool(args.output) == bool(args.output_dir):
        parser.error("Specify exactly one of --output or --output-dir.")

    config_path = _resolve_path(args.config)
    tasks = load_tasks(config_path)
    num_tasks = len(tasks) if str(args.num_tasks).strip().lower() == "all" else int(args.num_tasks)
    if num_tasks <= 0:
        parser.error("--num-tasks must be a positive integer or all")

    if args.output:
        write_split(Path(args.output), args.split, num_tasks, args.prompt, args.start_task, tasks)
        return

    output_dir = Path(args.output_dir)
    for split in args.splits:
        write_split(output_dir / f"{split}_{num_tasks}.jsonl", split, num_tasks, args.prompt, args.start_task, tasks)


if __name__ == "__main__":
    main()
