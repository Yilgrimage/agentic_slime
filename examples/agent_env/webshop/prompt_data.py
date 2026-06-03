import argparse
import json
from pathlib import Path


def write_split(path: Path, split: str, num_tasks: int, prompt: str, start_task: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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
    parser = argparse.ArgumentParser(description="Create WebShop prompt data for slime rollouts/eval.")
    parser.add_argument("--output", help="Write one jsonl file for --split. Kept for backward compatibility.")
    parser.add_argument("--output-dir", help="Write one <split>_<num_tasks>.jsonl file per split.")
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--split", default="train", help="Split used with --output.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "valid_seen", "valid_unseen"),
        help="Splits used with --output-dir.",
    )
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    if bool(args.output) == bool(args.output_dir):
        parser.error("Specify exactly one of --output or --output-dir.")

    if args.output:
        write_split(Path(args.output), args.split, args.num_tasks, args.prompt, args.start_task)
        return

    output_dir = Path(args.output_dir)
    for split in args.splits:
        write_split(output_dir / f"{split}_{args.num_tasks}.jsonl", split, args.num_tasks, args.prompt, args.start_task)


if __name__ == "__main__":
    main()
