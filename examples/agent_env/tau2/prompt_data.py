from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _placeholder_prompt() -> list[dict[str, str]]:
    # Real tau2 rollout prompts are built after env reset from policy + observation.
    # This placeholder keeps slime's chat-template dataset path message-shaped.
    return [{"role": "user", "content": "Start the task."}]


def _split_csv(value: str | None, default: Sequence[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_weights(value: str | None, domains: Sequence[str]) -> dict[str, float]:
    weights = {domain: 1.0 for domain in domains}
    if not value:
        return weights
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --domain-weights item {item!r}; expected domain=weight")
        domain, weight = item.split("=", 1)
        domain = domain.strip()
        if domain not in weights:
            raise ValueError(f"--domain-weights contains unknown domain {domain!r}; domains={list(domains)!r}")
        weights[domain] = float(weight)
    if any(weight < 0 for weight in weights.values()):
        raise ValueError(f"Domain weights must be non-negative: {weights}")
    if sum(weights.values()) <= 0:
        raise ValueError(f"At least one domain weight must be positive: {weights}")
    return weights


def _task_set_for(domain: str, explicit: dict[str, str]) -> str:
    return explicit.get(domain, domain)


def _parse_task_sets(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    result: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --task-sets item {item!r}; expected domain=task_set")
        domain, task_set = item.split("=", 1)
        result[domain.strip()] = task_set.strip()
    return result


def _load_tasks(domain: str, task_set: str, split: str | None, num_tasks: int | None) -> list[Any]:
    from tau2.run import get_tasks

    return list(get_tasks(task_set, task_split_name=split, num_tasks=num_tasks))


def _domain_rows(domain: str, task_set: str, split: str | None, tasks: Sequence[Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    for task_index, task in enumerate(tasks):
        task_id = getattr(task, "id", None)
        rows.append(
            {
                "prompt": _placeholder_prompt(),
                "metadata": {
                    "env": "tau2",
                    "data_source": "official",
                    "domain": domain,
                    "task_set": task_set,
                    "split": split or "all",
                    "task_index": task_index,
                    "task_id": task_id,
                    "seed": seed,
                    "task_ref": {
                        "type": "index",
                        "source": "official",
                        "domain": domain,
                        "task_set": task_set,
                        "split": split or "all",
                        "index": task_index,
                    },
                },
            }
        )
    return rows


def _weighted_take(rows_by_domain: dict[str, list[dict[str, Any]]], weights: dict[str, float], total: int, rng: random.Random) -> list[dict[str, Any]]:
    shuffled = {domain: list(rows) for domain, rows in rows_by_domain.items()}
    for rows in shuffled.values():
        rng.shuffle(rows)

    cursors = {domain: 0 for domain in shuffled}
    domains = [domain for domain, rows in shuffled.items() if rows and weights.get(domain, 0.0) > 0]
    domain_weights = [weights[domain] for domain in domains]
    result: list[dict[str, Any]] = []

    while len(result) < total and domains:
        domain = rng.choices(domains, weights=domain_weights, k=1)[0]
        rows = shuffled[domain]
        cursor = cursors[domain]
        if cursor >= len(rows):
            rng.shuffle(rows)
            cursor = 0
        row = json.loads(json.dumps(rows[cursor], ensure_ascii=False))
        row["metadata"]["epoch_cursor"] = cursor
        result.append(row)
        cursors[domain] = cursor + 1
    return result


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.source == "areal_synthetic":
        return build_areal_rows(args)

    if args.data_dir:
        os.environ["TAU2_DATA_DIR"] = str(Path(args.data_dir).expanduser())

    domains = _split_csv(args.domains, ["retail"])
    task_sets = _parse_task_sets(args.task_sets)
    weights = _parse_weights(args.domain_weights, domains)
    per_domain_limit = args.max_tasks_per_domain
    rng = random.Random(args.seed)

    rows_by_domain: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for domain in domains:
        task_set = _task_set_for(domain, task_sets)
        tasks = _load_tasks(domain, task_set, args.split, per_domain_limit)
        rows = _domain_rows(domain, task_set, args.split, tasks, args.seed)
        rows_by_domain[domain] = rows
        counts[domain] = len(rows)

    if args.num_tasks is not None:
        rows = _weighted_take(rows_by_domain, weights, args.num_tasks, rng)
    else:
        rows = [row for domain in domains for row in rows_by_domain[domain]]
        rng.shuffle(rows)

    for order, row in enumerate(rows):
        row["metadata"]["shuffle_seed"] = args.seed
        row["metadata"]["shuffle_order"] = order
        row["metadata"]["domain_task_count"] = counts.get(row["metadata"]["domain"], 0)
    return rows


def _safe_name(value: Any) -> str:
    text = str(value or "task").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "task"


def _infer_areal_domain(row: dict[str, Any]) -> str:
    scenario = row.get("user_scenario") or {}
    instructions = scenario.get("instructions") if isinstance(scenario, dict) else {}
    if isinstance(instructions, dict) and instructions.get("domain"):
        return str(instructions["domain"])
    task_id = str(row.get("id") or "")
    if "_" in task_id:
        return task_id.split("_", 1)[0]
    db_path = str(row.get("db_path") or "")
    for domain in ("airline", "retail", "telecom"):
        if domain in db_path:
            return domain
    return "unknown"


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_areal_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.areal_root:
        raise ValueError("--areal-root is required when --source areal_synthetic")
    rng = random.Random(args.seed)
    areal_root = Path(args.areal_root).expanduser().resolve()
    input_path = Path(args.areal_input).expanduser()
    if not input_path.is_absolute():
        input_path = areal_root / input_path
    task_dir = Path(args.task_file_dir).expanduser() if args.task_file_dir else areal_root / "tasks"
    if not task_dir.is_absolute():
        task_dir = areal_root / task_dir

    domains = set(_split_csv(args.domains, []))
    raw_rows = _iter_jsonl(input_path)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for source_index, raw in enumerate(raw_rows):
        domain = _infer_areal_domain(raw)
        if domains and domain not in domains:
            continue
        task_id = str(raw.get("id") or f"{domain}_{source_index}")
        task_path = task_dir / domain / f"{_safe_name(task_id)}.json"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_payload = dict(raw)
        task_payload["_data_source"] = "areal_synthetic"
        task_payload["_data_root"] = str(areal_root)
        task_path.write_text(json.dumps(task_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        rel_task_path = task_path.relative_to(areal_root)
        row = {
            "prompt": _placeholder_prompt(),
            "metadata": {
                "env": "tau2",
                "data_source": "areal_synthetic",
                "domain": domain,
                "task_set": "areal_rl",
                "split": args.split,
                "task_index": source_index,
                "task_id": task_id,
                "seed": args.seed,
                "task_ref": {
                    "type": "file",
                    "source": "areal_synthetic",
                    "root": str(areal_root),
                    "path": str(rel_task_path),
                },
            },
        }
        rows.append(row)
        counts[domain] = counts.get(domain, 0) + 1

    if args.num_tasks is not None:
        if not rows:
            raise ValueError(f"No AReaL rows selected from {input_path}")
        selected = []
        shuffled = list(rows)
        while len(selected) < args.num_tasks:
            rng.shuffle(shuffled)
            selected.extend(json.loads(json.dumps(row, ensure_ascii=False)) for row in shuffled)
        rows = selected[: args.num_tasks]
    else:
        rng.shuffle(rows)

    for order, row in enumerate(rows):
        domain = row["metadata"]["domain"]
        row["metadata"]["shuffle_seed"] = args.seed
        row["metadata"]["shuffle_order"] = order
        row["metadata"]["domain_task_count"] = counts.get(domain, 0)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tau2 prompt-data JSONL with deterministic domain mixing.")
    parser.add_argument("--source", choices=("official", "areal_synthetic"), default=os.environ.get("TAU2_DATA_SOURCE", "official"))
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--data-dir", default=os.environ.get("TAU2_DATA_DIR", ""), help="tau2 data root.")
    parser.add_argument("--areal-root", default=os.environ.get("TAU2_AREAL_ROOT", ""), help="AReaL tau2 data root for --source areal_synthetic.")
    parser.add_argument("--areal-input", default=os.environ.get("TAU2_AREAL_INPUT", "tau2_rl_train.jsonl"), help="AReaL RL JSONL relative to --areal-root.")
    parser.add_argument("--task-file-dir", default=os.environ.get("TAU2_TASK_FILE_DIR", ""), help="Directory for normalized AReaL task files.")
    parser.add_argument("--domains", default=os.environ.get("TAU2_DOMAINS", "retail"), help="Comma-separated domains, e.g. retail,airline.")
    parser.add_argument("--task-sets", default=os.environ.get("TAU2_TASK_SETS", ""), help="Optional mapping: domain=task_set,domain=task_set.")
    parser.add_argument("--domain-weights", default=os.environ.get("TAU2_DOMAIN_WEIGHTS", ""), help="Optional sampling weights: retail=2,airline=1.")
    parser.add_argument("--split", default=os.environ.get("TAU2_SPLIT", "train"), help="tau2 task split name.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")), help="Deterministic shuffle seed.")
    parser.add_argument("--num-tasks", type=int, default=None, help="Total output rows. If omitted, write each selected task once.")
    parser.add_argument("--max-tasks-per-domain", type=int, default=None, help="Cap loaded tasks per domain before mixing.")
    args = parser.parse_args()

    rows = build_rows(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    domains = {}
    for row in rows:
        domain = row["metadata"]["domain"]
        domains[domain] = domains.get(domain, 0) + 1
    print(json.dumps({"output": str(output), "rows": len(rows), "domains": domains, "seed": args.seed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
