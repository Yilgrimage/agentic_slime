from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from examples.agent_env.alfworld.task_ids import normalize_alfworld_task_id


def read_json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    with archive.open(name) as handle:
        value = json.loads(handle.read().decode("utf-8"))
    return value if isinstance(value, dict) else {}


def read_jsonl_member(archive: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with archive.open(name) as handle:
        for raw in handle:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def compact_teacher_trace(rows: list[dict[str, Any]], init_obs: str = "") -> str:
    parts = []
    if init_obs:
        parts.append(f"Initial observation:\n{init_obs.strip()}")
    for idx, row in enumerate(rows, start=1):
        response = str(row.get("response") or "").strip()
        action = str(row.get("action") or "").strip()
        observation = str(row.get("observation") or "").strip()
        step_parts = [f"Step {idx}:"]
        if response:
            step_parts.append(f"Teacher response:\n{response}")
        if action and f"Action: {action}" not in response:
            step_parts.append(f"Action: {action}")
        if observation:
            step_parts.append(f"Observation after action:\n{observation}")
        parts.append("\n".join(step_parts))
    return "\n\n".join(parts)


def build_rows(zip_path: Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        trajectory_names = sorted(name for name in archive.namelist() if name.endswith("/trajectory.jsonl"))
        for trajectory_name in trajectory_names:
            if archive.getinfo(trajectory_name).file_size <= 0:
                continue
            success_name = trajectory_name[: -len("trajectory.jsonl")] + "success.json"
            success = read_json_member(archive, success_name) if success_name in archive.namelist() else {}
            if success and not bool(success.get("success", False)):
                continue
            rows = read_jsonl_member(archive, trajectory_name)
            if not rows:
                continue
            info = rows[0].get("info") if isinstance(rows[0].get("info"), dict) else {}
            game_file = None
            if isinstance(info, dict):
                raw_game_file = info.get("extra.gamefile")
                if isinstance(raw_game_file, list):
                    game_file = raw_game_file[0] if raw_game_file else None
                else:
                    game_file = raw_game_file
            task_id = normalize_alfworld_task_id(str(game_file or ""))
            if not task_id:
                continue
            if task_id in seen_task_ids:
                continue
            seen_task_ids.add(task_id)
            teacher_trace = compact_teacher_trace(rows, init_obs=str(success.get("init_obs") or ""))
            row = {
                "task_id": task_id,
                "teacher_full_trace_text": teacher_trace,
                "teacher_actions": [str(item.get("action") or "") for item in rows if item.get("action")],
                "teacher_success": True,
                "source": "gpt-oss-120b_treact_success_traj",
                "source_zip": str(zip_path),
                "source_member": trajectory_name,
            }
            output.append(row)
            if max_records is not None and len(output) >= max_records:
                break
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ALFWorld ROPD teacher JSONL from GPT-OSS success trajectories.")
    parser.add_argument("--input-zip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    rows = build_rows(Path(args.input_zip), max_records=args.max_records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
