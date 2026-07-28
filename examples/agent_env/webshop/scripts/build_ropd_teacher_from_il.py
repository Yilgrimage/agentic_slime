#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any


def _repo_dir() -> Path:
    return Path(__file__).resolve().parents[4]


def _root_dir() -> Path:
    raw = os.environ.get("ROOT_DIR")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return _repo_dir().parents[1]


def _default_webshop_lib() -> Path:
    for raw in (
        os.environ.get("WEBSHOP_LIB"),
        os.environ.get("WEBSHOP_ROOT"),
        f"{os.environ.get('LOCAL_RUNTIME_DIR', '/tmp/server-ops-runtime')}/code/WebShop",
        f"{_root_dir()}/code/WebShop",
    ):
        if raw:
            path = Path(os.path.expandvars(str(raw))).expanduser()
            if (path / "web_agent_site").exists():
                return path
    return _root_dir() / "code" / "WebShop"


def _normalize_instruction(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .")
    value = re.sub(r",?\s*and price lower than \d+(?:\.\d+)? dollars$", "", value)
    return value.strip(" .")


def _instruction_from_state(state: Any) -> str:
    text = str(state or "")
    match = re.search(r"Instruction:\s*(.*?)(?:\n\s*\[button\]|\Z)", text, flags=re.S)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def _limit_text(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if max_chars <= 20:
        return value[:max_chars]
    head = max(1, (max_chars - 17) // 2)
    tail = max(1, max_chars - 17 - head)
    return value[:head] + "\n...[truncated]...\n" + value[-tail:]


def _teacher_tool_trace(
    row: dict[str, Any],
    *,
    max_observation_chars: int,
    max_action_chars: int,
) -> tuple[str, dict[str, Any]]:
    states = row.get("states") if isinstance(row.get("states"), list) else []
    actions = row.get("actions") if isinstance(row.get("actions"), list) else []
    original_observation_chars = sum(len(str(state or "").strip()) for state in states)
    original_action_chars = sum(len(str(action or "").strip()) for action in actions)
    observation_truncated = max_observation_chars > 0 and any(len(str(state or "").strip()) > max_observation_chars for state in states)
    action_truncated = max_action_chars > 0 and any(len(str(action or "").strip()) > max_action_chars for action in actions)
    lines: list[str] = []
    if states:
        lines.append("Initial observation:\n" + _limit_text(states[0], max_observation_chars))
    for idx, action in enumerate(actions, start=1):
        parts = [f"Step {idx}:", "Tool call:", _limit_text(action, max_action_chars)]
        if idx < len(states):
            parts.extend(["Tool response:", _limit_text(states[idx], max_observation_chars)])
        lines.append("\n".join(parts))
    metadata = {
        "teacher_observation_chars_original": original_observation_chars,
        "teacher_action_chars_original": original_action_chars,
        "teacher_observation_truncated": bool(observation_truncated),
        "teacher_action_truncated": bool(action_truncated),
    }
    if max_observation_chars > 0:
        metadata["teacher_max_observation_chars"] = int(max_observation_chars)
    if max_action_chars > 0:
        metadata["teacher_max_action_chars"] = int(max_action_chars)
    return "\n\n".join(lines).strip(), metadata


def _read_il_rows(zip_path: Path, member: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        name = member
        if not name:
            names = [item for item in archive.namelist() if item.endswith(".jsonl")]
            if len(names) != 1:
                raise ValueError(f"{zip_path} must contain exactly one jsonl member or pass --il-member")
            name = names[0]
        with archive.open(name) as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{zip_path}:{name}:{line_no}: expected a JSON object")
                rows.append(value)
    return rows


def _best_il_by_instruction(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        instruction = _normalize_instruction(_instruction_from_state((row.get("states") or [""])[0]))
        if not instruction:
            continue
        actions = row.get("actions") if isinstance(row.get("actions"), list) else []
        success = bool(actions and str(actions[-1]).strip().lower() == "click[buy now]")
        candidate = {
            "row": row,
            "success": success,
            "action_count": len(actions),
        }
        current = best.get(instruction)
        if current is None:
            best[instruction] = candidate
            continue
        current_rank = (bool(current["success"]), -int(current["action_count"]))
        candidate_rank = (success, -len(actions))
        if candidate_rank > current_rank:
            best[instruction] = candidate
    return best


def _goal_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo_dir = _repo_dir()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    webshop_lib = Path(args.webshop_lib).expanduser()
    if str(webshop_lib) not in sys.path:
        sys.path.insert(0, str(webshop_lib))

    from examples.agent_env.webshop.server import _install_text_env_import_stubs, _load_text_env_class

    _install_text_env_import_stubs()
    WebAgentTextEnv = _load_text_env_class(str(webshop_lib))
    if args.attr_file:
        import web_agent_site.engine.engine as engine
        import web_agent_site.utils as utils

        engine.DEFAULT_ATTR_PATH = str(args.attr_file)
        utils.DEFAULT_ATTR_PATH = str(args.attr_file)
    if args.product_file:
        import web_agent_site.utils as utils

        utils.DEFAULT_FILE_PATH = str(args.product_file)
    env = WebAgentTextEnv(
        observation_mode="text",
        file_path=str(args.product_file),
        num_products=args.num_products,
        human_goals=True,
    )
    goals = list(getattr(getattr(env, "server", None), "goals", []) or [])
    if not goals:
        raise RuntimeError("WebShop env reported no goals")
    rows: list[dict[str, Any]] = []
    for task_index, goal in enumerate(goals):
        instruction = str(goal.get("instruction_text") or "")
        rows.append(
            {
                "task_index": task_index,
                "task_id": f"webshop:{args.split}:{task_index}",
                "instruction_text": instruction,
                "instruction_key": _normalize_instruction(instruction),
                "asin": goal.get("asin"),
                "goal_options": goal.get("goal_options"),
            }
        )
    if hasattr(env, "close"):
        env.close()
    return rows


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    il_rows = _read_il_rows(Path(args.il_zip), args.il_member)
    best = _best_il_by_instruction(il_rows)
    goals = _goal_rows(args)
    output: list[dict[str, Any]] = []
    missing = 0
    for goal in goals:
        il = best.get(str(goal["instruction_key"]))
        if il is None:
            missing += 1
            continue
        row = il["row"]
        actions = row.get("actions") if isinstance(row.get("actions"), list) else []
        trace, trace_metadata = _teacher_tool_trace(
            row,
            max_observation_chars=args.max_observation_chars,
            max_action_chars=args.max_action_chars,
        )
        teacher_row = {
            "task_id": goal["task_id"],
            "task_index": goal["task_index"],
            "split": args.split,
            "teacher_full_trace_text": trace,
            "teacher_tool_trace": trace,
            "teacher_success": bool(il["success"]),
            "teacher_score": 1.0 if il["success"] else 0.0,
            "teacher_actions": actions,
            "teacher_instruction": goal["instruction_text"],
            "teacher_source": "webshop_official_il_trajs_finalized_images",
            "teacher_source_zip": str(args.il_zip),
            "teacher_action_count": len(actions),
            "asin": goal.get("asin"),
            "goal_options": goal.get("goal_options"),
        }
        teacher_row.update(trace_metadata)
        output.append(teacher_row)
    summary = {
        "goals": len(goals),
        "il_rows": len(il_rows),
        "il_instruction_keys": len(best),
        "matched": len(output),
        "missing": missing,
        "coverage": len(output) / len(goals) if goals else 0.0,
        "success_rows": sum(1 for row in output if row.get("teacher_success")),
        "rows_with_observation_truncation": sum(1 for row in output if row.get("teacher_observation_truncated")),
        "rows_with_action_truncation": sum(1 for row in output if row.get("teacher_action_truncated")),
        "split": args.split,
        "output": str(args.output),
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    root_dir = _root_dir()
    data_dir = Path(os.environ.get("AGENT_ENV_DATA_DIR") or root_dir / "data" / "webshop")
    parser = argparse.ArgumentParser(description="Build WebShop ROPD teacher JSONL from official IL trajectories.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--il-zip",
        default=str(root_dir / "code" / "WebShop" / "baseline_models" / "data" / "il_trajs_finalized_images.zip"),
    )
    parser.add_argument("--il-member", default="")
    parser.add_argument("--webshop-lib", default=str(_default_webshop_lib()))
    parser.add_argument("--data-dir", default=str(data_dir))
    parser.add_argument("--product-file", default=str(data_dir / "data" / "items_shuffle.json"))
    parser.add_argument("--attr-file", default=str(data_dir / "data" / "items_ins_v2.json"))
    parser.add_argument("--num-products", type=int, default=100000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-observation-chars", type=int, default=0)
    parser.add_argument("--max-action-chars", type=int, default=0)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("il_zip", "webshop_lib", "product_file", "attr_file"):
        value = Path(getattr(args, name)).expanduser()
        if not value.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {value}")
        setattr(args, name, value)
    rows, summary = build_rows(args)
    if not rows:
        raise RuntimeError("No WebShop teacher rows matched current task order")
    if summary["coverage"] < args.min_coverage:
        raise RuntimeError(f"Teacher coverage {summary['coverage']:.3f} < --min-coverage {args.min_coverage:.3f}")
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
