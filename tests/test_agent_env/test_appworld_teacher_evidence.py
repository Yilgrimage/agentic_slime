from __future__ import annotations

import argparse
import json

import pytest

from examples.agent_env.appworld.reward_evidence import (
    SCHEMA_VERSION,
    build_api_call_evidence,
    build_execution_evidence,
)
from examples.agent_env.scripts.collect_env_rollouts import _select_teacher_record, _teacher_row
from examples.agent_env.scripts.validate_teacher_jsonl import validate
from examples.agent_env.rewards.ropd import _teacher_answers
from slime.utils.types import Sample


def _record() -> dict:
    evidence = build_execution_evidence(
        observation="Execution successful.",
        api_calls=[
            build_api_call_evidence(
                app_name="calendar",
                api_name="create_event",
                arguments={"title": "Weekly sync"},
                result={"event_id": 17},
            )
        ],
    )
    return {
        "task_id": "task-1",
        "task_index": 0,
        "split": "train",
        "request_id": "request-1",
        "elapsed_s": 1.5,
        "input": {
            "prompt": "prompt",
            "metadata": {
                "env_name": "appworld",
                "task_id": "task-1",
                "task_prompt": "Create a weekly sync event.",
            },
        },
        "result": {
            "status": "completed",
            "success": True,
            "score": 1.0,
            "metadata": {
                "env_name": "appworld",
                "task_id": "task-1",
                "task_prompt": "Create a weekly sync event.",
                "turns": [
                    {
                        "assistant_message": {
                            "role": "assistant",
                            "content": "```python\napis.calendar.create_event(title='Weekly sync')\n```",
                        },
                        "action": {
                            "type": "tool_call",
                            "name": "execute",
                            "arguments": {"code": "apis.calendar.create_event(title='Weekly sync')"},
                        },
                        "env_step": {
                            "observation": "Execution successful.",
                            "info": {"reward_evidence": evidence},
                        },
                    }
                ],
            },
        },
    }


def _validation_args(path: str) -> argparse.Namespace:
    return argparse.Namespace(
        teacher_jsonl=path,
        env="appworld",
        split="train",
        prompt_data="",
        expected_count=1,
        min_coverage=0.0,
        min_success_rate=0.0,
        require_success_only=True,
    )


def test_appworld_teacher_materialization_separates_policy_and_reward_traces(tmp_path) -> None:
    row = _teacher_row(_record(), max_chars=0)

    assert row["teacher_reward_evidence_schema"] == SCHEMA_VERSION
    assert row["teacher_reward_trace_payload"]["env_name"] == "appworld"
    assert len(row["teacher_reward_trace_payload"]["turns"]) == 1
    assert "apis.calendar.create_event" in row["teacher_response"]
    assert "Execution successful." in row["teacher_response"]
    assert "calendar.create_event" in row["teacher_trace"]
    assert "apis.calendar.create_event" not in row["teacher_trace"]
    assert "Execution successful." not in row["teacher_trace"]

    teacher_path = tmp_path / "teacher.jsonl"
    teacher_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary, errors = validate(_validation_args(str(teacher_path)))

    assert errors == []
    assert summary["execution_evidence_rows"] == 1
    assert summary["invalid_execution_evidence_rows"] == 0
    assert summary["missing_raw_policy_session_rows"] == 0
    assert summary["missing_training_policy_session_rows"] == 0
    assert summary["invalid_structured_payload_rows"] == 0


def test_appworld_teacher_materialization_removes_invalid_turn_only_from_training_views() -> None:
    record = _record()
    record["result"]["metadata"]["turns"].insert(
        0,
        {
            "assistant_message": {"role": "assistant", "content": "I forgot the required code block."},
            "action": {
                "type": "tool_call",
                "name": "format_error",
                "arguments": {"response": "I forgot the required code block."},
            },
            "format_valid": False,
            "env_step": {
                "observation": "Invalid response format. Respond with exactly one markdown Python code block.",
                "info": {"format_error": True},
            },
        },
    )

    row = _teacher_row(record, max_chars=0)

    assert "I forgot the required code block." in row["teacher_raw_trace_text"]
    assert "I forgot the required code block." not in row["teacher_response"]
    assert "format_error" not in row["teacher_trace"]
    assert row["teacher_turn_count_original"] == 2
    assert row["teacher_turn_count"] == 1
    assert row["teacher_removed_format_error_turns"] == 1


def test_appworld_teacher_validator_rejects_legacy_success_text(tmp_path) -> None:
    row = {
        "task_id": "task-1",
        "task_index": 0,
        "split": "train",
        "teacher_success": True,
        "teacher_score": 1.0,
        "teacher_response": "```python\npass\n```",
        "teacher_trace": (
            "Initial observation:\nTask:\nDo the task.\n\n"
            "Step 1:\nTool call:\nexecute({})\nTool response:\nExecution successful."
        ),
    }
    teacher_path = tmp_path / "legacy.jsonl"
    teacher_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary, errors = validate(_validation_args(str(teacher_path)))

    assert summary["execution_evidence_rows"] == 0
    assert summary["invalid_execution_evidence_rows"] == 1
    assert any("lack valid structured execution evidence" in error for error in errors)


def test_teacher_materialization_refuses_character_truncation() -> None:
    with pytest.raises(ValueError, match="refusing to truncate"):
        _teacher_row(_record(), max_chars=32)


def test_teacher_selection_prefers_strict_success_then_cleaner_shorter_attempt() -> None:
    partial = _record()
    partial.update({"ok": True, "attempt_index": 0})
    partial["result"].update({"success": False, "score": 0.8})
    partial["result"]["metadata"].update({"format_errors": 0, "turn_count": 8})

    successful = _record()
    successful.update({"ok": True, "attempt_index": 1})
    successful["result"].update({"success": True, "score": 1.0})
    successful["result"]["metadata"].update({"format_errors": 1, "turn_count": 12})

    assert _select_teacher_record([partial, successful]) is successful


def test_ropd_teacher_payload_is_rendered_with_runtime_appworld_compression() -> None:
    row = _teacher_row(_record(), max_chars=0)
    args = argparse.Namespace(
        env_name="appworld",
        reward={
            "ropd": {
                "answer_mode": "trace",
                "teacher_answer_keys": ["teacher_reward_trace_payload"],
                "strip_reasoning": True,
                "strip_tool_call": 512,
                "strip_tool_response": 120,
                "strip_assistant_response": True,
                "strip_system_prompt": True,
            }
        },
    )
    sample = Sample(
        prompt="prompt",
        metadata={
            "env_name": "appworld",
            "task_id": "task-1",
            "task_prompt": "Create a weekly sync event.",
            "teacher_reward_trace_payload": row["teacher_reward_trace_payload"],
        },
    )

    answers = _teacher_answers(args, sample)

    assert len(answers) == 1
    assert "AppWorld execution evidence" in answers[0]
    assert "calendar.create_event" in answers[0]
    assert "apis.calendar.create_event" not in answers[0]


def test_ropd_teacher_payload_is_loaded_from_teacher_index(tmp_path) -> None:
    row = _teacher_row(_record(), max_chars=0)
    teacher_path = tmp_path / "teacher.jsonl"
    teacher_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        env_name="appworld",
        reward={
            "ropd": {
                "answer_mode": "trace",
                "join_key": "task_id",
                "teacher_index_path": str(teacher_path),
                "teacher_index_keys": ["task_id"],
                "teacher_answer_keys": ["teacher_reward_trace_payload"],
                "strip_reasoning": True,
                "strip_tool_call": 512,
                "strip_tool_response": 120,
                "strip_assistant_response": True,
                "strip_system_prompt": True,
            }
        },
    )
    sample = Sample(
        prompt="prompt",
        metadata={
            "env_name": "appworld",
            "task_id": "task-1",
            "task_prompt": "Create a weekly sync event.",
        },
    )

    answers = _teacher_answers(args, sample)

    assert len(answers) == 1
    assert "AppWorld execution evidence" in answers[0]
    assert "calendar.create_event" in answers[0]
