from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from .config import resolve_path, reward_cfg_path
from . import naive
from .extractors import (
    bool_value,
    float_value,
    int_value,
    metadata,
    prediction_text,
    reference_values,
    runtime_env,
    task_prompt,
    truncate,
)
from .llm_client import call_json_judge_with_metadata, judge_mode
from .types import RewardResult

RUBRIC_SCHEMA_VERSION = "ropd.rubric.v1"
BATCH_VERIFIER_SCHEMA_VERSION = "ropd.batch_verifier.v2"

RUBRIC_SYSTEM_PROMPT = "你是一名教育评估与共享评分细则设计专家。只返回 JSON 对象本身。"
JUDGE_SYSTEM_PROMPT = "你是一名答案评分专家。只返回 JSON 对象本身。"

RUBRICATOR_PROMPT_TEMPLATE = """Rubricator 角色提示词

PROMPT:
你是一名教育评估与共享评分细则设计专家。你的任务是结合一道题目、多份参考回答和多份学生回答进行分析：
- reference responses：高置信度的参考答案，但不保证绝对正确，也不应被视为题目所有有效优点、关键点或合理解法的完整清单
- student responses：需要在同一套共享 rubric 下被评估的回答，它们可能暴露出有价值的答案质量差距

你的目标是生成一份共享 rubric：既要保持对题目本身的一般性评估有效性，又要利用参考回答提供的高质量信号作为蒸馏参照，使同一套 rubric 能稳定评估全部回答，而不是把任何一份参考回答当成不可质疑的金标准。这份 rubric 应当优先奖励那些具有教学价值、并且能在全部回答上形成稳定区分信号的答案层面优点。

# 输入数据
[Question]: {question}
[Reference Responses]: {teacher_response}
[Student Responses]: {student_response}
[Additional Instructions]: {extra_rubric_instructions}

# 核心目标
生成一份 rubric，帮助模型围绕具有教学价值的答案层面质量信号去评估所有回答。但必须满足以下要求：
- 不要奖励对参考回答措辞、表面风格或具体方法的照搬。
- 不要假定任何参考回答一定完全正确。
- 不要定义必须匹配某份参考回答最终答案的标准。
- 不要定义只有在验证阶段直接与参考回答对比时才能判断的标准。

每一条 criterion 都必须能够仅凭单个 response 独立评估。后续 verifier 会在同一份 rubric 下，分别对每一份回答单独打分。

# 硬性要求
每条 criterion 都必须满足：
1. 具体、自包含且可衡量：criterion 本身应明确写出 verifier 需要检查的具体内容，使人无需额外推断就能理解它在评什么；避免使用“回答正确”“解释充分”“表达清楚”“推理合理”这类未说明检查对象的空泛表述。
2. 可二元判定：verifier 应能仅基于单个 response 将其判为 True 或 False。
3. 具有教学价值：它应当指出一个有意义的改进方向。
4. 对替代方法安全：如果回答采用了不同但同样合理的方法，只要体现出相同优点，也应被奖励。
5. 尽可能具有区分度：优先选择那些更强的参考回答或学生回答明显具备、而更弱回答明显缺失的优点，但这些优点仍须保持为一般性的答案质量标准。
6. 基于外显回答可评估：优先选择那些能够基于回答中可直接观察到的内容来评估答案质量的标准。

# 必须参考的三类内容场景
请在 `category` 字段中填写最匹配该 criterion 的内容场景名称：
1. Task Fulfillment and Requirement Compliance Scenario
2. Observable Response Quality Scenario
3. General Reasoning Quality Scenario

# 分值锚点与权重分配
不要把分值在各条 criterion 之间平均分配。请用 points 制造清晰区分：
- 5 分：决定性瓶颈。
- 4 分：强区分瓶颈。
- 2 分：支撑性优点。
- 1 分：低风险常规要求。
- 3 分：罕见的中间权重，应谨慎使用。

纯表层格式类 criterion 最高只能给 1 分；除非题目本身确实存在多个同等决定性的失败模式，否则不要平均铺分。

# 禁止出现的 criterion 模式
不要写出如下类型的 criterion：
- “使用与某份参考回答相同的方法”
- “与某份参考回答的最终答案一致”
- “与某份参考回答在措辞/风格/结构上相同”
- 编码了某份参考回答里可能错误的中间结论
- 只适用于某一个非常具体的步骤编号或偶然措辞
- 主要奖励篇幅长、展开多或表面文风表现的标准

# 输出格式
返回一个 JSON 对象，结构如下：
```json
{
  "schema_version": "ropd.rubric.v1",
  "rubrics": [
    {
      "criterion_id": "c1",
      "category": "Task Fulfillment and Requirement Compliance Scenario",
      "criterion": "XXX",
      "points": 5
    }
  ],
  "maximum_score": 5
}
```

# 输出约束
- `schema_version` 必须严格等于 `ropd.rubric.v1`。
- 生成 4 到 12 条 criterion。
- `criterion_id` 必须唯一，格式为 `c1`、`c2`、`c3`……
- `points` 必须是 1 到 5 的整数。
- `maximum_score` 必须等于所有 rubric points 之和。
- 不应把参考回答视为天然完美。
- 每条 criterion 都必须可以只基于单个 response 判断。

只返回 JSON 对象本身，不要附加任何解释性文字。
"""

VERIFIER_PROMPT_TEMPLATE = """Prompt of the verifier role

PROMPT:
你是一名答案评分专家。你的任务是针对同一道题，在给定共享 rubric 的情况下，一次性评价多个回答。

[题目]
{question}

[Rubrics]
{rubrics}

[回答]
{answers}

[额外评分要求]
{extra_scoring_instructions}

# 核心评分规则
对每个回答，按 rubric 中 criterion 的顺序逐条判断：

- 如果该回答明确满足某条 criterion，则对应 judgement 为 `true`。
- 如果该回答没有明确满足某条 criterion，则对应 judgement 为 `false`。
- 如果一条 criterion 包含多个明确条件，只有当所有条件都满足时，才能判为 `true`。
- 如果回答采用了不同但有效的方法，只要满足 criterion 描述的要求，也应判为 `true`。
- 不要引入 rubric 和题目之外的额外评分标准。
- 不要因为回答更长、更自信、措辞更像标准答案或风格更好就判为 `true`，除非 criterion 明确要求这些属性。
- 不要比较不同回答之间谁更好。
- 每个回答都必须独立评分。

# 批量评分要求
请保持回答的输入顺序。

对每个回答输出：
- `answer_index`：回答的 1-based 索引，从 1 开始。
- `judgement`：布尔列表，长度必须等于 rubric criterion 数量，顺序必须与 rubric 完全一致。
- `final_score`：该回答所有 judgement 为 `true` 的 criterion points 之和。

criterion 内部没有部分分。每条 criterion 只能是 true 或 false。

# 输出格式
请返回一个 JSON object，结构如下：

```json
{
  "schema_version": "ropd.batch_verifier.v2",
  "answers": [
    {
      "answer_index": 1,
      "judgement": [true, false, true],
      "final_score": 7
    }
  ]
}
```

# 输出约束
- `schema_version` 必须严格等于 `ropd.batch_verifier.v2`。
- `answers` 数组必须包含每个输入回答各一项。
- `answer_index` 必须从 1 开始，并按输入顺序覆盖所有回答。
- 每个回答的 `judgement` 长度必须等于 rubric criterion 数量。
- `final_score` 必须等于该回答所有 `true` criterion 对应 points 的总和。
- 最终只输出 JSON object，不要输出解释、Markdown 或其他文本。
"""

_DUMP_COUNTS: dict[str, int] = {}
_STAGE_SEMAPHORES: dict[tuple[int, str, int], asyncio.Semaphore] = {}


def _cfg(args: Any, name: str, default: Any = None) -> Any:
    return reward_cfg_path(args, f"ropd.{name}", default)


def _env_cfg(args: Any, env_name: str, cfg_name: str, default: Any = None) -> Any:
    value = runtime_env(args, env_name, "").strip()
    if value != "":
        return value
    return _cfg(args, cfg_name, default)


def _cfg_bool(args: Any, env_name: str, cfg_name: str, default: bool) -> bool:
    return bool_value(_env_cfg(args, env_name, cfg_name, default), default)


def _cfg_float(args: Any, env_name: str, cfg_name: str, default: float) -> float:
    return float_value(_env_cfg(args, env_name, cfg_name, default), default)


def _cfg_int(args: Any, env_name: str, cfg_name: str, default: int) -> int:
    return int_value(_env_cfg(args, env_name, cfg_name, default), default)


def _cfg_choice(args: Any, env_name: str, cfg_name: str, default: str, choices: set[str]) -> str:
    value = str(_env_cfg(args, env_name, cfg_name, default) or default).strip().lower()
    return value if value in choices else default


def _list_value(value: Any, default: tuple[str, ...] = ()) -> list[str]:
    if value in (None, "", []):
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _metadata_value(sample: Sample, keys: list[str]) -> Any:
    sample_metadata = metadata(sample)
    for key in keys:
        value = sample_metadata.get(key)
        if value not in (None, "", []):
            return value
    return None


def _dump_limit(args: Any) -> int:
    raw = runtime_env(args, "AGENT_ENV_ROPD_DUMP_N", "").strip()
    if raw == "":
        raw = str(_cfg(args, "dump_n", "") or "").strip()
    return max(0, int_value(raw, 0))


def _dump_dir(args: Any) -> Path | None:
    if _dump_limit(args) <= 0:
        return None
    raw = runtime_env(args, "AGENT_ENV_ROPD_DUMP_DIR", "").strip()
    if raw == "":
        raw = str(_cfg(args, "dump_dir", "") or "").strip()
    if raw:
        return resolve_path(args, raw)
    run_root = runtime_env(args, "RUN_ROOT", "").strip()
    if not run_root:
        return None
    return Path(run_root) / "reward_artifacts" / "ropd"


def _dump_artifact(args: Any, stage: str, record: dict[str, Any]) -> None:
    output_dir = _dump_dir(args)
    if output_dir is None:
        return
    limit = _dump_limit(args)
    key = f"{stage}:{os.getpid()}"
    count = _DUMP_COUNTS.get(key, 0)
    if count >= limit:
        return
    _DUMP_COUNTS[key] = count + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stage}_pid{os.getpid()}.jsonl"
    payload = {
        "schema_version": "agent_env.ropd_artifact.v1",
        "stage": stage,
        "pid": os.getpid(),
        "time": time.time(),
        "index": count,
        **record,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _role_api_key_path(args: Any, role: str) -> str | None:
    upper = role.upper()
    path = runtime_env(args, f"AUX_{upper}_API_KEY_PATH", "").strip()
    return str(resolve_path(args, path)) if path else None


def _role_endpoint(args: Any, role: str) -> dict[str, str]:
    upper = role.upper()
    values: dict[str, str] = {}
    for key in ("provider", "base_url", "model"):
        key_upper = key.upper()
        value = runtime_env(args, f"AUX_{upper}_{key_upper}", "").strip()
        if value:
            values[key] = value
    return values


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _base_concurrency(args: Any) -> int:
    raw = runtime_env(args, "AGENT_ENV_ROPD_CONCURRENCY", "").strip()
    if raw == "":
        raw = runtime_env(args, "AGENT_ENV_REWARD_CONCURRENCY", "").strip()
    if raw == "":
        raw = _cfg(args, "concurrency", reward_cfg_path(args, "concurrency", 8))
    return _positive_int(raw, 8)


def _stage_concurrency(args: Any, stage: str) -> int:
    base = _base_concurrency(args)
    stage = stage.lower()
    env_keys = {
        "rubric": ("AGENT_ENV_ROPD_RUBRIC_CONCURRENCY",),
        "judge": ("AGENT_ENV_ROPD_JUDGE_CONCURRENCY",),
    }.get(stage, ())
    raw: Any = ""
    for key in env_keys:
        raw = runtime_env(args, key, "").strip()
        if raw != "":
            break
    if raw == "":
        raw = _cfg(args, f"{stage}_concurrency", 0)
    return _positive_int(raw, base)


def _stage_semaphore(args: Any, stage: str) -> asyncio.Semaphore:
    limit = _stage_concurrency(args, stage)
    loop_id = id(asyncio.get_running_loop())
    key = (loop_id, stage, limit)
    semaphore = _STAGE_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        _STAGE_SEMAPHORES[key] = semaphore
    return semaphore


async def _with_stage_limit(args: Any, stage: str, func: Any, *func_args: Any, **func_kwargs: Any) -> Any:
    async with _stage_semaphore(args, stage):
        return await func(*func_args, **func_kwargs)


def _cache_key(sample: Sample) -> str:
    sample_metadata = metadata(sample)
    raw = {
        "task_id": sample_metadata.get("task_id"),
        "prompt": task_prompt(sample),
        "references": reference_values(sample),
    }
    text = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _join_key(args: Any) -> str:
    return str(_cfg(args, "join_key", "task_id") or "task_id")


def _join_value(args: Any, sample: Sample) -> str:
    sample_metadata = metadata(sample)
    key = _join_key(args)
    for candidate in (key, "task_id", "id", "task_index"):
        value = sample_metadata.get(candidate)
        if value not in (None, "", []):
            return str(value)
    return _cache_key(sample)


def _configured_metadata_value(args: Any, sample: Sample, cfg_name: str, default_keys: tuple[str, ...]) -> Any:
    keys = _list_value(_cfg(args, cfg_name, None), default_keys)
    return _metadata_value(sample, keys)


def _render_answer_block(label: str, answers: str | list[str] | tuple[str, ...], *, start_index: int = 0, force_labels: bool = False) -> str:
    if isinstance(answers, str):
        normalized = (answers,)
    else:
        normalized = tuple(str(answer) for answer in answers if str(answer).strip())
    if not normalized:
        return ""
    if len(normalized) == 1 and not force_labels:
        return normalized[0]
    return "\n\n".join(
        f"{label} {idx}:\n{answer}" for idx, answer in enumerate(normalized, start=start_index)
    )


def _limit_text(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0:
        return value
    return truncate(value, max_chars)


def _trim_answer_for_judge(answer: Any) -> str:
    answer_text = str(answer or "")
    if "</think>" not in answer_text:
        return answer_text
    trimmed = answer_text.rsplit("</think>", 1)[1].strip()
    return trimmed or answer_text


def _sanitize_teacher_answer_for_anonymous_verifier(answer: Any) -> str:
    text = _trim_answer_for_judge(answer)
    replacements = (
        (r"\bTeacher response\b", "Response"),
        (r"\bTeacher action\b", "Action"),
        (r"\bTeacher actions\b", "Actions"),
        (r"\bteacher response\b", "response"),
        (r"\bteacher action\b", "action"),
        (r"\bteacher actions\b", "actions"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _strip_tool_response_text(text: str) -> str:
    if not text:
        return ""
    lines: list[str] = []
    skip_block = False
    for line in text.splitlines():
        normalized = line.strip().lower()
        if normalized.startswith("[observation") or normalized.startswith("observation after action"):
            skip_block = True
            continue
        if skip_block and (normalized.startswith("[assistant") or normalized.startswith("[action") or normalized.startswith("step ")):
            skip_block = False
        if not skip_block:
            lines.append(line)
    return "\n".join(lines).strip()


def _trace_text(sample: Sample, *, omit_tool_responses: bool = False) -> str:
    sample_metadata = metadata(sample)
    turns = sample_metadata.get("turns")
    if isinstance(turns, list) and turns:
        lines: list[str] = []
        for idx, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                lines.append(f"Step {idx}:\n{_message_text(turn)}")
                continue
            response_text = turn.get("parser_text") or turn.get("response_text") or ""
            action = turn.get("action")
            observation = turn.get("observation")
            parts = [f"Step {idx}:"]
            if response_text:
                parts.append(f"Response:\n{_message_text(response_text)}")
            if action not in (None, "", []):
                parts.append(f"Action:\n{_message_text(action)}")
            if not omit_tool_responses and observation not in (None, "", []):
                parts.append(f"Observation after action:\n{_message_text(observation)}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)
    trace_value = _metadata_value(sample, ["trace", "trajectory", "rollout_trace", "student_trace"])
    trace_text = _message_text(trace_value) if trace_value not in (None, "", []) else str(getattr(sample, "response", "") or "")
    return _strip_tool_response_text(trace_text) if omit_tool_responses else trace_text


def _combine_final_and_trace(prediction: str, trace: str, *, trace_label: str) -> str:
    prediction = prediction.strip()
    trace = trace.strip()
    if prediction and trace and prediction != trace:
        return f"[Final Answer]\n{prediction}\n\n[{trace_label}]\n{trace}"
    return prediction or trace


def _answer_mode(args: Any) -> str:
    return str(_env_cfg(args, "AGENT_ENV_ROPD_ANSWER_MODE", "answer_mode", "full") or "full").strip().lower()


def _answer_for_judge(args: Any, sample: Sample) -> str:
    prediction = _trim_answer_for_judge(prediction_text(sample))
    mode = _answer_mode(args)
    if mode == "final":
        return prediction
    if mode in {"trace", "full_trace"}:
        return _trim_answer_for_judge(_trace_text(sample))
    if mode in {
        "final_without_tool_response",
        "final_without_tool_responses",
        "final_with_no_tool_trace",
        "final_with_no_tool_response_trace",
        "final_with_trace_no_tool_response",
    }:
        trace = _trace_text(sample, omit_tool_responses=True)
        return _trim_answer_for_judge(_combine_final_and_trace(prediction, trace, trace_label="Rollout Trace Without Tool Responses"))
    if mode in {"trace_without_tool_response", "trace_without_tool_responses", "no_tool_trace", "no_tool_response_trace", "trace_no_tool_response"}:
        return _trim_answer_for_judge(_trace_text(sample, omit_tool_responses=True) or prediction)
    return _trim_answer_for_judge(_combine_final_and_trace(prediction, _trace_text(sample), trace_label="Rollout Trace"))


def _student_answer(args: Any, sample: Sample) -> str:
    value = _metadata_value(
        sample,
        _list_value(
            _cfg(args, "student_answer_keys", None),
            ("student_response", "student_answer", "student_final_answer"),
        ),
    )
    if value not in (None, "", []):
        return _trim_answer_for_judge(value)
    return _answer_for_judge(args, sample)


def _teacher_answers(args: Any, sample: Sample) -> tuple[str, ...]:
    keys = _list_value(
        _cfg(args, "teacher_answer_keys", None),
        ("teacher_response", "teacher_answer", "teacher_final_answer"),
    )
    values: list[str] = []
    metadata_value = _metadata_value(sample, keys)
    if metadata_value not in (None, "", []):
        if isinstance(metadata_value, (list, tuple)):
            values.extend(str(item) for item in metadata_value if str(item).strip())
        else:
            values.append(str(metadata_value))
    if values:
        return tuple(
            dict.fromkeys(_sanitize_teacher_answer_for_anonymous_verifier(value) for value in values if value.strip())
        )
    return ()


def _extra_rubric_instructions(args: Any) -> str:
    return str(_cfg(args, "extra_rubric_instructions", "") or "")


def _extra_scoring_instructions(args: Any) -> str:
    return str(_cfg(args, "extra_scoring_instructions", "") or "")


def _rubric_items(rubric: Any) -> list[dict[str, Any]]:
    if isinstance(rubric, dict):
        items = rubric.get("rubrics")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        legacy_items = rubric.get("rubric")
        if isinstance(legacy_items, list):
            return [
                {
                    "criterion_id": f"c{idx}",
                    "category": "Task Fulfillment and Requirement Compliance Scenario",
                    "criterion": str(item),
                    "points": 1,
                }
                for idx, item in enumerate(legacy_items, start=1)
            ]
    if isinstance(rubric, list):
        return [
            item
            if isinstance(item, dict)
            else {
                "criterion_id": f"c{idx}",
                "category": "Task Fulfillment and Requirement Compliance Scenario",
                "criterion": str(item),
                "points": 1,
            }
            for idx, item in enumerate(rubric, start=1)
        ]
    return []


def _normalize_rubric(payload: Any) -> dict[str, Any] | None:
    items = _rubric_items(payload)
    normalized_items: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        criterion = str(item.get("criterion") or item.get("description") or "").strip()
        if not criterion:
            continue
        try:
            points = int(item.get("points", 1))
        except (TypeError, ValueError):
            points = 1
        points = max(1, min(5, points))
        normalized_items.append(
            {
                "criterion_id": str(item.get("criterion_id") or f"c{idx}").strip() or f"c{idx}",
                "category": str(item.get("category") or "Task Fulfillment and Requirement Compliance Scenario"),
                "criterion": criterion,
                "points": points,
            }
        )
    if not normalized_items:
        return None
    maximum_score = sum(int(item["points"]) for item in normalized_items)
    if isinstance(payload, dict):
        try:
            supplied_maximum = int(payload.get("maximum_score", maximum_score))
        except (TypeError, ValueError):
            supplied_maximum = maximum_score
        if supplied_maximum == maximum_score:
            maximum_score = supplied_maximum
    return {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubrics": normalized_items,
        "maximum_score": maximum_score,
    }


def _maximum_score(rubric: Any) -> float:
    normalized = _normalize_rubric(rubric)
    if normalized is None:
        return 0.0
    return float(normalized["maximum_score"])


def _rubric_hash(rubric: Any) -> str:
    text = json.dumps(rubric, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _build_rubricator_prompt(args: Any, samples: list[Sample], teacher_answers: tuple[str, ...]) -> str:
    question_max_chars = _cfg_int(args, "AGENT_ENV_ROPD_QUESTION_MAX_CHARS", "question_max_chars", 8000)
    student_max_chars = _cfg_int(args, "AGENT_ENV_ROPD_STUDENT_RUBRIC_MAX_CHARS", "student_rubric_max_chars", 6000)
    reference_max_chars = _cfg_int(args, "AGENT_ENV_ROPD_REFERENCE_MAX_CHARS", "reference_max_chars", 0)
    return _render_template(
        RUBRICATOR_PROMPT_TEMPLATE,
        {
            "question": _limit_text(task_prompt(samples[0]), question_max_chars),
            "teacher_response": _render_answer_block(
                "Reference",
                [_limit_text(answer, reference_max_chars) for answer in teacher_answers],
            ),
            "student_response": _render_answer_block(
                "Student",
                [_limit_text(_student_answer(args, sample), student_max_chars) for sample in samples],
                start_index=0,
                force_labels=True,
            ),
            "extra_rubric_instructions": _extra_rubric_instructions(args),
        },
    )


def _build_verifier_prompt(
    args: Any,
    sample: Sample,
    *,
    rubric: dict[str, Any],
    answers: tuple[str, ...],
) -> str:
    question_max_chars = _cfg_int(args, "AGENT_ENV_ROPD_QUESTION_MAX_CHARS", "question_max_chars", 8000)
    answer_max_chars = _cfg_int(args, "AGENT_ENV_ROPD_VERIFIER_ANSWER_MAX_CHARS", "verifier_answer_max_chars", 6000)
    return _render_template(
        VERIFIER_PROMPT_TEMPLATE,
        {
            "question": _limit_text(task_prompt(sample), question_max_chars),
            "rubrics": json.dumps(rubric["rubrics"], ensure_ascii=False, indent=2),
            "answers": _render_answer_block(
                "Answer",
                [_limit_text(answer, answer_max_chars) for answer in answers],
                start_index=1,
                force_labels=True,
            ),
            "extra_scoring_instructions": _extra_scoring_instructions(args),
        },
    )


def _answer_shuffle_key(bucket_key: str, source: str, source_index: int, text: str) -> tuple[str, str, int]:
    digest = hashlib.sha256(f"{bucket_key}\x1f{source}\x1f{source_index}\x1f{text}".encode("utf-8")).hexdigest()
    return digest, source, source_index


def _anonymous_answer_items(
    *,
    args: Any,
    bucket_key: str,
    teacher_answers: tuple[str, ...],
    student_answers: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    del args
    items = [
        {"source": "teacher", "source_index": idx, "text": answer}
        for idx, answer in enumerate(teacher_answers)
    ]
    items.extend([
        {"source": "student", "source_index": idx, "text": answer}
        for idx, answer in enumerate(student_answers)
    ])
    return tuple(
        sorted(
            items,
            key=lambda item: _answer_shuffle_key(
                bucket_key,
                str(item["source"]),
                int(item["source_index"]),
                str(item["text"]),
            ),
        )
    )


def _parse_batch_scores(payload: Any, *, rubric: dict[str, Any], expected: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("ROPD verifier response must be a JSON object")
    if payload.get("schema_version") != BATCH_VERIFIER_SCHEMA_VERSION:
        raise ValueError("ROPD verifier schema_version mismatch")
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != expected:
        raise ValueError(f"ROPD verifier returned {0 if not isinstance(answers, list) else len(answers)} scores for {expected} answers")
    rubric_items = _rubric_items(rubric)
    scores: list[dict[str, Any]] = []
    for expected_index, item in enumerate(answers, start=1):
        if not isinstance(item, dict):
            raise ValueError("ROPD verifier answer item must be an object")
        if int(item.get("answer_index", -1)) != expected_index:
            raise ValueError("ROPD verifier answer_index must cover 1..n in order")
        judgement = item.get("judgement")
        if not isinstance(judgement, list) or len(judgement) != len(rubric_items):
            raise ValueError("ROPD verifier judgement length does not match rubric length")
        bool_judgement = []
        for value in judgement:
            if isinstance(value, str):
                bool_judgement.append(value.strip().lower() in {"1", "true", "yes"})
            else:
                bool_judgement.append(bool(value))
        final_score = float(
            sum(
                int(criterion.get("points", 1))
                for criterion, passed in zip(rubric_items, bool_judgement, strict=True)
                if passed
            )
        )
        scores.append(
            {
                "answer_index": expected_index,
                "judgement": bool_judgement,
                "final_score": final_score,
            }
        )
    return scores


def _existing_rubric(args: Any, sample: Sample) -> Any:
    return _configured_metadata_value(
        args,
        sample,
        "rubric_keys",
        ("rubric", "reward_rubric", "ropd_rubric"),
    )


def _weight(args: Any) -> float:
    raw = runtime_env(args, "AGENT_ENV_ROPD_TASK_SUCCESS_WEIGHT", "")
    if raw == "":
        raw = _cfg(args, "task_success_weight", reward_cfg_path(args, "outcome", 10.0))
    return float_value(raw, 10.0)


def _clip(args: Any, score: float) -> float:
    lo = float_value(_cfg(args, "clip_min", -3.0), -3.0)
    hi = float_value(_cfg(args, "clip_max", 3.0), 3.0)
    return max(lo, min(hi, float(score)))


def _sample_std(values: list[float], mean: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return max(variance, 0.0) ** 0.5


def _score_list_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"values": [], "mean": 0.0, "std": 0.0, "count": 0}
    mean = sum(values) / len(values)
    return {
        "values": values,
        "mean": mean,
        "std": _sample_std(values, mean),
        "count": len(values),
    }


def _luffy_mode(args: Any) -> str:
    return _cfg_choice(args, "AGENT_ENV_ROPD_LUFFY_MODE", "luffy_mode", "off", {"off", "reward_anchor", "token_loss"})


def _luffy_enabled(args: Any) -> bool:
    return _cfg_bool(args, "AGENT_ENV_ROPD_LUFFY_ENABLE", "luffy_enable", False)


def _reward_group_reference(args: Any) -> str:
    default_reference = "teacher_plus_students" if _luffy_enabled(args) and _luffy_mode(args) == "reward_anchor" else "students"
    return _cfg_choice(
        args,
        "AGENT_ENV_ROPD_REWARD_GROUP_REFERENCE",
        "reward_group_reference",
        default_reference,
        {"students", "teacher_plus_students"},
    )


def _reward_mode(args: Any) -> str:
    return _cfg_choice(
        args,
        "AGENT_ENV_ROPD_REWARD_MODE",
        "reward_mode",
        "answer_only",
        {"answer_only", "group_centered", "group_zscore"},
    )


def _validate_reward_config(args: Any) -> None:
    if _luffy_enabled(args) and _luffy_mode(args) == "token_loss":
        raise RuntimeError(
            "AGENT_ENV_ROPD_LUFFY_MODE=token_loss requires actor-side off-policy teacher-token loss integration. "
            "The agentic Slime ROPD reward module can only provide scalar rewards."
        )


def _group_stats(args: Any, student_scores: list[float], teacher_scores: tuple[float, ...]) -> dict[str, Any]:
    reference_values_for_stats = list(student_scores)
    reference = _reward_group_reference(args)
    if reference == "teacher_plus_students":
        reference_values_for_stats = [float(score) for score in teacher_scores] + reference_values_for_stats
    stats = _score_list_stats(reference_values_for_stats)
    stats.update(
        {
            "student_scores": list(student_scores),
            "teacher_scores": [float(score) for score in teacher_scores],
            "reference": reference,
        }
    )
    return stats


def _select_train_score(
    args: Any,
    *,
    answer_score: float,
    group_stats: dict[str, Any],
) -> tuple[float, str]:
    reward_mode = _reward_mode(args)
    if _luffy_enabled(args) and _luffy_mode(args) == "reward_anchor" and reward_mode == "answer_only":
        reward_mode = "group_centered"
    if reward_mode == "group_centered":
        return _clip(args, answer_score - float(group_stats.get("mean", 0.0))), reward_mode
    if reward_mode == "group_zscore":
        std = float(group_stats.get("std", 0.0))
        if std <= 0:
            return 0.0, reward_mode
        return _clip(args, (answer_score - float(group_stats.get("mean", 0.0))) / (std + 1e-6)), reward_mode
    return _clip(args, answer_score), "answer_only"


async def _rubric_for_bucket(
    args: Any,
    samples: list[Sample],
) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None, tuple[str, ...]]:
    sample = samples[0]
    teacher_answers = _teacher_answers(args, sample)
    if not teacher_answers:
        return None, "missing_teacher", None, ()

    existing = _existing_rubric(args, sample)
    existing_rubric = _normalize_rubric(existing)
    if existing_rubric is not None:
        return existing_rubric, "teacher", None, teacher_answers

    allow_online_raw = runtime_env(args, "AGENT_ENV_ROPD_ALLOW_ONLINE_RUBRIC", "")
    if allow_online_raw == "":
        allow_online_raw = _cfg(args, "allow_online_rubric", False)
    allow_online = str(allow_online_raw).strip().lower() in {"1", "true", "yes", "on"}
    if not allow_online or judge_mode(args) != "aux":
        return None, "missing", None, teacher_answers

    prompt = _build_rubricator_prompt(args, samples, teacher_answers)
    try:
        payload, call_metadata = await call_json_judge_with_metadata(
            args,
            prompt,
            system_prompt=RUBRIC_SYSTEM_PROMPT,
            api_key_path=_role_api_key_path(args, "rubric"),
            **_role_endpoint(args, "rubric"),
        )
        call_metadata = {
            **call_metadata,
            "role": "rubric",
            "concurrency_limit": _stage_concurrency(args, "rubric"),
        }
    except Exception as exc:
        _dump_artifact(
            args,
            "rubricator",
            {
                "join_key": _join_key(args),
                "join_value": _join_value(args, sample),
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "prompt": prompt,
                "teacher_answer_count": len(teacher_answers),
                "student_answer_count": len(samples),
            },
        )
        return None, "rubric_error", None, teacher_answers
    rubric = _normalize_rubric(payload)
    _dump_artifact(
        args,
        "rubricator",
        {
            "join_key": _join_key(args),
            "join_value": _join_value(args, sample),
            "status": "ok" if rubric is not None else "invalid_rubric",
            "prompt": prompt,
            "payload": payload,
            "rubric": rubric,
            "call": call_metadata,
            "teacher_answer_count": len(teacher_answers),
            "student_answer_count": len(samples),
        },
    )
    if rubric is None:
        return None, "invalid_online_rubric", call_metadata, teacher_answers
    return rubric, "online", call_metadata, teacher_answers


def _fallback_result(
    args: Any,
    sample: Sample,
    reason: str,
    rubric_source: str = "",
    details: dict[str, Any] | None = None,
) -> RewardResult:
    raw = {
        "fallback": reason,
        "rubric_source": rubric_source or reason,
        "join_key": _join_key(args),
        "join_value": _join_value(args, sample),
    }
    if details:
        raw["fallback_details"] = details
    return RewardResult(
        score=0.0,
        components={"rubric_task_success": 0.0},
        raw=raw,
        reason=reason,
        returns_total=True,
        reward_version="ropd_v1_fallback",
    )


def _result(
    args: Any,
    *,
    sample: Sample,
    rubric: Any,
    rubric_source: str,
    rubric_call: dict[str, Any] | None,
    judge_call: dict[str, Any] | None,
    maximum_score: float,
    teacher_scores: tuple[float, ...],
    group_stats: dict[str, Any],
    student_score: float,
    student_item: dict[str, Any],
    student_position: int,
    teacher_below_student: bool,
) -> RewardResult:
    if maximum_score <= 0:
        bounded = 0.0
    else:
        bounded = max(0.0, min(1.0, float(student_score) / maximum_score))
    train_score, effective_reward_mode = _select_train_score(
        args,
        answer_score=bounded,
        group_stats=group_stats,
    )
    weighted = train_score * _weight(args)
    raw = {
        "rubric": rubric,
        "rubric_hash": _rubric_hash(rubric),
        "rubric_source": rubric_source,
        "judge": student_item,
        "student_score": float(student_score),
        "teacher_scores": [float(score) for score in teacher_scores],
        "maximum_score": float(maximum_score),
        "reward_score": float(train_score),
        "answer_score": float(bounded),
        "student_answer_position": int(student_position),
        "teacher_below_student": bool(teacher_below_student),
        "ropd_group_reference": group_stats.get("reference", "students"),
        "ropd_group_reference_mean": float(group_stats.get("mean", 0.0)),
        "ropd_group_reference_std": float(group_stats.get("std", 0.0)),
        "ropd_student_group_scores": group_stats.get("student_scores", []),
        "ropd_teacher_scores": group_stats.get("teacher_scores", []),
        "ropd_train_reward_mode": effective_reward_mode,
        "ropd_reward_mode_requested": _reward_mode(args),
        "ropd_luffy_enabled": bool(_luffy_enabled(args)),
        "ropd_luffy_mode": _luffy_mode(args),
        "answer_mode": _answer_mode(args),
    }
    if rubric_call is not None:
        raw["rubric_call"] = rubric_call
    if judge_call is not None:
        raw["judge_call"] = judge_call
    return RewardResult(
        score=weighted,
        components={"rubric_task_success": weighted, "ropd_answer_score": bounded},
        raw=raw,
        reason="",
        returns_total=True,
        reward_version="ropd_v1",
    )


async def _score_bucket(
    args: Any,
    samples: list[Sample],
    *,
    bucket_key: str,
    rubric: dict[str, Any],
    rubric_source: str,
    rubric_call: dict[str, Any] | None,
    teacher_answers: tuple[str, ...],
) -> list[RewardResult]:
    student_answers = tuple(_student_answer(args, sample) for sample in samples)
    answer_items = _anonymous_answer_items(
        args=args,
        bucket_key=bucket_key,
        teacher_answers=teacher_answers,
        student_answers=student_answers,
    )
    answers = tuple(str(item["text"]) for item in answer_items)
    prompt = _build_verifier_prompt(args, samples[0], rubric=rubric, answers=answers)
    answer_item_records = [
        {
            "answer_index": idx,
            "source": item["source"],
            "source_index": item["source_index"],
            "text": item["text"],
        }
        for idx, item in enumerate(answer_items, start=1)
    ]
    try:
        payload, judge_call = await call_json_judge_with_metadata(
            args,
            prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            api_key_path=_role_api_key_path(args, "judge"),
            **_role_endpoint(args, "judge"),
        )
        judge_call = {
            **judge_call,
            "role": "judge",
            "concurrency_limit": _stage_concurrency(args, "judge"),
        }
        scored_items = _parse_batch_scores(payload, rubric=rubric, expected=len(answer_items))
    except Exception as exc:
        details = {"stage": "verifier", "type": type(exc).__name__, "message": str(exc)}
        _dump_artifact(
            args,
            "verifier",
            {
                "join_key": _join_key(args),
                "join_value": _join_value(args, samples[0]),
                "bucket_key": bucket_key,
                "status": "error",
                "error": details,
                "prompt": prompt,
                "rubric": rubric,
                "answer_items": answer_item_records,
            },
        )
        return [_fallback_result(args, sample, "judge_error", rubric_source, details) for sample in samples]

    teacher_scores_by_index: list[float | None] = [None] * len(teacher_answers)
    student_scores_by_index: list[tuple[float, dict[str, Any], int] | None] = [None] * len(student_answers)
    for position, (answer_item, score_item) in enumerate(zip(answer_items, scored_items, strict=True), start=1):
        score = float(score_item["final_score"])
        if answer_item["source"] == "teacher":
            teacher_scores_by_index[int(answer_item["source_index"])] = score
        elif answer_item["source"] == "student":
            student_scores_by_index[int(answer_item["source_index"])] = (score, score_item, position)

    if any(score is None for score in teacher_scores_by_index):
        raise ValueError("ROPD verifier did not return every teacher score")
    if any(score is None for score in student_scores_by_index):
        raise ValueError("ROPD verifier did not return every student score")

    teacher_scores = tuple(float(score) for score in teacher_scores_by_index if score is not None)
    maximum_score = _maximum_score(rubric)
    student_answer_scores = [
        max(0.0, min(1.0, float(item[0]) / maximum_score)) if item is not None and maximum_score > 0 else 0.0
        for item in student_scores_by_index
    ]
    teacher_answer_scores = tuple(
        max(0.0, min(1.0, float(score) / maximum_score)) for score in teacher_scores
    ) if maximum_score > 0 else ()
    group_stats = _group_stats(args, student_answer_scores, teacher_answer_scores)
    teacher_below_student = bool(teacher_scores and min(teacher_scores) < max(item[0] for item in student_scores_by_index if item is not None))
    _dump_artifact(
        args,
        "verifier",
        {
            "join_key": _join_key(args),
            "join_value": _join_value(args, samples[0]),
            "bucket_key": bucket_key,
            "status": "ok",
            "prompt": prompt,
            "rubric": rubric,
            "answer_items": answer_item_records,
            "payload": payload,
            "scored_items": scored_items,
            "teacher_scores": teacher_scores,
            "student_scores": [None if item is None else item[0] for item in student_scores_by_index],
            "maximum_score": maximum_score,
            "group_stats": group_stats,
            "teacher_below_student": teacher_below_student,
            "call": judge_call,
        },
    )
    results: list[RewardResult] = []
    for idx, item in enumerate(student_scores_by_index):
        if item is None:
            results.append(_fallback_result(args, samples[idx], "missing_student_score", rubric_source))
            continue
        student_score, student_item, student_position = item
        # Attach call metadata only once so aggregate metrics count actual LLM
        # calls, not samples scored by the same batched call. Teacher answers are
        # scored in the same call but are diagnostics only.
        results.append(
            _result(
                args,
                sample=samples[idx],
                rubric=rubric,
                rubric_source=rubric_source,
                rubric_call=rubric_call if idx == 0 else None,
                judge_call=judge_call if idx == 0 else None,
                maximum_score=maximum_score,
                teacher_scores=teacher_scores,
                group_stats=group_stats,
                student_score=student_score,
                student_item=student_item,
                student_position=student_position,
                teacher_below_student=teacher_below_student,
            )
        )
    return results


async def score(args: Any, samples: list[Sample], *, single: bool = False) -> list[RewardResult]:
    if judge_mode(args) != "aux":
        fallback_results = await naive.score(args, samples, single=single)
        for result in fallback_results:
            result.reward_version = "ropd_v1_fallback_naive"
        return fallback_results
    _validate_reward_config(args)

    buckets: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        buckets.setdefault(_join_value(args, sample), []).append(idx)

    results: list[RewardResult | None] = [None] * len(samples)
    bucket_items = list(buckets.items())
    rubric_infos = await asyncio.gather(
        *[
            _with_stage_limit(args, "rubric", _rubric_for_bucket, args, [samples[idx] for idx in indices])
            for _, indices in bucket_items
        ]
    )

    judge_tasks: list[
        tuple[list[int], str, list[Sample], dict[str, Any], str, dict[str, Any] | None, tuple[str, ...]]
    ] = []
    for (bucket_key, indices), (rubric, rubric_source, rubric_call, teacher_answers) in zip(
        bucket_items, rubric_infos, strict=True
    ):
        if rubric is None:
            for idx in indices:
                results[idx] = _fallback_result(args, samples[idx], "missing_rubric", rubric_source)
            continue
        judge_tasks.append(
            (
                indices,
                bucket_key,
                [samples[idx] for idx in indices],
                rubric,
                rubric_source,
                rubric_call,
                teacher_answers,
            )
        )

    if judge_tasks:
        limited_judge_tasks = [
            _with_stage_limit(
                args,
                "judge",
                _score_bucket,
                args,
                bucket_samples,
                bucket_key=bucket_key,
                rubric=rubric,
                rubric_source=rubric_source,
                rubric_call=rubric_call,
                teacher_answers=teacher_answers,
            )
            for _indices, bucket_key, bucket_samples, rubric, rubric_source, rubric_call, teacher_answers in judge_tasks
        ]
        for (indices, *_), bucket_results in zip(judge_tasks, await asyncio.gather(*limited_judge_tasks), strict=True):
            for idx, result in zip(indices, bucket_results, strict=True):
                results[idx] = result
    return [
        result if result is not None else _fallback_result(args, sample, "missing_result")
        for result, sample in zip(results, samples, strict=True)
    ]
