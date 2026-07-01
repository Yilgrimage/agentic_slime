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

_TEACHER_CACHE: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
_DUMP_COUNTS: dict[str, int] = {}


def _cfg(args: Any, name: str, default: Any = None) -> Any:
    return reward_cfg_path(args, f"ropd.{name}", default)


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


def _cache_path(args: Any) -> Path | None:
    path = runtime_env(args, "AGENT_ENV_ROPD_RUBRIC_CACHE_PATH", "").strip()
    if not path:
        path = str(_cfg(args, "rubric_cache_path", "") or "").strip()
    return resolve_path(args, path) if path else None


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


def _cache_key(sample: Sample) -> str:
    sample_metadata = metadata(sample)
    raw = {
        "task_id": sample_metadata.get("task_id"),
        "prompt": task_prompt(sample),
        "references": reference_values(sample),
    }
    text = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_cache(path)
    current.update(cache)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
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
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        return [item for item in value["items"] if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("tasks"), list):
        return [item for item in value["tasks"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported ROPD teacher file format: {path}")


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


def _teacher_files(args: Any) -> list[Path]:
    override = runtime_env(args, "AGENT_ENV_ROPD_TEACHER_FILE", "").strip()
    raw = override or _cfg(args, "teacher_file", None) or _cfg(args, "teacher_files", None)
    return [resolve_path(args, item) for item in _list_value(raw)]


def _teacher_index(args: Any) -> dict[str, dict[str, Any]]:
    files = _teacher_files(args)
    if not files:
        return {}
    key = _join_key(args)
    cache_key = ("|".join(str(path) for path in files), key)
    if cache_key in _TEACHER_CACHE:
        return _TEACHER_CACHE[cache_key]

    output: dict[str, dict[str, Any]] = {}
    for path in files:
        for row in _load_json_or_jsonl(path):
            row_key = row.get(key)
            if row_key in (None, "", []):
                row_key = row.get("task_id") or row.get("id") or row.get("task_index")
            if row_key in (None, "", []):
                continue
            output[str(row_key)] = row
    _TEACHER_CACHE[cache_key] = output
    return output


def _teacher_data(args: Any, sample: Sample) -> dict[str, Any]:
    return _teacher_index(args).get(_join_value(args, sample), {})


def _teacher_or_metadata_value(args: Any, sample: Sample, teacher_data: dict[str, Any], cfg_name: str, default_keys: tuple[str, ...]) -> Any:
    keys = _list_value(_cfg(args, cfg_name, None), default_keys)
    value = _metadata_value(sample, keys)
    if value not in (None, "", []):
        return value
    for key in keys:
        value = teacher_data.get(key)
        if value not in (None, "", []):
            return value
    return None


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

    sample_metadata = metadata(sample)
    turns = sample_metadata.get("turns")
    if isinstance(turns, list) and turns:
        lines: list[str] = []
        for idx, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                continue
            response_text = turn.get("parser_text") or turn.get("response_text") or ""
            action = turn.get("action")
            observation = turn.get("observation")
            parts = [f"Step {idx}:"]
            if response_text:
                parts.append(f"Response:\n{response_text}")
            if action not in (None, "", []):
                parts.append(f"Action: {action}")
            if observation not in (None, "", []):
                parts.append(f"Observation after action:\n{observation}")
            lines.append("\n".join(parts))
        if lines:
            return _trim_answer_for_judge("\n\n".join(lines))

    return _trim_answer_for_judge(prediction_text(sample))


def _teacher_answers(args: Any, sample: Sample, teacher_data: dict[str, Any]) -> tuple[str, ...]:
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
    for key in keys:
        value = teacher_data.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value if str(item).strip())
        else:
            values.append(str(value))
    if values:
        return tuple(
            dict.fromkeys(_sanitize_teacher_answer_for_anonymous_verifier(value) for value in values if value.strip())
        )

    refs = reference_values(sample)
    return tuple(dict.fromkeys(_sanitize_teacher_answer_for_anonymous_verifier(value) for value in refs if value.strip()))


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


def _rubric_cache_key(args: Any, samples: list[Sample], teacher_answers: tuple[str, ...]) -> str:
    raw = {
        "join_key": _join_key(args),
        "join_value": _join_value(args, samples[0]),
        "prompt": task_prompt(samples[0]),
        "teacher_answers": teacher_answers,
        "student_answers": [_student_answer(args, sample) for sample in samples],
    }
    text = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _build_rubricator_prompt(args: Any, samples: list[Sample], teacher_answers: tuple[str, ...]) -> str:
    return _render_template(
        RUBRICATOR_PROMPT_TEMPLATE,
        {
            "question": truncate(task_prompt(samples[0]), 8000),
            "teacher_response": _render_answer_block("Reference", teacher_answers),
            "student_response": _render_answer_block(
                "Student",
                [truncate(_student_answer(args, sample), 6000) for sample in samples],
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
    return _render_template(
        VERIFIER_PROMPT_TEMPLATE,
        {
            "question": truncate(task_prompt(sample), 8000),
            "rubrics": json.dumps(rubric["rubrics"], ensure_ascii=False, indent=2),
            "answers": _render_answer_block(
                "Answer",
                [truncate(answer, 6000) for answer in answers],
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
    bucket_key: str,
    teacher_answers: tuple[str, ...],
    student_answers: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    items = [
        {"source": "teacher", "source_index": idx, "text": answer}
        for idx, answer in enumerate(teacher_answers)
    ] + [
        {"source": "student", "source_index": idx, "text": answer}
        for idx, answer in enumerate(student_answers)
    ]
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


def _existing_rubric(args: Any, sample: Sample, teacher_data: dict[str, Any]) -> Any:
    return _teacher_or_metadata_value(
        args,
        sample,
        teacher_data,
        "rubric_keys",
        ("rubric", "reward_rubric", "ropd_rubric"),
    )


def _weight(args: Any) -> float:
    raw = runtime_env(args, "AGENT_ENV_ROPD_TASK_SUCCESS_WEIGHT", "")
    if raw == "":
        raw = _cfg(args, "task_success_weight", reward_cfg_path(args, "outcome", 10.0))
    return float_value(raw, 10.0)


async def _rubric_for_bucket(
    args: Any,
    samples: list[Sample],
    cache: dict[str, Any],
) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None, tuple[str, ...]]:
    sample = samples[0]
    teacher_data = _teacher_data(args, sample)
    teacher_answers = _teacher_answers(args, sample, teacher_data)
    if not teacher_answers:
        return None, "missing_teacher", None, ()

    existing = _existing_rubric(args, sample, teacher_data)
    existing_rubric = _normalize_rubric(existing)
    if existing_rubric is not None:
        return existing_rubric, "teacher", None, teacher_answers

    key = _rubric_cache_key(args, samples, teacher_answers)
    if key in cache:
        cached_rubric = _normalize_rubric(cache[key])
        if cached_rubric is not None:
            return cached_rubric, "cache", None, teacher_answers

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
    except Exception as exc:
        _dump_artifact(
            args,
            "rubricator",
            {
                "join_key": _join_key(args),
                "join_value": _join_value(args, sample),
                "cache_key": key,
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
            "cache_key": key,
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
    cache[key] = rubric
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
    rubric: Any,
    rubric_source: str,
    rubric_call: dict[str, Any] | None,
    judge_call: dict[str, Any] | None,
    maximum_score: float,
    teacher_scores: tuple[float, ...],
    student_score: float,
    student_item: dict[str, Any],
    student_position: int,
    teacher_below_student: bool,
) -> RewardResult:
    if maximum_score <= 0:
        bounded = 0.0
    else:
        bounded = max(0.0, min(1.0, float(student_score) / maximum_score))
    weighted = bounded * _weight(args)
    raw = {
        "rubric": rubric,
        "rubric_hash": _rubric_hash(rubric),
        "rubric_source": rubric_source,
        "judge": student_item,
        "student_score": float(student_score),
        "teacher_scores": [float(score) for score in teacher_scores],
        "maximum_score": float(maximum_score),
        "reward_score": bounded,
        "student_answer_position": int(student_position),
        "teacher_below_student": bool(teacher_below_student),
    }
    if rubric_call is not None:
        raw["rubric_call"] = rubric_call
    if judge_call is not None:
        raw["judge_call"] = judge_call
    return RewardResult(
        score=weighted,
        components={"rubric_task_success": weighted},
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
                rubric=rubric,
                rubric_source=rubric_source,
                rubric_call=rubric_call if idx == 0 else None,
                judge_call=judge_call if idx == 0 else None,
                maximum_score=maximum_score,
                teacher_scores=teacher_scores,
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

    cache_path = _cache_path(args)
    cache = _load_cache(cache_path)
    buckets: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        buckets.setdefault(_join_value(args, sample), []).append(idx)

    results: list[RewardResult | None] = [None] * len(samples)
    bucket_items = list(buckets.items())
    rubric_infos = await asyncio.gather(
        *[_rubric_for_bucket(args, [samples[idx] for idx in indices], cache) for _, indices in bucket_items]
    )
    changed = any(rubric_source == "online" for _, rubric_source, _, _ in rubric_infos)

    judge_tasks = []
    judge_task_keys: list[list[int]] = []
    for (bucket_key, indices), (rubric, rubric_source, rubric_call, teacher_answers) in zip(
        bucket_items, rubric_infos, strict=True
    ):
        if rubric is None:
            for idx in indices:
                results[idx] = _fallback_result(args, samples[idx], "missing_rubric", rubric_source)
            continue
        judge_task_keys.append(indices)
        judge_tasks.append(
            _score_bucket(
                args,
                [samples[idx] for idx in indices],
                bucket_key=bucket_key,
                rubric=rubric,
                rubric_source=rubric_source,
                rubric_call=rubric_call,
                teacher_answers=teacher_answers,
            )
        )

    if judge_tasks:
        for indices, bucket_results in zip(judge_task_keys, await asyncio.gather(*judge_tasks), strict=True):
            for idx, result in zip(indices, bucket_results, strict=True):
                results[idx] = result
    if changed:
        _save_cache(cache_path, cache)
    return [
        result if result is not None else _fallback_result(args, sample, "missing_result")
        for result, sample in zip(results, samples, strict=True)
    ]
