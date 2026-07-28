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

from examples.agent_env.dump import record_dump_step_label, reserve_dump_slot, sample_dump_step_label
from examples.agent_env.trace_rendering import (
    TraceCompressionOptions,
    compress_trace_text,
    render_answer_for_reward,
    render_teacher_trace_for_reward,
)

from .config import resolve_path, reward_cfg_path
from .extractors import (
    bool_value,
    explicit_task_prompt,
    float_value,
    int_value,
    metadata,
    prediction_text,
    reference_values,
    runtime_env,
    truncate,
)
from .llm_client import call_json_judge_with_metadata, judge_mode
from .types import RewardResult

RUBRIC_SCHEMA_VERSION = "ropd.rubric.v1"
BATCH_VERIFIER_SCHEMA_VERSION = "ropd.batch_verifier.v2"
ANSWER_PROCESS_RUBRIC_SCHEMA_VERSION = "ropd.answer_process_rubric.v1"
ANSWER_PROCESS_VERIFIER_SCHEMA_VERSION = "ropd.answer_process_compact_batch_verifier.v1"
TEACHER_TRACE_FIELDS = (
    "teacher_tool_trace",
    "teacher_trace",
    "teacher_trajectory",
    "teacher_action_observation_trace",
    "teacher_full_trace_text",
    "reference_trace",
    "teacher_trace_tool_io",
    "teacher_no_tool_response_text",
    "teacher_trace_no_tool_response",
    "teacher_no_tool_trace",
    "reference_no_tool_trace",
)

_TEACHER_INDEX_CACHE: dict[str, tuple[int, int, dict[str, dict[str, Any]]]] = {}

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
7. 对 agent 任务，skill 名称只是能力线索而不是硬性约束；如果回答使用了等价 skill、MCP 或工具能力并正确执行，rubric 不应要求精确匹配某个历史 skill 名。
8. 对 agent 任务，最终任务结论或完成状态应是最高权重信号；过程、证据链和工具使用是支撑信号，不能让一个最终结论错误的回答仅靠过程相似拿到高分。

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
- 不要因为 skill 名称不同而扣分；如果回答使用了等价 skill、MCP 或工具能力并正确执行，应按 criterion 实质判定。
- 当回答清楚使用了与参考回答相同或等价的 MCP/tool 路径、参数合理、并且因为 timeout、rate limit、5xx 或 internal service error 等瞬时服务问题失败时，不要仅因服务没有返回数据而判定过程/tool-use criterion 失败。
- 上述瞬时服务例外不适用于：工具选错、参数错误、缺鉴权/无权限、跳过必要取证、编造工具结果、或最终结论没有证据支持。
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

RUBRICATOR_ANSWER_PROCESS_PROMPT_TEMPLATE = """你是一名 agentic task 评估专家。你的任务是为同一道业务问题生成一套 answer-first 的共享评分细则。

输入包含：
- teacher trajectory：高置信度参考轨迹，但不保证绝对正确，也不是唯一解法。
- student trajectory：待评估轨迹，可能暴露当前模型的质量缺口。

请生成两组 rubric：
1. answer_rubrics：评价最终回答质量。`a1` 必须是核心结论正确性，后续 `a2...` 评价最终回答里的证据、关键细节、边界披露和风险说明。
2. process_rubrics：评价过程、工具、证据链、取证可靠性和错误处理。该分数暂不进入最终 reward，只用于后续 trace 内部信用分配和诊断。

# 输入数据
[Question]
{question}

[Teacher Trajectory]
{teacher_response}

[Student Trajectory]
{student_response}

[Additional Instructions]
{extra_rubric_instructions}

# 打分口径
后续 verifier 会对每条 rubric 给 0 到 5 分：
- 5：完全满足。
- 4：基本满足，只有轻微遗漏。
- 3：部分满足，但缺少重要细节。
- 2：只有少量相关内容。
- 1：极弱相关或基本不可用。
- 0：不满足。

最终 answer reward 的计算方式固定为：
- `core_score = a1.score`，范围 0 到 5。
- `support_score = weighted_average(a2..an scores)`，范围 0 到 5。
- `answer_reward = core_score + support_score`，范围 0 到 10。
- 也就是核心结论正确性占 answer reward 的 50%，其他 answer rubric 加权归一后占 50%。

process score 独立计算为 `weighted_average(p1..pn scores)`，范围 0 到 5，暂不参与 `answer_reward`。

# 核心原则
- `a1` 必须只评价最终回答是否解决用户核心问题和核心结论是否正确。它允许 0 到 5 的部分分：例如多对象任务里答对部分对象、结论方向正确但缺关键限定、或只完成部分子问题。
- `a2...` 只评价最终回答中的必要证据、关键细节、置信边界、不可达信息披露和风险说明。不要把工具路径本身写进 answer rubric。
- process_rubrics 只评价过程中是否正确选择 skill/tool、是否基于可见证据推理、是否处理工具错误、是否避免编造工具结果、是否能定位到具体 step。
- 如果最终回答没有回答用户核心问题、对象类型错误、关键事实错误、编造结果、只有过程没有结论、或没有最终答案，`a1` 应接近 0。
- 如果正确处理方式是披露鉴权、权限、数据不可达或证据不足，`a1` 应奖励“明确披露阻断点且不编造业务结论”；自信给出具体分析但没有相应证据，应低分。
- 不要要求复刻 teacher 的措辞、格式或具体工具路径；等价能力、等价证据和等价结论应被允许。
- 若 trajectory 只声称用了工具但没有可见证据，process rubric 不能自动给高分。

# rubric 设计要求
- `answer_rubrics` 生成 2 到 5 条。
- `answer_rubrics[0]` 必须是 `a1`，category 必须是 `Core Answer Correctness`，`max_score` 必须是 5，`weight` 必须是 1。
- `answer_rubrics[1:]` 每条 `max_score` 必须是 5，`weight` 为正数，用来控制 support_score 的加权平均。
- `process_rubrics` 生成 2 到 5 条。
- 每条 process rubric 的 `max_score` 必须是 5，`weight` 为正数。
- process rubric 必须能定位到具体 step；不要写“整体过程合理”“证据充分”这类无法定位到具体 step 的 criterion。

# 禁止项
不要写：
- “与 teacher 答案一致”
- “使用和 teacher 相同的工具”
- “措辞/结构和 teacher 相同”
- 只能通过直接比较 teacher 和 student 才能判断的 criterion
- 奖励长篇幅、泛化流程描述或表面自信的 criterion

# 输出格式
只返回 JSON 对象，结构必须为：
```json
{
  "schema_version": "ropd.answer_process_rubric.v1",
  "answer_rubrics": [
    {
      "criterion_id": "a1",
      "category": "Core Answer Correctness",
      "criterion": "0-5 分核心结论正确性标准",
      "max_score": 5,
      "weight": 1
    },
    {
      "criterion_id": "a2",
      "category": "Answer Support",
      "criterion": "0-5 分最终回答证据/关键细节/边界披露标准",
      "max_score": 5,
      "weight": 1
    }
  ],
  "process_rubrics": [
    {
      "criterion_id": "p1",
      "category": "Process Evidence",
      "criterion": "0-5 分、可定位 step 的过程/证据标准",
      "max_score": 5,
      "weight": 1
    }
  ],
  "maximum_scores": {
    "answer_core": 5,
    "answer_support": 5,
    "answer": 10,
    "process": 5,
    "total": 10
  },
  "score_policy": {
    "answer_first": true,
    "answer_core_weight": 0.5,
    "answer_support_weight": 0.5,
    "answer_weight": 1.0,
    "process_weight": 0.0,
    "process_for_credit_assignment": true,
    "final_reward_uses": "answer_only",
    "rubric_score_range": "0_to_5_per_criterion"
  }
}
```

# 输出约束
- `schema_version` 必须严格等于 `ropd.answer_process_rubric.v1`。
- `answer_rubrics[].criterion_id` 必须为 `a1`, `a2`, ...
- `process_rubrics[].criterion_id` 必须为 `p1`, `p2`, ...
- 每条 rubric 的 `max_score` 必须等于 5。
- 每条 rubric 的 `weight` 必须是正数。
- `maximum_scores.answer_core` 必须等于 5。
- `maximum_scores.answer_support` 必须等于 5。
- `maximum_scores.answer` 必须等于 10。
- `maximum_scores.process` 必须等于 5。
- `maximum_scores.total` 必须等于 10。
- 只返回 JSON，不要输出解释、Markdown 或额外文本。
"""

VERIFIER_ANSWER_PROCESS_PROMPT_TEMPLATE = """你是一名 agentic task 评分专家。你的任务是针对同一道题，在给定 answer/process 分离 rubric 的情况下，一次性评价多个匿名 trajectory。

[Question]
{question}

[Rubrics]
{rubrics}

[Anonymous Trajectories]
{answers}

[Additional Scoring Instructions]
{extra_scoring_instructions}

# 核心评分规则
对每个 trajectory 独立评分，不能比较不同 trajectory，也不能猜哪个是 teacher 或 student。

你必须分别输出：
1. `answer_scores`：对每条 answer rubric 给 0 到 5 分。
2. `process_scores`：对每条 process rubric 给 0 到 5 分。
3. `final_answer_quality` 和 `fatal_error`。

不要输出 rationale、evidence、step_indices 或长文本解释；这些会显著拖慢正式训练。

# 0-5 分含义
- 5：完全满足。
- 4：基本满足，只有轻微遗漏。
- 3：部分满足，但缺少重要细节。
- 2：只有少量相关内容。
- 1：极弱相关或基本不可用。
- 0：不满足。

# Answer-first 规则
- 最终 answer 是当前 final reward 的唯一来源。
- `a1` 是核心结论正确性，必须按最终回答的核心结论给 0 到 5 分。多对象、多子问题任务可以给部分分；没有最终答案、编造核心事实、对象错配或答非所问应接近 0。
- `a2...` 评价最终回答的证据、关键细节、边界披露和风险说明。它们不应替代 `a1`，也不能因为过程看起来努力就抬高核心结论分。
- 如果正确处理方式是披露鉴权/权限/数据不可达/证据不足，则明确披露阻断点且不编造业务结论可以获得相应 answer 分；自信给出具体业务归因、对象判断或数值结论但缺少可见证据，应低分并可设 `fatal_error=true`。

# Process 规则
- process_rubrics 只用于过程诊断和后续信用分配，不参与当前 final answer reward。
- process_rubrics 评价工具/skill 选择、证据链、错误处理、是否基于可见证据、是否避免编造工具结果。
- 如果 trajectory 声称查过工具，但没有可见证据支撑，不能给高 process 分。
- 如果工具失败但 trajectory 明确披露阻断点且没有编造结论，可以给对应 process 分，但不能因此自动给 answer 分。

# 质量枚举
为每个 trajectory 给出 `final_answer_quality`：
- `correct`：最终回答完整且核心事实正确。
- `mostly_correct`：核心结论正确，仅有轻微遗漏或轻微不确定。
- `partial`：部分回答了问题，但缺关键事实、关键对象或关键归因。
- `wrong`：核心结论错误、对象类型错误、答非所问或明显幻觉。
- `no_answer`：没有可用最终回答，或只有“无法完成/turn budget/继续查询”等非答案。

如果存在严重错误（明显编造工具结果、把空结果说成有结果、越权利用隐藏信息、核心对象错配），`fatal_error` 设为 true。

# 输出格式
只返回 JSON object：
```json
{
  "schema_version": "ropd.answer_process_compact_batch_verifier.v1",
  "answers": [
    {
      "answer_index": 1,
      "answer_scores": [
        {
          "criterion_id": "a1",
          "score": 4
        },
        {
          "criterion_id": "a2",
          "score": 3
        }
      ],
      "process_scores": [
        {
          "criterion_id": "p1",
          "score": 4
        },
        {
          "criterion_id": "p2",
          "score": 0
        }
      ],
      "final_answer_quality": "partial",
      "fatal_error": false
    }
  ]
}
```

# 输出约束
- `schema_version` 必须严格等于 `ropd.answer_process_compact_batch_verifier.v1`。
- `answers` 数组必须包含每个输入 trajectory 各一项。
- `answer_index` 必须从 1 开始，并按输入顺序覆盖所有 trajectory。
- `answer_scores` 长度必须等于 answer_rubrics 数量，criterion_id 顺序必须与 answer_rubrics 完全一致。
- `process_scores` 长度必须等于 process_rubrics 数量，criterion_id 顺序必须与 process_rubrics 完全一致。
- 所有 `score` 必须是 0 到 5 的数字。
- 不要输出未要求字段，尤其不要输出 rationale、evidence、step_indices。
- 只返回 JSON，不要输出解释、Markdown 或其他文本。
"""

_STAGE_SEMAPHORES: dict[tuple[int, str, int], asyncio.Semaphore] = {}


def _cfg(args: Any, name: str, default: Any = None) -> Any:
    return reward_cfg_path(args, f"ropd.{name}", default)


def _cfg_bool(args: Any, cfg_name: str, default: bool) -> bool:
    return bool_value(_cfg(args, cfg_name, default), default)


def _cfg_int(args: Any, cfg_name: str, default: int) -> int:
    return int_value(_cfg(args, cfg_name, default), default)


def _cfg_choice(args: Any, cfg_name: str, default: str, choices: set[str]) -> str:
    value = str(_cfg(args, cfg_name, default) or default).strip().lower()
    return value if value in choices else default


def _schema_mode(args: Any) -> str:
    return _cfg_choice(args, "schema_mode", "binary", {"binary", "answer_process_50_50"})


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


def _dump_total_limit(args: Any) -> int:
    raw = runtime_env(args, "AGENT_ENV_ROPD_DUMP_TOTAL_N", "").strip()
    if raw == "":
        raw = str(_cfg(args, "dump_total_n", "") or "").strip()
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


def _sample_rollout_label(sample: Sample) -> str:
    return sample_dump_step_label(sample)


def _record_rollout_label(record: dict[str, Any]) -> str:
    return record_dump_step_label(record)


def _dump_artifact(args: Any, stage: str, record: dict[str, Any]) -> None:
    output_dir = _dump_dir(args)
    if output_dir is None:
        return
    limit = _dump_limit(args)
    total_limit = _dump_total_limit(args)
    dump_step = _record_rollout_label(record)
    slot = reserve_dump_slot(
        namespace="reward_artifacts:ropd",
        stage=stage,
        dump_step=dump_step,
        per_step_limit=limit,
        total_limit=total_limit,
    )
    if slot is None:
        return
    count, total_count = slot
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stage}_pid{os.getpid()}.jsonl"
    payload = {
        "schema_version": "agent_env.ropd_artifact.v1",
        "stage": stage,
        "pid": os.getpid(),
        "time": time.time(),
        "dump_step": dump_step,
        "index": total_count,
        **record,
    }
    payload["dump_step"] = dump_step
    payload["index_in_step"] = count
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


def _role_cfg(args: Any, role: str, name: str, default: Any = None) -> Any:
    role_value = _cfg(args, f"{role}_{name}", None)
    if role_value is not None:
        return role_value
    return _cfg(args, name, default)


def _role_int(args: Any, role: str, name: str, default: int) -> int:
    return _positive_int(_role_cfg(args, role, name, default), default)


def _role_float(args: Any, role: str, name: str, default: float) -> float:
    return float_value(_role_cfg(args, role, name, default), default)


def _role_json(args: Any, role: str, name: str) -> dict[str, Any] | None:
    value = _role_cfg(args, role, name, None)
    if value in (None, "", {}):
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"reward.ropd.{role}_{name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"reward.ropd.{role}_{name} must be a JSON object")
    return parsed


def _role_request_options(args: Any, role: str, *, default_max_tokens: int) -> dict[str, Any]:
    return {
        "max_tokens": _role_int(args, role, "max_tokens", default_max_tokens),
        "max_attempts": _role_int(args, role, "max_attempts", _role_int(args, role, "attempts", 6)),
        "rate_limit_backoff_s": _role_float(args, role, "rate_limit_backoff_s", 30.0),
        "retry_backoff_s": _role_float(args, role, "retry_backoff_s", 1.0),
        "response_format": _role_cfg(args, role, "response_format", "json_object"),
        "extra_body": _role_json(args, role, "extra_body_json"),
    }


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _base_concurrency(args: Any) -> int:
    raw = _cfg(args, "concurrency", reward_cfg_path(args, "concurrency", 8))
    return _positive_int(raw, 8)


def _stage_concurrency(args: Any, stage: str) -> int:
    base = _base_concurrency(args)
    stage = stage.lower()
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
        "task_prompt": explicit_task_prompt(sample),
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


def _teacher_index_path(args: Any) -> Path | None:
    raw = str(_cfg(args, "teacher_index_path", "") or runtime_env(args, "AGENT_ENV_ROPD_TEACHER_INDEX_PATH", "")).strip()
    return resolve_path(args, raw) if raw else None


def _candidate_values_from_mapping(mapping: dict[str, Any], keys: list[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = mapping.get(key)
        if value in (None, "", []):
            continue
        for item in _list_value(value):
            values.append(item)
            if "::" in item:
                values.append(item.split("::", 1)[0])
    return values


def _teacher_index_key_candidates(args: Any, sample: Sample) -> list[str]:
    sample_metadata = metadata(sample)
    keys = _list_value(
        _cfg(args, "teacher_index_keys", None),
        (
            "teacher_trace_key",
            "teacher_index_key",
            "teacher_rollout_key",
            "rollout_key",
            "source_sample_id",
            "global_index",
            "prompt_index",
            "row_index",
            "sample_id",
            "task_id",
            "id",
            "task_index",
        ),
    )
    candidates = _candidate_values_from_mapping(sample_metadata, keys)
    source_row = sample_metadata.get("source_row")
    if isinstance(source_row, dict):
        candidates.extend(_candidate_values_from_mapping(source_row, keys))
    sample_index = getattr(sample, "index", None)
    if sample_index is not None:
        candidates.append(str(sample_index))
    join_value = _join_value(args, sample)
    if join_value:
        candidates.append(join_value)
    return list(dict.fromkeys(value for value in candidates if value))


def _first_mapping_text(mapping: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return str(value).strip()
    return ""


def _teacher_index_row_text(args: Any, row: dict[str, Any]) -> str:
    keys = _list_value(_cfg(args, "teacher_answer_keys", None), TEACHER_TRACE_FIELDS)
    for source in (row, row.get("evidence"), row.get("adapter_result"), row.get("source_row")):
        if isinstance(source, dict):
            text = _first_mapping_text(source, keys)
            if text:
                return text
    return ""


def _teacher_index_row_status(row: dict[str, Any], *, has_text: bool) -> str:
    for source in (row, row.get("evidence"), row.get("adapter_result")):
        if isinstance(source, dict):
            for success_key in ("teacher_success", "success", "completed"):
                success = source.get(success_key)
                if success is True:
                    return "completed"
            value = source.get("status") or source.get("teacher_status") or source.get("episode_status")
            if value:
                status = str(value).strip().lower()
                if status in {"ok", "success", "succeeded", "done"}:
                    return "completed"
                return status
    return "completed" if has_text else "missing"


def _read_teacher_index(args: Any, path: Path) -> dict[str, dict[str, Any]]:
    stat = path.stat()
    cache_key = str(path.resolve())
    cached = _TEACHER_INDEX_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            if all(isinstance(item, dict) for item in value.values()):
                rows = [dict(item, _index_key=str(key)) for key, item in value.items()]
            elif isinstance(value.get("items"), list):
                rows = [item for item in value["items"] if isinstance(item, dict)]
            else:
                rows = [value]
        elif isinstance(value, list):
            rows = [item for item in value if isinstance(item, dict)]
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    if not line.endswith("\n"):
                        break
                    raise ValueError(f"Invalid JSON in teacher index {path}:{line_no}: {exc}") from exc
                if isinstance(value, dict):
                    rows.append(value)

    key_fields = _list_value(
        _cfg(args, "teacher_index_keys", None),
        (
            "teacher_trace_key",
            "teacher_index_key",
            "teacher_rollout_key",
            "rollout_key",
            "source_sample_id",
            "global_index",
            "prompt_index",
            "row_index",
            "sample_id",
            "task_id",
            "id",
            "task_index",
        ),
    )
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidates = _candidate_values_from_mapping(row, key_fields)
        source_row = row.get("source_row")
        if isinstance(source_row, dict):
            candidates.extend(_candidate_values_from_mapping(source_row, key_fields))
        if row.get("_index_key") not in (None, ""):
            candidates.append(str(row["_index_key"]))
        for key in dict.fromkeys(candidates):
            index.setdefault(key, row)

    _TEACHER_INDEX_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, index)
    return index


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


def _limit_reward_text(args: Any, text: Any, max_chars: int, *, field: str) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if _cfg_bool(args, "allow_text_truncation", False):
        return truncate(value, max_chars)
    raise ValueError(
        f"ROPD {field} has {len(value)} chars, exceeding {max_chars}. "
        "Reward inputs must not be silently truncated; use trace strip_* options "
        "or explicitly set ropd.allow_text_truncation=true for a debug-only run."
    )


def _ropd_question(sample: Sample) -> str:
    question = explicit_task_prompt(sample).strip()
    if question:
        return question
    sample_metadata = metadata(sample)
    task_id = sample_metadata.get("task_id")
    raise ValueError(
        "ROPD requires explicit task question metadata (query/task_prompt/instruction/question). "
        f"sample.prompt is not accepted as a fallback because it may be a policy/system template; task_id={task_id!r}"
    )


def _trace_options(args: Any) -> TraceCompressionOptions:
    return TraceCompressionOptions(
        strip_reasoning=_cfg_bool(args, "strip_reasoning", True),
        strip_tool_response=_cfg_bool(args, "strip_tool_response", False),
        strip_assistant_response=_cfg_bool(args, "strip_assistant_response", True),
        strip_system_prompt=_cfg_bool(args, "strip_system_prompt", True),
    )


def _sanitize_teacher_answer_for_anonymous_verifier(args: Any, answer: Any) -> str:
    text = render_teacher_trace_for_reward(
        answer,
        options=_trace_options(args),
        check_reasoning_presence=True,
        reasoning_context="ropd_teacher_answer",
    )
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


def _answer_mode(args: Any) -> str:
    mode = str(_cfg(args, "answer_mode", "trace") or "trace").strip().lower()
    if mode in {"final", "trace"}:
        return mode
    deprecated = {
        "tool_trace",
        "tool_io",
        "tool_call_response_trace",
        "action_observation_trace",
        "final_with_tool_trace",
        "final_with_tool_io",
        "final_with_action_observation_trace",
        "full_trace",
        "full",
        "final_without_tool_response",
        "final_without_tool_responses",
        "final_with_no_tool_trace",
        "final_with_no_tool_response_trace",
        "final_with_trace_no_tool_response",
        "trace_without_tool_response",
        "trace_without_tool_responses",
        "no_tool_trace",
        "no_tool_response_trace",
        "trace_no_tool_response",
    }
    if mode in deprecated:
        raise RuntimeError(
            f"ropd.answer_mode={mode!r} is deprecated. Use answer_mode=trace plus strip_* switches, "
            "or answer_mode=final."
        )
    raise RuntimeError(f"Unsupported ropd.answer_mode={mode!r}; expected 'trace' or 'final'")


def _answer_for_judge(args: Any, sample: Sample) -> str:
    mode = _answer_mode(args)
    if mode == "final":
        return compress_trace_text(
            prediction_text(sample),
            options=_trace_options(args),
            strip_assistant_response=False,
            check_reasoning_presence=True,
            reasoning_context="ropd_student_final_answer",
        )
    return render_answer_for_reward(
        sample,
        final_answer=_explicit_final_answer(sample),
        answer_mode=mode,
        options=_trace_options(args),
        check_reasoning_presence=True,
    )


def _student_answer(args: Any, sample: Sample) -> str:
    return _answer_for_judge(args, sample)


def _explicit_final_answer(sample: Sample) -> str:
    sample_metadata = metadata(sample)
    for key in ("final_answer", "answer"):
        value = sample_metadata.get(key)
        if value not in (None, "", []):
            return str(value)
    env_eval = sample_metadata.get("env_evaluate")
    if isinstance(env_eval, dict):
        info = env_eval.get("info")
        if isinstance(info, dict) and info.get("final_answer") not in (None, ""):
            return str(info["final_answer"])
    turns = sample_metadata.get("turns")
    if isinstance(turns, list):
        for turn in reversed(turns):
            if not isinstance(turn, dict):
                continue
            env_step = turn.get("env_step")
            info = env_step.get("info") if isinstance(env_step, dict) and isinstance(env_step.get("info"), dict) else {}
            if info.get("final_answer") not in (None, ""):
                return str(info["final_answer"])
    return ""


def _teacher_answers(args: Any, sample: Sample) -> tuple[str, ...]:
    keys = _list_value(
        _cfg(args, "teacher_answer_keys", None),
        TEACHER_TRACE_FIELDS,
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
            dict.fromkeys(
                _sanitize_teacher_answer_for_anonymous_verifier(args, value)
                for value in values
                if value.strip()
            )
        )
    index_path = _teacher_index_path(args)
    if index_path is not None:
        index = _read_teacher_index(args, index_path)
        for key in _teacher_index_key_candidates(args, sample):
            row = index.get(key)
            if row is None:
                continue
            text = _teacher_index_row_text(args, row)
            status = _teacher_index_row_status(row, has_text=bool(text))
            if status != "completed" or not text:
                return ()
            return (_sanitize_teacher_answer_for_anonymous_verifier(args, text),)
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


def _positive_float(value: Any, default: float = 1.0) -> float:
    parsed = float_value(value, default)
    return parsed if parsed > 0 else default


def _normalize_answer_process_items(
    items: Any,
    *,
    prefix: str,
    default_category: str,
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        criterion = str(item.get("criterion") or item.get("description") or "").strip()
        if not criterion:
            continue
        expected_id = f"{prefix}{idx}"
        criterion_id = str(item.get("criterion_id") or expected_id).strip() or expected_id
        if not re.fullmatch(rf"{prefix}[1-9][0-9]*", criterion_id):
            criterion_id = expected_id
        normalized.append(
            {
                "criterion_id": criterion_id,
                "category": str(item.get("category") or default_category),
                "criterion": criterion,
                "max_score": 5,
                "weight": _positive_float(item.get("weight", 1.0), 1.0),
            }
        )
    return normalized


def _normalize_answer_process_rubric(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != ANSWER_PROCESS_RUBRIC_SCHEMA_VERSION:
        return None
    answer_items = _normalize_answer_process_items(
        payload.get("answer_rubrics"),
        prefix="a",
        default_category="Answer Support",
    )
    process_items = _normalize_answer_process_items(
        payload.get("process_rubrics"),
        prefix="p",
        default_category="Process Evidence",
    )
    if len(answer_items) < 2 or not process_items:
        return None
    answer_items[0] = {
        **answer_items[0],
        "criterion_id": "a1",
        "category": "Core Answer Correctness",
        "max_score": 5,
        "weight": 1.0,
    }
    for idx, item in enumerate(answer_items[1:], start=2):
        item["criterion_id"] = f"a{idx}"
        item["max_score"] = 5
    for idx, item in enumerate(process_items, start=1):
        item["criterion_id"] = f"p{idx}"
        item["max_score"] = 5
    return {
        "schema_version": ANSWER_PROCESS_RUBRIC_SCHEMA_VERSION,
        "answer_rubrics": answer_items,
        "process_rubrics": process_items,
        "maximum_scores": {
            "answer_core": 5,
            "answer_support": 5,
            "answer": 10,
            "process": 5,
            "total": 10,
        },
        "score_policy": {
            "answer_first": True,
            "answer_core_weight": 0.5,
            "answer_support_weight": 0.5,
            "answer_weight": 1.0,
            "process_weight": 0.0,
            "process_for_credit_assignment": True,
            "final_reward_uses": "answer_only",
            "rubric_score_range": "0_to_5_per_criterion",
        },
    }


def _normalize_rubric_for_mode(args: Any, payload: Any) -> dict[str, Any] | None:
    if _schema_mode(args) == "answer_process_50_50":
        return _normalize_answer_process_rubric(payload)
    return _normalize_rubric(payload)


def _maximum_score(rubric: Any) -> float:
    if isinstance(rubric, dict) and rubric.get("schema_version") == ANSWER_PROCESS_RUBRIC_SCHEMA_VERSION:
        maximum_scores = rubric.get("maximum_scores")
        if isinstance(maximum_scores, dict):
            return float_value(maximum_scores.get("answer"), 10.0)
        return 10.0
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
    question_max_chars = _cfg_int(args, "question_max_chars", 0)
    student_max_chars = _cfg_int(args, "student_rubric_max_chars", 0)
    reference_max_chars = _cfg_int(args, "reference_max_chars", 0)
    template = (
        RUBRICATOR_ANSWER_PROCESS_PROMPT_TEMPLATE
        if _schema_mode(args) == "answer_process_50_50"
        else RUBRICATOR_PROMPT_TEMPLATE
    )
    return _render_template(
        template,
        {
            "question": _limit_reward_text(args, _ropd_question(samples[0]), question_max_chars, field="question"),
            "teacher_response": _render_answer_block(
                "Reference",
                [_limit_reward_text(args, answer, reference_max_chars, field="teacher_response") for answer in teacher_answers],
            ),
            "student_response": _render_answer_block(
                "Student",
                [
                    _limit_reward_text(args, _student_answer(args, sample), student_max_chars, field="student_response")
                    for sample in samples
                ],
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
    question_max_chars = _cfg_int(args, "question_max_chars", 0)
    answer_max_chars = _cfg_int(args, "verifier_answer_max_chars", 0)
    if isinstance(rubric, dict) and rubric.get("schema_version") == ANSWER_PROCESS_RUBRIC_SCHEMA_VERSION:
        template = VERIFIER_ANSWER_PROCESS_PROMPT_TEMPLATE
        rubric_payload = {
            "answer_rubrics": rubric.get("answer_rubrics", []),
            "process_rubrics": rubric.get("process_rubrics", []),
            "maximum_scores": rubric.get("maximum_scores", {}),
            "score_policy": rubric.get("score_policy", {}),
        }
        answer_label = "Trajectory"
    else:
        template = VERIFIER_PROMPT_TEMPLATE
        rubric_payload = rubric["rubrics"]
        answer_label = "Answer"
    return _render_template(
        template,
        {
            "question": _limit_reward_text(args, _ropd_question(sample), question_max_chars, field="question"),
            "rubrics": json.dumps(rubric_payload, ensure_ascii=False, indent=2),
            "answers": _render_answer_block(
                answer_label,
                [_limit_reward_text(args, answer, answer_max_chars, field="verifier_answer") for answer in answers],
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


def _score_0_to_5(value: Any) -> float:
    score = float_value(value, 0.0)
    return max(0.0, min(5.0, score))


def _score_item(raw: Any, *, criterion_id: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {
            "criterion_id": str(raw.get("criterion_id") or criterion_id),
            "score": _score_0_to_5(raw.get("score", raw.get("value", 0.0))),
            "rationale": str(raw.get("rationale") or raw.get("reason") or "").strip(),
        }
    raise ValueError("ROPD answer-process score item must be an object")


def _weighted_average_0_to_5(scores: list[float], items: list[dict[str, Any]]) -> float:
    if not scores or not items:
        return 0.0
    weights = [_positive_float(item.get("weight", 1.0), 1.0) for item in items]
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    return sum(score * weight for score, weight in zip(scores, weights, strict=True)) / total_weight


def _parse_binary_batch_scores(payload: Any, *, rubric: dict[str, Any], expected: int) -> list[dict[str, Any]]:
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


def _parse_answer_process_batch_scores(
    payload: Any,
    *,
    rubric: dict[str, Any],
    expected: int,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("ROPD answer-process verifier response must be a JSON object")
    if payload.get("schema_version") != ANSWER_PROCESS_VERIFIER_SCHEMA_VERSION:
        raise ValueError("ROPD answer-process verifier schema_version mismatch")
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != expected:
        raise ValueError(
            f"ROPD answer-process verifier returned "
            f"{0 if not isinstance(answers, list) else len(answers)} scores for {expected} answers"
        )
    answer_items = list(rubric.get("answer_rubrics", []))
    process_items = list(rubric.get("process_rubrics", []))
    answer_ids = [str(item.get("criterion_id")) for item in answer_items]
    process_ids = [str(item.get("criterion_id")) for item in process_items]
    scores: list[dict[str, Any]] = []
    for expected_index, item in enumerate(answers, start=1):
        if not isinstance(item, dict):
            raise ValueError("ROPD answer-process answer item must be an object")
        if int(item.get("answer_index", -1)) != expected_index:
            raise ValueError("ROPD answer-process answer_index must cover 1..n in order")
        answer_scores_raw = item.get("answer_scores")
        process_scores_raw = item.get("process_scores")
        if not isinstance(answer_scores_raw, list) or len(answer_scores_raw) != len(answer_items):
            raise ValueError("ROPD answer-process answer_scores length mismatch")
        if not isinstance(process_scores_raw, list) or len(process_scores_raw) != len(process_items):
            raise ValueError("ROPD answer-process process_scores length mismatch")

        answer_score_items = [
            _score_item(raw, criterion_id=criterion_id)
            for raw, criterion_id in zip(answer_scores_raw, answer_ids, strict=True)
        ]
        process_score_items = [
            _score_item(raw, criterion_id=criterion_id)
            for raw, criterion_id in zip(process_scores_raw, process_ids, strict=True)
        ]
        for score_item, criterion_id in zip(answer_score_items, answer_ids, strict=True):
            if score_item["criterion_id"] != criterion_id:
                raise ValueError("ROPD answer-process answer criterion_id mismatch")
        for score_item, criterion_id in zip(process_score_items, process_ids, strict=True):
            if score_item["criterion_id"] != criterion_id:
                raise ValueError("ROPD answer-process process criterion_id mismatch")

        evidence_raw = item.get("process_step_evidence", [])
        if evidence_raw is None:
            evidence_raw = []
        if evidence_raw and (not isinstance(evidence_raw, list) or len(evidence_raw) != len(process_items)):
            raise ValueError("ROPD answer-process process_step_evidence length mismatch")
        evidence_items: list[dict[str, Any]] = []
        if evidence_raw:
            for raw_evidence, criterion_id, process_score in zip(
                evidence_raw, process_ids, process_score_items, strict=True
            ):
                if not isinstance(raw_evidence, dict):
                    raise ValueError("ROPD answer-process evidence item must be an object")
                if str(raw_evidence.get("criterion_id", criterion_id)) != criterion_id:
                    raise ValueError("ROPD answer-process evidence criterion_id mismatch")
                raw_steps = raw_evidence.get("step_indices", [])
                if not isinstance(raw_steps, list):
                    raise ValueError("ROPD answer-process step_indices must be a list")
                step_indices = [int(step) for step in raw_steps if isinstance(step, int) or str(step).isdigit()]
                if process_score["score"] <= 0 and step_indices:
                    raise ValueError("ROPD answer-process zero process score cannot have step evidence")
                evidence_items.append(
                    {
                        "criterion_id": criterion_id,
                        "satisfied": bool(raw_evidence.get("satisfied", process_score["score"] > 0)),
                        "score": float(process_score["score"]),
                        "step_indices": step_indices,
                        "evidence": str(raw_evidence.get("evidence") or "").strip(),
                    }
                )

        answer_values = [float(score_item["score"]) for score_item in answer_score_items]
        process_values = [float(score_item["score"]) for score_item in process_score_items]
        answer_core_score = answer_values[0] if answer_values else 0.0
        answer_support_score = _weighted_average_0_to_5(answer_values[1:], answer_items[1:])
        answer_score = answer_core_score + answer_support_score
        process_score = _weighted_average_0_to_5(process_values, process_items)
        final_answer_quality = str(item.get("final_answer_quality") or "").strip().lower()
        if final_answer_quality not in {"correct", "mostly_correct", "partial", "wrong", "no_answer"}:
            raise ValueError("ROPD answer-process final_answer_quality mismatch")
        if "fatal_error" not in item:
            raise ValueError("ROPD answer-process fatal_error is required")
        scores.append(
            {
                "answer_index": expected_index,
                "answer_scores": answer_score_items,
                "process_scores": process_score_items,
                "process_step_evidence": evidence_items,
                "answer_core_score": answer_core_score,
                "answer_support_score": answer_support_score,
                "answer_score": answer_score,
                "process_score": process_score,
                "final_score": answer_score,
                "final_answer_quality": final_answer_quality,
                "fatal_error": bool(item.get("fatal_error", False)),
            }
        )
    return scores


def _parse_batch_scores(payload: Any, *, rubric: dict[str, Any], expected: int) -> list[dict[str, Any]]:
    if rubric.get("schema_version") == ANSWER_PROCESS_RUBRIC_SCHEMA_VERSION:
        return _parse_answer_process_batch_scores(payload, rubric=rubric, expected=expected)
    return _parse_binary_batch_scores(payload, rubric=rubric, expected=expected)


def _existing_rubric(args: Any, sample: Sample) -> Any:
    return _configured_metadata_value(
        args,
        sample,
        "rubric_keys",
        ("rubric", "reward_rubric", "ropd_rubric"),
    )


def _weight(args: Any) -> float:
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
    return _cfg_choice(args, "luffy_mode", "off", {"off", "reward_anchor", "token_loss"})


def _luffy_enabled(args: Any) -> bool:
    return _cfg_bool(args, "luffy_enable", False)


def _reward_group_reference(args: Any) -> str:
    default_reference = "teacher_plus_students" if _luffy_enabled(args) and _luffy_mode(args) == "reward_anchor" else "students"
    return _cfg_choice(
        args,
        "reward_group_reference",
        default_reference,
        {"students", "teacher_plus_students"},
    )


def _reward_mode(args: Any) -> str:
    return _cfg_choice(
        args,
        "reward_mode",
        "answer_only",
        {"answer_only", "group_centered", "group_zscore"},
    )


def _validate_reward_config(args: Any) -> None:
    if _luffy_enabled(args) and _luffy_mode(args) == "token_loss":
        raise RuntimeError(
            "ropd.luffy_mode=token_loss requires actor-side off-policy teacher-token loss integration. "
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
    existing_rubric = _normalize_rubric_for_mode(args, existing)
    if existing_rubric is not None:
        return existing_rubric, "teacher", None, teacher_answers

    allow_online = bool_value(_cfg(args, "allow_online_rubric", False), False)
    if not allow_online or judge_mode(args) != "aux":
        return None, "missing", None, teacher_answers

    prompt = _build_rubricator_prompt(args, samples, teacher_answers)
    try:
        payload, call_metadata = await call_json_judge_with_metadata(
            args,
            prompt,
            system_prompt=RUBRIC_SYSTEM_PROMPT,
            api_key_path=_role_api_key_path(args, "rubric"),
            **_role_request_options(args, "rubric", default_max_tokens=32768),
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
                "dump_step": _sample_rollout_label(sample),
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
    rubric = _normalize_rubric_for_mode(args, payload)
    _dump_artifact(
        args,
        "rubricator",
        {
            "dump_step": _sample_rollout_label(sample),
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
    discard_reason = f"ropd_{reason}"
    raw = {
        "fallback": reason,
        "remove_sample": True,
        "discard_reason": discard_reason,
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
        reason=discard_reason,
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
    teacher_below_any_student: bool,
    teacher_reference_score: float | None,
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
    trace_options = _trace_options(args)
    weighted = train_score * _weight(args)
    raw = {
        "rubric": rubric,
        "rubric_hash": _rubric_hash(rubric),
        "ropd_schema_mode": _schema_mode(args),
        "rubric_source": rubric_source,
        "judge": student_item,
        "student_score": float(student_score),
        "teacher_scores": [float(score) for score in teacher_scores],
        "maximum_score": float(maximum_score),
        "reward_score": float(train_score),
        "answer_score": float(bounded),
        "student_answer_position": int(student_position),
        "teacher_below_student": bool(teacher_below_student),
        "group_teacher_below_any_student": bool(teacher_below_any_student),
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
        "strip_reasoning": trace_options.strip_reasoning,
        "strip_tool_response": trace_options.strip_tool_response,
        "strip_assistant_response": trace_options.strip_assistant_response,
        "strip_system_prompt": trace_options.strip_system_prompt,
    }
    for key in (
        "answer_core_score",
        "answer_support_score",
        "process_score",
        "answer_scores",
        "process_scores",
        "process_step_evidence",
        "final_answer_quality",
        "fatal_error",
    ):
        if key in student_item:
            raw[key] = student_item[key]
    if teacher_reference_score is not None:
        raw["teacher_reference_score"] = float(teacher_reference_score)
    if rubric_call is not None:
        raw["rubric_call"] = rubric_call
    if judge_call is not None:
        raw["judge_call"] = judge_call
    return RewardResult(
        score=weighted,
        components={
            "rubric_task_success": weighted,
            "ropd_answer_score": bounded,
        },
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
            **_role_request_options(args, "judge", default_max_tokens=32768),
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
                "dump_step": _sample_rollout_label(samples[0]),
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
    teacher_reference_score = min(teacher_scores) if teacher_scores else None
    student_above_teacher_flags = [
        bool(teacher_reference_score is not None and item is not None and float(item[0]) > teacher_reference_score)
        for item in student_scores_by_index
    ]
    teacher_below_any_student = any(student_above_teacher_flags)
    _dump_artifact(
        args,
        "verifier",
        {
            "dump_step": _sample_rollout_label(samples[0]),
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
            "teacher_reference_score": teacher_reference_score,
            "student_above_teacher_flags": student_above_teacher_flags,
            "group_teacher_below_any_student": teacher_below_any_student,
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
                teacher_below_student=student_above_teacher_flags[idx],
                teacher_below_any_student=teacher_below_any_student,
                teacher_reference_score=teacher_reference_score,
            )
        )
    return results


async def score(args: Any, samples: list[Sample], *, single: bool = False) -> list[RewardResult]:
    if judge_mode(args) != "aux":
        raise RuntimeError("reward.impl=ropd requires reward.judge_mode=aux")
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
            reason = "missing_teacher" if not teacher_answers or rubric_source == "missing_teacher" else "missing_rubric"
            for idx in indices:
                results[idx] = _fallback_result(args, samples[idx], reason, rubric_source)
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
