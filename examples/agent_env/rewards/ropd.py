from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from examples.agent_env import credit_assignment
from examples.agent_env.appworld.reward_evidence import parse_execution_evidence_trace
from examples.agent_env.dump import record_dump_step_label, reserve_dump_slot, sample_dump_step_label
from examples.agent_env.trace_rendering import (
    TraceCompressionOptions,
    compress_trace_text,
    render_answer_for_reward,
    render_teacher_trace_for_reward,
    resolve_trace_env_name,
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

logger = logging.getLogger(__name__)

RUBRIC_SCHEMA_VERSION = "ropd.rubric.v1"
BATCH_VERIFIER_SCHEMA_VERSION = "ropd.batch_verifier.v2"
ANSWER_PROCESS_RUBRIC_SCHEMA_VERSION = "ropd.answer_process_rubric.v1"
ANSWER_PROCESS_VERIFIER_SCHEMA_VERSION = "ropd.answer_process_compact_batch_verifier.v1"
RUBRIC_SHAPING_RUBRIC_SCHEMA_VERSION = "ropd.rubric_shaping_rubric.v1"
RUBRIC_SHAPING_VERIFIER_SCHEMA_VERSION = "ropd.rubric_shaping_batch_verifier.v1"
CA_COMPACT_RUBRIC_SCHEMA_VERSION = "ropd.ca_compact_rubric.v2"
CA_COMPACT_VERIFIER_SCHEMA_VERSION = "ropd.ca_compact_batch_verifier.v2"
TASA_STATE_VERIFIER_SCHEMA_VERSION = "ropd.tasa_state_batch_verifier.v1"
TEACHER_TRACE_FIELDS = (
    "teacher_reward_trace_payload",
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
CA_JUDGE_SYSTEM_PROMPT = "你是一名 agentic trajectory credit-assignment judge。输出必须是单个 JSON 对象。"

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

RUBRICATOR_SHAPING_PROMPT_TEMPLATE = """你是一名 agentic trajectory 评估专家。你的任务是为同一道任务生成一套共享 trajectory-quality rubric。

这套 rubric 只用于给未成功轨迹提供弱 shaping 信号；任务最终是否成功由外部环境 verifier 的 env_success 决定，不由你判断。因此不要生成“最终答案正确”“任务已经完成”“complete_task 调用成功”等核心正确性 rubric。

# 输入数据
[Question]
{question}

[Teacher Trajectory]
{teacher_response}

[Student Trajectory]
{student_response}

[Additional Instructions]
{extra_rubric_instructions}

# rubric 目标
请生成 2 到 6 条可观察、可打分、可泛化的过程质量标准。优先覆盖：
- 是否选择了与任务相关的工具/API/action。
- 是否使用了工具返回的可见证据，而不是编造结果。
- 是否遵守任务约束、用户偏好和环境反馈。
- 是否能在失败、空结果、报错或不确定时做合理修正。
- 是否避免重复无效动作、无关探索和过早宣布完成。

# 禁止项
- 不要奖励轨迹仅仅声称任务完成。
- 不要把 teacher 的具体文本、具体参数或具体中间值写成唯一正确答案。
- 不要把格式整洁、篇幅长、礼貌表达作为主要得分点。
- 不要要求 verifier 访问外部知识或环境真值；只能基于给定 trajectory 打分。

# 输出格式
返回一个 JSON 对象，结构如下：
```json
{
  "schema_version": "ropd.rubric_shaping_rubric.v1",
  "rubrics": [
    {
      "criterion_id": "r1",
      "category": "Evidence-Grounded Tool Use",
      "criterion": "The trajectory uses visible tool/API/action observations to justify the next step instead of inventing unavailable results.",
      "max_score": 5,
      "weight": 1.0
    }
  ],
  "score_policy": {
    "score_range": "0_to_5_per_criterion",
    "env_success_overrides_reward": true,
    "failure_reward_scale_beta": {shaping_beta}
  }
}
```

# 输出约束
- `schema_version` 必须严格等于 `ropd.rubric_shaping_rubric.v1`。
- `rubrics[].criterion_id` 必须为 `r1`, `r2`, ...
- 每条 rubric 的 `max_score` 必须等于 5。
- 每条 rubric 的 `weight` 必须是正数。
- 只返回 JSON 对象本身，不要输出 Markdown 或解释。
"""

VERIFIER_SHAPING_PROMPT_TEMPLATE = """你是一名 agentic trajectory 评分专家。你的任务是基于共享 rubric 给多个匿名 trajectory 打分。

任务最终是否成功由外部环境 verifier 的 env_success 决定，不由你判断。你只评价 trajectory 的可观察过程质量，用于给失败轨迹提供弱 shaping 信号。

[Question]
{question}

[Rubrics]
{rubrics}

[Trajectories]
{answers}

[Additional Scoring Instructions]
{extra_scoring_instructions}
{process_step_evidence_instructions}

# 打分规则
- 对每条 rubric 给 0 到 5 分：
  - 5：完全满足。
  - 4：基本满足，只有轻微遗漏。
  - 3：部分满足，但缺少重要细节。
  - 2：只有少量相关内容。
  - 1：极弱相关或基本不可用。
  - 0：不满足。
- 不要因为 trajectory 调用了 `complete_task`、声明任务完成、或工具调用本身返回 execution successful 就给高分；这些只说明动作被执行，不说明任务成功。
- 如果 trajectory 编造工具结果、跳过必要证据、使用明显错误的工具/API/action，相关 rubric 应低分。
- 如果 trajectory 为空、严重损坏、无法看出任何有效动作，`fatal_error=true` 且总分为 0。
- 每条 trajectory 独立评分，不要互相比高低。

# 输出格式
返回一个 JSON 对象：
```json
{
  "schema_version": "ropd.rubric_shaping_batch_verifier.v1",
  "answers": [
    {
      "answer_index": 1,
      "rubric_scores": [
        {"criterion_id": "r1", "score": 4, "rationale": "short reason"}
      ],
      "trajectory_quality": "partial",
      "fatal_error": false
    }
  ]
}
```

# 输出约束
- `schema_version` 必须严格等于 `ropd.rubric_shaping_batch_verifier.v1`。
- `answers` 必须按输入顺序覆盖所有 trajectory。
- `rubric_scores` 长度和顺序必须与 rubric 完全一致。
- 如果要求输出 `process_step_evidence`，其长度和顺序必须与 rubric 完全一致；step 编号必须使用 trajectory 文本里的 `Step N` 编号。
- 所有 `score` 必须是 0 到 5 的数字。
- `trajectory_quality` 必须是 `strong`, `useful`, `partial`, `weak`, `invalid` 之一。
- 只返回 JSON 对象本身。
"""

CA_COMPACT_VERIFIER_PROMPT_TEMPLATE = """你是一名 agentic trajectory credit-assignment judge。请一次性完成 behavior mining 和 student step-index 标注。

# 任务
1. 根据任务、参考轨迹和 student trajectories，提炼少量可迁移的关键进展 good behavior 与明确错误 bad behavior。
2. 为每条 student trajectory 标出命中各项 behavior 的 Step 编号。

# 判定口径
- 最终任务是否成功由外部 env verifier 判断；本输出只描述可观察的过程行为及其 student 命中位置。
- good hit 表示由可见 action、tool call 或 tool response 直接确认的关键任务进展或 milestone。
- bad hit 表示由可见过程直接确认的错误调用、编造结果、忽略 observation、重复无效动作、过早终止或参数错误。
- 终止调用、状态文字和意图陈述只作为上下文；它们需要可见的任务状态变化或执行证据才能构成 good hit。
- 被压缩或省略的内容不构成命中证据；对应 `step_indices` 保持为空。
- 每个 hit 都必须对应输入文本中实际存在的 `Step N`。

[Question]
{question}

[Reference Trajectories]
{references}

[Student Trajectories]
{students}

[Student Trajectory IDs]
{student_trajectory_ids}

{extra_scoring_section}

# 输出格式
返回一个 JSON object：
```json
{
  "schema_version": "ropd.ca_compact_batch_verifier.v2",
  "behaviors": [
    {
      "behavior_id": "g1",
      "polarity": "good",
      "description": "one concise observable behavior",
      "hits_by_trajectory": [
        {"trajectory_id": "S0_STUDENT", "step_indices": [1, 3]},
        {"trajectory_id": "S1_STUDENT", "step_indices": []}
      ]
    },
    {
      "behavior_id": "b1",
      "polarity": "bad",
      "description": "one concise observable failure behavior",
      "hits_by_trajectory": [
        {"trajectory_id": "S0_STUDENT", "step_indices": [2]},
        {"trajectory_id": "S1_STUDENT", "step_indices": []}
      ]
    }
  ]
}
```

# 输出约束
- `schema_version` 必须严格等于 `ropd.ca_compact_batch_verifier.v2`。
- 生成 3 到 8 条 behavior，good 和 bad 都至少 1 条。
- `behavior_id` 必须以 `g` 或 `b` 开头并唯一；`polarity` 只能是 `good` 或 `bad`。
- 每条 `hits_by_trajectory` 必须完整覆盖 [Student Trajectory IDs]，且顺序与该列表一致。
- Step 编号只能来自对应 trajectory 文本里的 `Step N`。
- 输出字段只能是 schema 中出现的字段。
"""

TASA_STATE_VERIFIER_PROMPT_TEMPLATE = """你是一名 agentic trajectory semantic-state judge。请基于任务、参考轨迹和所有 student trajectories，一次性定义少量可复用 milestone state，并标注每条 student 的状态建立与退化位置。

# 核心语义
- milestone 是执行某个 action 后成立的、可由可见执行证据确认的任务状态事实，不是 action 文本，也不是 agent 的意图或自我声明。
- milestone 集合只描述 reference 成功路径中实际成立的正向任务进展，并作为从 root 到完整成功状态的坐标系。
- 最高 progress 的最后一个 milestone 必须描述任务完整成功，而不是部分完成、尝试执行或单个子目标完成。
- 错误目标、违规操作、无效尝试和其它负向事实不定义为 milestone；它们只有在破坏已有正向状态时才通过对应 milestone 的 `unset_step` 表示。
- `set_step=t` 表示执行完 `Step t` 的 action 后，该 milestone 从 false 变为 true。
- `unset_step=t` 表示执行完 `Step t` 的 action 后，该 milestone 从 true 退化为 false。
- 同一 milestone 可以在一条 trajectory 中多次 set/unset。
- student 可以采用与 reference 不同的路径；只要达成同一状态事实，就应命中同一 milestone。
- 终止调用、`Execution successful`、格式整洁或 agent 声称完成任务都不能单独证明 milestone 成立。
- 只能依据该 step 中可见的结构化执行证据判断状态变化；代码文本、计划、注释和预期结果都不是执行证据。执行报错或证据缺失时不得 set。
- predicate 若声称覆盖“全部”“目标集合”或完整核验，抽查单个/部分对象不足以 set；必须有可见结果证明所需范围已完整覆盖。
- `set_steps` 只记录 false→true 的首次转变；milestone 保持为 true 时不得在后续 step 重复 set，除非中间先有对应 unset。
- 被压缩或省略的内容不能作为状态证据。

# Milestone schema
- 生成 3 到 8 个 milestone，id 严格使用 `M1`, `M2`, ...。
- `predicate` 用一句简洁、可观察、可跨轨迹复用的状态事实表达。
- `requires` 字段必须显式出现；没有依赖时写 `[]`。
- `progress` 是 reference 路径中的正整数粗粒度顺序，不是成功概率。
- 高阶 milestone 的 `requires` 必须引用更低 progress 的 milestone。
- 每个 milestone 都必须在至少一条 reference trajectory 中由可见执行证据建立；student 中反复出现但 reference 未建立的失败模式不能进入 schema。

[Question]
{question}

[Reference Trajectories]
{references}

[Student Trajectories]
{students}

[Student Trajectory IDs]
{student_trajectory_ids}

{extra_scoring_section}

# 输出格式
只返回如下 JSON object：
```json
{
  "schema_version": "ropd.tasa_state_batch_verifier.v1",
  "milestones": [
    {
      "id": "M1",
      "predicate": "正确目标用户已经由可见 API 结果确认",
      "requires": [],
      "progress": 1,
      "transitions_by_trajectory": [
        {"trajectory_id": "S0_STUDENT", "set_steps": [2], "unset_steps": []},
        {"trajectory_id": "S1_STUDENT", "set_steps": [], "unset_steps": []}
      ]
    }
  ]
}
```

# 严格约束
- `schema_version` 必须严格等于 `ropd.tasa_state_batch_verifier.v1`。
- 顶层字段只能有 `schema_version` 和 `milestones`。
- 每个 milestone 的字段必须完整且只能是示例中的五项。
- 每个 `transitions_by_trajectory` 必须按 [Student Trajectory IDs] 顺序完整覆盖所有 student。
- `set_steps`/`unset_steps` 只能包含对应 trajectory 文本里实际存在的 `Step N`。
- 同一 milestone 在同一个 step 不能同时 set 和 unset。
"""

RUBRICATOR_ANSWER_PROCESS_PROMPT_TEMPLATE = """你是一名 agentic task 评估专家。你的任务是为同一道业务问题生成一套 answer-first 的共享评分细则。

输入包含：
- teacher trajectory：高置信度参考轨迹，但不保证绝对正确，也不是唯一解法。
- student trajectory：待评估轨迹，可能暴露当前模型的质量缺口。

请生成两组 rubric：
1. answer_rubrics：评价最终回答质量。`a1` 必须是核心结论正确性，后续 `a2...` 评价最终回答里的证据、关键细节、边界披露和风险说明。
2. process_rubrics：评价过程、工具、证据链、取证可靠性和错误处理。该分数会按配置进入最终 reward，同时保留为 trace 内部信用分配和诊断信号。

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

最终 answer reward 的计算方式由配置控制：
- `core_score = a1.score`，范围 0 到 5。
- `support_score = weighted_average(a2..an scores)`，范围 0 到 5。
- `answer_support_weight_for_reward = {answer_support_weight_for_reward}`。
- `answer_reward = 2 * (core_score + answer_support_weight_for_reward * support_score) / (1 + answer_support_weight_for_reward)`，范围 0 到 10。
- 当 `answer_support_weight_for_reward=0` 时，`a2...` 只保留为诊断信号，不进入训练 reward。

process score 独立计算为 `weighted_average(p1..pn scores)`，范围 0 到 5。

最终 reward points 的计算方式固定为：
- `final_reward_points = {answer_weight} * answer_reward + {process_weight} * (process_score * 2)`。
- `process_score * 2` 是为了把 process 从 0 到 5 映射到 0 到 10 后再和 answer 对齐。
- `final_reward_points` 范围仍为 0 到 10。

# 核心原则
- `a1` 必须只评价最终回答是否解决用户核心问题和核心结论是否正确。它允许 0 到 5 的部分分：例如多对象任务里答对部分对象、结论方向正确但缺关键限定、或只完成部分子问题。
- `a2...` 只评价最终回答中的必要证据、关键细节、置信边界、不可达信息披露和风险说明。不要把工具路径本身写进 answer rubric。
- process_rubrics 只评价过程中是否正确选择 skill/tool、是否基于可见证据推理、是否处理工具错误、是否避免编造工具结果、是否能定位到具体 step。
- 如果最终回答没有回答用户核心问题、对象类型错误、关键事实错误、编造结果、只有过程没有结论、或没有最终答案，`a1` 应接近 0。
- 只有当题目本身允许“无法完成/证据不足”作为合格答案，或 trajectory 已经展示出外部权限、数据不可达、服务限制等非 agent 自身造成的真实阻断时，`a1` 才应奖励明确披露阻断点且不编造业务结论。对于有明确可执行解法的任务，格式错误、变量错误、鉴权遗漏、工具选错、参数错误、跳过必要取证、主动放弃或只调用任务结束/提交接口，都不是合格答案，`a1` 应接近 0。
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
    "answer_core_weight": {answer_core_weight_for_reward},
    "answer_support_weight": {answer_support_weight_for_reward_normalized},
    "answer_support_weight_for_reward": {answer_support_weight_for_reward},
    "answer_weight": {answer_weight},
    "process_weight": {process_weight},
    "process_for_credit_assignment": true,
    "final_reward_uses": "{final_reward_uses}",
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

# Answer/Process reward 规则
- 最终 reward 同时使用 answer 和 process：`final_reward_points = {answer_weight} * answer_reward + {process_weight} * (process_score * 2)`，范围仍为 0 到 10。
- `answer_reward = 2 * (a1.score + {answer_support_weight_for_reward} * support_score) / (1 + {answer_support_weight_for_reward})`。当该 support 权重为 0 时，`a2...` 只作为诊断，不进入训练 reward。
- `a1` 是核心结论正确性，必须按最终回答的核心结论给 0 到 5 分。多对象、多子问题任务可以给部分分；没有最终答案、编造核心事实、对象错配或答非所问应接近 0。
- `a2...` 评价最终回答的证据、关键细节、边界披露和风险说明。它们不应替代 `a1`，也不能因为过程看起来努力就抬高核心结论分。
- 只有当题目本身允许“无法完成/证据不足”作为合格答案，或 trajectory 已经展示出外部权限、数据不可达、服务限制等非 agent 自身造成的真实阻断时，明确披露阻断点且不编造业务结论才可以获得相应 answer 分。对于有明确可执行解法的任务，格式错误、变量错误、鉴权遗漏、工具选错、参数错误、跳过必要取证、主动放弃或只调用任务结束/提交接口，都不是合格 answer，`a1` 应接近 0，并可设 `fatal_error=true`。

# Process 规则
- process_rubrics 参与当前 final reward，但只能评价可见过程证据，不能替代 answer correctness。
- process_rubrics 评价工具/skill 选择、证据链、错误处理、是否基于可见证据、是否避免编造工具结果。
- 如果 trajectory 声称查过工具，但没有可见证据支撑，不能给高 process 分。
- 如果工具失败但 trajectory 明确披露了非自身造成的真实阻断点且没有编造结论，可以给对应 process 分，但不能因此自动给 answer 分。由 agent 自身错误造成的失败不应因“披露失败”获得高 process 或 answer 分。

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


def _cfg_trace_part_spec(args: Any, cfg_name: str, default: bool | int) -> bool | int:
    value = _cfg(args, cfg_name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"ropd.{cfg_name} must be a boolean or positive integer character limit")
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"ropd.{cfg_name} must be a boolean or positive integer character limit") from exc
    if parsed <= 0:
        raise ValueError(f"ropd.{cfg_name} must be a boolean or positive integer character limit")
    return parsed


def _cfg_int(args: Any, cfg_name: str, default: int) -> int:
    return int_value(_cfg(args, cfg_name, default), default)


def _cfg_choice(args: Any, cfg_name: str, default: str, choices: set[str]) -> str:
    value = str(_cfg(args, cfg_name, default) or default).strip().lower()
    return value if value in choices else default


def _schema_mode(args: Any) -> str:
    value = str(_cfg(args, "schema_mode", "binary") or "binary").strip().lower()
    choices = {"binary", "answer_process_50_50", "rubric_shaping", "ca_compact"}
    if value not in choices:
        raise ValueError(f"Unsupported reward.ropd.schema_mode={value!r}; expected one of {sorted(choices)}")
    return value


def _rubric_shaping_beta(args: Any) -> float:
    beta = float_value(_cfg(args, "shaping_beta", 0.2), 0.2)
    if beta < 0:
        raise ValueError("reward.ropd.shaping_beta must be non-negative")
    return beta


def _failure_shaping_beta(args: Any) -> float:
    beta = float_value(_cfg(args, "failure_shaping_beta", _cfg(args, "shaping_beta", 1.0)), 1.0)
    if beta < 0:
        raise ValueError("reward.ropd.failure_shaping_beta must be non-negative")
    return beta


def _env_success_overrides_reward(args: Any) -> bool:
    return _cfg_bool(args, "env_success_overrides_reward", False)


def _answer_process_weights(args: Any) -> tuple[float, float]:
    answer_weight = float_value(_cfg(args, "answer_weight", 1.0), 1.0)
    process_weight = float_value(_cfg(args, "process_weight", 0.0), 0.0)
    if answer_weight < 0 or process_weight < 0:
        raise ValueError("reward.ropd.answer_weight and reward.ropd.process_weight must be non-negative")
    total = answer_weight + process_weight
    if total <= 0:
        raise ValueError("reward.ropd.answer_weight + reward.ropd.process_weight must be positive")
    return answer_weight / total, process_weight / total


def _answer_support_weight_for_reward(args: Any) -> float:
    weight = float_value(_cfg(args, "answer_support_weight_for_reward", 1.0), 1.0)
    if weight < 0:
        raise ValueError("reward.ropd.answer_support_weight_for_reward must be non-negative")
    return weight


def _answer_process_final_reward_uses(args: Any) -> str:
    _, process_weight = _answer_process_weights(args)
    return "answer_process_weighted" if process_weight > 0 else "answer_only"


def _answer_process_score_policy(args: Any) -> dict[str, Any]:
    answer_weight, process_weight = _answer_process_weights(args)
    support_weight = _answer_support_weight_for_reward(args)
    answer_norm = 1.0 + support_weight
    return {
        "answer_first": True,
        "answer_core_weight": 1.0 / answer_norm,
        "answer_support_weight": support_weight / answer_norm,
        "answer_support_weight_for_reward": support_weight,
        "answer_weight": answer_weight,
        "process_weight": process_weight,
        "process_for_credit_assignment": True,
        "final_reward_uses": _answer_process_final_reward_uses(args),
        "rubric_score_range": "0_to_5_per_criterion",
    }


def _answer_process_answer_score(args: Any, *, core_score: float, support_score: float) -> float:
    support_weight = _answer_support_weight_for_reward(args)
    raw = float(core_score) + support_weight * float(support_score)
    return max(0.0, min(10.0, 2.0 * raw / (1.0 + support_weight)))


def _answer_process_final_score(args: Any, *, answer_score: float, process_score: float) -> float:
    answer_weight, process_weight = _answer_process_weights(args)
    process_score_scaled = max(0.0, min(10.0, process_score * 2.0))
    answer_score_bounded = max(0.0, min(10.0, answer_score))
    return max(0.0, min(10.0, answer_weight * answer_score_bounded + process_weight * process_score_scaled))


def _list_value(value: Any, default: tuple[str, ...] = ()) -> list[str]:
    if value in (None, "", []):
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _metadata_field_values(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
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


def _role_endpoint_pool_path(args: Any, role: str) -> str | None:
    upper = role.upper()
    for key in (f"AUX_{upper}_POOL_PATH", f"AUX_{upper}_ENDPOINT_POOL_PATH", "AUX_ENDPOINT_POOL_PATH"):
        path = runtime_env(args, key, "").strip()
        if path:
            return str(resolve_path(args, path))
    value = _role_cfg(args, role, "endpoint_pool_path", None)
    if value:
        return str(resolve_path(args, value))
    return None


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
    candidates = (key, "id", "task_index") if key == "task_id" else (key,)
    for candidate in candidates:
        value = sample_metadata.get(candidate)
        if value not in (None, "", []):
            return str(value)
    if key != "task_id":
        raise ValueError(f"ROPD join_key={key!r} is missing from sample metadata; refusing to fall back to task_id")
    return _cache_key(sample)


def _configured_metadata_value(args: Any, sample: Sample, cfg_name: str, default_keys: tuple[str, ...]) -> Any:
    keys = _list_value(_cfg(args, cfg_name, None), default_keys)
    return _metadata_value(sample, keys)


def _teacher_index_path(args: Any) -> Path | None:
    raw = str(runtime_env(args, "AGENT_ENV_ROPD_TEACHER_INDEX_PATH", "") or _cfg(args, "teacher_index_path", "")).strip()
    return resolve_path(args, raw) if raw else None


def _candidate_values_from_mapping(mapping: dict[str, Any], keys: list[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = mapping.get(key)
        if value in (None, "", []):
            continue
        for item in _metadata_field_values(value):
            item_text = str(item).strip()
            if not item_text:
                continue
            values.append(item_text)
            normalized = re.sub(r"\s+", " ", item_text).lower()
            if normalized and normalized != item_text:
                values.append(normalized)
    return values


def _teacher_index_key_candidates(args: Any, sample: Sample) -> list[str]:
    sample_metadata = metadata(sample)
    configured_keys = _cfg(args, "teacher_index_keys", None)
    keys = _list_value(
        configured_keys,
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
    if configured_keys in (None, "", []):
        sample_index = getattr(sample, "index", None)
        if sample_index is not None:
            candidates.append(str(sample_index))
    join_value = _join_value(args, sample)
    if join_value:
        candidates.append(join_value)
    return list(dict.fromkeys(value for value in candidates if value))


def _first_mapping_value(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value in (None, "", []):
            continue
        return value
    return None


def _teacher_index_row_answer(args: Any, row: dict[str, Any]) -> Any:
    keys = _list_value(_cfg(args, "teacher_answer_keys", None), TEACHER_TRACE_FIELDS)
    for source in (row, row.get("evidence"), row.get("adapter_result"), row.get("source_row")):
        if isinstance(source, dict):
            value = _first_mapping_value(source, keys)
            if value not in (None, "", []):
                return value
    return None


def _teacher_index_row_status(row: dict[str, Any], *, has_text: bool) -> str:
    for source in (row, row.get("evidence"), row.get("adapter_result")):
        if isinstance(source, dict):
            if source.get("can_be_teacher") is False or source.get("teacher_eligible") is False:
                return "not_teacher"
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
        strip_reasoning=_cfg_trace_part_spec(args, "strip_reasoning", True),
        strip_tool_call=_cfg_trace_part_spec(args, "strip_tool_call", False),
        strip_tool_response=_cfg_trace_part_spec(args, "strip_tool_response", False),
        strip_assistant_response=_cfg_trace_part_spec(args, "strip_assistant_response", True),
        strip_system_prompt=_cfg_trace_part_spec(args, "strip_system_prompt", True),
    )


def _trace_env_name_from_args(args: Any) -> str:
    for key in ("env_name", "agent_env_name", "environment"):
        value = str(getattr(args, key, "") or "").strip().lower()
        if value:
            return value
    return str(os.environ.get("ENV_NAME", "") or os.environ.get("AGENT_ENV_NAME", "")).strip().lower()


def _sanitize_teacher_answer_for_anonymous_verifier(args: Any, answer: Any, *, sample: Sample) -> str:
    env_name = resolve_trace_env_name(sample, explicit_env_name=_trace_env_name_from_args(args))
    text = render_teacher_trace_for_reward(
        answer,
        options=_trace_options(args),
        env_name=env_name,
        context_metadata=metadata(sample),
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
    _require_appworld_execution_evidence(args, text, role="teacher", env_name=env_name)
    return text


def _require_appworld_execution_evidence(args: Any, text: str, *, role: str, env_name: str) -> None:
    if env_name != "appworld" or _answer_mode(args) != "trace":
        return
    try:
        parse_execution_evidence_trace(text)
    except ValueError as exc:
        raise ValueError(
            f"AppWorld ROPD {role} trace lacks valid structured execution evidence; "
            "legacy raw-code/Execution successful traces are not accepted"
        ) from exc


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
    env_name = resolve_trace_env_name(sample, explicit_env_name=_trace_env_name_from_args(args))
    text = render_answer_for_reward(
        sample,
        final_answer=_explicit_final_answer(sample),
        answer_mode=mode,
        options=_trace_options(args),
        env_name=env_name,
        check_reasoning_presence=True,
    )
    _require_appworld_execution_evidence(args, text, role="student", env_name=env_name)
    return text


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
    values: list[Any] = []
    metadata_value = _metadata_value(sample, keys)
    if metadata_value not in (None, "", []):
        if isinstance(metadata_value, (list, tuple)):
            values.extend(item for item in metadata_value if item not in (None, "", []))
        else:
            values.append(metadata_value)
    if values:
        rendered = [
            _sanitize_teacher_answer_for_anonymous_verifier(args, value, sample=sample)
            for value in values
        ]
        return tuple(dict.fromkeys(value for value in rendered if value.strip()))
    index_path = _teacher_index_path(args)
    if index_path is not None:
        index = _read_teacher_index(args, index_path)
        for key in _teacher_index_key_candidates(args, sample):
            row = index.get(key)
            if row is None:
                continue
            answer = _teacher_index_row_answer(args, row)
            status = _teacher_index_row_status(row, has_text=bool(answer))
            if status != "completed" or answer in (None, "", []):
                return ()
            return (_sanitize_teacher_answer_for_anonymous_verifier(args, answer, sample=sample),)
    return ()


def _extra_rubric_instructions(args: Any) -> str:
    return str(_cfg(args, "extra_rubric_instructions", "") or "")


def _extra_scoring_instructions(args: Any) -> str:
    return str(_cfg(args, "extra_scoring_instructions", "") or "")


def _process_step_evidence_instructions(args: Any) -> str:
    if not credit_assignment.request_process_step_evidence(args):
        return ""
    return """

# Process step evidence
同时为每个 trajectory 输出 `process_step_evidence`，用于把过程质量信号定位到具体 action step。
每条 evidence 必须与 rubric_scores 一一对应，结构如下：
```json
{
  "criterion_id": "r1",
  "positive_step_indices": [2],
  "negative_step_indices": [4],
  "evidence": "short evidence"
}
```
规则：
- `positive_step_indices` 只填满足对应 rubric 的关键 Step N。
- `negative_step_indices` 只填明显违反对应 rubric 的关键 Step N。
- step 编号必须严格使用 trajectory 文本中的 `Step N` 编号；没有可定位证据就返回空数组。
- 不要把同一个 step 同时放入 positive 和 negative。
- `evidence` 只写一句短证据，不要长篇解释。
"""


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


def _normalize_answer_process_rubric(args: Any, payload: Any) -> dict[str, Any] | None:
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
        "score_policy": _answer_process_score_policy(args),
    }


def _normalize_rubric_shaping_rubric(args: Any, payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != RUBRIC_SHAPING_RUBRIC_SCHEMA_VERSION:
        return None
    items = payload.get("rubrics")
    if not isinstance(items, list):
        return None
    normalized_items: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        criterion = str(item.get("criterion") or item.get("description") or "").strip()
        if not criterion:
            continue
        normalized_items.append(
            {
                "criterion_id": f"r{idx}",
                "category": str(item.get("category") or "Trajectory Quality").strip() or "Trajectory Quality",
                "criterion": criterion,
                "max_score": 5,
                "weight": _positive_float(item.get("weight", 1.0), 1.0),
            }
        )
    if not normalized_items:
        return None
    return {
        "schema_version": RUBRIC_SHAPING_RUBRIC_SCHEMA_VERSION,
        "rubrics": normalized_items,
        "maximum_score": 1.0,
        "score_policy": {
            "score_range": "0_to_5_per_criterion",
            "env_success_overrides_reward": True,
            "failure_reward_scale_beta": _rubric_shaping_beta(args),
        },
    }


def _ca_compact_rubric(args: Any) -> dict[str, Any]:
    return {
        "schema_version": CA_COMPACT_RUBRIC_SCHEMA_VERSION,
        "score_policy": {
            "judge_output": "good_bad_step_masks_only",
            "scalar_reward_source": _cfg(args, "ca_scalar_reward_source", "env_success"),
            "single_call_behavior_mining": True,
            "process_step_evidence_for_credit_assignment": True,
            "tasa_state_evidence_for_credit_assignment": credit_assignment.request_tasa_state_evidence(args),
        },
    }


def _normalize_rubric_for_mode(args: Any, payload: Any) -> dict[str, Any] | None:
    schema_mode = _schema_mode(args)
    if schema_mode == "answer_process_50_50":
        return _normalize_answer_process_rubric(args, payload)
    if schema_mode == "rubric_shaping":
        return _normalize_rubric_shaping_rubric(args, payload)
    if schema_mode == "ca_compact":
        if isinstance(payload, dict) and payload.get("schema_version") == CA_COMPACT_RUBRIC_SCHEMA_VERSION:
            return _ca_compact_rubric(args)
        return None
    return _normalize_rubric(payload)


def _maximum_score(rubric: Any) -> float:
    if isinstance(rubric, dict) and rubric.get("schema_version") == RUBRIC_SHAPING_RUBRIC_SCHEMA_VERSION:
        return 1.0
    if isinstance(rubric, dict) and rubric.get("schema_version") == ANSWER_PROCESS_RUBRIC_SCHEMA_VERSION:
        maximum_scores = rubric.get("maximum_scores")
        if isinstance(maximum_scores, dict):
            return float_value(maximum_scores.get("total"), 10.0)
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
    schema_mode = _schema_mode(args)
    if schema_mode == "answer_process_50_50":
        template = RUBRICATOR_ANSWER_PROCESS_PROMPT_TEMPLATE
    elif schema_mode == "rubric_shaping":
        template = RUBRICATOR_SHAPING_PROMPT_TEMPLATE
    else:
        template = RUBRICATOR_PROMPT_TEMPLATE
    answer_weight, process_weight = _answer_process_weights(args)
    support_weight = _answer_support_weight_for_reward(args)
    answer_norm = 1.0 + support_weight
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
            "answer_weight": f"{answer_weight:.6g}",
            "process_weight": f"{process_weight:.6g}",
            "answer_core_weight_for_reward": f"{1.0 / answer_norm:.6g}",
            "answer_support_weight_for_reward": f"{support_weight:.6g}",
            "answer_support_weight_for_reward_normalized": f"{support_weight / answer_norm:.6g}",
            "final_reward_uses": _answer_process_final_reward_uses(args),
            "shaping_beta": f"{_rubric_shaping_beta(args):.6g}",
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
    elif isinstance(rubric, dict) and rubric.get("schema_version") == RUBRIC_SHAPING_RUBRIC_SCHEMA_VERSION:
        template = VERIFIER_SHAPING_PROMPT_TEMPLATE
        rubric_payload = {
            "rubrics": rubric.get("rubrics", []),
            "score_policy": rubric.get("score_policy", {}),
        }
        answer_label = "Trajectory"
    else:
        template = VERIFIER_PROMPT_TEMPLATE
        rubric_payload = rubric["rubrics"]
        answer_label = "Answer"
    answer_weight, process_weight = _answer_process_weights(args)
    support_weight = _answer_support_weight_for_reward(args)
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
            "process_step_evidence_instructions": _process_step_evidence_instructions(args),
            "answer_weight": f"{answer_weight:.6g}",
            "process_weight": f"{process_weight:.6g}",
            "answer_support_weight_for_reward": f"{support_weight:.6g}",
            "final_reward_uses": _answer_process_final_reward_uses(args),
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


def _student_trajectory_id(item: dict[str, Any]) -> str:
    source = str(item.get("source") or "")
    if source != "student":
        raise ValueError(f"compact CA trajectory ids are student-only, got source={source!r}")
    source_index = int(item.get("source_index", 0) or 0)
    return f"S{source_index}_STUDENT"


def _render_ca_compact_trajectory_block(answer_items: tuple[dict[str, Any], ...]) -> str:
    blocks = []
    for item in answer_items:
        blocks.append(f"[{_student_trajectory_id(item)}]\n{item.get('text')}")
    return "\n\n".join(blocks)


def _render_ca_compact_reference_block(answer_items: tuple[dict[str, Any], ...]) -> str:
    if not answer_items:
        return "None"
    return "\n\n".join(
        f"[REFERENCE {index}]\n{item.get('text')}"
        for index, item in enumerate(answer_items, start=1)
    )


def _build_ca_compact_verifier_prompt(
    args: Any,
    sample: Sample,
    *,
    answer_items: tuple[dict[str, Any], ...],
) -> str:
    question_max_chars = _cfg_int(args, "question_max_chars", 0)
    answer_max_chars = _cfg_int(args, "verifier_answer_max_chars", 0)
    limited_items = tuple(
        {
            **item,
            "text": _limit_reward_text(args, item.get("text"), answer_max_chars, field="verifier_answer"),
        }
        for item in answer_items
    )
    reference_items = tuple(item for item in limited_items if str(item.get("source") or "") == "teacher")
    student_items = tuple(item for item in limited_items if str(item.get("source") or "") == "student")
    extra_scoring_instructions = _extra_scoring_instructions(args).strip()
    extra_scoring_section = (
        f"[Additional Scoring Instructions]\n{extra_scoring_instructions}"
        if extra_scoring_instructions
        else ""
    )
    template = (
        TASA_STATE_VERIFIER_PROMPT_TEMPLATE
        if credit_assignment.request_tasa_state_evidence(args)
        else CA_COMPACT_VERIFIER_PROMPT_TEMPLATE
    )
    return _render_template(
        template,
        {
            "question": _limit_reward_text(args, _ropd_question(sample), question_max_chars, field="question"),
            "references": _render_ca_compact_reference_block(reference_items),
            "students": _render_ca_compact_trajectory_block(student_items),
            "student_trajectory_ids": ", ".join(_student_trajectory_id(item) for item in student_items),
            "extra_scoring_section": extra_scoring_section,
        },
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
    args: Any,
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
        answer_score = _answer_process_answer_score(
            args,
            core_score=answer_core_score,
            support_score=answer_support_score,
        )
        process_score = _weighted_average_0_to_5(process_values, process_items)
        process_score_scaled = max(0.0, min(10.0, process_score * 2.0))
        final_score = _answer_process_final_score(
            args,
            answer_score=answer_score,
            process_score=process_score,
        )
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
                "process_score_scaled": process_score_scaled,
                "final_score": final_score,
                "answer_process_answer_weight": _answer_process_weights(args)[0],
                "answer_process_process_weight": _answer_process_weights(args)[1],
                "answer_support_weight_for_reward": _answer_support_weight_for_reward(args),
                "answer_process_final_reward_uses": _answer_process_final_reward_uses(args),
                "final_answer_quality": final_answer_quality,
                "fatal_error": bool(item.get("fatal_error", False)),
            }
        )
    return scores


def _parse_step_indices(raw: Any) -> list[int]:
    if raw in (None, "", []):
        return []
    if not isinstance(raw, list):
        raise ValueError("ROPD rubric-shaping step indices must be a list")
    values: list[int] = []
    for item in raw:
        if isinstance(item, int):
            step = item
        elif isinstance(item, str) and item.strip().isdigit():
            step = int(item.strip())
        else:
            raise ValueError("ROPD rubric-shaping step indices must be integers")
        if step < 0:
            raise ValueError("ROPD rubric-shaping step indices must be non-negative")
        values.append(step)
    return list(dict.fromkeys(values))


def _step_numbers(text: str) -> set[int]:
    return {int(value) for value in re.findall(r"(?m)^Step\s+(\d+)\b", str(text or ""))}


def _parse_tasa_state_id_list(raw: Any, *, field: str) -> list[str]:
    if raw in (None, "", []):
        return []
    if not isinstance(raw, list):
        raise ValueError(f"ROPD compact TASA {field} must be a list")
    ids: list[str] = []
    for item in raw:
        item_id = str(item or "").strip()
        if not item_id:
            continue
        ids.append(item_id)
    return list(dict.fromkeys(ids))


def _parse_tasa_state_schema(args: Any, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("ROPD compact TASA state schema must be an object")
    milestones = payload.get("milestones")
    if not isinstance(milestones, list):
        raise ValueError("ROPD compact TASA state schema requires a milestones list")
    if not (3 <= len(milestones) <= 8):
        raise ValueError("ROPD compact TASA state schema must contain 3..8 milestones")
    normalized_milestones: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_item in milestones:
        if not isinstance(raw_item, dict):
            raise ValueError("ROPD compact TASA milestone must be an object")
        unknown_fields = sorted(set(raw_item) - {"id", "predicate", "requires", "progress"})
        if unknown_fields:
            raise ValueError(f"ROPD compact TASA milestone returned unsupported fields: {unknown_fields}")
        item_id = str(raw_item.get("id") or "").strip()
        if not re.fullmatch(r"M[1-9][0-9]*", item_id):
            raise ValueError("ROPD compact TASA milestone id must match M<N>")
        if item_id in seen_ids:
            raise ValueError("ROPD compact TASA predicate ids must be unique")
        predicate = str(raw_item.get("predicate") or "").strip()
        if not predicate:
            raise ValueError("ROPD compact TASA milestone predicate is required")
        if "requires" not in raw_item:
            raise ValueError(f"ROPD compact TASA milestone {item_id} requires an explicit requires list")
        raw_progress = raw_item.get("progress")
        if raw_progress in (None, ""):
            raise ValueError(f"ROPD compact TASA-GAE milestone {item_id} requires progress")
        else:
            progress = float_value(raw_progress, 0.0)
            if progress <= 0:
                raise ValueError(f"ROPD compact TASA milestone {item_id} progress must be positive")
        seen_ids.add(item_id)
        normalized_milestones.append(
            {
                "id": item_id,
                "predicate": predicate,
                "requires": _parse_tasa_state_id_list(raw_item["requires"], field=f"{item_id}.requires"),
                "progress": progress,
            }
        )

    milestone_ids = {item["id"] for item in normalized_milestones}
    progress_by_id = {str(item["id"]): float(item["progress"]) for item in normalized_milestones}
    for item in normalized_milestones:
        unknown_requires = sorted(set(item["requires"]) - milestone_ids)
        if unknown_requires:
            raise ValueError(f"ROPD compact TASA milestone requires unknown ids: {unknown_requires}")
        invalid_progress_requires = [
            required_id for required_id in item["requires"] if progress_by_id[required_id] >= float(item["progress"])
        ]
        if invalid_progress_requires:
            raise ValueError(
                "ROPD compact TASA milestone requires dependencies with non-increasing progress: "
                f"{item['id']} requires {invalid_progress_requires}"
            )

    return {"milestones": normalized_milestones}


def _parse_tasa_milestone_batch(
    args: Any,
    payload: dict[str, Any],
    *,
    answer_items: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    if payload.get("schema_version") != TASA_STATE_VERIFIER_SCHEMA_VERSION:
        raise ValueError("ROPD compact TASA verifier schema_version mismatch")
    unknown_top_level = sorted(set(payload) - {"schema_version", "milestones"})
    if unknown_top_level:
        raise ValueError(f"ROPD compact TASA verifier returned unsupported fields: {unknown_top_level}")

    student_items = tuple(item for item in answer_items if str(item.get("source") or "") == "student")
    expected_ids = [_student_trajectory_id(item) for item in student_items]
    expected_id_set = set(expected_ids)
    valid_steps = {
        _student_trajectory_id(item): _step_numbers(str(item.get("text") or ""))
        for item in student_items
    }
    raw_milestones = payload.get("milestones")
    schema = _parse_tasa_state_schema(
        args,
        {
            "milestones": [
                {
                    key: item[key]
                    for key in ("id", "predicate", "requires", "progress")
                    if key in item
                }
                if isinstance(item, dict)
                else item
                for item in (raw_milestones if isinstance(raw_milestones, list) else [])
            ]
        },
    )

    changes_by_trajectory: dict[str, dict[int, dict[str, list[str]]]] = {
        trajectory_id: {} for trajectory_id in expected_ids
    }
    for raw_item, schema_item in zip(raw_milestones, schema["milestones"], strict=True):
        if not isinstance(raw_item, dict):
            raise ValueError("ROPD compact TASA milestone must be an object")
        unknown_fields = sorted(
            set(raw_item) - {"id", "predicate", "requires", "progress", "transitions_by_trajectory"}
        )
        if unknown_fields:
            raise ValueError(f"ROPD compact TASA milestone returned unsupported fields: {unknown_fields}")
        transitions = raw_item.get("transitions_by_trajectory")
        if not isinstance(transitions, list):
            raise ValueError("ROPD compact TASA transitions_by_trajectory must be a list")
        transition_ids = [str(item.get("trajectory_id") or "").strip() for item in transitions if isinstance(item, dict)]
        if transition_ids != expected_ids:
            raise ValueError(
                "ROPD compact TASA transitions_by_trajectory must exactly match Student Trajectory IDs in order"
            )
        if set(transition_ids) != expected_id_set or len(transitions) != len(expected_ids):
            raise ValueError("ROPD compact TASA transitions_by_trajectory has duplicate or missing students")
        milestone_id = str(schema_item["id"])
        for transition in transitions:
            if not isinstance(transition, dict):
                raise ValueError("ROPD compact TASA transition must be an object")
            unknown_transition_fields = sorted(set(transition) - {"trajectory_id", "set_steps", "unset_steps"})
            if unknown_transition_fields:
                raise ValueError(
                    f"ROPD compact TASA transition returned unsupported fields: {unknown_transition_fields}"
                )
            trajectory_id = str(transition.get("trajectory_id") or "").strip()
            set_steps = _parse_step_indices(transition.get("set_steps", []))
            unset_steps = _parse_step_indices(transition.get("unset_steps", []))
            overlap = sorted(set(set_steps).intersection(unset_steps))
            if overlap:
                raise ValueError(
                    f"ROPD compact TASA milestone {milestone_id} cannot set and unset on the same steps: {overlap}"
                )
            invalid = [step for step in set_steps + unset_steps if step not in valid_steps[trajectory_id]]
            if invalid:
                raise ValueError(
                    f"ROPD compact TASA steps do not exist for {trajectory_id}: {sorted(set(invalid))}"
                )
            for step in set_steps:
                event = changes_by_trajectory[trajectory_id].setdefault(step, {"set": [], "unset": []})
                event["set"].append(milestone_id)
            for step in unset_steps:
                event = changes_by_trajectory[trajectory_id].setdefault(step, {"set": [], "unset": []})
                event["unset"].append(milestone_id)

    scores: list[dict[str, Any]] = []
    for expected_index, item in enumerate(answer_items, start=1):
        if str(item.get("source") or "") != "student":
            continue
        trajectory_id = _student_trajectory_id(item)
        changes = [
            {"step": step, "set": event["set"], "unset": event["unset"]}
            for step, event in sorted(changes_by_trajectory[trajectory_id].items())
            if event["set"] or event["unset"]
        ]
        scores.append(
            {
                "answer_index": expected_index,
                "source_index": int(item.get("source_index", 0) or 0),
                "trajectory_id": trajectory_id,
                "process_step_evidence": [],
                "behaviors": [],
                "tasa_state_schema": schema,
                "tasa_state_changes": changes,
            }
        )
    return scores


def _parse_ca_compact_batch_masks(
    args: Any,
    payload: Any,
    *,
    answer_items: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("ROPD compact CA verifier response must be a JSON object")
    if credit_assignment.request_tasa_state_evidence(args):
        return _parse_tasa_milestone_batch(args, payload, answer_items=answer_items)
    if payload.get("schema_version") != CA_COMPACT_VERIFIER_SCHEMA_VERSION:
        raise ValueError("ROPD compact CA verifier schema_version mismatch")
    allowed_top_level = {"schema_version", "behaviors"}
    unknown_top_level = sorted(set(payload) - allowed_top_level)
    if unknown_top_level:
        raise ValueError(f"ROPD compact CA verifier returned unsupported fields: {unknown_top_level}")
    student_items = tuple(item for item in answer_items if str(item.get("source") or "") == "student")
    expected_ids = [_student_trajectory_id(item) for item in student_items]
    valid_student_steps = {
        _student_trajectory_id(item): _step_numbers(str(item.get("text") or ""))
        for item in student_items
    }

    raw_behaviors = payload.get("behaviors")
    if not isinstance(raw_behaviors, list) or not (3 <= len(raw_behaviors) <= 8):
        raise ValueError("ROPD compact CA behaviors must contain 3..8 items")
    seen_behavior_ids: set[str] = set()
    polarities: set[str] = set()
    behavior_items: list[dict[str, Any]] = []
    evidence_by_trajectory = {trajectory_id: [] for trajectory_id in expected_ids}
    for index, raw_behavior in enumerate(raw_behaviors, start=1):
        if not isinstance(raw_behavior, dict):
            raise ValueError("ROPD compact CA behavior item must be an object")
        unknown_behavior_fields = sorted(
            set(raw_behavior) - {"behavior_id", "polarity", "description", "hits_by_trajectory"}
        )
        if unknown_behavior_fields:
            raise ValueError(f"ROPD compact CA behavior returned unsupported fields: {unknown_behavior_fields}")
        behavior_id = str(raw_behavior.get("behavior_id") or "").strip()
        polarity = str(raw_behavior.get("polarity") or "").strip().lower()
        if polarity not in {"good", "bad"}:
            raise ValueError("ROPD compact CA behavior polarity must be good or bad")
        if not re.fullmatch(r"[gb][1-9][0-9]*", behavior_id):
            raise ValueError("ROPD compact CA behavior_id must match gN or bN")
        if behavior_id in seen_behavior_ids:
            raise ValueError("ROPD compact CA behavior_id must be unique")
        seen_behavior_ids.add(behavior_id)
        polarities.add(polarity)
        description = str(raw_behavior.get("description") or "").strip()
        if not description:
            raise ValueError("ROPD compact CA behavior description is required")
        hits = raw_behavior.get("hits_by_trajectory")
        if not isinstance(hits, list):
            raise ValueError("ROPD compact CA hits_by_trajectory must be a list")
        hit_map: dict[str, list[int]] = {}
        for hit in hits:
            if not isinstance(hit, dict):
                raise ValueError("ROPD compact CA hit item must be an object")
            trajectory_id = str(hit.get("trajectory_id") or "").strip()
            if trajectory_id not in valid_student_steps:
                raise ValueError(f"ROPD compact CA unknown trajectory_id={trajectory_id!r}")
            step_indices = _parse_step_indices(hit.get("step_indices", []))
            invalid_steps = [step for step in step_indices if step not in valid_student_steps[trajectory_id]]
            if invalid_steps:
                raise ValueError(
                    f"ROPD compact CA step indices do not exist for {trajectory_id}: {invalid_steps}"
                )
            hit_map[trajectory_id] = step_indices
        missing_ids = [trajectory_id for trajectory_id in expected_ids if trajectory_id not in hit_map]
        if missing_ids:
            raise ValueError(f"ROPD compact CA hits missing trajectories: {missing_ids}")
        behavior_item = {
            "criterion_id": behavior_id,
            "behavior_id": behavior_id,
            "polarity": polarity,
            "description": description,
            "hits_by_trajectory": [
                {"trajectory_id": trajectory_id, "step_indices": hit_map[trajectory_id]}
                for trajectory_id in expected_ids
            ],
        }
        behavior_items.append(behavior_item)
        for trajectory_id, steps in hit_map.items():
            evidence_by_trajectory[trajectory_id].append(
                {
                    "criterion_id": behavior_id,
                    "positive_step_indices": steps if polarity == "good" else [],
                    "negative_step_indices": steps if polarity == "bad" else [],
                }
            )
    if polarities != {"good", "bad"}:
        raise ValueError("ROPD compact CA must include both good and bad behaviors")

    scores: list[dict[str, Any]] = []
    for expected_index, item in enumerate(answer_items, start=1):
        if str(item.get("source") or "") != "student":
            continue
        trajectory_id = _student_trajectory_id(item)
        scores.append(
            {
                "answer_index": expected_index,
                "source_index": int(item.get("source_index", 0) or 0),
                "trajectory_id": trajectory_id,
                "process_step_evidence": evidence_by_trajectory[trajectory_id],
                "behaviors": behavior_items,
            }
        )
    return scores


def _parse_rubric_shaping_evidence(
    args: Any,
    item: dict[str, Any],
    rubric_ids: list[str],
) -> list[dict[str, Any]]:
    raw_evidence = item.get("process_step_evidence")
    if raw_evidence in (None, ""):
        if credit_assignment.request_process_step_evidence(args):
            raise ValueError("ROPD rubric-shaping process_step_evidence is required by credit assignment")
        return []
    if not isinstance(raw_evidence, list) or len(raw_evidence) != len(rubric_ids):
        raise ValueError("ROPD rubric-shaping process_step_evidence length mismatch")
    parsed: list[dict[str, Any]] = []
    for raw_item, criterion_id in zip(raw_evidence, rubric_ids, strict=True):
        if not isinstance(raw_item, dict):
            raise ValueError("ROPD rubric-shaping evidence item must be an object")
        if str(raw_item.get("criterion_id") or criterion_id) != criterion_id:
            raise ValueError("ROPD rubric-shaping evidence criterion_id mismatch")
        positive = _parse_step_indices(raw_item.get("positive_step_indices", []))
        negative = _parse_step_indices(raw_item.get("negative_step_indices", []))
        if set(positive).intersection(negative):
            raise ValueError("ROPD rubric-shaping evidence cannot mark the same step positive and negative")
        parsed.append(
            {
                "criterion_id": criterion_id,
                "positive_step_indices": positive,
                "negative_step_indices": negative,
                "evidence": str(raw_item.get("evidence") or "").strip(),
            }
        )
    return parsed


def _parse_rubric_shaping_batch_scores(
    args: Any,
    payload: Any,
    *,
    rubric: dict[str, Any],
    expected: int,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("ROPD rubric-shaping verifier response must be a JSON object")
    if payload.get("schema_version") != RUBRIC_SHAPING_VERIFIER_SCHEMA_VERSION:
        raise ValueError("ROPD rubric-shaping verifier schema_version mismatch")
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != expected:
        raise ValueError(
            f"ROPD rubric-shaping verifier returned "
            f"{0 if not isinstance(answers, list) else len(answers)} scores for {expected} answers"
        )
    rubric_items = list(rubric.get("rubrics", []))
    rubric_ids = [str(item.get("criterion_id")) for item in rubric_items]
    scores: list[dict[str, Any]] = []
    for expected_index, item in enumerate(answers, start=1):
        if not isinstance(item, dict):
            raise ValueError("ROPD rubric-shaping answer item must be an object")
        if int(item.get("answer_index", -1)) != expected_index:
            raise ValueError("ROPD rubric-shaping answer_index must cover 1..n in order")
        raw_scores = item.get("rubric_scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != len(rubric_items):
            raise ValueError("ROPD rubric-shaping rubric_scores length mismatch")
        score_items = [
            _score_item(raw, criterion_id=criterion_id)
            for raw, criterion_id in zip(raw_scores, rubric_ids, strict=True)
        ]
        for score_item, criterion_id in zip(score_items, rubric_ids, strict=True):
            if score_item["criterion_id"] != criterion_id:
                raise ValueError("ROPD rubric-shaping criterion_id mismatch")
        quality = str(item.get("trajectory_quality") or "").strip().lower()
        if quality not in {"strong", "useful", "partial", "weak", "invalid"}:
            raise ValueError("ROPD rubric-shaping trajectory_quality mismatch")
        fatal_error = bool(item.get("fatal_error", False))
        evidence_items = _parse_rubric_shaping_evidence(args, item, rubric_ids)
        rubric_score = _weighted_average_0_to_5([float(score_item["score"]) for score_item in score_items], rubric_items)
        normalized_score = 0.0 if fatal_error else max(0.0, min(1.0, rubric_score / 5.0))
        scores.append(
            {
                "answer_index": expected_index,
                "rubric_scores": score_items,
                "process_step_evidence": evidence_items,
                "rubric_score": normalized_score,
                "rubric_score_0_to_5": rubric_score,
                "final_score": normalized_score,
                "trajectory_quality": quality,
                "fatal_error": fatal_error,
            }
        )
    return scores


def _parse_batch_scores(args: Any, payload: Any, *, rubric: dict[str, Any], expected: int) -> list[dict[str, Any]]:
    if rubric.get("schema_version") == ANSWER_PROCESS_RUBRIC_SCHEMA_VERSION:
        return _parse_answer_process_batch_scores(args, payload, rubric=rubric, expected=expected)
    if rubric.get("schema_version") == RUBRIC_SHAPING_RUBRIC_SCHEMA_VERSION:
        return _parse_rubric_shaping_batch_scores(args, payload, rubric=rubric, expected=expected)
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


def _reward_group_reference(args: Any) -> str:
    return _cfg_choice(
        args,
        "reward_group_reference",
        "students",
        {"students", "teacher_plus_students"},
    )


def _require_teacher(args: Any) -> bool:
    return _cfg_bool(args, "require_teacher", False)


def _reward_mode(args: Any) -> str:
    return _cfg_choice(
        args,
        "reward_mode",
        "answer_only",
        {"answer_only", "group_centered", "group_zscore"},
    )


def _validate_reward_config(args: Any) -> None:
    if credit_assignment.enabled(args) and _schema_mode(args) not in {"rubric_shaping", "ca_compact"}:
        raise ValueError(
            "reward.credit_assignment.enable currently requires "
            "reward.ropd.schema_mode in {rubric_shaping, ca_compact}"
        )
    return None


def _is_hard_discard_sample(sample: Sample) -> bool:
    sample_metadata = metadata(sample)
    if sample.status == Sample.Status.ABORTED:
        return True
    if bool(getattr(sample, "remove_sample", False)) or bool(sample_metadata.get("discard_sample", False)):
        return True
    return False


def _hard_discard_details(sample: Sample) -> dict[str, Any]:
    sample_metadata = metadata(sample)
    return {
        "status": str(getattr(sample.status, "value", sample.status)),
        "remove_sample": bool(getattr(sample, "remove_sample", False)),
        "discard_sample": bool(sample_metadata.get("discard_sample", False)),
        "discard_reason": sample_metadata.get("discard_reason") or "",
        "error": sample_metadata.get("error") or "",
    }


def _env_success_for_shaping(sample: Sample) -> bool:
    sample_metadata = metadata(sample)
    if "env_success" not in sample_metadata:
        raise ValueError(
            "reward.ropd.schema_mode with env-success override requires sample.metadata.env_success "
            f"for active sample index={sample.index} group_index={sample.group_index} "
            f"status={getattr(sample.status, 'value', sample.status)} "
            f"discard_reason={sample_metadata.get('discard_reason')!r}"
        )
    return bool(sample_metadata.get("env_success"))


def _env_score_for_shaping(sample: Sample) -> float:
    sample_metadata = metadata(sample)
    if "env_score" not in sample_metadata:
        raise ValueError(
            "reward.ropd.ca_scalar_reward_source=env_score requires sample.metadata.env_score "
            f"for active sample index={sample.index} group_index={sample.group_index} "
            f"status={getattr(sample.status, 'value', sample.status)} "
            f"discard_reason={sample_metadata.get('discard_reason')!r}"
        )
    return max(0.0, min(1.0, float_value(sample_metadata.get("env_score"), 0.0)))


def _ca_compact_scalar_reward(args: Any, sample: Sample) -> tuple[float, str]:
    source = _cfg_choice(
        args,
        "ca_scalar_reward_source",
        "env_success",
        {"env_success", "success", "env_score", "score"},
    )
    if source in {"env_success", "success"}:
        return (1.0 if _env_success_for_shaping(sample) else 0.0), "ca_compact_env_success"
    return _env_score_for_shaping(sample), "ca_compact_env_score"


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
    sample: Sample,
    answer_score: float,
    group_stats: dict[str, Any],
) -> tuple[float, str]:
    schema_mode = _schema_mode(args)
    if schema_mode == "ca_compact":
        return _ca_compact_scalar_reward(args, sample)
    if schema_mode == "rubric_shaping":
        reason = f"env_success_else_{schema_mode}"
        if _env_success_for_shaping(sample):
            return 1.0, reason
        shaped = _rubric_shaping_beta(args) * max(0.0, min(1.0, answer_score))
        return max(0.0, min(1.0, shaped)), reason
    if schema_mode == "answer_process_50_50" and _env_success_overrides_reward(args):
        reason = f"env_success_else_{schema_mode}"
        if _env_success_for_shaping(sample):
            return 1.0, reason
        shaped = _failure_shaping_beta(args) * max(0.0, min(1.0, answer_score))
        return max(0.0, min(1.0, shaped)), reason
    reward_mode = _reward_mode(args)
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
    if _schema_mode(args) == "ca_compact":
        if _require_teacher(args) and not teacher_answers:
            return None, "missing_teacher", None, teacher_answers
        source = "ca_compact" if teacher_answers else "ca_compact_no_reference"
        return _ca_compact_rubric(args), source, None, teacher_answers
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
            endpoint_pool_path=_role_endpoint_pool_path(args, "rubric"),
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
        sample=sample,
        answer_score=bounded,
        group_stats=group_stats,
    )
    trace_options = _trace_options(args)
    weighted = train_score * _weight(args)
    schema_mode = _schema_mode(args)
    raw = {
        "rubric": rubric,
        "rubric_hash": _rubric_hash(rubric),
        "ropd_schema_mode": schema_mode,
        "rubric_source": rubric_source,
        "judge": student_item,
        "student_score": float(student_score),
        "teacher_scores": [float(score) for score in teacher_scores],
        "maximum_score": float(maximum_score),
        "reward_score": float(train_score),
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
        "answer_mode": _answer_mode(args),
        "strip_reasoning": trace_options.strip_reasoning,
        "strip_tool_call": trace_options.strip_tool_call,
        "strip_tool_response": trace_options.strip_tool_response,
        "strip_assistant_response": trace_options.strip_assistant_response,
        "strip_system_prompt": trace_options.strip_system_prompt,
    }
    if schema_mode == "rubric_shaping":
        raw["rubric_score"] = float(bounded)
        raw["env_success_for_reward"] = _env_success_for_shaping(sample)
    elif schema_mode == "answer_process_50_50":
        raw["answer_process_score"] = float(bounded)
        raw["env_success_overrides_reward"] = _env_success_overrides_reward(args)
        if _env_success_overrides_reward(args):
            raw["env_success_for_reward"] = _env_success_for_shaping(sample)
            raw["failure_shaping_beta"] = _failure_shaping_beta(args)
    if schema_mode == "rubric_shaping":
        raw["ropd_shaping_beta"] = _rubric_shaping_beta(args)
    else:
        raw["answer_score"] = float(bounded)
    for key in (
        "answer_core_score",
        "answer_support_score",
        "process_score",
        "process_score_scaled",
        "final_score",
        "rubric_score",
        "rubric_score_0_to_5",
        "rubric_scores",
        "answer_process_answer_weight",
        "answer_process_process_weight",
        "answer_process_final_reward_uses",
        "answer_scores",
        "process_scores",
        "process_step_evidence",
        "final_answer_quality",
        "trajectory_quality",
        "fatal_error",
        "trajectory_id",
        "behaviors",
        "tasa_state_schema",
        "tasa_state_changes",
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
            (
                "ropd_rubric_score"
                if schema_mode == "rubric_shaping"
                else "ropd_answer_score"
            ): bounded,
        },
        raw=raw,
        reason="",
        returns_total=True,
        reward_version="ropd_v1",
    )


def _ca_compact_result(
    args: Any,
    *,
    sample: Sample,
    rubric: dict[str, Any],
    rubric_source: str,
    rubric_call: dict[str, Any] | None,
    judge_call: dict[str, Any] | None,
    student_item: dict[str, Any],
    student_position: int,
) -> RewardResult:
    """Build a mask-only CA result without synthetic judge scores."""

    train_score, effective_reward_mode = _ca_compact_scalar_reward(args, sample)
    weighted = train_score * _weight(args)
    trace_options = _trace_options(args)
    raw = {
        "rubric": rubric,
        "rubric_hash": _rubric_hash(rubric),
        "ropd_schema_mode": "ca_compact",
        "rubric_source": rubric_source,
        "judge": student_item,
        "reward_score": float(train_score),
        "student_answer_position": int(student_position),
        "ropd_train_reward_mode": effective_reward_mode,
        "ropd_reward_mode_requested": _reward_mode(args),
        "env_success_for_reward": _env_success_for_shaping(sample),
        "answer_mode": _answer_mode(args),
        "strip_reasoning": trace_options.strip_reasoning,
        "strip_tool_call": trace_options.strip_tool_call,
        "strip_tool_response": trace_options.strip_tool_response,
        "strip_assistant_response": trace_options.strip_assistant_response,
        "strip_system_prompt": trace_options.strip_system_prompt,
    }
    for key in (
        "process_step_evidence",
        "trajectory_id",
        "behaviors",
        "tasa_state_schema",
        "tasa_state_changes",
    ):
        if key in student_item:
            raw[key] = student_item[key]
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
        reward_version="ropd_ca_mask_v2",
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
    compact_mode = rubric.get("schema_version") == CA_COMPACT_RUBRIC_SCHEMA_VERSION
    if compact_mode:
        prompt = _build_ca_compact_verifier_prompt(args, samples[0], answer_items=answer_items)
    else:
        prompt = _build_verifier_prompt(args, samples[0], rubric=rubric, answers=answers)
    answer_item_records = []
    for idx, item in enumerate(answer_items, start=1):
        record = {
            "answer_index": idx,
            "source": item["source"],
            "source_index": item["source_index"],
            "text": item["text"],
        }
        if item["source"] == "student":
            record["trajectory_id"] = _student_trajectory_id(item)
        answer_item_records.append(record)

    def validate_payload(payload: Any) -> None:
        if compact_mode:
            _parse_ca_compact_batch_masks(args, payload, answer_items=answer_items)
        else:
            _parse_batch_scores(args, payload, rubric=rubric, expected=len(answer_items))

    try:
        payload, judge_call = await call_json_judge_with_metadata(
            args,
            prompt,
            system_prompt=CA_JUDGE_SYSTEM_PROMPT if compact_mode else JUDGE_SYSTEM_PROMPT,
            api_key_path=_role_api_key_path(args, "judge"),
            endpoint_pool_path=_role_endpoint_pool_path(args, "judge"),
            **_role_request_options(args, "judge", default_max_tokens=32768),
            **_role_endpoint(args, "judge"),
            payload_validator=validate_payload,
        )
        judge_call = {
            **judge_call,
            "role": "judge",
            "concurrency_limit": _stage_concurrency(args, "judge"),
        }
        if compact_mode:
            scored_items = _parse_ca_compact_batch_masks(args, payload, answer_items=answer_items)
        else:
            scored_items = _parse_batch_scores(args, payload, rubric=rubric, expected=len(answer_items))
    except Exception as exc:
        details = {"stage": "verifier", "type": type(exc).__name__, "message": str(exc)}
        logger.warning(
            "ROPD verifier exhausted retries bucket=%s sample_index=%s error_type=%s error=%s",
            bucket_key,
            samples[0].index,
            type(exc).__name__,
            str(exc),
        )
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

    if compact_mode:
        student_items_by_index: list[tuple[dict[str, Any], int] | None] = [None] * len(student_answers)
        for score_item in scored_items:
            source_index = int(score_item["source_index"])
            student_items_by_index[source_index] = (score_item, int(score_item["answer_index"]))
        if any(item is None for item in student_items_by_index):
            raise ValueError("ROPD compact CA verifier did not return every student trajectory")
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
                "call": judge_call,
            },
        )
        return [
            _ca_compact_result(
                args,
                sample=sample,
                rubric=rubric,
                rubric_source=rubric_source,
                rubric_call=rubric_call if idx == 0 else None,
                judge_call=judge_call if idx == 0 else None,
                student_item=item[0],
                student_position=item[1],
            )
            for idx, (sample, item) in enumerate(zip(samples, student_items_by_index, strict=True))
            if item is not None
        ]

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

    results: list[RewardResult | None] = [None] * len(samples)
    active_indices: list[int] = []
    for idx, sample in enumerate(samples):
        if _is_hard_discard_sample(sample):
            results[idx] = _fallback_result(
                args,
                sample,
                "discarded_sample",
                "discarded",
                details=_hard_discard_details(sample),
            )
            continue
        if _schema_mode(args) in {"rubric_shaping", "ca_compact"}:
            _env_success_for_shaping(sample)
        active_indices.append(idx)

    buckets: dict[str, list[int]] = {}
    for idx in active_indices:
        sample = samples[idx]
        buckets.setdefault(_join_value(args, sample), []).append(idx)

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
