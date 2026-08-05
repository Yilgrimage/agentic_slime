# Agent Env Profiles

This directory contains launchable experiment profiles for the external
agent-env backend. Keep it thin and auditable.

## Ownership

- `runs/*.env`: selects env, env config, reward profile, model profile, train
  profile, topology profile, optional aux profile, and experiment naming.
- `models/*.env`: model identity, model args, loss-mask family, dropout
  defaults, and model compatibility defaults.
- `train/*.env`: env-specific training baseline, algorithm, sync/full-async
  mode, batch/token budget, TP/CP, actor/rollout allocation, checkpointing,
  filtering, and Slime training flags.
- `topology/*.env`: node indexes, visible GPUs, and ports only.
- `aux/*.env`: optional auxiliary inference endpoint only.
- `rewards/*.yaml`: reward composition, env score scale, format/truncation
  penalties, LLM-as-judge method selection, and method-specific settings such
  as ROPD rubric/verifier behavior.

Environment semantics belong in `examples/agent_env/<env>/env_config.yaml`.
Reward semantics belong in the selected `REWARD_PROFILE`. Use `impl` and
`judge_mode` for the active RM path, and keep ROPD teacher/rubric files or
Valleydance process-reward weights out of generic train profiles. ROPD reward
concurrency is configured under `ropd.concurrency`; optional
`ropd.rubric_concurrency` and `ropd.judge_concurrency` override per-stage
limits, with `0` meaning "follow the global ROPD concurrency".
ROPD training semantics such as `answer_mode`, `reward_mode`,
`reward_group_reference`, and `luffy_*` also belong under `ropd.*`.
When `ropd.answer_mode=trace`, student inputs must be rendered by the shared
agent-env trace renderer from structured trajectory fields such as `turns`,
`messages`, `token_segments`, `reward_trace`, `judge_trace`, or `ropd_trace`.
Do not use historical answer fields such as `student_response` as trace
fallbacks. Teacher data for trace mode must likewise go through the shared
teacher trace renderer and expose canonical trace fields, preferably
`teacher_tool_trace` or `teacher_trace`; raw audit transcripts may be stored as
`teacher_raw_trace_text` but must not be the primary judge input.
`teacher_full_trace_text` is accepted only after canonicalization through the
renderer. `teacher_response`/`teacher_answer` should not be used as trace keys.
ROPD questions must come from explicit task metadata such as
`task_prompt`, `instruction`, `query`, or `question`. Do not let ROPD fall back
to the policy/system prompt, and do not silently truncate reward inputs with
character budgets. If prompt size must be reduced, use the structured trace
compression switches (`strip_reasoning`, `strip_tool_response`,
`strip_assistant_response`, `strip_system_prompt`) so the reward contract stays
auditable.
The verifier always scores teacher and student answers anonymously in the same
batch; `reward_group_reference` only controls whether teacher scores enter the
group baseline. The default V0 path is answer-only LLM rubric/judge reward;
teacher-anchored baselines are explicit opt-in knobs.
`ropd.schema_mode` selects the verifier schema. Use `binary` for the older
boolean rubric path, or `answer_process_50_50` when the reward should follow
the answer-first ROPD design: core answer correctness is 50% of the answer
score, answer support is the other 50%, and process scores are diagnostics only
unless a future reward profile explicitly changes that contract. The
answer-process verifier should keep its output compact in formal training:
numeric answer/process scores plus coarse quality flags are enough; long
per-criterion rationales and step evidence should be reserved for sampled dumps
or separate analysis jobs.
Durable reward variants should select a dedicated reward profile, such as
`configs/agent_env/rewards/alfworld_ropd_seed.yaml`, instead of copying a full
env config or exporting semantic reward environment variables.
For handoff teacher data, `AGENT_ENV_ROPD_TEACHER_INDEX_PATH` may override the
selected reward profile's `ropd.teacher_index_path` at launch time. This is a
runtime artifact path override, not a new reward variant; validate the file
with `examples/agent_env/scripts/validate_teacher_jsonl.py` and keep task
selection aligned through prompt-data generation.

## Native Eval

Formal checkpoint or training-time eval should use Slime's native eval path,
not a task-specific driver script. Enable it from a run/train override with
`EVAL_INTERVAL` and, when needed, `EVAL_CONFIG`. If `EVAL_CONFIG` is omitted and
`examples/agent_env/<env>/eval_config.yaml` exists, the train adapter uses it.
The adapter regenerates eval prompt-data under
`${RUN_ROOT}/prompt_data/eval/`, exports the dataset variables referenced by
the eval config, and sets `EVAL_FUNCTION_PATH` to Slime's stock
`slime.rollout.sglang_rollout.generate_rollout` so full-async training still
uses Slime's native eval loop.

Use `EVAL_SPLITS` and `EVAL_PROMPT_NUM_TASKS` only as explicit experiment
overrides. Formal comparisons must evaluate the full intended split and record
the generated prompt-data files with the run artifacts.

## Rules

- Do not create a copied profile to change one scalar for a one-off run. Change
  the existing owner file, or create a new profile only for a durable baseline.
- Train profile names are model-agnostic:

```text
<env>_<algorithm>_<sync|fullasync>_<nodes>x<gpus>.env
```

- Real IPs live only in local ignored node files such as
  `configs/nodes/agent_env_all.txt`. Commit only templates such as
  `configs/nodes/agent_env_all.txt.example`; topology profiles select
  zero-based indexes from the local file.
- Values meant to be overridden should be declared as `${VAR:-default}` and
  must appear in the resolved profile under `RUN_ROOT/logs/`.
- Keep credentials, W&B keys, run logs, checkpoints, data, models, and runtime
  packs out of git.

Launch through:

```bash
bash scripts/utils/launch_agentic_training.sh \
  configs/agent_env/runs/alfworld_qwen3_4b_grpo_fullasync_3x8.env
```

For design notes and known pitfalls, read `docs/agent_env/README.md`.
