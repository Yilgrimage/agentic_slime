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
and `reward_group_reference` also belong under `ropd.*`. Luffy is an
independent teacher-token loss extension and belongs under top-level
`luffy.*`; it may be combined with ROPD or with env reward profiles, but ROPD
must not own Luffy settings.
TASA modes use one milestone-centric judge schema and do not also request the
generic good/bad behavior schema. `tasa_use_teacher_prior` controls only the
teacher value-prior mask: set it to `false` for the pure student-MC ablation.
Full TASA-GAE uses coverage-adaptive shrinkage: for each student and state,
`B` is the number of other on-policy students in the group and
`V=(1-N/B)*V_teacher+(N/B)*V_MC` with strict LOO evidence. The teacher fills
only missing peer slots and exits completely at full coverage. The
`tasa_min_peer_support` hard gate applies only when the teacher prior is
disabled (and to the older TASA-GRPO mode). `tasa_enforce_prerequisites`
controls dependency projection. TASA-GAE does not mean-center advantages; use
RMS/std-only scaling if needed.
When `ropd.answer_mode=trace`, student inputs must be rendered by the shared
agent-env trace renderer from structured trajectory fields such as `turns`,
`messages`, `token_segments`, `reward_trace`, `judge_trace`, or `ropd_trace`.
Do not use historical answer fields such as `student_response` as trace
fallbacks. Teacher data for trace mode must likewise go through the shared
teacher trace renderer and expose canonical trace fields, preferably
`teacher_tool_trace` or `teacher_trace`; raw audit transcripts may be stored as
`teacher_raw_trace_text` but must not be the primary judge input.
AppWorld is stricter: reward profiles read only `teacher_reward_trace_payload`,
which contains captured structured turns and is rendered at runtime through
the same environment adapter and compression profile as student traces.
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
`ropd.schema_mode` selects the verifier schema. Use `rubric_shaping` for the
default agentic ROPD path: env success owns task correctness, while the LLM
rubric gives only weak failure shaping through `shaping_beta`. Use `binary` for
the older boolean rubric path. `answer_process_50_50` is a historical
answer/process schema and should not be used for new AppWorld/WebShop runs
unless the experiment explicitly studies that split.
Durable reward variants should select a dedicated reward profile, such as
`configs/agent_env/rewards/alfworld_ropd_seed.yaml`, instead of copying a full
env config or exporting semantic reward environment variables.
For handoff teacher data, `AGENT_ENV_ROPD_TEACHER_INDEX_PATH` may override the
selected reward profile's `ropd.teacher_index_path` at launch time. This is a
runtime artifact path override, not a new reward variant; validate the file
with `examples/agent_env/scripts/validate_teacher_jsonl.py` and keep task
selection aligned through prompt-data generation. For Luffy-only or
Luffy+ROPD runs, `AGENT_ENV_LUFFY_TEACHER_INDEX_PATH` may similarly override
`luffy.teacher_index_path`; prompt-data task selection still must be aligned
with the selected teacher file.

## Native Eval

Formal checkpoint or training-time eval should use Slime's native eval path,
not a task-specific driver script. Enable it from a run/train override with
`EVAL_INTERVAL` and, when needed, `EVAL_CONFIG`. If `EVAL_CONFIG` is omitted and
`examples/agent_env/<env>/eval_config.yaml` exists, the train adapter uses it.
The adapter regenerates eval prompt-data under
`${RUN_ROOT}/prompt_data/eval/`, resolves those paths into the selected eval
config, and sets `EVAL_FUNCTION_PATH` to Slime's stock
`slime.rollout.sglang_rollout.generate_rollout` so full-async training still
uses Slime's native eval loop.

The selected eval config is the only owner of eval datasets and sampling
semantics. Do not use legacy `EVAL_PROMPT_DATA`, `EVAL_MAX_RESPONSE_LEN`,
`EVAL_TEMPERATURE`, `EVAL_TOP_P`, or `EVAL_TOP_K` overrides; agent-env launchers
reject them instead of silently changing the protocol.

Use `EVAL_SPLITS` and `EVAL_PROMPT_NUM_TASKS` only as explicit experiment
overrides. Formal comparisons must evaluate the full intended split and record
the generated prompt-data files with the run artifacts.

For checkpoint sweeps, use the same launcher in eval-sweep mode rather than
writing a wrapper. Env-specific eval datasets, custom generate functions, and
sampling defaults live in `examples/agent_env/<env>/eval_config.yaml`; the
sweep launcher only fans out checkpoints to nodes and selects eval splits.

Example:

```bash
EVAL_SOURCE_RUN=/mnt/bn/.../runs/Qwen3-4B_appworld_ropd_grpo/run-name \
EVAL_CKPT_STEPS="49 99 149 199" \
EVAL_NODE_INDICES="0 1 2 3" \
bash scripts/utils/launch_agentic_training.sh \
  configs/agent_env/runs/appworld_qwen3_4b_grpo_fullasync_4x8_success.env \
  --eval-sweep
```

This starts one native eval run per checkpoint. Each child run still goes
through `launch_agentic_training.sh`, generates eval prompt-data, loads the
checkpoint via `LOAD_DIR`, and writes resolved configs and logs under the sweep
directory. Override `EVAL_SPLITS`, `EVAL_ROOT`, or specific `EVAL_*` variables
only when the env eval config is intentionally insufficient.

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
