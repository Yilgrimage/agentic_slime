---
name: agentic-slime-discipline
description: "Use when editing, reviewing, launching, or debugging the agentic Slime RL training stack: configs/agent_env profiles, examples/agent_env rollout/reward/env code, prompt-data alignment, Slime CLI adapters, launch boundaries, reward/filter/padding logic, sync/full-async behavior, and non-invasive integration with upstream Slime."
---

# Agentic Slime Discipline

Keep the agentic Slime stack small, non-invasive, and auditable. Prefer one
clear owner for each behavior. Remove obsolete compatibility paths instead of
adding compensating wrappers.

## Boundaries

- Do not modify Slime core for agent-env behavior unless explicitly asked.
- Keep agentic extensions outside core Slime, mainly under `examples/agent_env`,
  `configs/agent_env`, and thin launch/adapter scripts.
- Launchers orchestrate. Profiles own experiment semantics. Env code owns env
  behavior. Server keepalive/materialization belongs to server ops scripts.
- Use `${ROOT_DIR}/scripts/run_bench.sh`,
  `${ROOT_DIR}/scripts/gpu_idle_watchdog.sh`, and
  `${ROOT_DIR}/scripts/prepare_node_runtime.sh`; do not duplicate them here.
- Prepare task data through `${ROOT_DIR}/scripts/prepare_data.sh`, pack it with
  `${ROOT_DIR}/scripts/pack_data.sh`, and let node materialization only unpack
  or copy validated data. Agentic Slime launch code should not download data.

## Configuration Ownership

- `configs/agent_env/runs/*.env`: choose env, env config, reward profile,
  model profile, train profile, topology profile, optional aux profile, and
  run naming.
- `configs/agent_env/models/*.env`: model identity, model args, loss-mask
  family, dropout/model compatibility defaults.
- `configs/agent_env/train/*.env`: env-specific training baseline, algorithm,
  sync/full-async mode, rollout filters, batch/token budgets, TP/CP,
  actor/rollout allocation, checkpointing, logging, and Slime flags.
- `configs/agent_env/topology/*.env`: node indexes, visible GPUs, and ports
  only. Keep algorithm, model, batch, token, aux, and env semantics out.
- `configs/agent_env/aux/*.env`: auxiliary endpoint identity and credentials
  only.
- `configs/agent_env/rewards/*.yaml`: reward implementation, env-score scale,
  format/truncation penalties, LLM-as-judge mode, and method-specific settings
  such as ROPD. Do not mirror these settings in env configs or train profiles.
- `configs/nodes/*.txt`: local IP files only; commit examples, not real IPs.
- `examples/agent_env/<env>/env_config.yaml`: environment semantics, parser,
  task/data settings, and env server settings. Do not put reward composition
  here.

Train profile names should be:

```text
<env>_<algorithm>_<sync|fullasync>_<nodes>x<gpus>.env
```

Do not include model names or aux providers in train profile names.

## Runtime Contract

- Runtime/data pack creation, verification, backup, and provider-neutral
  cold-start restore are owned by server ops. Follow
  `../server-ops-discipline/references/runtime_pack_build.md`; training launch
  code must never become a second package installer or downloader.
- Support image and conda-pack Slime runtimes through
  `scripts/utils/slime_runtime.sh`.
- Treat Megatron-LM as part of the Slime training runtime, not as a separate
  materialized source. A valid runtime is either an image that already contains
  Slime plus Megatron, or a Slime pack that contains `src/Megatron-LM`.
- Do not add `--sources Megatron-LM`, `MEGATRON_LOCAL_PATH`, NAS checkout
  fallbacks, or repo-specific Megatron paths to agentic training scripts.
- `MEGATRON_PATH` may appear only as the resolved backend path exported by the
  Slime runtime resolver and used in `PYTHONPATH`.
- W&B is an optional reporting runtime. If W&B is enabled, it must come from a
  separate materialized `wandb` env pack. Do not fall back to Slime, image,
  system, user-site, or local Python W&B installs, and do not install W&B during
  training startup.

## Launch Contract

- Public launch resolves run/model/train/topology/aux profiles once and writes
  run-local artifacts under `RUN_ROOT/logs`.
- Internal head/worker roles consume `resolved_launch.env`; the train adapter
  consumes `resolved_train.env`. They must not re-parse git profiles.
- `resolved_train.env` is the final train adapter contract: Slime CLI args,
  env semantics, model paths, runtime overrides, and training runtime paths.
- `resolved_launch.env` is launch-only state: SSH, node selectors, Ray ports,
  env/router/aux lifecycle, bench/watchdog settings, and run directories.
- Keep resolved files in run directories and out of git. Do not freeze arbitrary
  ambient shell variables into them.
- Use `${ROOT_DIR}/models` for durable model checkpoints and `${ROOT_DIR}/runs`
  for run outputs. If a cluster needs those artifacts in another NAS quota
  tree, handle that with symlinks at the server-ops layer; do not add alternate
  artifact roots to training profiles or scripts.

## Rollout And Reward

- Env wrappers own prompt rendering, parser/action semantics, reset/step calls,
  success/score interpretation, and env-specific metadata.
- Prompt data is a run-local contract, not a cache. Regenerate prompt-data
  JSONL at every training launch, then validate it before passing it to Slime.
- Prompt data rows must have a non-empty `prompt` and a metadata object with
  explicit task identity. For known envs, `prompt` is the policy/system prompt;
  env server composes the first user message from the selected task plus reset
  observation/actions/docs.
- Do not add fallback prompts. If a prompt template, task id, task ref, policy
  document, or task payload cannot be loaded, fail fast instead of substituting
  generic model context.
- Env servers must honor task identity from prompt data. Prefer stable
  `task_id`, `task_ref`, or task payload over implicit index; if metadata and
  env data disagree, raise an error.
- Generic rollout code owns message/token accounting, env HTTP lease lifecycle,
  sample shape, common dumping, infra discard, and Slime data contracts.
- Reward implementations own final reward composition. Select them through the
  run's `REWARD_PROFILE`, and keep reward semantics out of env config, train
  profile, aux profile, launch scripts, and ambient shell variables.
- Reward post-process should adapt already-computed RM rewards to Slime's reward
  tensor contract. It should not secretly append env, format, or truncation
  rewards unless that is the selected reward implementation.
- When comparing reward methods such as env reward, 0-1 success reward, ROPD,
  or LUFFY variants, use the environment metrics (`<env>/success_rate` and
  `<env>/env_reward_mean`) as the primary task-performance axis. ROPD or judge
  `rollout/raw_reward` is a reward-quality/debug signal and is not directly
  comparable to env reward scales.
- Route agent-env W&B metrics through the common metrics logger and bind every
  emitted key to `rollout/step` or `eval/step`; one-level namespace globs do
  not cover nested or environment-specific metric names.
- Do not change the historical meaning of `<env>/env_reward_mean` mid-project.
  If a reward profile scales or binarizes the environment score and raw partial
  credit is also needed, add/use `<env>/env_score_mean` from metadata
  `env_score` instead of repurposing `env_reward_mean`.
- Dynamic sampling drops zero-variance reward groups after RM reward computation.
  It does not resample individual samples.
- GLM-style padding is for infra-discard recovery: discard bad samples, then
  pad kept groups from valid samples. Never train on discarded samples as real
  data.
- Do not lightweight formal eval. Smoke/debug eval may use small prompt subsets,
  but any result used to judge training quality or compare runs must evaluate
  the full official split(s), record the dataset size/hash, and log explicit
  metrics such as `success_rate` and `env_reward_mean` rather than ambiguous
  aggregate reward keys.
- Preserve eval outcomes in `${ROOT_DIR}/runs/eval_summaries`: copy compact
  TSV/JSON summaries into `<env>/summaries/`, link raw run/eval directories from
  `<env>/raw_links/`, and add an `index.tsv` row. Label dev-only and smoke evals
  explicitly so they are not confused with full official eval.
- Before using remembered AppWorld numbers, read
  `../server-ops-discipline/references/appworld_eval_results.md`. It records the
  portable accepted protocol, checkpoint ancestry, formal results, and known
  invalid historical evaluations without requiring the original NAS.
- Checkpoint eval must use a path that actually loads the checkpoint weights.
  Megatron dist checkpoints need the actor-load plus weight-sync eval path;
  rollout-only eval is valid only for HF checkpoints already loadable by the
  rollout engine.
- Checkpoint sweeps must use the native eval launch path directly. Do not write
  run-local wrapper scripts or duplicate eval loops after native eval exists.
  Assign one checkpoint per node by overriding `NODE_INDICES`, `RUN_ROOT`, and
  `LOAD_DIR`, then parse metrics from the native train log, for example the
  `eval 0:` line.
- `examples/agent_env/<env>/eval_config.yaml` (or one explicitly selected
  `EVAL_CONFIG`) is the sole owner of native-eval datasets and sampling. The
  launcher may select checkpoints, nodes, and splits, but must not inject
  `EVAL_PROMPT_DATA`, temperature, response length, top-p, or top-k overrides.
  Fail on these legacy duplicate sources instead of silently changing the eval
  protocol.
- Train and native eval must share the same single policy-turn request
  lifecycle. Do not retry `/generate` inside an env episode: an inner retry can
  outlive the env request timeout and create orphaned policy sessions. If eval
  retry is required, retry a failed whole task with a fresh episode/session and
  report the initial failure separately.

## Checkpoints And Artifacts

- Every formal run must declare its checkpoint policy in the train profile or
  explicit launch override: save interval, max retained checkpoints, and
  whether optimizer/RNG state is saved.
- Formal resumable training should save optimizer/RNG state. If a run is
  intentionally weight-only, label it as eval-only or non-resumable in the run
  name or notes.
- Debug/smoke runs should disable checkpointing or cap retained checkpoints to
  one or two. Do not let failed launch/debug attempts accumulate checkpoints.
- For checkpoint-sweep experiments, keep all planned checkpoints only when
  storage budget is confirmed. Use sparse intervals such as 25 or 50 steps, not
  dense default saves.
- Do not silently change checkpoint cadence or retention between comparable
  runs. If a run resumes from checkpoint, record the source checkpoint and the
  intended total training step target.
- After a run is declared failed or superseded, remove its checkpoints after
  preserving the minimal evidence needed for debugging: resolved configs, W&B
  run id, selected logs, cleanup manifest, and any eval summary. If no eval was
  run, state that explicitly before deleting checkpoints.

## Credit Assignment Experiments

- Reward profiles are the sole owner of credit-assignment semantics. Run
  profiles may choose a reward profile and own group size, lifecycle, and
  bounded dump cadence, but must not duplicate TASA/ROPD/LUFFY parameters.
- All teacher and student judge inputs must pass through the shared structured
  trace renderer. Environment-specific cleanup belongs in the env adapter;
  generic ROPD/TASA code must not reconstruct traces from legacy text fields or
  add fallback renderers.
- Full TASA-GAE uses strict leave-one-out student evidence and
  coverage-adaptive teacher fill. With peer budget `B`, reached peers `N`, and
  peer outcome mean `V_MC`, use
  `V=(1-N/B)*V_teacher+(N/B)*V_MC`. Derive `B` from actual on-policy students,
  excluding LUFFY teacher samples.
- Keep group semantics explicit in comparable runs. The canonical pure and
  student-MC profiles use 16 on-policy students (`B=15`); LUFFY
  `replace_anchor` uses 15 students plus one teacher (`B=14`). Compare methods
  only against baselines with the same student count and LUFFY insertion mode.
- Full TASA has a teacher prior at every valid state and does not use the hard
  peer-support gate. The no-teacher/student-MC ablation keeps
  `tasa_min_peer_support` and ignores under-supported states as value anchors.
- A formal TASA run must retain bounded judge and credit dumps and emit the
  `reward/tasa/*` namespace. Before trusting a curve, audit strict LOO peer
  histograms excluding ROOT, state reuse by milestone depth, prerequisite
  validity, non-terminal set/unset precision, terminal advantage sign, and
  response-token segment alignment.
- Treat new credit estimators as experimental until one-step dump checks and a
  formal train/eval comparison both pass. Unit tests prove data contracts and
  arithmetic, not judge quality or task performance.

## Training Pitfalls

- Disable dropout for PPO/GRPO/RL training.
- For Slime PPO, expect a separate critic role. Configure role-specific
  Megatron entries or resolved overrides so actor and critic save to disjoint
  sibling directories, for example `checkpoints/actor` and
  `checkpoints/critic`. Fail fast if the paths overlap, because a critic save
  can otherwise overwrite actor weights and make eval appear broken.
- Keep PPO role-save fixes in launch/profile layers. Do not modify Slime core
  merely to separate actor and critic checkpoints.
- When using critic warmup such as `num_critic_only_steps`, do not expect actor
  metrics or actor checkpoints before warmup finishes. Judge early PPO health
  from rollout validity, reward/success/truncation, critic value loss, and
  critic grad norm; actor grad norm appears only after actor updates start.
- For PPO eval, load the actor checkpoint path, not the top-level checkpoint
  root or critic role path. If a checkpoint has a scalar value head such as
  `output_layer.weight` shaped `[1, hidden]`, treat it as critic-contaminated
  for policy eval and preserve logs before deleting/re-running.
- In full-async training, use rollout-time logprobs as the old policy
  denominator, for example `USE_ROLLOUT_LOGPROBS=1`.
- Treat `train_rollout_logprob_abs_diff=0` under rollout-logprob mode as a
  metrics wart if it compares rollout logprobs to themselves. Cross-check PPO
  KL, clipfrac, reward, length, and grad norm.
- Align `GLOBAL_BATCH_SIZE`, `ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT`, and
  full-async in-flight capacity unless deliberately testing staleness.
- Keep checkpointing sparse and capped during debugging.
- Keep debug dumps bounded. A dump setting used to inspect one rollout bug
  should not become a default for 100+ step training.
- When comparing agentic RL papers or repos, distinguish true multi-turn env
  interaction from pseudo multi-turn single-step training. Avoid treating
  `verl-agent`/GiGPO-style pipelines as evidence for our setting when they
  replay a short history window inside each prompt, train one action at a time,
  or do not let the environment own state transitions between policy calls.
  Prefer references where a rollout is one complete interactive episode and
  rewards are assigned from the environment trajectory.

## Change Discipline

- Locate the current owner of a setting before editing.
- Change the existing owner instead of adding wrappers, copied profiles, ad hoc
  env vars, or temporary YAML/env files.
- Create a new profile only for a durable reusable baseline.
- Remove dead compatibility paths and abandoned scripts.
- Before finishing, check that Slime core is untouched, run profiles stay thin,
  runtime paths are not hard-coded, and resolved files can audit the run.
