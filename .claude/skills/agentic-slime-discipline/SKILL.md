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

- `configs/agent_env/runs/*.env`: choose env, env config, model profile, train
  profile, topology profile, optional aux profile, and run naming.
- `configs/agent_env/models/*.env`: model identity, model args, loss-mask
  family, dropout/model compatibility defaults.
- `configs/agent_env/train/*.env`: env-specific training baseline, algorithm,
  sync/full-async mode, rollout filters, batch/token budgets, TP/CP,
  actor/rollout allocation, checkpointing, logging, and Slime flags.
- `configs/agent_env/topology/*.env`: node indexes, visible GPUs, and ports
  only. Keep algorithm, model, batch, token, aux, and env semantics out.
- `configs/agent_env/aux/*.env`: auxiliary endpoint identity and credentials
  only.
- `configs/nodes/*.txt`: local IP files only; commit examples, not real IPs.
- `examples/agent_env/<env>/env_config.yaml`: environment semantics, parser,
  task/data settings, env server settings, and reward configuration.

Train profile names should be:

```text
<env>_<algorithm>_<sync|fullasync>_<nodes>x<gpus>.env
```

Do not include model names or aux providers in train profile names.

## Runtime Contract

- Support image and conda-pack Slime runtimes through
  `scripts/utils/slime_runtime.sh`.
- Treat Megatron-LM as part of the Slime training runtime, not as a separate
  materialized source. A valid runtime is either an image that already contains
  Slime plus Megatron, or a Slime pack that contains `src/Megatron-LM`.
- Do not add `--sources Megatron-LM`, `MEGATRON_LOCAL_PATH`, NAS checkout
  fallbacks, or repo-specific Megatron paths to agentic training scripts.
- `MEGATRON_PATH` may appear only as the resolved backend path exported by the
  Slime runtime resolver and used in `PYTHONPATH`.
- W&B is an optional reporting runtime. Prefer a separate `wandb` env pack when
  cluster images ship incompatible SDKs, then fall back to the Slime runtime,
  then to a version-compatible local Python. Do not install W&B during training
  startup.

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
- Generic rollout code owns message/token accounting, env HTTP lease lifecycle,
  sample shape, common dumping, infra discard, and Slime data contracts.
- Reward implementations own final reward composition. Select them through
  `env_config.yaml` under `reward.impl` and task-specific reward fields.
- Reward post-process should adapt already-computed RM rewards to Slime's reward
  tensor contract. It should not secretly append env, format, or truncation
  rewards unless that is the selected reward implementation.
- Dynamic sampling drops zero-variance reward groups after RM reward computation.
  It does not resample individual samples.
- GLM-style padding is for infra-discard recovery: discard bad samples, then
  pad kept groups from valid samples. Never train on discarded samples as real
  data.

## Training Pitfalls

- Disable dropout for PPO/GRPO/RL training.
- In full-async training, use rollout-time logprobs as the old policy
  denominator, for example `USE_ROLLOUT_LOGPROBS=1`.
- Treat `train_rollout_logprob_abs_diff=0` under rollout-logprob mode as a
  metrics wart if it compares rollout logprobs to themselves. Cross-check PPO
  KL, clipfrac, reward, length, and grad norm.
- Align `GLOBAL_BATCH_SIZE`, `ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT`, and
  full-async in-flight capacity unless deliberately testing staleness.
- Keep checkpointing sparse and capped during debugging.

## Change Discipline

- Locate the current owner of a setting before editing.
- Change the existing owner instead of adding wrappers, copied profiles, ad hoc
  env vars, or temporary YAML/env files.
- Create a new profile only for a durable reusable baseline.
- Remove dead compatibility paths and abandoned scripts.
- Before finishing, check that Slime core is untouched, run profiles stay thin,
  runtime paths are not hard-coded, and resolved files can audit the run.
