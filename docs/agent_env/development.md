# Agent Env Development Guide

This guide explains how to extend the stack without turning it into a patch
pile. The core rule is: put behavior in the component that owns it, and use
Slime's import hooks instead of editing Slime source.

## Development Principles

1. Do not edit Slime source for agent-env behavior.
2. Prefer one source of truth for each setting.
3. Keep run profiles thin.
4. Keep train profiles model-agnostic.
5. Keep aux profiles task-agnostic.
6. Keep env configs responsible for env semantics.
7. Add new profiles only for durable reusable baselines.
8. Do not copy a whole config to change one scalar.
9. Resolve and inspect effective configs before debugging training math.
10. Commit source, configs, scripts, and docs; never commit generated run data.

## Adding a New Environment

Assume the new environment is named `scienceworld`.

Create:

```text
examples/agent_env/scienceworld/__init__.py
examples/agent_env/scienceworld/env_config.yaml
examples/agent_env/scienceworld/server.py
examples/agent_env/scienceworld/rollout.py
examples/agent_env/scienceworld/prompt_data.py
examples/agent_env/scienceworld/smoke_test.py
```

The env-specific `rollout.py` should be small. It should define the environment
specification and call the shared rollout loop. Keep generic multi-turn logic in
`examples/agent_env/rollout.py`.

The env-specific `server.py` should expose the standard server protocol through
the shared server implementation. Keep backend imports and reset/step/evaluate
logic inside the env folder.

The env config should own:

- data roots
- task splits
- max turns
- parser/action mode
- prompt template flags
- reward scale
- format/truncation penalties
- env server pool/concurrency settings

Do not add scienceworld settings to aux configs or launch scripts.

## Updating the Train Adapter for a New Env

Only add minimal env defaults in:

```text
examples/agent_env/scripts/run_agent_env_train.sh
```

The adapter may need:

```bash
CUSTOM_GENERATE_FUNCTION_PATH=examples.agent_env.scienceworld.rollout.generate
CUSTOM_CONFIG_PATH=examples/agent_env/scienceworld/env_config.yaml
DATA_DIR=${MLF_LOCAL_ROOT}/data/scienceworld
PROMPT_DATA_SCRIPT=${REPO_DIR}/examples/agent_env/scienceworld/prompt_data.py
```

Do not add env-specific URL variables. The launcher computes the router URL and
passes it to the train adapter as `--env-server-url`; the adapter passes it to
Slime unchanged, and rollout code reads `args.env_server_url`.

Do not put algorithm, topology, judge prompt, model identity, or reward weights
in this adapter.

## Adding Server Support in the Launcher

Add a case in:

```text
scripts/utils/launch_agentic_training.sh
```

Only start the env server with the correct Python env, PYTHONPATH, data env vars,
host, port, and config path. Do not put experiment semantics in the launcher.

If the env needs a new runtime pack, add materialization support in:

```text
scripts/utils/materialize_node_runtime.sh
scripts/utils/prepare_agentic_runtime.sh
scripts/utils/build_<env>_env.sh
scripts/utils/pack_agent_data.sh
```

## Adding a New Reward Judge

The judge integration belongs in:

```text
examples/agent_env/group_rm.py
```

The rollout loop should not know about a judge provider. It should only provide
raw trajectory data in `Sample` and `sample.metadata`.

The group RM function must support both forms:

```python
async def group_reward(args, samples: Sample | list[Sample], **kwargs):
    ...
```

For group mode, return one float per sample in the same order. For single mode,
return one float.

Provider config should enter through environment variables resolved from aux
profiles or external secrets:

```text
AGENT_ENV_JUDGE_MODE=aux
AUX_ENDPOINT_PROVIDER=...
AUX_ENDPOINT_MODEL=...
AUX_ENDPOINT_BASE_URL=...
AUX_ENDPOINT_API_KEY_PATH=...
```

`AGENT_ENV_JUDGE_MODE=none` should be deterministic and return zero judge score.
`AGENT_ENV_JUDGE_MODE=aux` should fail fast if endpoint or model is missing.

## Reward Post-Processing

Final reward composition belongs in:

```text
examples/agent_env/reward_post_process.py
```

This is the only place that should combine:

- environment reward
- judge score
- format adjustment
- truncation adjustment

The intended formula is:

```text
raw_reward = env_reward + judge_score + format_reward + truncated_reward
```

Keep each component recorded in metadata so W&B dumps and rollout cases can be
audited.

Do not duplicate reward composition in rollout, env server, group RM, or launch
scripts.

## Dynamic Filtering

Dynamic filtering for GRPO should remove groups that cannot produce useful
advantages. The current filter:

```text
examples.agent_env.reward_post_process.check_reward_nonzero_std
```

uses the raw reward formula and drops groups with zero standard deviation across
active samples.

Sync Slime applies this hook directly. Full-async profiles must use:

```text
examples.agent_env.fully_async_rollout.generate_rollout_fully_async
```

so filtering happens before actor consumption.

When adding new reward components, update both:

- `post_process_rewards`
- `raw_reward_for_filter`

They should remain mathematically aligned.

## Full-Async Wrapper Maintenance

The external full-async wrapper is:

```text
examples/agent_env/fully_async_rollout.py
```

It exists because full-async training needs dynamic filtering before actor
consumption. It deliberately avoids editing Slime source.

Maintenance rules:

1. Keep the wrapper small.
2. Reuse Slime's worker and sample types.
3. Do not duplicate env rollout logic.
4. Do not call env servers directly from the wrapper.
5. Record filter metrics through Slime's metric gatherer.
6. If upstream Slime changes the private worker API, update this wrapper.

## Model Profiles

Add or edit model profiles under:

```text
configs/agent_env/models/
```

Model profiles should set dropout to zero for RL unless deliberately testing
dropout:

```bash
MODEL_EXTRA_ARGS=${MODEL_EXTRA_ARGS:-"--attention-dropout 0.0 --hidden-dropout 0.0"}
```

Dropout can make rollout-time logprobs and train-time recomputed logprobs
inconsistent, which breaks PPO-style ratios. Treat non-zero dropout as a
training algorithm change, not a harmless model default.

Loss mask belongs here because it is model-family behavior:

```bash
LOSS_MASK_TYPE=qwen3
```

Do not move loss mask to run or topology profiles.

## Train Profiles

Train profiles are the right place for:

- `TRAIN_ENTRYPOINT`
- `AGENT_ENV_TRAIN_LOOP`
- `ROLLOUT_FUNCTION_PATH`
- `USE_ROLLOUT_LOGPROBS`
- `ADVANTAGE_ESTIMATOR`
- `NORMALIZE_ADVANTAGES`
- `DYNAMIC_SAMPLING_FILTER_PATH`
- `GLOBAL_BATCH_SIZE`
- `ROLLOUT_BATCH_SIZE`
- `N_SAMPLES_PER_PROMPT`
- token budgets
- TP/CP sizes
- actor and rollout GPU counts
- SGLang concurrency

Train profiles may be split by env, sync/full-async, and resource scale. Avoid
splitting by model.

## Topology Profiles

Topology profiles should stay small. They select nodes and ports:

```bash
NODES_FILE=configs/nodes/agent_env_all.txt
NODE_INDICES=1,2,3
ENV_PORT=18180
ROUTER_PORT=19000
RAY_PORT=6379
RAY_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS_PER_NODE_FOR_RAY=8
```

Do not put actor GPU counts or rollout GPU counts here. Those are train
profile settings because they affect Slime placement groups and batch sizing.

## Env Configs

Env configs should be readable without inspecting bash scripts. They should
answer:

- which data/task split is being used?
- what is the environment max turn limit?
- what output parser is expected?
- what reward scale and penalties are used?
- what server pool/concurrency does the env need?

If an env setting affects semantics, it belongs in the env config even if a
bash profile could technically export it.

## Testing Checklist

Before launching multi-node training:

1. Compile changed Python:

   ```bash
   /tmp/mlf-envs/slime/bin/python -m py_compile \
     examples/agent_env/train_entrypoint.py \
     examples/agent_env/rollout.py \
     examples/agent_env/fully_async_rollout.py \
     examples/agent_env/group_rm.py \
     examples/agent_env/reward_post_process.py
   ```

2. Check bash syntax:

   ```bash
   bash -n scripts/utils/launch_agentic_training.sh
   bash -n scripts/utils/prepare_agentic_runtime.sh
   bash -n examples/agent_env/scripts/run_agent_env_train.sh
   ```

3. Run env smoke test on a prepared node.
4. Launch a one-node sync run.
5. Launch a full-async run with `USE_ROLLOUT_LOGPROBS=1`.
6. Check first rollout case dumps.
7. Check dynamic filter metrics.
8. Check `ppo_kl`, `clipfrac`, `raw_reward`, and normalized `rewards`.

## Debugging Training Quality

When reward does not improve, inspect in this order:

1. `rollout/raw_reward`: is the env reward non-zero and task-scale sensible?
2. group variance: are most groups all correct or all wrong?
3. dynamic filter: are zero-variance groups dropped?
4. `USE_ROLLOUT_LOGPROBS`: enabled for full-async?
5. dropout: disabled in model profile?
6. loss mask: correct for model family?
7. format/truncation rates: are penalties dominating env reward?
8. env server: are actions valid and observations meaningful?
9. prompt: does it ask for the intended output format without duplicating model
   native thinking behavior?
10. sync baseline: does sync train when full-async does not?

Avoid jumping straight to hyperparameter tuning before these checks pass.

## Code Review Checklist

Before committing:

- no changes under Slime internals unless explicitly intended;
- no generated runs, W&B files, checkpoints, packs, or secrets;
- no new one-off copied configs for a single scalar;
- new keys appear in resolved profiles;
- docs mention any new public profile or operational command;
- `scripts/utils` is used in references, not the old `scripts/mlf` path;
- bash scripts pass `bash -n`;
- changed Python compiles in the Slime env.

## Commit Strategy

Prefer commits that separate:

- source code changes;
- config/profile changes;
- docs/operations changes;
- mechanical path renames.

For handoff to another cluster agent, it is acceptable to make one snapshot
commit when the user explicitly wants the current working stack uploaded. In
that case, mention that the commit is a migration snapshot and list the main
areas it contains.
