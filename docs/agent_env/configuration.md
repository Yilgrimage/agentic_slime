# Agent Env Configuration

The configuration tree is designed to make every parameter have one owner.
When changing a setting, edit the owner file instead of adding a wrapper,
temporary override file, or copied profile.

## Ownership Rules

```text
configs/agent_env/runs/*.env
```

Run profiles are thin manifests. They select:

- `ENV_NAME`
- `ENV_CONFIG`
- `MODEL_PROFILE`
- `TRAIN_PROFILE`
- `TOPOLOGY_PROFILE`
- optional `AUX_PROFILE`
- `EXP_PROJECT`
- `RUN_NAME` / `EXP_NAME`

Run profiles should not contain algorithm knobs, batch size, token budget,
model args, or environment semantics.

```text
configs/agent_env/models/*.env
```

Model profiles own:

- model basename
- HF checkpoint directory
- Megatron torch-dist directory
- Megatron model args script
- model-family loss mask
- dropout/model compatibility defaults
- model extra args

Model profiles should not select sync/full-async mode, topology, aux endpoints,
or task reward semantics.

```text
configs/agent_env/train/*.env
```

Train profiles own:

- environment-specific training baseline
- algorithm: GRPO, R++, etc.
- sync or full-async entrypoint
- rollout function path
- dynamic sampling filter
- global batch size and rollout batch size
- samples per prompt
- context/response token budgets
- rollout temperature
- TP/CP/PP and recompute settings
- actor and rollout GPU allocation
- SGLang engine settings
- checkpoint interval and total steps
- W&B on/off
- Slime training behavior such as `USE_ROLLOUT_LOGPROBS`

Train profiles are allowed to differ by environment because ALFWorld, WebShop,
tau2, and AppWorld naturally need different token budgets, rollout lengths, and
sampling behavior.

```text
configs/agent_env/topology/*.env
```

Topology profiles own only physical placement:

- `NODES_FILE`
- `NODE_INDICES`
- optional `AUX_NODES_FILE`
- optional `AUX_NODE_INDICES`
- ports
- `RAY_CUDA_VISIBLE_DEVICES`
- `NUM_GPUS_PER_NODE_FOR_RAY`

Topology profiles should not own global batch size, rollout batch size, TP/CP,
actor/rollout GPU counts, or algorithm choices. Those are train-profile
settings because they affect training semantics and performance.

```text
configs/agent_env/aux/*.env
```

Aux profiles own optional OpenAI-compatible inference endpoint details:

- provider
- model name
- serving env
- port
- TP size
- memory fraction
- max tokens
- thinking flags
- API key path

They should not contain tau2-specific or ALFWorld-specific env semantics.

```text
examples/agent_env/<env>/env_config.yaml
```

Env configs own environment behavior:

- task/data paths
- server pool/concurrency settings
- parser/action mode
- prompt templates or prompt flags
- reward scale and penalties
- max turns
- env backend settings

Env configs do not own model identity, topology, Ray ports, or algorithm
selection.

## Profile Naming

Train profile names use:

```text
<env>_<algorithm>_<sync|fullasync>_<nodes>x<gpus>.env
```

Examples:

```text
alfworld_grpo_sync_1x8.env
alfworld_grpo_sync_2x8.env
alfworld_grpo_fullasync_3x8.env
tau2_rpp_fullasync_3x8.env
webshop_grpo_fullasync_4x8.env
```

Do not include model names or aux providers in train profile names. Model names
belong in run profiles through `MODEL_PROFILE`; aux providers belong in run
profiles through `AUX_PROFILE`.

Run profile names can include model and env because they are launchable
experiment manifests:

```text
alfworld_qwen3_4b_grpo_fullasync_3x8.env
tau2_qwen35_4b_grpo_qwen27_3train_1aux.env
```

## Node Files and Index Selection

The current train node IPs live in:

```text
configs/nodes/agent_env_all.txt
```

This is operational state. After cluster reset, update this file and avoid
editing every topology or run profile.

Topology profiles select nodes by zero-based index:

```bash
NODES_FILE=configs/nodes/agent_env_all.txt
NODE_INDICES=1,2,3
```

Aux nodes use the same pattern:

```bash
AUX_NODES_FILE=configs/nodes/agent_env_all.txt
AUX_NODE_INDICES=0
```

This makes node replacement a one-file operation.

## Declared Override Pattern

Most profile keys are written as:

```bash
KEY=${KEY:-default}
```

This allows explicit launch-time overrides while still making the default
auditable in the profile.

An override is considered clean only if:

1. the key is declared in the relevant profile;
2. the launcher or train adapter consumes it;
3. it appears in `RUN_ROOT/logs/resolved_*.env`;
4. the override is used for a short experiment, not as a hidden permanent
   behavior.

If a key should persist, edit the profile owner file.

## Important Run Profile Fields

Minimal run profile:

```bash
ENV_NAME=alfworld
ENV_CONFIG=examples/agent_env/alfworld/env_config.yaml
MODEL_PROFILE=configs/agent_env/models/qwen3_4b.env
TRAIN_PROFILE=configs/agent_env/train/alfworld_grpo_fullasync_3x8.env
TOPOLOGY_PROFILE=configs/agent_env/topology/3train_8gpu.env
EXP_PROJECT=Qwen3-4B_alfworld_grpo
RUN_NAME=alfworld-Qwen3-4B-grpo-fullasync-3x8
```

Optional aux:

```bash
AUX_PROFILE=configs/agent_env/aux/qwen36_27b_sglang.env
```

## Important Train Profile Fields

Training entrypoint:

```bash
TRAIN_ENTRYPOINT=train.py        # sync
TRAIN_ENTRYPOINT=train_async.py  # full-async
```

Rollout function:

```bash
ROLLOUT_FUNCTION_PATH=slime.rollout.sglang_rollout.generate_rollout
ROLLOUT_FUNCTION_PATH=examples.agent_env.fully_async_rollout.generate_rollout_fully_async
```

Dynamic filter:

```bash
DYNAMIC_SAMPLING_FILTER_PATH=examples.agent_env.reward_post_process.check_reward_nonzero_std
```

Reward hooks:

```bash
GROUP_RM=1
CUSTOM_RM_PATH=examples.agent_env.group_rm.group_reward
CUSTOM_REWARD_POST_PROCESS_PATH=examples.agent_env.reward_post_process.post_process_rewards
AGENT_ENV_JUDGE_MODE=none
```

Async correctness:

```bash
USE_ROLLOUT_LOGPROBS=1
```

Batch and rollout:

```bash
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=8
ROLLOUT_MAX_CONTEXT_LEN=32768
ROLLOUT_MAX_RESPONSE_LEN=8192
ROLLOUT_TEMPERATURE=1
```

Actor performance:

```bash
TP_SIZE=4
CP_SIZE=1
MAX_TOKENS_PER_GPU=16384
RECOMPUTE_GRANULARITY=full
RECOMPUTE_METHOD=uniform
RECOMPUTE_NUM_LAYERS=1
```

Resource allocation:

```bash
ACTOR_NUM_NODES=1
ACTOR_GPUS=8
ROLLOUT_GPUS=16
NUM_GPUS=8
ROLLOUT_TP_SIZE=1
SGLANG_SERVER_CONCURRENCY=16
SGLANG_MEM_FRACTION_STATIC=0.55
```

## Sync Profile Guidance

Use sync profiles for correctness checks:

- one or two nodes;
- colocated actor/rollout;
- stock Slime sync rollout;
- dynamic filter enabled;
- `USE_ROLLOUT_LOGPROBS=0` is acceptable.

Sync is useful to answer: "Can this env/model/reward setup train at all without
full-async staleness?"

## Full-Async Profile Guidance

Use full-async profiles for throughput:

- one actor node and one or more rollout nodes;
- external agent-env full-async rollout wrapper;
- dynamic filter enabled;
- `USE_ROLLOUT_LOGPROBS=1`;
- actor/rollout split sized from observed GPU utilization.

Full-async is useful only after sync correctness is established.

## Adding a New Model

1. Add `configs/agent_env/models/<model>.env`.
2. Point `MODEL_ARGS_SCRIPT` at an existing or new `scripts/models/*.sh`.
3. Set `MODEL_BASENAME`, `MODEL_DIR`, `TORCH_DIST_DIR`.
4. Set `LOSS_MASK_TYPE` for the model family.
5. Set dropout to zero unless there is a deliberate reason not to.
6. Convert HF checkpoint to Megatron torch-dist using
   `scripts/utils/convert_model_to_torch_dist.sh` or the documented Slime
   conversion command.
7. Select this model from a run profile.

Do not copy train profiles just to switch model family.

## Adding a New Train Baseline

Create a new train profile only when the baseline is durable and reusable, for
example:

- new env-specific token budget;
- sync versus full-async;
- different node/GPU resource scale;
- algorithm family that changes more than one knob;
- a stable ablation that must run repeatedly.

Do not create a full new profile to change one scalar for a temporary run. Edit
the existing owner or pass a declared override for that run.

## Adding a New Environment

New env config:

```text
examples/agent_env/<env>/env_config.yaml
```

New env code:

```text
examples/agent_env/<env>/rollout.py
examples/agent_env/<env>/server.py
examples/agent_env/<env>/prompt_data.py
examples/agent_env/<env>/smoke_test.py
```

New train profile:

```text
configs/agent_env/train/<env>_grpo_fullasync_<nodes>x<gpus>.env
```

New run profile:

```text
configs/agent_env/runs/<env>_<model>_grpo_fullasync_<nodes>x<gpus>.env
```

Add env defaults in `examples/agent_env/scripts/run_agent_env_train.sh` only for
paths and env-url variable names needed by the generic adapter. Do not put
reward semantics or task-specific prompt settings in the adapter.

## Resolved Profiles

Always inspect:

```bash
sed -n '1,220p' "$RUN_ROOT/logs/resolved_launch.env"
sed -n '1,260p' "$RUN_ROOT/logs/resolved_train_profile.env"
```

These files reveal what the launch script actually used. They also help another
cluster reproduce a run without reverse-engineering a long shell command.

## Common Configuration Mistakes

- Putting real IPs into several run profiles. Use `configs/nodes/agent_env_all.txt`.
- Creating `env_config_format0.yaml` or similar full copies for one scalar.
- Putting model names into train profiles.
- Putting aux provider settings into env configs.
- Running full-async with `USE_ROLLOUT_LOGPROBS=0`.
- Running GRPO without a dynamic filter when most groups are all-correct or
  all-wrong.
- Adding launch-script fallback behavior for one failing experiment instead of
  fixing the owner config.
