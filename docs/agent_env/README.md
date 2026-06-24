# Agent Env Training Stack

This document set describes the non-invasive agent-environment training stack
built on top of Slime. It is intended for developers moving the code to another
cluster or extending it with new environments, reward models, or launch
profiles.

The current implementation keeps Slime source code untouched. All integration
points are external modules selected through Slime CLI options such as
`--custom-generate-function-path`, `--custom-reward-post-process-path`,
`--custom-rm-path`, `--dynamic-sampling-filter-path`, and
`--rollout-function-path`.

## Documents

- [Architecture](architecture.md): component boundaries, launch flow, rollout
  flow, reward flow, sync/full-async behavior, and the exact non-invasive Slime
  hooks.
- [Configuration](configuration.md): ownership rules for run/model/train/topology
  and env configs, profile naming, important fields, and how to add new
  profiles without config sprawl.
- [Operations](operations.md): cluster bootstrap, runtime materialization,
  Ray/env-server launch, GPU keepalive, monitoring, reset, and migration
  checklist.
- [Development](development.md): how to add environments, models, reward judges,
  metrics, and tests while keeping responsibilities clean.

## Current Repo Layout

```text
configs/agent_env/
  runs/             Thin launch manifests.
  models/           Model identity, model args, loss mask, dropout defaults.
  train/            Algorithm, sync/full-async mode, batch/token/TP layout.
  topology/         Node index selection, ports, visible GPUs.
  aux/              Optional auxiliary OpenAI-compatible inference endpoint.

configs/nodes/
  agent_env_all.txt Single source of truth for current cluster node IPs.

examples/agent_env/
  server.py         Generic process-pool environment HTTP server.
  router.py         Generic multi-node lease router.
  rollout.py        Generic Slime custom-generate agent loop.
  fully_async_rollout.py
                    External full-async rollout wrapper that preserves Slime's
                    worker model while applying dynamic filtering.
  reward_post_process.py
                    Final reward combiner and dynamic sampling filter.
  group_rm.py       Optional group/single sample reward-model judge hook.
  metrics.py        Generic rollout metric aggregation.
  <env>/            Environment-specific prompt/action/backend logic.

examples/agent_env/scripts/
  run_agent_env_train.sh
                    Train adapter that translates resolved profiles into a
                    Slime CLI invocation.

scripts/utils/
  launch_agentic_training.sh
                    Profile-driven distributed launcher.
  prepare_agentic_runtime.sh
                    Multi-node runtime materialization helper.
  materialize_node_runtime.sh
                    Single-node env/data/source materializer.
  aux_endpoint.sh   Optional auxiliary inference server manager.
  build_*.sh, pack_*.sh, publish_*.sh
                    Pack creation and publication utilities.
```

The old path `scripts/mlf/` has been renamed to `scripts/utils/`. Future commands
and documentation should use `scripts/utils/...`.

## Quick Start

Prepare node-local runtimes after updating `configs/nodes/agent_env_all.txt`:

```bash
cd /mnt/bn/jixf-nas-lq/mlf/code/slime
bash scripts/utils/prepare_agentic_runtime.sh \
  --all-nodes \
  --orchestrator head \
  --nodes configs/nodes/agent_env_all.txt \
  --node 0,1,2,3 \
  --envs slime,alfworld,webshop,tau2 \
  --data alfworld,webshop,tau2 \
  --sources webshop,tau2 \
  --models none
```

Launch a profile-driven training run:

```bash
cd /mnt/bn/jixf-nas-lq/mlf/code/slime
bash scripts/utils/launch_agentic_training.sh \
  configs/agent_env/runs/alfworld_qwen3_4b_grpo_fullasync_3x8.env
```

Inspect the run directory:

```bash
run=/mnt/bn/jixf-nas-lq/mlf/runs/Qwen3-4B_alfworld_grpo/alfworld-Qwen3-4B-grpo-fullasync-3x8
ls "$run/logs"
tail -f "$run/logs/alfworld_train.log"
```

Every run writes resolved config files under `RUN_ROOT/logs/`:

```text
resolved_launch.env
resolved_train_profile.env
resolved_aux_profile.env  # only when AUX_PROFILE is selected
```

These resolved files are the best first artifact to check when debugging a run
or moving it to another cluster.

## What Is Non-Invasive Here

No Slime source file under `slime/`, `slime_plugins/`, `train.py`, or
`train_async.py` is required to be modified for agent-env training.

The external integration points are:

```text
--custom-generate-function-path examples.agent_env.<env>.rollout.generate
--custom-reward-post-process-path examples.agent_env.reward_post_process.post_process_rewards
--dynamic-sampling-filter-path examples.agent_env.reward_post_process.check_reward_nonzero_std
--group-rm
--custom-rm-path examples.agent_env.group_rm.group_reward
--rollout-function-path examples.agent_env.fully_async_rollout.generate_rollout_fully_async
```

For sync training, Slime's stock rollout function can be used:

```text
slime.rollout.sglang_rollout.generate_rollout
```

For full-async training, the agent-env external rollout wrapper is used so that
dynamic sampling filtering is applied before the actor consumes a batch. This is
still non-invasive because it is selected by `--rollout-function-path` rather
than by editing Slime internals.

## Current Known Training Patterns

ALFWorld diagnostics currently use two useful profiles:

- `alfworld_qwen3_4b_grpo_sync_1x8.env`: one-node colocated sync baseline.
- `alfworld_qwen3_4b_grpo_fullasync_3x8.env`: three-node full-async baseline
  with one actor node and two rollout nodes.

The full-async profile currently enables:

- `USE_ROLLOUT_LOGPROBS=1`, required for correct PPO/GRPO ratios when rollout
  data can be off-policy relative to the current actor.
- `DYNAMIC_SAMPLING_FILTER_PATH=examples.agent_env.reward_post_process.check_reward_nonzero_std`,
  which drops zero-variance reward groups before actor training.
- `GROUP_RM=1` with `AGENT_ENV_JUDGE_MODE=none` by default. This keeps the RM
  hook structurally active but returns zero judge score unless an aux judge is
  explicitly enabled.

## Git Hygiene

This branch is the development branch for the agent-env backend:

```bash
git branch --show-current
# agentic-env-backend
```

The intended push target is the user's fork:

```text
origin  https://github.com/Yilgrimage/agentic_slime.git
upstream https://github.com/THUDM/slime.git
```

Use `origin/agentic-env-backend` for development pushes. Keep `upstream` read
only so that new Slime releases can be fetched and merged or rebased without
accidentally pushing local agent-env work to the upstream project.
