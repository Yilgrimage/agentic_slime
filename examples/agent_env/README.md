# Agent Env Runtime

This folder contains the external agent-env integration used by Slime custom
hooks. Keep generic orchestration here and task-specific behavior inside each
environment folder.

## Shared Modules

- `server.py`: generic process-pool environment HTTP server.
- `router.py`: generic multi-node lease router.
- `episode.py`: Slime-side policy gateway, token ledger, env episode RPC, sample
  dumping, and common shape handling.
- `env_episode.py`: env-side helpers for calling the policy gateway from a
  server-owned episode loop.
- `rollout.py`: shared rollout utilities used by server-owned episode
  adapters.
- `fully_async_rollout.py`: external full-async rollout wrapper with dynamic
  filtering and GLM-style padding support.
- `group_rm.py`: single/group RM hook that dispatches to reward
  implementations.
- `reward_post_process.py`: adapter from RM-produced rewards to Slime's reward
  tensor path, including optional grouped normalization.
- `metrics.py`: generic rollout/eval metric aggregation.
- `train_entrypoint.py`: registers agent-env CLI args and dispatches to Slime
  sync/full-async train loops.
- `scripts/run_agent_env_train.sh`: translates resolved profiles into the Slime
  CLI.

## Environment Folders

`<env>/` owns prompt rendering, parser/action semantics, server-side episode
loops, backend internals, success/score interpretation, task data settings,
and env-specific smoke tests.

Env servers are pure environment RPC services. They own process isolation,
leases, episode loops, and backend state. The public training contract is one
episode request: `POST /run_episode`. Env servers call the Slime-side policy
gateway when they need a model action. They must not own chat templating,
tokenizer state, rollout logprobs, loss masks, or Slime `Sample` construction.
Those stay in Slime-side rollout code so training tokens and masks follow the
same contract as the rest of Slime.

The shared rollout layer should not know task data layout or reward semantics.
Keep task/env semantics in `<env>/env_config.yaml`, and keep reward composition
in the selected `configs/agent_env/rewards/*.yaml` profile.

## Boundaries

- Slime core is not edited for agent-env behavior; use CLI hook paths.
- Runtime env/router URL is passed as the explicit `--env-server-url` argument.
  Do not rely on hidden environment variables for this.
- Env score, format reward, truncation penalty, and optional judge reward are
  composed by the selected RM implementation, not by rollout post-processing.
- Infra-discard and padding exist to keep bad samples out of actor training,
  not to make invalid samples real training data.

For config ownership and common training pitfalls, read
`docs/agent_env/README.md`.
