# Agent Env Runtime

This folder contains the external agent-env integration used by Slime custom
hooks. Keep generic orchestration here and task-specific behavior inside each
environment folder.

## Shared Modules

- `server.py`: generic process-pool environment HTTP server.
- `router.py`: generic multi-node lease router.
- `rollout.py`: generic multi-turn custom-generate loop, token ledger, env HTTP
  calls, sample dumping, and common shape handling.
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

`<env>/` owns prompt rendering, parser/action semantics, backend reset/step,
success/score interpretation, task data settings, and env-specific smoke tests.

The shared rollout layer should not know task data layout or environment reward
semantics. Keep those in `<env>/env_config.yaml` and `<env>/rollout.py`.

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
