# Agent Environment Runtime

This folder contains the environment-agnostic pieces shared by agentic examples.
ALFWorld, WebShop, and future ScienceWorld adapters should keep only
environment-specific reset/step logic in their own folders.

Shared modules:

- `router.py`: generic multi-worker lease router.
- `server.py`: generic process-pool lease server and HTTP protocol.
- `rollout.py`: generic Slime custom-generate agent loop, including policy
  calls, text-action/tool-call parsing, context accounting, env HTTP calls,
  token reward alignment, and lease cleanup.
- `metrics.py`: generic rollout/eval metric aggregation for Slime
  logging.
- `scripts/run_agent_env_grpo.sh`: generic Slime GRPO runner shared by
  environment wrappers.

Environment folders provide a small `AgentEnvSpec` plus a backend implementation:

- prompt and observation rendering
- available/admissible action extraction
- invalid-action fallback policy
- success/outcome interpretation
- env-specific reset payload and metadata
- backend-specific imports, reset/step/evaluate behavior, and data paths

Generic endpoints:

- `GET /health` or `GET /healthz`
- `GET /status`
- `POST /allocate`
- `POST /reset`
- `POST /step`
- `POST /evaluate`
- `POST /close`

Allocation requires a stable `task_key` when possible. The router maps
`sha1(task_key) % num_workers` to a primary worker and falls back to later
workers if the primary worker is unreachable or capacity constrained. The worker
lease is encoded as `<worker_idx>:<worker_lease_id>`, which makes `reset`,
`step`, `evaluate`, and `close` sticky to the worker that owns the environment.

Example:

```bash
python -m examples.agent_env.router \
  --host 0.0.0.0 \
  --port 18080 \
  --workers http://node0:18180,http://node1:18180,http://node2:18180,http://node3:18180
```

Keep environment-specific behavior in the env server:

- reset semantics and task ordering
- available actions
- action validation at the environment boundary
- observations, scores, done/success flags
- environment-local metadata

Keep generic behavior in this folder:

- worker discovery from configured URLs
- task-key based worker selection
- capacity/unreachable fallback
- global lease encoding and sticky forwarding
- aggregate health/status
- process-pool worker lifecycle
- lease allocation, idempotency, TTL, and release
- multi-turn rollout bookkeeping
- token-level reward list shape management
- common rollout/eval logging

Launch/config convention:

- env behavior belongs in `<env>/train_config.yaml`;
- train/model/topology/aux parameters belong in `configs/agent_env/**/*.env`;
- optional auxiliary inference parameters belong in an aux config, not in the
  top-level launch script;
- use `examples/agent_env/scripts/launch_agent_env.sh <run-profile.env>` as the
  top-level launch entrypoint.

## Current source of truth

Use the GitHub fork as the code source of truth and the NAS checkout as the
runtime checkout:

- Git remote: `https://github.com/Yilgrimage/agentic_slime.git`
- NAS checkout: `/mnt/bn/jixf-nas-lq/mlf/code/slime`
- Node-local runtime: `/tmp/mlf-envs` and `/tmp/mlf-runtime`

When moving to a cloud development machine with the NAS mounted, develop
directly in the NAS checkout or clone the GitHub fork there and keep that clone
as the only editable copy. Avoid editing both a laptop checkout and the NAS
checkout in parallel; sync drift is otherwise hard to reason about during
training.

## Profile-driven launch

Normal tau2 launch:

```bash
cd /mnt/bn/jixf-nas-lq/mlf/code/slime
bash examples/agent_env/scripts/launch_agent_env.sh \
  configs/agent_env/runs/tau2_qwen35_4b_grpo_m2p7_3train_1aux.env
```

The selected run profile sources:

- one topology profile from `configs/agent_env/topology/`;
- one train profile from `configs/agent_env/train/`;
- optionally one aux profile from `configs/agent_env/aux/`;
- one env config from `examples/agent_env/<env>/train_config.yaml`.

The launcher writes the resolved profiles into the run log directory before
starting Ray, env servers, the router, optional aux inference, and the Slime
train driver. If a parameter should be considered part of an experiment, add it
to the appropriate profile rather than hiding it in ad-hoc shell variables.

## Environment responsibilities

The shared rollout layer handles HTTP leases, token ledger bookkeeping, common
format-reward accounting, sample dumping, and Slime custom generation. It should
not contain environment-specific reset semantics.

Environment folders should own:

- task selection and split handling;
- prompt/observation/action rendering;
- backend reset/step/evaluate calls;
- success and score interpretation;
- optional tool-use or text-action parser choice.

For tool-use environments, keep policy-facing tool schemas and usersim behavior
inside the env-specific adapter. For text-action environments such as ALFWorld
and WebShop, keep tag parsing local to the env adapter while still using the
shared message/token ledger.
