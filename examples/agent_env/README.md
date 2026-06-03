# Agent Environment Runtime

This folder contains the environment-agnostic pieces shared by agentic examples.
ALFWorld, WebShop, and future ScienceWorld adapters should keep only
environment-specific reset/step logic in their own folders.

Shared modules:

- `router.py`: generic multi-worker lease router.
- `server.py`: generic process-pool lease server and HTTP protocol.
- `rollout.py`: generic Slime custom-generate agent loop, including policy
  calls, `<think>/<action>` parsing, context trimming, env HTTP calls, token
  reward alignment, and lease cleanup.
- `metrics.py`: generic rollout/eval metric aggregation for Slime
  logging.

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
