# Agent Environment Router

`env_router.py` is a generic lease router for agentic environment pool servers.
It is intentionally environment-agnostic: ALFWorld, WebShop, and future
ScienceWorld servers expose the same small HTTP lease protocol, while the router
only chooses a worker at allocation time and forwards later requests to the same
worker.

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
python -m examples.agent_env.env_router \
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

Keep generic behavior in this router:

- worker discovery from configured URLs
- task-key based worker selection
- capacity/unreachable fallback
- global lease encoding and sticky forwarding
- aggregate health/status
