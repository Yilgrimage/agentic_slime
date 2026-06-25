# Agent Env Profiles

This directory contains launchable experiment profiles for the external
agent-env backend. Keep it thin and auditable.

## Ownership

- `runs/*.env`: selects env, env config, model profile, train profile, topology
  profile, optional aux profile, and experiment naming.
- `models/*.env`: model identity, model args, loss-mask family, dropout
  defaults, and model compatibility defaults.
- `train/*.env`: env-specific training baseline, algorithm, sync/full-async
  mode, batch/token budget, TP/CP, actor/rollout allocation, checkpointing,
  filtering, and Slime training flags.
- `topology/*.env`: node indexes, visible GPUs, and ports only.
- `aux/*.env`: optional auxiliary inference endpoint only.

Environment semantics belong in `examples/agent_env/<env>/env_config.yaml`.

## Rules

- Do not create a copied profile to change one scalar for a one-off run. Change
  the existing owner file, or create a new profile only for a durable baseline.
- Train profile names are model-agnostic:

```text
<env>_<algorithm>_<sync|fullasync>_<nodes>x<gpus>.env
```

- Real IPs live in `configs/nodes/agent_env_all.txt`; topology profiles select
  zero-based indexes from it.
- Values meant to be overridden should be declared as `${VAR:-default}` and
  must appear in the resolved profile under `RUN_ROOT/logs/`.
- Keep credentials, W&B keys, run logs, checkpoints, data, models, and runtime
  packs out of git.

Launch through:

```bash
bash scripts/utils/launch_agentic_training.sh \
  configs/agent_env/runs/alfworld_qwen3_4b_grpo_fullasync_3x8.env
```

For design notes and known pitfalls, read `docs/agent_env/README.md`.
