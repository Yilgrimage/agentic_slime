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
For `mcp_server`, each task family should use its own env config, for example
`examples/agent_env/mcp_server/env_config_ipr_product_check.yaml`. The run
profile selects that file through `ENV_CONFIG`; train and launch profiles must
not hard-code IPR, search, or other task-family semantics.
Reward implementation and task-specific reward data also belong under the
selected env config's `reward:` section. Use `reward.impl` and
`reward.judge_mode` for the active RM path, and keep ROPD teacher/rubric files
or Valleydance process-reward weights out of generic train profiles.

## Rules

- Do not create a copied profile to change one scalar for a one-off run. Change
  the existing owner file, or create a new profile only for a durable baseline.
- Train profile names are model-agnostic:

```text
<env>_<algorithm>_<sync|fullasync>_<nodes>x<gpus>.env
```

- Real IPs live only in local ignored node files such as
  `configs/nodes/agent_env_all.txt`. Commit only templates such as
  `configs/nodes/agent_env_all.txt.example`; topology profiles select
  zero-based indexes from the local file.
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
