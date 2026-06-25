---
name: agent-env-ops-discipline
description: Use when operating the local agent-env Slime training cluster, preparing node runtimes, managing env/data packs, launching distributed runs, preserving GPUs with bench/watchdog, or moving the workflow to another cluster.
---

# Agent Env Ops Discipline

Keep server operations reproducible and separate from experiment semantics.
Prefer reusable packs, node-indexed topology, and explicit launch profiles over
manual per-node fixes.

## Runtime Model

- NAS is the durable source for code, data, models, packs, secrets, and runs.
- Node-local state under `/tmp/mlf-envs` and `/tmp/mlf-runtime` is disposable.
- Training should never install Python packages during startup. Build or refresh
  reusable packs first, then materialize them on nodes.
- Model checkpoints stay on NAS unless a specific experiment proves local copy
  is required.

Canonical layout:

```text
/mnt/bn/jixf-nas-lq/mlf/code
/mnt/bn/jixf-nas-lq/mlf/data
/mnt/bn/jixf-nas-lq/mlf/models
/mnt/bn/jixf-nas-lq/mlf/packs
/mnt/bn/jixf-nas-lq/mlf/runs
/mnt/bn/jixf-nas-lq/mlf/secrets
/tmp/mlf-envs
/tmp/mlf-runtime
```

## Node And IP Handling

- Real node IPs live in `configs/nodes/agent_env_all.txt`.
- Topology profiles select zero-based node indexes from that file.
- Do not hard-code IPs in run/train/env/aux profiles or scripts unless the file
  is explicitly operational node state.
- After a resource reset, update the node file first, then materialize runtime,
  then launch training.

## Environment Pack Discipline

- Use `scripts/utils/prepare_agentic_runtime.sh` for multi-node preparation.
- Use `scripts/utils/materialize_node_runtime.sh` only as the single-node worker.
- Use `scripts/utils/build_*.sh`, `pack_*.sh`, and `publish_*.sh` to refresh
  reusable packs.
- Keep env packs separate by dependency domain: slime, alfworld, webshop, tau2,
  appworld, plus data/source packs as needed.
- Do not patch dependency checkouts manually during a run. Keep repeatable
  dependency changes as patch files under `scripts/utils/patches/`.

## Launch Discipline

- Launch through `scripts/utils/launch_agentic_training.sh <run-profile.env>`.
- The launcher orchestrates Ray, env servers, router, optional aux, and train
  submission. It should not become a training hyperparameter owner.
- The train adapter translates resolved profiles into Slime CLI args. It should
  not grow another override language.
- Effective values must be inspectable in resolved profiles under
  `RUN_ROOT/logs/`.

## GPU Keepalive

- Keepalive is cluster bootstrap logic, not Slime training logic.
- `run_bench.sh` and `gpu_idle_watchdog.sh` live under the NAS `bash/` tree,
  outside the Slime repo.
- Watchdog should protect idle GPUs by utilization only. Do not make it depend
  on Ray, Slime, tmux training sessions, or process-name heuristics.
- Before launching training, stop bench on training nodes. After failures or
  normal exits, watchdog/bench should restore occupancy if GPUs stay idle.

## Git And Artifact Hygiene

- Git contains source, small configs, scripts, skills, and concise docs.
- Git must not contain run logs, checkpoints, W&B outputs, packs, data, model
  weights, secrets, or node-local runtime materialization.
- Keep `upstream` read-only for Slime updates. Push local agent-env work to the
  user fork branch, currently `origin/agentic-env-backend`.

## Debugging Priority

When a run fails, check in this order:

1. `RUN_ROOT/logs/resolved_*.env` for wrong effective config.
2. `multi_head.log`, Ray worker logs, env server logs, and train log for startup
   failures.
3. GPU utilization and tmux sessions to distinguish startup, rollout, actor, and
   keepalive states.
4. W&B metrics only after confirming the pipeline is actually training.

Avoid adding shell patches before proving which owner should change.
