---
name: server-ops-discipline
description: "Use when preparing remote servers or GPU nodes, managing reusable runtime or data packs, materializing node-local environments, handling node IP files, launching distributed jobs, preserving idle GPUs with watchdog/bench, separating durable NAS state from disposable local state, or moving this workflow to another cluster."
---

# Server Ops Discipline

Keep server operations reproducible. Durable state belongs on shared storage or
in git; node-local state is disposable and should be rebuilt from packs or
scripts.

## Bundled Scripts

This skill ships reusable ops scripts under `scripts/`. Install or sync them to
the shared cluster root before wiring project launchers to them:

```bash
mkdir -p "${ROOT_DIR}/scripts"
cp -a server-ops-discipline/scripts/. "${ROOT_DIR}/scripts/"
chmod +x "${ROOT_DIR}/scripts/"*.sh
```

Use `ROOT_DIR` as the portable root for the cluster workspace. Existing MLF
deployments may still export `MLF_NAS_ROOT`; the bundled scripts accept it as a
compatibility alias, but new project code should locate ops helpers through
`${ROOT_DIR}/scripts`.

Stable interfaces:

```text
${ROOT_DIR}/scripts/run_bench.sh start|stop|status|restart [--nodes nodes.txt --node 0,1]
${ROOT_DIR}/scripts/gpu_idle_watchdog.sh start|stop|status|restart
${ROOT_DIR}/scripts/prepare_node_runtime.sh --local-only|--all-nodes ...
${ROOT_DIR}/scripts/materialize_node_runtime.sh --envs ... --data ... --sources ...
${ROOT_DIR}/scripts/pack_data.sh
```

Launchers that need `bench_on_exit` must call `${ROOT_DIR}/scripts/run_bench.sh`
instead of embedding their own keepalive implementation. Watchdogs must call the
same `run_bench.sh` interface. Repo-specific launch code may set `ROOT_DIR`,
`LOCAL_ENVS_DIR`, `LOCAL_ROOT`, `PACK_DIR`, SSH variables, and node selectors,
but should not duplicate these scripts.

## State Model

- Treat shared storage as the source of truth for code, data, models, packs,
  secrets, and run outputs.
- Treat node-local disks as cache: envs, source mirrors, data mirrors, and temp
  runtime state can be deleted and materialized again.
- Do not install packages during normal training startup. Build reusable packs
  first, then materialize them onto nodes.
- Keep generated outputs out of git: run logs, checkpoints, W&B outputs, data,
  model weights, env packs, secrets, and node-local runtime files.

Current MLF layout:

```text
/mnt/bn/jixf-nas-lq/mlf/code
/mnt/bn/jixf-nas-lq/mlf/data
/mnt/bn/jixf-nas-lq/mlf/models
/mnt/bn/jixf-nas-lq/mlf/packs
/mnt/bn/jixf-nas-lq/mlf/runs
/mnt/bn/jixf-nas-lq/mlf/secrets
/mnt/bn/jixf-nas-lq/mlf/scripts
/tmp/mlf-envs
/tmp/mlf-runtime
```

When moving clusters, preserve the same roles even if the paths change.

## Node And IP Handling

- Maintain one node file as the source of truth for current IPs.
- Let topology or launch profiles select nodes by index from that file.
- Do not scatter literal IPs through training configs, env configs, aux configs,
  or scripts.
- After resource reset, update the node file first, then prepare runtime, then
  launch jobs.
- Prefer idempotent per-node setup commands. A repeated setup should either skip
  unchanged packs or rebuild only what is stale.

## Runtime Pack Discipline

- Split runtime packs by dependency domain. Do not create one giant mutable
  environment that every task edits.
- Use `${ROOT_DIR}/scripts/pack_data.sh`,
  `${ROOT_DIR}/scripts/prepare_node_runtime.sh`, and
  `${ROOT_DIR}/scripts/materialize_node_runtime.sh` as the shared vocabulary for
  pack creation and node-local installation.
- Keep data/source packs separate from conda or Python environment packs.
- Keep model checkpoints on shared storage unless local copies are explicitly
  needed and documented.
- Do not hot-edit dependency checkouts for temporary fixes. Store repeatable
  dependency changes as patch files and provide apply/check/reverse commands.
- Materialization scripts should be explicit about what they install, copy, or
  skip; avoid hidden package installation inside launch scripts.

## Launch And Process Discipline

- Separate runtime preparation from job launch.
- Launchers orchestrate services and submit jobs; they should not own algorithm
  or experiment semantics.
- Prefer tmux/system sessions with clear names and logs for long-running remote
  processes.
- Always leave an auditable run directory with resolved config, launch logs,
  service logs, and train logs.
- When resetting experiments, stop old train/Ray/env processes on selected nodes
  before starting the new run. Do not kill unrelated user processes.

## GPU Keepalive

- Keep GPU keepalive independent of the training framework.
- Watchdog should protect idle GPUs by utilization only, not by checking Slime,
  Ray, tmux session names, or process names.
- Bench should be a replaceable keepalive workload. Starting bench should not
  destroy unrelated tmux sessions or user processes.
- Put bench and watchdog under `${ROOT_DIR}/scripts`, not under a repo-specific
  `bash/` or `utils/` directory.
- Before launching training, stop bench on nodes used by the job. If a job fails
  and GPUs stay idle, watchdog should restart bench after the configured window.

## Debugging Order

When a remote run behaves strangely:

1. Confirm the node file and selected indexes are correct.
2. Confirm env/data packs are materialized on every selected node.
3. Check service logs and train logs before changing code.
4. Check GPU memory/utilization to distinguish startup, rollout, actor, and
   keepalive states.
5. Check resolved config files to catch shell/env pollution or stale values.
6. Add code or script changes only after locating the owning layer.
