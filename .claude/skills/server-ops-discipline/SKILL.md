---
name: server-ops-discipline
description: "Use when preparing remote GPU nodes, building or publishing runtime/data packs, materializing node-local envs/data/source mirrors, handling node IP files, launching distributed jobs, preserving idle GPUs with watchdog/bench, separating durable shared state from disposable local state, or moving this workflow to another cluster."
---

# Server Ops Discipline

Keep server operations reproducible. Shared storage is the source of truth;
node-local disks are disposable caches rebuilt from packs and scripts.

## Root And Scripts

- Confirm the cluster workspace root with the user before first use. Store it in
  a local, non-git config:

```bash
server-ops-discipline/scripts/configure_root.sh /path/to/root
```

- Install reusable scripts under `${ROOT_DIR}/scripts`:

```bash
mkdir -p "${ROOT_DIR}/scripts"
cp -a server-ops-discipline/scripts/. "${ROOT_DIR}/scripts/"
chmod +x "${ROOT_DIR}/scripts/"*.sh
```

- Use `ROOT_DIR`, not project- or cluster-branded root variables. Do not commit
  concrete root paths.
- Stable interfaces:

```text
${ROOT_DIR}/scripts/run_bench.sh start|stop|status|restart [--nodes nodes.txt --node 0,1]
${ROOT_DIR}/scripts/gpu_idle_watchdog.sh start|stop|status|restart
${ROOT_DIR}/scripts/prepare_data.sh --data alfworld,webshop,tau2,mcp_server
${ROOT_DIR}/scripts/pack_data.sh --data alfworld,webshop,tau2,mcp_server
${ROOT_DIR}/scripts/prepare_node_runtime.sh --local-only|--all-nodes ...
${ROOT_DIR}/scripts/materialize_node_runtime.sh --envs ... --data ... --sources ...
```

Project launchers may call these scripts, but must not reimplement bench,
watchdog, node materialization, SSH fanout, or root discovery.

## State Model

- Durable: code, packs, data, models, secrets, runs, and node files live under
  `${ROOT_DIR}`.
  If a cluster puts a smaller quota on the workspace tree, keep the same public
  layout and make `${ROOT_DIR}/models` or `${ROOT_DIR}/runs` symlinks to the
  real shared-storage locations.
- Disposable: extracted envs, source mirrors, data mirrors, Ray temp state, and
  service scratch live on node-local disks such as `/tmp`.
- Do not install packages during normal training startup. Build packs first,
  then materialize them onto nodes.
- Keep generated outputs, real node IP files, secrets, packs, data, models,
  checkpoints, and run logs out of git.

Recommended layout:

```text
${ROOT_DIR}/code
${ROOT_DIR}/data
${ROOT_DIR}/models
${ROOT_DIR}/packs
${ROOT_DIR}/runs
${ROOT_DIR}/secrets
${ROOT_DIR}/scripts
${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}
${LOCAL_RUNTIME_DIR:-/tmp/server-ops-runtime}
```

On the current A100 cluster, `${ROOT_DIR}` is
`/mnt/bn/jixf-nas-lq/mlf`. Models and run outputs are physically stored outside
that quota tree, but remain available through standard symlinks:

```bash
${ROOT_DIR}/models -> /mnt/bn/jixf-nas-lq/yanjingyuan_models
${ROOT_DIR}/runs   -> /mnt/bn/jixf-nas-lq/yanjingyuan_runs
```

Treat this as an A100-local storage layout. H100 or other clusters should use
their own shared NAS target paths behind the same `${ROOT_DIR}/models` and
`${ROOT_DIR}/runs` names, not these concrete A100 paths.

## Runtime Packs

- Prefer image-provided foundation training stacks when the image is complete
  and compatible.
- If the image is not complete, use one complete conda pack per foundation
  stack. For Slime, the pack must include Slime, SGLang/vLLM/Torch dependencies
  needed by that stack, and bundled Megatron source at `src/Megatron-LM`.
- Do not materialize Megatron-LM as a standalone source mirror for Slime
  training. It is part of the Slime runtime, not a separate runtime component.
- Keep task envs separate: ALFWorld, WebShop, tau2, AppWorld, MCP servers, and
  other env dependencies get their own packs.
- Keep task data separate from env packs. Data preparation is a shared-storage
  phase: run `prepare_data.sh --data ...` to download or construct
  `${ROOT_DIR}/data/<name>`, then `pack_data.sh --data ...` to write immutable
  `${ROOT_DIR}/packs/<name>-data.tar.gz` artifacts. Node materialization never
  downloads data; it only unpacks a data pack or copies an already prepared
  `${ROOT_DIR}/data/<name>` directory.
- Data packs must contain every file an env server can need on any selected
  node. For tau2, `prepare_data.sh --data tau2` must produce both raw
  official/AReaL data and portable generated AReaL task/prompt files; training
  launch should not generate node-local tau2 task files.
- Private MCP server datasets may come from another NAS checkout rather than a
  public downloader. Treat that checkout as an input source only: copy the
  required JSON/JSONL task files and referenced assets into
  `${ROOT_DIR}/data/mcp_server`, pack them as `mcp_server-data.tar.gz`, and have
  training read the materialized `${AGENT_ENV_DATA_DIR}` copy. Set
  `MCP_SERVER_REQUIRED_FILES` during preparation/packing for task-family
  specific files that must exist, for example an IPR product-check task JSONL.
- Validate task data before packing and after materialization. Domain-specific
  datasets should fail fast on missing required files; for example ALFWorld
  needs paired `game.tw-pddl`/`traj_data.json` files plus `logic/alfred.pddl`
  and `logic/alfred.twl2`. Use `--validate-data-load` when bringing up a new
  cluster to run supported env load smoke tests.
- Keep optional reporting clients isolated when cluster images disagree. A
  `wandb` conda pack may be materialized like any other env pack; training
  repos should prefer that pack, fall back to the foundation runtime, then to a
  version-compatible local Python if they choose to support W&B.
- Keep task source mirrors only when the env itself needs a checkout, for
  example WebShop. Source mirrors are not a replacement for Python runtime
  dependencies.
- Publish immutable pack artifacts to `${ROOT_DIR}/packs/<name>.tar.gz` plus
  `.sha256` and `.revision`; materialize them with `--force` when replacing a
  node-local env.
- Validate a pack before use: check hashes, run `conda-unpack`, import critical
  modules, and run an env-specific smoke test.
- Do not run Python directly from a NAS env directory. Use node-local extracted
  packs to avoid slow small-file IO and stale absolute prefixes.

## Node And Launch

- Maintain one current node file as the IP source of truth. Launch/topology
  profiles select nodes by index.
- For multi-node GPU jobs, make the routable training network interface an
  explicit launch contract. Prefer `SOCKET_IFNAME=${MLP_SOCKET_IFNAME:-eth0}`
  or a cluster-specific override, then propagate `NCCL_SOCKET_IFNAME`,
  `GLOO_SOCKET_IFNAME`, and `TP_SOCKET_IFNAME` into Ray actor runtime envs.
  Do not rely on outer-shell exports reaching Ray actors.
- After resource reset: update node file, materialize runtime/data, start
  watchdog/bench, then launch jobs.
- Before launching training, stop bench on selected nodes. If training fails and
  GPUs idle, watchdog should restart bench by utilization threshold.
- `bench_on_exit` must call `${ROOT_DIR}/scripts/run_bench.sh`; do not embed a
  separate keepalive implementation in training repos.
- Reset experiments by stopping only selected train/Ray/env processes. Do not
  kill unrelated user processes.

## GPU Keepalive

- Watchdog protects idle GPUs by utilization only. It should not special-case
  Slime, Ray, tmux names, or process names.
- Bench is a replaceable keepalive workload. Starting bench must not destroy
  unrelated tmux sessions or user processes.
- Keep watchdog and bench independent of training framework state.

## Debugging Order

1. Confirm node file and selected indexes.
2. Confirm env/data/source materialization on every selected node.
3. Check service logs and train logs.
4. Check GPU utilization and memory to distinguish rollout, actor, aux, and
   keepalive behavior.
5. Check resolved configs for stale values or shell pollution.
6. Change code only after locating the owning layer.
