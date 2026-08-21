---
name: server-ops-discipline
description: "Use when preparing remote GPU nodes, building or publishing runtime/data packs, materializing node-local envs/data/source mirrors, handling node IP files, launching distributed jobs, preserving idle GPUs with watchdog/bench, separating durable shared state from disposable local state, or moving this workflow to another cluster."
---

# Server Ops Discipline

Keep GPU-node operations reproducible. Shared storage is the source of truth;
node-local disks are disposable caches rebuilt from packs and scripts.

## Core Model

- `ROOT_DIR` is the portable workspace root. Confirm it with the user on a new
  cluster and store it outside git with `scripts/configure_root.sh`.
- Durable state lives under `ROOT_DIR`: code, packs, data, models, secrets,
  node files, runs, checkpoints, and logs selected for long-term analysis.
- Disposable state lives on node-local storage: extracted envs, copied data,
  model/source caches, Ray scratch, runtime-env caches, service scratch, and
  core/temp files.
- Treat the development/control host as a jump box. It may orchestrate jobs and
  prepare lightweight JSONL metadata, but must not materialize task envs/data or
  run env-dependent rollout/eval loops. Start env servers and offline teacher
  collection on worker/GPU nodes using node-local runtimes.
- If a cluster has separate quotas, keep public names stable and use symlinks
  such as `${ROOT_DIR}/models` or `${ROOT_DIR}/runs` behind the scenes. Do not
  add cluster-specific artifact roots to training repos.
- Normal training startup must not install packages. Build packs first, then
  materialize them onto nodes.

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

## Shared Scripts

Install reusable scripts under `${ROOT_DIR}/scripts` and call them from
project launchers instead of reimplementing server ops:

```bash
mkdir -p "${ROOT_DIR}/scripts"
cp -a server-ops-discipline/scripts/. "${ROOT_DIR}/scripts/"
chmod +x "${ROOT_DIR}/scripts/"*.sh
```

Stable entrypoints:

```text
bootstrap_micromamba.sh
build_runtime_packs.sh --runtime slime,alfworld,appworld,tau2,webshop,wandb
verify_pack_bundle.sh --runtime ...
run_bench.sh start|stop|status|restart [--nodes nodes.txt --node 0,1]
gpu_idle_watchdog.sh start|stop|status|restart
prepare_data.sh --data <env>
pack_data.sh --data <env>
build_wandb_env.sh
prepare_node_runtime.sh --local-only|--all-nodes ...
materialize_node_runtime.sh --envs ... --data ... --sources ...
```

Project code may call these scripts, but should not duplicate bench, watchdog,
node materialization, SSH fanout, or root discovery logic.

## Packs And Data

- Before rebuilding or handing packs to a new cluster, read
  `references/runtime_pack_build.md`. It is the provider-neutral cold-start,
  verification, backup, and restore contract; no historical pack or specific
  NAS mount is a prerequisite.
- Prefer a complete training image when available. Otherwise use one complete
  conda pack per foundation stack. A Slime pack must include Slime, training
  dependencies, and bundled Megatron source at `src/Megatron-LM`.
- Keep task envs separate from foundation runtimes. ALFWorld, WebShop, tau2,
  AppWorld, OpenClaw, and future envs get their own env/data packs as needed.
- Keep task data separate from env packs. Prepare data on shared storage,
  publish immutable `${ROOT_DIR}/packs/<name>-data.tar.gz` artifacts, and let
  node materialization only unpack/copy validated data.
- Data packs must contain every file an env server can need on any selected
  node. Domain-specific validators should fail fast before training launch.
- W&B is a separate optional runtime pack. Formal runs must enable W&B through
  the materialized `wandb` pack; `ENABLE_WANDB=0` is only for smoke/debug runs.
- Some materialized packs are not conda envs, for example Codex runtime,
  skills, or helper bundles. Run `conda-unpack` only when the target provides
  it.
- Do not run Python directly from NAS env directories. Use node-local extracted
  packs to avoid slow small-file IO and stale absolute prefixes.
- A newly published runtime/data pack is complete only with its tarball,
  portable checksum, revision, and manifest. Back up all four files to any
  available durable cloud/shared storage after verification. Source rebuild is
  the recovery path, not a substitute for preserving an exact working bundle.

## Nodes And SSH

- Maintain one current node file as the IP source of truth. Topology profiles
  select nodes by index.
- For an already-launched run, do not guess SSH options. Read
  `logs/resolved_launch.env` and use its `SSH_USER`, `SSH_PORT`, `SSH_KEY`,
  `SSH_IPV6`, `SSH_JUMP`, `NODES_FILE`, `NODE_INDICES`, and
  `AUX_NODE_INDICES`.
- Plain `ssh <node>` may fail on Trail/Merlin-style GPU nodes even when the
  node is healthy. Use the resolved launch contract, for example:
  `ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$host"`.
- For multi-node jobs, set the routable network interface explicitly and
  propagate `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and
  `TP_SOCKET_IFNAME` into Ray actor runtime envs. Outer-shell exports are not
  enough.
- Preserve unset-vs-empty semantics in launch scripts. Use
  `[[ -z "${VAR+x}" ]]` before applying defaults when empty has meaning, such
  as `WORKER_HOSTS=''` for a single-node launch.

Cluster path hints are handoff aids only; confirm before use:

```text
H100 ecomcommonnas: ROOT_DIR=/mnt/bn/ecomcommonnas/yanjingyuan, SSH_PORT=10413
A100 jixf-nas-lq:  ROOT_DIR=/mnt/bn/jixf-nas-lq/mlf, SSH_PORT=10413,
                   SSH_KEY=${ROOT_DIR}/secrets/byte_id_rsa
```

## Launch Hygiene

- After resource reset: update node file, materialize runtime/data, start
  watchdog or bench, then launch jobs.
- Before launching training, stop bench on selected nodes. If training exits
  and GPUs become idle, watchdog should restart bench by utilization threshold.
- `bench_on_exit` must call `${ROOT_DIR}/scripts/run_bench.sh`; do not embed a
  second keepalive implementation in training repos.
- When stopping a training run to release GPUs, read that run's
  `logs/resolved_launch.env`, stop only the selected train/Ray/env/router
  tmux sessions and processes, then decide bench explicitly. If the user wants
  idle GPUs, stop bench after train exit; if keepalive is acceptable, start or
  leave `${ROOT_DIR}/scripts/run_bench.sh` running.
- Reset only selected train/Ray/env processes. Do not kill unrelated user
  processes.
- Avoid `pkill -f <pattern>` in SSH one-liners; it can match and kill the SSH
  command itself. Prefer named tmux sessions or exact PID filtering.
- For repeated large-model startup, prefer a node-local model cache. NAS reads
  are acceptable for smoke tests, but repeated SGLang/actor loads should use a
  disposable local cache.
- After stop/start actions, verify with `nvidia-smi --query-compute-apps`,
  GPU memory/utilization, `tmux ls`, and targeted `pgrep` patterns for train,
  Ray, SGLang, env/router, and bench. Do not rely on a single SSH command that
  may have been killed by process cleanup.
- To stop a distributed run, first read its `resolved_launch.env`, then target
  only its selected nodes and named train/Ray/env/router sessions. Request a
  normal train exit before stopping that run's Ray daemons. Do not stop the GPU
  watchdog. After cleanup, verify the run status file, confirm no owned
  processes remain, and start bench immediately rather than waiting for the
  watchdog idle window.
- Before handing a run to another operator or cluster, push the exact code
  commit, preserve resolved configs and bounded debug evidence, record the last
  complete checkpoint, and state which algorithm changes were not exercised by
  that run. A live process that imported an older checkout is not evidence for
  a later commit even when both used the same shared path.

## Ray And Local Scratch

- `/tmp/ray` and `${LOCAL_RUNTIME_DIR}/ray` are scratch, not durable records.
  Durable evidence belongs in run logs, resolved configs, W&B, dumps, eval
  summaries, and checkpoints.
- Leaving Ray at default `/tmp/ray` is acceptable when it is node-local,
  monitored, and cleaned between launches.
- If overriding Ray temp, do it at daemon startup with `ray start --temp-dir`
  on both head and worker nodes. A train adapter variable or Python CLI flag
  parsed after Ray starts does not move Ray session directories.
- Launchers that reset Ray may clean stale `/tmp/ray/session_*` and
  `session_latest` after `ray stop --force`, but only after confirming no Ray
  daemon processes remain for the current user.
- Do not route Ray temp, object spilling, runtime-env cache, or high-churn
  service scratch to NAS as a first response. Fix the writer or clean local
  scratch instead.
- If node disk grows, inspect `/tmp/ray`, Ray logs, object spilling,
  runtime-env cache, core files, and debug dumps before changing training
  semantics.
- A healthy Slime/agent-env run should not require `/tmp/ray` to grow without
  bound. If it jumps from small scratch usage to hundreds of GB, capture
  `du -sh /tmp/ray`, `df -h /tmp`, deleted-open files, Ray object spill logs,
  and the writing process before relaunching on another node.

## Keepalive

- Watchdog protects idle GPUs by utilization only. It should not special-case
  Slime, Ray, tmux names, or process names.
- Bench is a replaceable keepalive workload. Starting bench must not destroy
  unrelated tmux sessions or user processes.
- Keep watchdog and bench independent of training framework state.

## Debug Order

1. Confirm root, node file, selected indexes, and resolved SSH contract.
2. Confirm env/data/source materialization on every selected node.
3. Check service logs and train logs.
4. Check GPU utilization and memory to distinguish rollout, actor, aux, and
   keepalive behavior.
5. Check local disk pressure from Ray scratch, runtime caches, core files, and
   debug dumps.
6. Check resolved configs for stale values or shell pollution.
7. Change code only after locating the owning layer.

## Eval Records

- Keep a small eval index under `${ROOT_DIR}/runs/eval_summaries`.
- For AppWorld, read `references/appworld_eval_results.md` before comparing,
  deleting, or handing off experiments. It is the portable accepted-results
  ledger when the original NAS is unavailable.
- Copy TSV/JSON summary files into `eval_summaries/<env>/summaries/` and add
  raw symlinks under `eval_summaries/<env>/raw_links/`.
- Update `eval_summaries/index.tsv` for every result used in discussion,
  cleanup, or follow-up experiments. Include whether it is full eval, dev-only,
  smoke, token probe, or train-metric-only.
- Do not rely on chat history or W&B alone for remembered eval outcomes.
- Before deleting checkpoints, verify the corresponding eval summary is indexed.
  If no eval was run, record that explicitly in the cleanup manifest.
- Before releasing a cluster or trail, copy the compact accepted metrics,
  protocol, checkpoint ancestry, and validity caveats into a tracked skill
  reference. Raw symlinks and run-local files are not a portable handoff.
