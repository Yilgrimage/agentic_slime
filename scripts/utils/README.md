# Agent Env Utility Scripts

These scripts manage profile-driven training launch and repo-specific build
helpers. Runtime materialization and GPU keepalive live in the root server-ops
scripts, not in this repo.

## Runtime Layout

- workspace root: `${ROOT_DIR}`; infer it from the repo location or set it
  explicitly in the shell/local machine config, not in git
- reusable packs: `${ROOT_DIR}/packs`
- reusable source/data/model assets: `${ROOT_DIR}/{code,data,models}`
- node-local envs: `/tmp/server-ops-envs`
- node-local runtime assets: `/tmp/server-ops-runtime`
- run outputs: `${ROOT_DIR}/runs`

## Main Entrypoints

- `launch_agentic_training.sh`: profile-driven distributed launcher. It resolves
  run/topology/model/train/aux profiles, starts services, and submits the train
  adapter.
- `aux_endpoint.sh`: optional OpenAI-compatible aux inference server manager.
- `build_*.sh`, `pack_*.sh`, `publish_*.sh`: build and publish reusable packs.
- Runtime materialization: use `${ROOT_DIR}/scripts/prepare_node_runtime.sh`.
- Data packing: use `${ROOT_DIR}/scripts/pack_data.sh`.

Example:

```bash
${ROOT_DIR}/scripts/prepare_node_runtime.sh \
  --all-nodes \
  --nodes configs/nodes/agent_env_all.txt \
  --node 0,1,2,3 \
  --envs slime,alfworld,webshop,tau2 \
  --data alfworld,webshop,tau2 \
  --models none \
  --sources webshop,tau2

bash scripts/utils/launch_agentic_training.sh \
  configs/agent_env/runs/alfworld_qwen3_4b_grpo_fullasync_3x8.env
```

Create `configs/nodes/agent_env_all.txt` locally from
`configs/nodes/agent_env_all.txt.example`. The real node file is intentionally
git-ignored because it contains cluster-specific IPs or hostnames.

Task data is not bundled into env packs. `prepare_node_runtime.sh --data ...`
materializes data from `${ROOT_DIR}/data/<name>` or
`${ROOT_DIR}/packs/<name>-data.tar.gz`; supported datasets can be fetched with
`--auto-download-data`, and new clusters should use `--validate-data-load` once
to run heavier env load smoke checks.

Multi-node distributed training must use a routable socket interface. The
launcher resolves `SOCKET_IFNAME=${MLP_SOCKET_IFNAME:-eth0}` unless overridden,
then propagates `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and
`TP_SOCKET_IFNAME` through the run-local resolved env and Ray actor runtime env.
Override `SOCKET_IFNAME` or the specific backend variables for clusters whose
business network is not `eth0`.

## Keepalive

GPU keepalive is intentionally outside this repo and should be called through
the root-level ops interface:

```bash
${ROOT_DIR}/scripts/run_bench.sh start|stop|status
${ROOT_DIR}/scripts/gpu_idle_watchdog.sh start|status|stop
```

The watchdog should protect idle GPUs by utilization only. It should not depend
on Slime, Ray, tmux training sessions, or process-name heuristics.

## Hygiene

- Do not install Python packages during normal training startup.
- Do not commit packs, node-local materialization, model weights, data, run
  logs, checkpoints, or secrets.
- Keep optional third-party fixes as patch files, for example under
  `scripts/utils/patches/`, instead of editing dependency source trees directly.

For design notes and known pitfalls, read `docs/agent_env/README.md`.
