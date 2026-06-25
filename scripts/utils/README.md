# Agent Env Utility Scripts

These scripts manage runtime materialization and profile-driven training launch.
They are operational glue, not a second experiment configuration layer.

## Runtime Layout

- NAS root: `/mnt/bn/jixf-nas-lq/mlf`
- reusable packs: `${MLF_NAS_ROOT}/packs`
- reusable source/data/model assets: `${MLF_NAS_ROOT}/{code,data,models}`
- node-local envs: `/tmp/mlf-envs`
- node-local runtime assets: `/tmp/mlf-runtime`
- run outputs: `${MLF_NAS_ROOT}/runs`

## Main Entrypoints

- `prepare_agentic_runtime.sh`: materialize selected env/data/source packs on
  one or more nodes.
- `materialize_node_runtime.sh`: single-node materializer called by the prepare
  script.
- `launch_agentic_training.sh`: profile-driven distributed launcher. It resolves
  run/topology/model/train/aux profiles, starts services, and submits the train
  adapter.
- `aux_endpoint.sh`: optional OpenAI-compatible aux inference server manager.
- `build_*.sh`, `pack_*.sh`, `publish_*.sh`: build and publish reusable packs.

Example:

```bash
bash scripts/utils/prepare_agentic_runtime.sh \
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

## Keepalive

GPU keepalive is intentionally outside this repo and should be called through
the root-level ops interface:

```bash
${MLF_NAS_ROOT}/scripts/run_bench.sh start|stop|status
${MLF_NAS_ROOT}/scripts/gpu_idle_watchdog.sh start|status|stop
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
