# Agent Env Operations

This file describes how to bring up the stack on a fresh cluster, run
experiments, preserve GPUs, and debug failures.

## Cluster Assumptions

The current deployment assumes:

- a shared NAS-like root mounted at `MLF_NAS_ROOT`;
- node-local fast storage under `/tmp`;
- SSH access from the submit/head node to all train and aux nodes;
- one editable Slime checkout at `${MLF_NAS_ROOT}/code/slime`;
- conda/env packs and data packs stored under `${MLF_NAS_ROOT}/packs`;
- models stored under `${MLF_NAS_ROOT}/models`;
- run outputs stored under `${MLF_NAS_ROOT}/runs`;
- secrets stored under `${MLF_NAS_ROOT}/secrets`.

Current defaults:

```bash
MLF_NAS_ROOT=/mnt/bn/jixf-nas-lq/mlf
MLF_LOCAL_ENVS=/tmp/mlf-envs
MLF_LOCAL_ROOT=/tmp/mlf-runtime
REPO_DIR=${MLF_NAS_ROOT}/code/slime
```

These variable names still include `MLF` because they describe the current NAS
layout. The reusable script directory has been renamed to `scripts/utils/`.

## Fresh Node Checklist

1. Update current IPs:

   ```bash
   vim configs/nodes/agent_env_all.txt
   ```

2. Confirm SSH:

   ```bash
   ssh -6 -p 10413 -i /mnt/bn/jixf-nas-lq/mlf/secrets/byte_id_rsa tiger@<node-ip> hostname
   ```

3. Prepare runtimes:

   ```bash
   cd /mnt/bn/jixf-nas-lq/mlf/code/slime
   bash scripts/utils/prepare_agentic_runtime.sh \
     --all-nodes \
     --orchestrator head \
     --nodes configs/nodes/agent_env_all.txt \
     --node 0,1,2,3 \
     --envs slime,alfworld,webshop,tau2 \
     --data alfworld,webshop,tau2 \
     --sources webshop,tau2 \
     --models none
   ```

4. Check prepare status on each node:

   ```bash
   ssh ... 'tail -80 /tmp/mlf_prepare.log; ls /tmp/mlf-envs; ls /tmp/mlf-runtime/data'
   ```

5. Start or verify the external GPU watchdog:

   ```bash
   bash /mnt/bn/jixf-nas-lq/mlf/bash/gpu_idle_watchdog.sh status
   ```

6. Stop bench on nodes selected for a real training run. The launcher does this
   automatically for the selected train nodes, but manual checks are useful:

   ```bash
   bash /mnt/bn/jixf-nas-lq/mlf/bash/run_bench.sh status \
     --nodes configs/nodes/agent_env_all.txt \
     --node 0,1,2,3
   ```

## Runtime Materialization Scripts

The node setup scripts are in:

```text
scripts/utils/prepare_agentic_runtime.sh
scripts/utils/materialize_node_runtime.sh
```

`prepare_agentic_runtime.sh` is the multi-node orchestrator. It can run locally
or submit work to selected remote nodes.

`materialize_node_runtime.sh` is single-node only. It unpacks env/data/source
packs into `/tmp/mlf-envs` and `/tmp/mlf-runtime`.

The script is idempotent:

- env/data packs use local hash stamps;
- matching stamps skip reinstall;
- `--force` removes and recreates selected targets;
- `--no-check-hash` falls back to existence checks.

Normal use:

```bash
bash scripts/utils/prepare_agentic_runtime.sh \
  --all-nodes \
  --nodes configs/nodes/agent_env_all.txt \
  --node 0,1,2,3 \
  --envs slime,alfworld,webshop,tau2 \
  --data alfworld,webshop,tau2 \
  --sources webshop,tau2 \
  --models none
```

The current training scripts read models from NAS and do not materialize models
to `/tmp`:

```bash
--models none
```

## Pack Utilities

Pack scripts are also in `scripts/utils/`:

```bash
bash scripts/utils/publish_slime_pack.sh
bash scripts/utils/build_alfworld_env.sh
bash scripts/utils/build_webshop_env.sh
bash scripts/utils/build_tau2_env.sh
bash scripts/utils/build_appworld_env.sh
bash scripts/utils/pack_agent_data.sh
```

Only rebuild packs when dependencies or data assets change. Normal training
startup should not install Python packages or rebuild indexes.

## Launching Training

Use a run profile:

```bash
cd /mnt/bn/jixf-nas-lq/mlf/code/slime
bash scripts/utils/launch_agentic_training.sh \
  configs/agent_env/runs/alfworld_qwen3_4b_grpo_fullasync_3x8.env
```

The launcher performs:

- profile resolution;
- optional aux endpoint startup;
- selected train-node bench stop;
- selected train-node runtime reset;
- env server startup on each train node;
- Ray head/worker startup;
- router startup;
- train adapter startup.

Do not manually compose a long Slime CLI for normal experiments. Use profiles,
then inspect the resolved files.

## Dry Run

The launcher accepts:

```bash
bash scripts/utils/launch_agentic_training.sh <run-profile.env> --dry-run
```

Dry run is intended for inspecting generated commands. Do not run it while a
real training job is active unless you have confirmed the current implementation
will not reset active nodes. Prefer inspecting resolved profiles from a real run
or running dry run on unused nodes.

## Ray Isolation

Two Ray clusters do not conflict when they use different physical head nodes,
even if both use port `6379`.

Safe example:

```text
sync head  node0 10.x.x.142:6379
async head node1 10.x.x.74:6379
```

Unsafe example:

```text
two independent Ray heads on the same host with the same port and temp dir
```

If splitting one host across multiple jobs, configure different:

- `RAY_PORT`
- dashboard port if needed
- `RAY_MIN_WORKER_PORT` / `RAY_MAX_WORKER_PORT`
- `RAY_TEMP_DIR`
- tmux session names or isolated launch wrappers

The current profiles assume one training run owns all visible GPUs on each
selected node.

## Monitoring Commands

GPU status:

```bash
for n in $(awk 'NF && $1 !~ /^#/ {print $1}' configs/nodes/agent_env_all.txt); do
  echo "== $n =="
  ssh -6 -p 10413 -i /mnt/bn/jixf-nas-lq/mlf/secrets/byte_id_rsa tiger@"$n" \
    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits"
done
```

Ray status:

```bash
/tmp/mlf-envs/slime/bin/ray status --address=<head-ip>:6379
```

Run status:

```bash
cat "$RUN_ROOT/logs/"*"_train_status.env"
tail -200 "$RUN_ROOT/logs/"*"_train.log"
tail -200 "$RUN_ROOT/logs/multi_head.log"
```

Useful train log patterns:

```bash
grep -RInE "raw_reward|rollout/rewards|advantages|ppo_kl|clip|dynamic_filter|drop_|Traceback|RuntimeError|TimeoutError" \
  "$RUN_ROOT/logs/"*"_train.log"
```

## Interpreting Key Metrics

`rollout/raw_reward`:

Task-scale reward before GRPO normalization. This is the first metric to check
for actual environment progress.

`rollout/rewards`:

The reward/advantage tensor after post-process normalization. In GRPO this can
be near zero when a group has no reward variance.

`rollout/dynamic_filter/drop_*`:

Groups dropped before training because they have no useful advantage signal.
For sparse reward tasks this should often be non-zero. If it is always zero,
confirm the dynamic filter path is actually active.

`ppo_kl`:

Policy divergence metric. In full-async, very large or persistently abnormal
values can indicate missing rollout-time behavior logprobs, dropout mismatch,
or stale/off-policy data.

`clipfrac`:

High clip fraction suggests most samples are outside the PPO trust region.
Check `USE_ROLLOUT_LOGPROBS`, dropout, learning rate, and off-policy staleness.

Format/truncation rates:

These should be rates, not counts. Avoid adding redundant mean reward metrics
for each penalty if the rate already explains the failure mode.

## GPU Keepalive

Keepalive is external to this repo:

```text
/mnt/bn/jixf-nas-lq/mlf/bash/run_bench.sh
/mnt/bn/jixf-nas-lq/mlf/bash/gpu_idle_watchdog.sh
```

The Slime repo launcher only calls `run_bench.sh` at job boundaries:

- stop bench on selected train nodes before launch;
- start bench on selected train nodes when training exits, unless reset
  suppression is active;
- start bench when launch orchestration fails.

The watchdog should be simple and independent: it protects GPUs based on GPU
utilization, not based on whether a training process appears alive. If a process
hangs with no GPU usage, the watchdog should still start bench.

The current watchdog policy outside this repo is intended to be:

- sample maximum GPU utilization every 10 seconds by default;
- if utilization stays below 5% for 1800 seconds by default, run bench;
- do not bind watchdog behavior to training tmux/process liveness.

## Reset Behavior

At launch start, selected train nodes are reset when:

```bash
RESET_TRAIN_RUNTIME_ON_START=1
```

The reset kills known training tmux sessions, Ray processes, SGLang servers,
env servers, routers, and train drivers on the selected train nodes.

The reset also writes a short suppression timestamp so an old train-exit trap
does not immediately restart bench during a new launch.

Do not point a run profile at nodes used by a different active experiment.

## Aux Endpoint Operations

Aux endpoint startup is optional and selected by `AUX_PROFILE`.

If `AUX_PROFILE` is absent:

- no aux server is started;
- `AGENT_ENV_JUDGE_MODE=none` should be used unless the endpoint env vars are
  provided externally.

If `AUX_PROFILE` is present:

- launcher starts `scripts/utils/aux_endpoint.sh`;
- aux endpoint writes `AUX_ENV_FILE`;
- train driver sources that env file;
- group RM can call the endpoint when `AGENT_ENV_JUDGE_MODE=aux`.

Do not put task-specific tau2 data or ALFWorld data in aux configs. Aux configs
only describe the inference endpoint.

## Moving to Another Cluster

Checklist:

1. Clone the fork and checkout `agentic-env-backend`.
2. Set the new NAS root:

   ```bash
   export MLF_NAS_ROOT=/path/to/shared/root
   export REPO_DIR=$MLF_NAS_ROOT/code/slime
   ```

3. Copy or rebuild:

   ```text
   packs/
   models/
   data/
   secrets/
   ```

4. Update:

   ```text
   configs/nodes/agent_env_all.txt
   ```

5. Confirm SSH options. Override if needed:

   ```bash
   export SSH_USER=<user>
   export SSH_PORT=<port>
   export SSH_KEY=<path>
   export SSH_IPV6=0|1
   export SSH_JUMP=<optional jump host>
   ```

6. Prepare node runtimes with `scripts/utils/prepare_agentic_runtime.sh`.
7. Run env smoke tests before training.
8. Launch a one-node sync profile before a full-async profile.
9. Inspect `resolved_*.env` and W&B config for every new cluster run.

## Git Operations

Use:

```bash
git remote -v
git branch --show-current
git status --short --branch
```

Expected:

```text
branch: agentic-env-backend
origin: user's fork
upstream: THUDM/slime, fetch only
```

Push development work with:

```bash
git push -u origin agentic-env-backend
```

Do not push to upstream. Fetch upstream Slime separately when rebasing or
merging new Slime changes.

## Files That Should Not Be Committed

Do not commit:

- `runs/`
- `wandb/`
- model weights
- conda packs
- data packs
- API keys
- W&B secrets
- node-local runtime copies
- generated `__pycache__` files

Commit:

- source code under `examples/agent_env`
- profile configs under `configs/agent_env`
- current node index file only when it is intentionally operational state for
  this cluster
- utility scripts under `scripts/utils`
- documentation under `docs/agent_env`

## Failure Triage

Launch does not start:

1. Check `multi_head.log`.
2. Check SSH reachability.
3. Check `resolved_launch.env`.
4. Confirm `/tmp/mlf-envs/slime/bin/python` exists on selected nodes.

Env server not ready:

1. Check `<env>_env_server.log`.
2. Run the env-specific smoke test.
3. Confirm data path under `/tmp/mlf-runtime/data/<env>`.

Ray nodes missing:

1. Check `ray_head.log` and `ray_worker_*.log`.
2. Confirm no old Ray processes remain.
3. Confirm selected nodes are unique and reachable.

Actor trains but reward does not move:

1. Check `rollout/raw_reward`.
2. Check dynamic filter drops.
3. Check full-async `USE_ROLLOUT_LOGPROBS=1`.
4. Check dropout is disabled in model profile.
5. Check loss mask type.
6. Check group reward variance.
7. Compare sync and full-async profiles.

GPU utilization low:

1. Check whether rollout nodes are blocked on env server.
2. Check SGLang server concurrency and env pool size.
3. Check rollout batch size and samples per prompt.
4. Check actor train wait time.
5. Check whether dynamic filter is dropping too many groups and starving actor.
