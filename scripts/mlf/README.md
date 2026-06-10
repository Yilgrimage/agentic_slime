# MLF runtime and pack scripts

These scripts keep code, reusable packs, and node-local runtime state separate.

- NAS root: `/mnt/bn/jixf-nas-lq/mlf`
- reusable packs: `${MLF_NAS_ROOT}/packs`
- reusable source/data/model assets: `${MLF_NAS_ROOT}/{code,data,models}`
- node-local envs: `/tmp/mlf-envs`
- node-local source/data runtime: `/tmp/mlf-runtime`
- model checkpoints stay on NAS and are read from `${MLF_NAS_ROOT}/models`

Normal pack refresh is:

```bash
bash scripts/mlf/publish_slime_pack.sh
bash scripts/mlf/build_webshop_env.sh
bash scripts/mlf/build_alfworld_env.sh
bash scripts/mlf/build_tau2_env.sh
bash scripts/mlf/build_appworld_env.sh
bash scripts/mlf/pack_agent_data.sh
```

Normal node migration is now split from training launch.

Prepare only the current node:

```bash
bash scripts/mlf/prepare_agentic_runtime.sh \
  --local-only \
  --envs slime,alfworld,webshop \
  --data alfworld,webshop \
  --models none
```

Prepare only the lightweight text/tool-use envs:

```bash
bash scripts/mlf/prepare_agentic_runtime.sh \
  --local-only \
  --envs tau2,appworld \
  --data tau2,appworld \
  --sources tau2,appworld \
  --models none
```

Prepare every node listed in a node file from the current machine:

```bash
bash scripts/mlf/prepare_agentic_runtime.sh \
  --all-nodes \
  --nodes configs/nodes/agent_env_4x8.txt \
  --envs slime,webshop \
  --data webshop \
  --models none
```

`prepare_agentic_runtime.sh` calls `materialize_node_runtime.sh` on each target
node. `materialize_node_runtime.sh` is intentionally single-node only.

Training launch is separate:

```bash
bash scripts/mlf/launch_agentic_training.sh \
  --env webshop \
  --nodes configs/nodes/agent_env_4x8.txt \
  --env-pool-size 32 \
  --train-cmd '...'
```

The env packs are intentionally independent:

- `slime.tar.gz`: training, Ray, SGLang, Megatron, torch, CUDA toolkit.
- `webshop.tar.gz`: WebShop HTTP environment server dependencies.
- `alfworld.tar.gz`: ALFWorld/TextWorld HTTP environment server dependencies.
- `tau2.tar.gz`: tau2/tau3 text-mode and gym tool-use dependencies.
- `appworld.tar.gz`: AppWorld package dependencies.
- `webshop-data.tar.gz`: WebShop product JSON, human goals/attributes, and Lucene indexes.
- `alfworld-data.tar.gz`: ALFWorld/TextWorld game files and environment data.
- `tau2-data.tar.gz`: tau2/tau3 domain data bundled with tau2-bench.
- `appworld-data.tar.gz`: AppWorld databases and task assets.

The adapter code in `examples/` should consume these packs. It should not install
Python packages during normal training startup.

## Materialization checks

Conda env packs are installed under `/tmp/mlf-envs/{slime,alfworld,webshop,tau2,appworld}`.
Data/source runtime assets are copied under `/tmp/mlf-runtime`.

The materializer keeps simple local stamps for conda and data packs:

- If the target exists and the local stamp matches the NAS `.sha256`, it skips.
- If the NAS pack hash changes, it reinstalls that specific env/data pack.
- `--force` removes and recreates selected targets even when hashes match.
- `--no-check-hash` falls back to existence checks only.

Source mirrors currently use existence checks unless `--force` is given.
Model mirroring is disabled; pass `--models none` and point training scripts at
`${MLF_NAS_ROOT}/models`.

Data packs are deliberately separate from conda packs:

- Conda packs are architecture/runtime dependency bundles.
- Data packs are task assets and indexes. They can be large and are copied to
  `/tmp/mlf-runtime/data/...` before training.
- WebShop also needs its data and search indexes mirrored into
  `/tmp/mlf-runtime/code/WebShop/{data,search_engine}` because the upstream
  WebShop code resolves paths relative to its source tree.
- ALFWorld consumes the materialized data path from its env config; the data is
  not part of `alfworld.tar.gz`.
