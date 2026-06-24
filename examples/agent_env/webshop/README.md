# WebShop Agentic Rollout

This example keeps WebShop outside the slime training environment. The slime
process calls `examples.agent_env.webshop.rollout.generate`, which declares WebShop's
environment-specific `AgentEnvSpec` and delegates the common agent loop to
`examples.agent_env.rollout`. The WebShop env server owns process-isolated
workers and must run in the separate WebShop conda-pack runtime.

Directory layout:

- `rollout.py`: WebShop prompt/action/success spec for the shared rollout loop.
- `server.py`: WebShop backend for the shared process-pool lease server.
- `prompt_data.py`: WebShop prompt metadata generation.
- `env_config.yaml`: WebShop data, reward, interaction, and env server config.

Expected packs:

- slime: training, Ray, sglang, Megatron, torch
- webshop: Gym/WebShop and WebShop data/runtime dependencies
- alfworld: ALFWorld/TextWorld dependencies

Runtime convention:

- NAS stores reusable packs, source checkouts, and data under `/mnt/bn/jixf-nas-lq/mlf`.
- The launch script copies IO-heavy runtime pieces to `/tmp/mlf-runtime`.
- No WebShop Python dependency should be installed into the slime pack.

Runtime setup:

1. Ensure the WebShop pack exists at `/mnt/bn/jixf-nas-lq/mlf/packs/webshop.tar.gz`.
2. Ensure the WebShop data backup exists at `/mnt/bn/jixf-nas-lq/mlf/data/webshop`.
3. Materialize node-local runtime with `scripts/utils/prepare_agentic_runtime.sh`.

Use `scripts/utils/build_webshop_env.sh`, `scripts/utils/pack_webshop_env.sh`, and
`scripts/utils/pack_agent_data.sh` only when rebuilding NAS packs/data. They are
not part of normal training startup.

## Data Scope

Training uses the data files and `num_products` declared in
`examples/agent_env/webshop/env_config.yaml`. The current training config uses
the full 100k product scope: `items_shuffle.json`, `items_ins_v2.json`, and
`indexes_100k`.

Previously verified on node0 for the full setup:

- `items_shuffle.json` with `num_products=100000` loads the 100k product scope.
- `human_goals=true` loads 1021 human goals with the current data.
- A single full WebShop env took about 65s to initialize.
- A single full WebShop env used about 13.1 GiB steady RSS and 18.4 GiB peak
  RSS during initialization; VSZ was about 58 GiB and should not be used as the
  pool capacity estimate.
- `reset(session=0)` and a valid `search[...]` step both completed in
  milliseconds after initialization.

For full WebShop runs on the current 2 TiB nodes, `env_config.yaml` sets
`env_server.pool_size=32`. For smaller-memory nodes, lower that field after
checking peak RSS headroom. Use a small pool such as `2` only for conservative
smoke tests.

Do not claim full WebShop coverage until the NAS/runtime contains:

- `${MLF_NAS_ROOT}/data/webshop/data/items_shuffle.json`
- `${MLF_NAS_ROOT}/data/webshop/data/items_ins_v2.json`
- `${MLF_NAS_ROOT}/data/webshop/data/items_human_ins.json`
- `${MLF_NAS_ROOT}/data/webshop/search_engine/indexes_100k`

Pack full data with:

```bash
DATASETS=webshop bash scripts/utils/pack_agent_data.sh
```

## Shared backend direction

The current server starts one `WebAgentTextEnv` per worker process. Each worker
therefore loads its own `SimServer`, product list, goals, and Lucene searcher.
That is acceptable for the 1k smoke setup, but it is the wrong shape for full
WebShop.

The intended scalable design is:

- process-isolated episode workers keep lease/session lifecycle, observations,
  action parsing, and reset/step ownership;
- a shared WebShop backend service loads product data, goals, and Lucene search
  index once per node;
- workers call the backend for `SimServer.receive`-equivalent operations.

`WebAgentTextEnv` already accepts a `server=` argument, so this can be done
without editing the WebShop source. The main compatibility detail is that click
actions currently pass BeautifulSoup clickable nodes into `SimServer.receive`;
an HTTP backend must serialize only the needed clickable metadata and reconstruct
a small compatible object on the backend side.
