# WebShop Agentic Rollout

This example keeps WebShop outside the slime training environment. The slime
process calls `examples.agent_env.webshop.rollout.generate`, which declares WebShop's
environment-specific `AgentEnvSpec` and delegates episode execution to the
WebShop env server through the Slime-side policy gateway. The WebShop env
server owns process-isolated workers and must run in the separate WebShop
conda-pack runtime.

Directory layout:

- `rollout.py`: WebShop prompt/action/success spec for the server-episode adapter.
- `server.py`: WebShop backend for the shared process-pool lease server.
- `prompt_data.py`: WebShop prompt metadata generation.
- `env_config.yaml`: WebShop data, interaction, and env server config.

Expected packs:

- slime: training, Ray, sglang, Megatron, torch
- webshop: Gym/WebShop and WebShop data/runtime dependencies
- alfworld: ALFWorld/TextWorld dependencies

Runtime convention:

- Shared storage keeps reusable packs, source checkouts, and data under
  `${ROOT_DIR}`.
- Runtime materialization copies IO-heavy runtime pieces to
  `${LOCAL_RUNTIME_DIR:-/tmp/server-ops-runtime}`.
- No WebShop Python dependency should be installed into the slime pack.

Runtime setup:

1. Ensure the WebShop pack exists at `${ROOT_DIR}/packs/webshop.tar.gz`.
2. Ensure the WebShop data backup exists at `${ROOT_DIR}/data/webshop`.
3. Materialize node-local runtime with `${ROOT_DIR}/scripts/prepare_node_runtime.sh`.

Use `scripts/utils/build_webshop_env.sh`, `scripts/utils/pack_webshop_env.sh`, and
`${ROOT_DIR}/scripts/pack_data.sh` only when rebuilding shared packs/data. They
are not part of normal training startup.

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

- `${ROOT_DIR}/data/webshop/data/items_shuffle.json`
- `${ROOT_DIR}/data/webshop/data/items_ins_v2.json`
- `${ROOT_DIR}/data/webshop/data/items_human_ins.json`
- `${ROOT_DIR}/data/webshop/search_engine/indexes_100k`

Pack full data with:

```bash
DATASETS=webshop ${ROOT_DIR}/scripts/pack_data.sh
```

## ROPD Teacher Data

Do not use the historical
`${ROOT_DIR}/data/teacher_traces/processed/webshop_official_il_teacher_tool_trace.jsonl`
as a quality baseline. It was rebuilt from WebShop's official IL trajectories
by matching normalized instruction text against the current env goals. That
file is useful only as a smoke artifact:

- it covers only 137 of the current 1021 train goals;
- 83 of those 137 rows have instruction/price mismatch between the current goal
  and the teacher trace, because the old matcher removed the price constraint
  when building the instruction key;
- it is not a true rollout from the current env config.

Preferred WebShop ROPD teachers should be produced by running a strong policy
against the exact current WebShop env configuration:

- `items_shuffle.json`
- `items_ins_v2.json`
- `num_products=100000`
- the selected train task ids and split

Each teacher row must include `task_id` or `task_index`, `split`,
`teacher_tool_trace` or `teacher_trace`, `teacher_success`, and
`teacher_score`. Store raw rollout outputs separately if needed, but feed ROPD
the canonical trace rendered through the shared agent-env trace renderer.

Validate before training:

```bash
python examples/agent_env/scripts/validate_teacher_jsonl.py \
  --env webshop \
  --teacher-jsonl "${ROOT_DIR}/data/teacher_traces/processed/webshop_teacher.jsonl" \
  --expected-count 1021 \
  --min-coverage 0.9 \
  --min-success-rate 0.9 \
  --require-success-only
```

Launch with the standard WebShop ROPD profile and override only the teacher
file path:

```bash
AGENT_ENV_ROPD_TEACHER_INDEX_PATH="${ROOT_DIR}/data/teacher_traces/processed/webshop_teacher.jsonl" \
PROMPT_DATA_EXTRA_ARGS="--task-id-file ${ROOT_DIR}/data/teacher_traces/processed/webshop_teacher.jsonl" \
bash scripts/utils/launch_agentic_training.sh \
  configs/agent_env/runs/webshop_qwen3_4b_ropd_fullasync_3x8_qwen27.env
```

## Shared backend direction

The current server starts one `WebAgentTextEnv` per worker process. Each worker
therefore loads its own `SimServer`, product list, goals, and Lucene searcher.
That is acceptable for the 1k smoke setup, but it is the wrong shape for full
WebShop.

The intended scalable design is:

- process-isolated episode workers keep lease/session lifecycle, observations,
  action parsing, policy-gateway calls, and environment-loop ownership;
- a shared WebShop backend service loads product data, goals, and Lucene search
  index once per node;
- workers call the backend for `SimServer.receive`-equivalent operations.

`WebAgentTextEnv` already accepts a `server=` argument, so this can be done
without editing the WebShop source. The main compatibility detail is that click
actions currently pass BeautifulSoup clickable nodes into `SimServer.receive`;
an HTTP backend must serialize only the needed clickable metadata and reconstruct
a small compatible object on the backend side.
