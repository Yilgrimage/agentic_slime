# WebShop Agentic Rollout

This example keeps WebShop outside the slime training environment. The slime process calls `examples.webshop.generate.generate`, which talks to `env_server.py` over HTTP. The env server owns process-isolated WebShop workers and must run in the separate WebShop conda-pack runtime.

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
3. Materialize the runtime with `examples/webshop/materialize_webshop_runtime.sh`.

`prepare_webshop_data.sh` is only for rebuilding the NAS data backup from public sources. It is not part of normal machine migration or training startup.

## Current validation scope

The current runtime is a 1k-product WebShop smoke setup, not a full 100k-product
WebShop setup.

Verified on node0:

- `/tmp/mlf-runtime/code/WebShop/data/items_shuffle_1000.json` has 1000 products.
- `/tmp/mlf-runtime/code/WebShop/data/items_ins_v2_1000.json` has 1000 products.
- The checked `resources_100k/documents.jsonl` copy also has 1000 rows, so its
  name is misleading in this runtime.
- `human_goals=true` with the current 1k product file loads 13 goals.
- `pool_size=32` starts 32 process-isolated WebShop workers with about 12.7 GiB
  total child RSS before interaction and about 13.1 GiB after 32 concurrent
  reset/search interactions.
- A Qwen3-8B GRPO smoke run completed one training step on 4 GPUs with the
  WebShop server at `pool_size=32`.

Do not claim full WebShop coverage until the NAS/runtime contains the real 100k
product JSON, matching attributes/goals, and a matching Lucene index.

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
