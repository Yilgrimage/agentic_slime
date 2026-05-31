# ALFWorld agentic rollout for slime

This example plugs ALFWorld into slime through `--custom-generate-function-path`.
One rollout sample is one ALFWorld episode. Model action tokens use
`loss_mask=1`; environment observations and formatting tokens use `loss_mask=0`.
When `alfworld_return_logprob: true`, model action tokens keep SGLang rollout
logprobs and environment tokens get dummy `0.0` logprobs so OPD/GRPO tensor
lengths stay aligned.

## Setup on cn_server_0

The example config points at this workspace data directory:

`/mnt/bn/jixf-nas-lq/mlf/data/alfworld`

Install ALFWorld in the runtime environment used for rollout workers. For the
current server smoke tests it was installed into an isolated target directory:

```bash
python3 -m pip install --target /mnt/bn/jixf-nas-lq/mlf/pythonlibs/alfworld_text alfworld
```

Full ALFWorld TextWorld reset requires `game.tw-pddl` files under:

`/mnt/bn/jixf-nas-lq/mlf/data/alfworld/json_2.1.1/{train,valid_seen,valid_unseen}`

The JSON trajectories and tw-pddl files are present on this server. Direct
GitHub release downloads were unstable during setup; `gh-proxy.com` worked for
the tw-pddl package:

```bash
wget -c --tries=5 --timeout=30 \
  -O /mnt/bn/jixf-nas-lq/mlf/data/alfworld/json_2.1.2_tw-pddl.zip \
  https://gh-proxy.com/https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip
python -m zipfile -e \
  /mnt/bn/jixf-nas-lq/mlf/data/alfworld/json_2.1.2_tw-pddl.zip \
  /mnt/bn/jixf-nas-lq/mlf/data/alfworld
```

Create prompt data. Each row is one ALFWorld task id; slime duplicates each row
`--n-samples-per-prompt` times for GRPO groups.

```bash
python examples/alfworld/make_prompt_data.py \
  --output /mnt/bn/jixf-nas-lq/mlf/data/alfworld/train.jsonl \
  --num-tasks 100 \
  --split train

python examples/alfworld/make_prompt_data.py \
  --output-dir /mnt/bn/jixf-nas-lq/mlf/data/alfworld \
  --num-tasks 100 \
  --splits train valid_seen valid_unseen
```

## Runtime Design

This adapter keeps the training/rollout environment and ALFWorld runtime
decoupled:

- `generate_with_alfworld.py` stays on the slime side. It only talks to SGLang
  for model actions and to the ALFWorld HTTP server for `/reset`, `/step`, and
  `/close`.
- `env_server.py` owns the ALFWorld import, data path, and env lifecycle. The
  slime training environment does not need ALFWorld on its `PYTHONPATH`.
- The server prewarms a pool of ALFWorld env workers per split. A rollout episode
  leases one worker at reset time and returns it on close, so we avoid repeatedly
  constructing ALFWorld/TextWorld envs.
- The server is process-isolated. The HTTP process only owns lease routing; each
  warm ALFWorld/TextWorld env lives in a child process. Independent active
  episodes can reset/step concurrently without sharing parser state.
- If slime uses `router_policy=consistent_hashing`, the adapter forwards
  `sample.session_id` as the SGLang routing key.
- For high-throughput training with long-tail episode lengths, use slime
  `train_async.py` plus fully-async rollout so slow ALFWorld episodes do not
  block the next training batch.

Server-side knobs live in `alfworld_config.yaml` and
`alfworld_smoke_config.yaml`:

```yaml
alfworld_server_pool_size: 8
alfworld_server_acquire_timeout_s: 30
alfworld_server_session_ttl_s: 1800
alfworld_server_lease_ttl_s: 1800
alfworld_server_idempotency_ttl_s: 300
alfworld_server_reuse_envs: true
alfworld_server_reset_on_release: false
alfworld_server_worker_start_timeout_s: 120
alfworld_server_worker_request_timeout_s: 120
alfworld_server_prewarm_splits:
  - train
alfworld_server_honor_direct_game_file: true
```

When `alfworld_server_honor_direct_game_file: true`, direct-game reset retargets
a pooled worker to a single `game.tw-pddl`, so `task_index -> game_file` mapping
is exact while still reusing warm workers. Set it to `false` to use TextWorld's
native seeded shuffle/reset sequence. Set `alfworld_server_reuse_envs: false` if
you explicitly want dedicated envs.


### Lease API

The ALFWorld server follows a lease/pool API for large-scale rollout workers:

```text
POST /allocate   -> {lease_id, worker_id}
POST /reset      -> reset the leased worker to a task
POST /step       -> execute one action on the same worker
POST /evaluate   -> return the latest outcome state
POST /heartbeat  -> refresh lease TTL
POST /close      -> idempotently release the lease
GET  /status     -> pool, lease, and worker counters
```

`session_id` is still accepted as an alias for `lease_id` for compatibility with
older rollout code. `request_id` on `/allocate` is idempotent, which prevents a
rollout retry from accidentally occupying two workers. The server remains single
node today; a future router can encode worker identity into `lease_id` and proxy
all lease-scoped requests back to the owning env worker.

## GRPO

Add these rollout arguments to a normal slime GRPO script:

```bash
--prompt-data /mnt/bn/jixf-nas-lq/mlf/data/alfworld/train.jsonl \
--input-key prompt \
--metadata-key metadata \
--custom-generate-function-path examples.alfworld.generate_with_alfworld.generate \
--custom-config-path examples/alfworld/alfworld_config.yaml \
--advantage-estimator grpo
```

Add eval through slime's native eval dataset path. `alfworld_eval_config.yaml`
uses the same custom generate function and injects the ALFWorld split metadata:

```bash
--eval-interval 5 \
--eval-config examples/alfworld/alfworld_eval_config.yaml
```

For fully-async training, keep the same custom generate function and launch the
training entry with slime's fully-async rollout path:

```bash
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
```

## OPD on top of GRPO

For `--opd-type sglang`, the rollout stores ALFWorld environment reward in
sample metadata and leaves `sample.reward` for the teacher server. The
post-process hook then restores env rewards and attaches teacher logprobs.

```bash
--use-opd \
--opd-type sglang \
--opd-kl-coef 1.0 \
--reward-model-path slime.rollout.on_policy_distillation.reward_func \
--custom-reward-post-process-path examples.alfworld.opd.post_process_rewards
```

`--opd-type megatron` needs no ALFWorld-specific reward post-process because the
teacher logprobs are computed inside Megatron during actor training.

## Smoke tests

```bash
python -m py_compile \
  examples/alfworld/generate_with_alfworld.py \
  examples/alfworld/opd.py \
  examples/alfworld/make_prompt_data.py \
  examples/alfworld/smoke_test.py \
  examples/alfworld/real_env_smoke_test.py
PYTHONPATH=. python examples/alfworld/smoke_test.py
PYTHONPATH=/mnt/bn/jixf-nas-lq/mlf/pythonlibs/alfworld_text:. \
  python examples/alfworld/real_env_smoke_test.py
```

## Notes

- `alfworld_direct_game_file: true` is the default path for training. It selects
  the episode file directly from the cached split wrapper.
- Set `alfworld_restrict_to_admissible: true` only if you want invalid model
  actions rewritten before `env.step()`.
- Partial rollout is intentionally disabled for this adapter.
