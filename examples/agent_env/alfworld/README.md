# ALFWorld agentic rollout for slime

This example plugs ALFWorld into slime through `--custom-generate-function-path`.
One rollout sample is one ALFWorld episode. `rollout.py` now only
declares ALFWorld-specific prompt/action/success behavior and delegates the
common agent loop to `examples.agent_env.rollout`. Model action tokens use
`loss_mask=1`; environment observations use `loss_mask=0`. The shared rollout
keeps semantic `messages` during interaction and lets slime's native
`MultiTurnLossMaskGenerator` build final `tokens` and `loss_mask` at episode
end. Rollout logprobs are not emitted by the env wrapper; the training backend
recomputes old logprobs when needed.

Directory layout:

- `rollout.py`: ALFWorld `AgentEnvSpec` for the shared rollout
  loop.
- `server.py`: ALFWorld backend for the shared process-pool lease server.
- `prompt_data.py`: ALFWorld prompt metadata generation.
- `scripts/`: ALFWorld shell entrypoints.

## Setup on cn_server_0

The example config points at this workspace data directory:

`/mnt/bn/jixf-nas-lq/mlf/data/alfworld`

Install ALFWorld in the runtime environment used for rollout workers. For the
current server smoke tests it was installed into an isolated target directory:

```bash
python3 -m pip install --target /tmp/mlf-runtime/alfworld/pythonlibs/alfworld_text alfworld
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
python examples/agent_env/alfworld/prompt_data.py \
  --output /mnt/bn/jixf-nas-lq/mlf/data/alfworld/train.jsonl \
  --num-tasks 100 \
  --split train

python examples/agent_env/alfworld/prompt_data.py \
  --output-dir /mnt/bn/jixf-nas-lq/mlf/data/alfworld \
  --num-tasks 100 \
  --splits train valid_seen valid_unseen
```

## Runtime Design

This adapter keeps the training/rollout environment and ALFWorld runtime
decoupled:

- `rollout.py` stays on the slime side. It only talks to SGLang
  for model actions and to the ALFWorld HTTP server for `/reset`, `/step`, and
  `/close`.
- `server.py` owns the ALFWorld import, data path, and env lifecycle. The
  slime training environment does not need ALFWorld on its `PYTHONPATH`.
- The shared process-pool server prewarms a pool of ALFWorld env workers per
  split. A rollout episode leases one worker at reset time and returns it on
  close, so we avoid repeatedly constructing ALFWorld/TextWorld envs.
- The server is process-isolated. The HTTP process only owns lease routing and
  worker lifecycle; each warm ALFWorld/TextWorld env lives in a child process.
  Independent active episodes can reset/step concurrently without sharing parser
  state.
- If slime uses `router_policy=consistent_hashing`, the adapter forwards
  `sample.session_id` as the SGLang routing key.
- For high-throughput training with long-tail episode lengths, use slime
  `train_async.py` plus fully-async rollout so slow ALFWorld episodes do not
  block the next training batch.

Server-side knobs live in `train_config.yaml`:

```yaml
env_server:
  pool_size: 8
  acquire_timeout_s: 30
  lease_ttl_s: 1800
  idempotency_ttl_s: 300
  reuse_workers: true
  reset_on_release: false
  worker_start_timeout_s: 120
  worker_request_timeout_s: 120
  prewarm_splits:
    - train
  honor_direct_game_file: true
```

When `env_server.honor_direct_game_file: true`, direct-game reset retargets
a pooled worker to a single `game.tw-pddl`, so `task_index -> game_file` mapping
is exact while still reusing warm workers. Set it to `false` to use TextWorld's
native seeded shuffle/reset sequence. Set `env_server.reuse_workers: false` if
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
rollout retry from accidentally occupying two workers. Multi-node routing is
handled by `examples.agent_env.router`, which encodes worker identity into the
global `lease_id` and proxies all lease-scoped requests back to the owning env
worker.

## GRPO

Add these rollout arguments to a normal slime GRPO script:

```bash
--prompt-data /mnt/bn/jixf-nas-lq/mlf/data/alfworld/train.jsonl \
--input-key prompt \
--metadata-key metadata \
--custom-generate-function-path examples.agent_env.alfworld.rollout.generate \
--custom-rollout-log-function-path examples.agent_env.alfworld.rollout.log_rollout_data \
--custom-eval-rollout-log-function-path examples.agent_env.alfworld.rollout.log_eval_rollout_data \
--custom-config-path examples/agent_env/alfworld/train_config.yaml \
--advantage-estimator grpo
```

Add eval through slime's native eval dataset path. `eval_config.yaml`
uses the same custom generate function and injects the ALFWorld split metadata:

```bash
--eval-interval 5 \
--eval-config examples/agent_env/alfworld/eval_config.yaml
```

For fully-async training, keep the same custom generate function and launch the
training entry with slime's fully-async rollout path:

```bash
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
```

## Smoke tests

```bash
python -m py_compile \
  examples/agent_env/alfworld/rollout.py \
  examples/agent_env/alfworld/prompt_data.py \
  examples/agent_env/alfworld/smoke_test.py \
  examples/agent_env/alfworld/real_env_smoke_test.py
PYTHONPATH=. python examples/agent_env/alfworld/smoke_test.py
PYTHONPATH=/tmp/mlf-runtime/alfworld/pythonlibs/alfworld_text:. \
  python examples/agent_env/alfworld/real_env_smoke_test.py
```

## Notes

- `alfworld.direct_game_file: true` is the default path for training. It selects
  the episode file directly from the cached split wrapper.
- The canonical dense reward channel is `metadata["token_rewards"]`, a
  `response_length`-aligned list. Format rewards/penalties are placed on the
  final generated token of each turn; the environment outcome reward is placed on
  the final token of the rollout. `sample.reward` is the sum of this list for
  compatibility with slime's current scalar GRPO reward path.
- Generated assistant output is kept in the multi-turn training context as the
  token sequence returned by rollout. The current action is parsed from that
  generated turn, but invalid-format turns are not shortened or rewritten before
  being appended to context.
- The default reward scale is `reward.outcome: 10.0` for success and
  `reward.format.invalid: -0.1` for malformed action output.
- Per-rollout metadata is intentionally small: `actions`, `turn_count`,
  `format_ok`, `format_errors`, `env_score`, `env_success`, `env_reward`, and an
  `alfworld` block for task/server identifiers.
- `examples.agent_env.alfworld.rollout` aggregates metadata into slime's tracking path,
  including `alfworld/format_error_rate`, `alfworld/success_rate`, and eval
  variants under `eval/<dataset>/...`.
- Set `action.restrict_to_available: true` only if you want invalid model
  actions rewritten before `env.step()`.
- Partial rollout is intentionally disabled for this adapter.
