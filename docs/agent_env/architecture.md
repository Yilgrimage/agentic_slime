# Agent Env Architecture

This stack adds agentic environment training to Slime without modifying Slime
source code. The guiding principle is that Slime remains the trainer and rollout
framework, while `examples/agent_env` provides external functions and services
that Slime already knows how to import by path.

## Component Boundaries

### Slime Owns

- base training loops: `train.py` and `train_async.py`
- actor rollout/update scheduling
- Ray placement groups
- Megatron actor training
- SGLang policy rollout engines
- PPO/GRPO loss computation
- checkpoint save/load
- W&B logging
- public custom function hooks

### Agent Env Owns

- environment HTTP servers and router
- environment reset/step/evaluate adapters
- custom generation loop for multi-turn agent trajectories
- action/tool parsing and token-level bookkeeping
- rollout case dumping
- reward post-processing
- optional custom RM judge calls
- agent-env training entrypoint that registers external CLI args and dispatches
  to Slime's base training loops
- profile-driven launch and runtime materialization

### Cluster Utilities Own

- node-local environment/data materialization
- starting/stopping Ray and env servers across selected nodes
- optional auxiliary inference endpoint startup
- GPU keepalive integration through the external NAS `run_bench.sh`

## High-Level Launch Flow

The public entrypoint is:

```bash
bash scripts/utils/launch_agentic_training.sh <run-profile.env>
```

The launcher does this once on the submitting machine:

1. Source the selected run profile from `configs/agent_env/runs/`.
2. Source the selected topology, model, train, and optional aux profiles.
3. Resolve relative paths against `REPO_DIR`.
4. Write auditable resolved env files under `RUN_ROOT/logs/`.
5. Start optional aux inference endpoint when `AUX_PROFILE` is set.
6. Stop GPU bench on train nodes.
7. Reset stale train runtime on selected train nodes.
8. Start one head tmux on the first selected train node.

The head role then:

1. Starts a Ray head on the head node.
2. Starts one env server on the head node.
3. Starts one worker tmux per remaining selected train node.
4. Waits for worker env servers.
5. Waits for all Ray nodes to join.
6. Starts the env router on the head node.
7. Starts the train driver tmux.

Each worker role:

1. Starts its local env server.
2. Joins the head Ray cluster.

The train driver:

1. Sources optional aux endpoint env file.
2. Calls `examples/agent_env/scripts/run_agent_env_train.sh --env-server-url <router-url>`.

The train adapter translates resolved profiles to a Slime CLI command. It
passes the router endpoint as Slime arg `--env-server-url`, so rollout code
reads `args.env_server_url` instead of process environment variables.

## Environment Server and Router

Each training node runs one environment server:

```text
examples/agent_env/<env>/server.py
```

That env-specific server builds on the shared server protocol:

```text
examples/agent_env/server.py
```

The server exposes:

```text
GET  /health
GET  /status
POST /allocate
POST /reset
POST /step
POST /evaluate
POST /close
```

The router runs on the head node:

```text
examples/agent_env/router.py
```

The router is a lease router. Rollout sends a task key. The router maps that key
to a preferred worker with a stable hash and falls back to other workers when
the preferred worker is unavailable or capacity-limited. The returned lease id
encodes the worker index, so later reset/step/evaluate/close requests are sticky
to the owning worker.

This keeps environment execution outside Slime and outside SGLang. SGLang only
generates model responses; the env servers own task state and scoring.

## Custom Generation Flow

Slime imports the env custom generator through:

```text
--custom-generate-function-path examples.agent_env.<env>.rollout.generate
```

Each env-specific generator is intentionally thin. It provides the environment
specification and delegates common work to `examples.agent_env.rollout`.

The shared rollout loop handles:

- policy calls through Slime/SGLang interfaces
- prompt/message ledger construction
- multi-turn stopping
- response token accounting
- text-action or tool-call parsing
- HTTP env reset/step/evaluate calls
- env score/reward metadata capture
- format and truncation metadata
- rollout case dumping
- token-level reward list shape management

The env-specific adapter owns:

- prompt text and chat messages
- whether actions are text tags or tool calls
- parser details
- valid/admissible action extraction
- invalid-action handling
- success and score interpretation
- backend-specific reset/step/evaluate payloads

## Reward Flow

Reward handling is intentionally centralized.

1. The environment rollout records environment outcome in `sample.metadata`.
   Common fields include `env_score`, `env_reward`, `env_success`, `actions`,
   `turns`, format counters, and truncation counters.
2. The optional custom RM hook writes a judge score into `sample.reward` and
   records judge metadata in `sample.metadata`.
3. The custom reward post-process function combines all reward sources into the
   final train reward.

The final combiner is:

```text
examples.agent_env.reward_post_process.post_process_rewards
```

It computes:

```text
raw_reward = env_reward + judge_score + format_reward + truncated_reward
```

For removed samples it uses zero contribution. It also records:

```text
env_reward_for_train
judge_score_for_train
format_reward
truncated_reward
raw_reward
```

When Slime GRPO reward normalization is enabled, the post-process returns both
raw rewards and normalized rewards according to the active group. This keeps
W&B metrics interpretable: raw reward remains the task score scale, while
normalized rewards are the training advantages passed to the actor.

## Group RM and No-Judge Mode

The custom RM entrypoint is:

```text
examples.agent_env.group_rm.group_reward
```

It supports both Slime calling conventions:

- `Sample -> float` for non-group RM mode.
- `list[Sample] -> list[float]` for `--group-rm`.

By default:

```text
AGENT_ENV_JUDGE_MODE=none
```

In this mode the RM hook returns zero judge scores and records that judge was
skipped. This keeps the Slime RM path active and stable while making the final
reward come from env/format/truncation only.

To enable an aux judge:

```text
AGENT_ENV_JUDGE_MODE=aux
AUX_ENDPOINT_PROVIDER=sglang|vllm|deepseek|...
AUX_ENDPOINT_MODEL=<model-name>
AUX_ENDPOINT_BASE_URL=http://host:port/v1
AUX_ENDPOINT_API_KEY_PATH=<optional secret file>
```

When `AGENT_ENV_JUDGE_MODE=aux` is selected, missing endpoint/model config is a
hard error. This avoids accidentally running a judge experiment with a silent
zero-score fallback.

The judge prompt construction lives in `group_rm.py`, not in rollout code. This
keeps rollout independent of any particular judge provider.

## Dynamic Sampling Filter

The dynamic filter is:

```text
examples.agent_env.reward_post_process.check_reward_nonzero_std
```

It computes the same raw reward formula used by post-processing and drops a
group if all active samples have zero reward variance. For GRPO, a zero-std
group produces zero advantage and wastes actor compute. Dropping these groups
is the intended dynamic sampling behavior.

Sync Slime rollout calls `--dynamic-sampling-filter-path` directly.

Full-async Slime rollout does not apply this filter before actor consumption in
the stock path. To keep the fix outside Slime source code, full-async train
profiles use:

```text
--rollout-function-path examples.agent_env.fully_async_rollout.generate_rollout_fully_async
```

This external wrapper reuses Slime's full-async worker and queue, drains
completed groups, applies the dynamic filter, records dynamic filter metrics,
and keeps collecting until it has `rollout_batch_size` accepted groups.

The wrapper imports Slime's private `_get_global_worker`. This is the one
intentional compatibility risk in the non-invasive integration. If upstream
Slime changes that private symbol, update this wrapper rather than patching
Slime source.

## Sync Versus Full-Async

### Sync Training

Sync profiles use:

```text
TRAIN_ENTRYPOINT=examples/agent_env/train_entrypoint.py
AGENT_ENV_TRAIN_LOOP=sync
ROLLOUT_FUNCTION_PATH=slime.rollout.sglang_rollout.generate_rollout
COLOCATE=1
```

Typical behavior:

- actor and rollout engines colocate on the same Ray cluster resources
- rollout batch is generated before actor update
- dynamic sampling filter is applied in Slime's stock sync rollout path
- `USE_ROLLOUT_LOGPROBS` can be disabled because the rollout batch is effectively
  on-policy with respect to the current actor update

Sync is useful as a correctness baseline.

### Full-Async Training

Full-async profiles use:

```text
TRAIN_ENTRYPOINT=examples/agent_env/train_entrypoint.py
AGENT_ENV_TRAIN_LOOP=async
ROLLOUT_FUNCTION_PATH=examples.agent_env.fully_async_rollout.generate_rollout_fully_async
COLOCATE=0
USE_ROLLOUT_LOGPROBS=1
```

Typical behavior:

- actor GPUs and rollout GPUs are separately allocated
- rollout continues while actor trains
- samples can be generated by a previous policy
- rollout-time log probabilities must be retained so PPO/GRPO ratios use the
  behavior policy rather than an accidental recomputation with the current actor
- dynamic filter must run before actor consumption, which is why the external
  full-async rollout wrapper exists

Full-async is the intended throughput path after correctness is established.

## PPO/GRPO Log Probability Semantics

In PPO-style training there are three relevant policies:

- `pi_b`: behavior policy that generated a sampled trajectory.
- `pi_theta`: current trainable actor used for the gradient step.
- `pi_ref`: optional fixed reference model for KL penalty.

The PPO/GRPO ratio should compare `pi_theta` against `pi_b` for the sampled
tokens. In full-async training, `pi_b` may be older than `pi_theta`. Therefore
`USE_ROLLOUT_LOGPROBS=1` is required for async profiles so the rollout-time
behavior logprobs are available.

The reference model is different. It is only used when KL loss is enabled. A
large `ppo_kl` can still appear in metrics even when KL loss coefficient is
zero, but the ratio/clip path still needs the correct behavior logprobs.

## Runtime Outputs

Each run lives under:

```text
${MLF_NAS_ROOT}/runs/${EXP_PROJECT}/${EXP_NAME}
```

Important subdirectories:

```text
logs/
  resolved_launch.env
  resolved_train_profile.env
  resolved_aux_profile.env
  <env>_train.log
  <env>_train_status.env
  multi_head.log
  ray_head.log
  <env>_router.log
  <env>_env_server.log

rollout_cases/
  <env>/samples/
  <env>/discarded/

checkpoints/
wandb/
```

The Slime `--custom-config-path` points at the static env config selected by
the run profile, for example `examples/agent_env/alfworld/env_config.yaml`.
Dynamic service endpoints are not written back into YAML. The generated train
driver records the explicit `--env-server-url <router-url>` argument used for
that run, while resolved profiles record the static configuration inputs.

## Why This Is Not a Slime Fork

The implementation deliberately avoids editing:

```text
slime/
slime_plugins/
train.py
train_async.py
```

The only Slime dependency surface used is public CLI import hooks plus one
private full-async worker helper imported by the external wrapper. If future
Slime exposes dynamic filtering in full-async directly, the wrapper can be
removed and train profiles can point back to Slime's stock full-async rollout
function.
