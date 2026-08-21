# Agent Env Backend Notes

This backend is a non-invasive agent-environment layer on top of Slime. The
goal of this document is not to duplicate the implementation. It records the
design boundaries and failure modes that are easy to miss when reading code.

For details, read the source and the resolved run profiles under
`RUN_ROOT/logs/`.

For the current TASA-GAE implementation, reproducible profiles, required dump
audits, and known unvalidated items, see [TASA-GAE Handoff](tasa_handoff.md).

Portable agent operating rules live in `.claude/skills/`:

- `.claude/skills/agentic-slime-discipline/SKILL.md`: agentic Slime code,
  config, rollout, reward, and training discipline.
- `.claude/skills/server-ops-discipline/SKILL.md`: server operations, runtime
  packs, node handling, keepalive, and artifact hygiene.

## Design Boundary

- Do not modify Slime core under `slime/` for agent-env experiments. Agent-env
  behavior is selected through Slime CLI hooks.
- The top-level launcher orchestrates nodes, Ray, env servers, optional aux
  inference, and the train adapter. It should not invent experiment semantics.
- The train adapter translates resolved profiles into Slime CLI arguments. It
  should not become a second config system.
- Env wrappers own environment semantics. Generic rollout code should not know
  task-specific reward rules, task data layout, or action semantics.
- Generated run artifacts, W&B logs, checkpoints, packs, secrets, and node-local
  runtime files do not belong in git.

Important hooks:

```text
--custom-generate-function-path examples.agent_env.<env>.rollout.generate
--custom-reward-post-process-path examples.agent_env.reward_post_process.post_process_rewards
--dynamic-sampling-filter-path examples.agent_env.reward_post_process.check_reward_nonzero_std
--rollout-sample-filter-path examples.agent_env.rollout.glm_style_pad_groups_filter
--group-rm
--custom-rm-path examples.agent_env.group_rm.group_reward
--rollout-function-path examples.agent_env.fully_async_rollout.generate_rollout_fully_async
```

## Config Ownership

- `configs/agent_env/runs/*.env`: thin launch manifests. Select env, env config,
  reward profile, model profile, train profile, topology profile, optional aux
  profile, and experiment naming.
- `configs/agent_env/models/*.env`: model identity, model args, loss-mask
  family, dropout defaults, and model compatibility defaults.
- `configs/agent_env/train/*.env`: env-specific training baseline, algorithm,
  sync/full-async mode, batch sizes, token budgets, TP/CP, actor/rollout
  allocation, checkpointing, filtering, and Slime training flags.
- `configs/agent_env/topology/*.env`: node indexes, visible GPUs, and ports.
  Keep algorithm, batch, model, and reward settings out of topology files.
- `configs/agent_env/aux/*.env`: optional auxiliary inference endpoint.
- `configs/agent_env/rewards/*.yaml`: reward composition, env-score scale,
  format/truncation penalties, LLM-as-judge mode, and method-specific settings.
- `examples/agent_env/<env>/env_config.yaml`: environment semantics, parser
  settings, data/task settings, and env-server settings.

Prefer changing the existing owner of a parameter over adding a wrapper,
override layer, copied config, or one-off profile. If a new key matters for an
experiment, it must appear in the selected profile and in the resolved profile
written under `RUN_ROOT/logs/`.

Train profile names should stay model-agnostic:

```text
<env>_<algorithm>_<sync|fullasync>_<nodes>x<gpus>.env
```

## What To Read

- Launch and node orchestration: `scripts/utils/launch_agentic_training.sh`
- Slime CLI adapter: `examples/agent_env/scripts/run_agent_env_train.sh`
- Extra Slime args and sync/async dispatch: `examples/agent_env/train_entrypoint.py`
- Generic multi-turn generation: `examples/agent_env/rollout.py`
- Full-async wrapper and padding/filtering: `examples/agent_env/fully_async_rollout.py`
- Final reward combination and dynamic sampling filter:
  `examples/agent_env/reward_post_process.py`
- Optional single/group judge hook: `examples/agent_env/group_rm.py`
- Env-specific behavior: `examples/agent_env/<env>/`

## Key Pitfalls

- Qwen dropout must be disabled for RL training. Otherwise old/current logprob
  comparisons become noisy and PPO/GRPO metrics can look broken.
- For full-async training, keep `USE_ROLLOUT_LOGPROBS=1`. Rollout data may be
  off-policy relative to the actor when consumed, so rollout-time logprobs are
  needed as the old policy denominator.
- `train_rollout_logprob_abs_diff=0` with `USE_ROLLOUT_LOGPROBS=1` is a Slime
  metric wart: the metric compares rollout logprobs to themselves. Use PPO KL,
  clipfrac, reward, length, and grad norm together.
- `SGLANG_SERVER_CONCURRENCY` limits the Slime request submission side. It is
  not the same as SGLang server `max_running_requests`.
- In full-async, align `GLOBAL_BATCH_SIZE`, `ROLLOUT_BATCH_SIZE *
  N_SAMPLES_PER_PROMPT`, and the total in-flight sample cap unless intentionally
  testing staleness. Mismatches can manufacture stale batches.
- Full-async worker backlog truncation is a coarse queue cap, not a strict
  policy-version freshness guarantee.
- Dynamic sampling drops zero-variance reward groups after reward computation.
  It does not resample individual samples.
- GLM-style padding is for infra-discard recovery: keep enough valid samples
  after removing bad samples, then pad from valid samples. Do not let discarded
  samples enter actor training as real data.
- Reward source must stay clear. The selected RM implementation composes env
  score, format reward, truncation penalty, process reward, and optional judge
  reward. Reward post-process only adapts the RM output to Slime's reward tensor
  path.
- Watchdog and bench are cluster keepalive mechanisms. They should not inspect
  training process state; they protect idle GPUs by utilization only.
- Checkpointing is intentionally sparse and capped during debugging to avoid
  filling NAS.

## Operational State

- Current node IPs live in a local ignored node file such as
  `configs/nodes/agent_env_all.txt`.
- Topology profiles select node indexes from that file; do not hard-code IPs
  into run/train profiles.
- Runtime packs and data live on NAS and are materialized to
  `${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}` and
  `${LOCAL_RUNTIME_DIR:-/tmp/server-ops-runtime}`.
- Task data is prepared on shared storage before node materialization:
  `${ROOT_DIR}/scripts/prepare_data.sh --data ...` constructs
  `${ROOT_DIR}/data/<env>`, `${ROOT_DIR}/scripts/pack_data.sh --data ...`
  archives it, and `${ROOT_DIR}/scripts/prepare_node_runtime.sh --data ...
  --validate-data-load` only unpacks or copies already prepared data.
- W&B should be provided by a compatible `wandb` env pack when the base image's
  SDK differs from Slime's expectations. The train adapter falls back to the
  Slime runtime and then to a version-compatible local Python.
- Multi-node NCCL/Gloo/TP traffic uses the launch-resolved socket interface.
  Set `SOCKET_IFNAME` or backend-specific `NCCL_SOCKET_IFNAME`,
  `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` when the routable interface is not
  `${MLP_SOCKET_IFNAME:-eth0}`; the launcher propagates these into Ray training
  actors through `TRAIN_ENV_VARS_JSON`.
- The development branch is `agentic-env-backend`; push local agent-env work to
  `origin`, and keep `upstream` read-only for fetching Slime updates.
