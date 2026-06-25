---
name: agentic-slime-discipline
description: "Use when editing, reviewing, launching, or debugging the agentic Slime RL training stack: configs/agent_env profiles, examples/agent_env rollout/reward/env code, Slime CLI adapters, launch profile boundaries, reward/filter/padding logic, sync/full-async training behavior, and non-invasive integration with upstream Slime."
---

# Agentic Slime Discipline

Keep the agentic Slime stack small, non-invasive, and auditable. Prefer one
clear owner for each behavior over copied profiles, wrappers, fallback branches,
or patch-style fixes.

## Core Boundary

- Do not modify Slime core for agent-env behavior unless explicitly asked.
  Prefer external hooks selected by Slime CLI arguments.
- Keep upstream Slime updateable. Agentic extensions live outside core Slime
  packages, usually under `examples/agent_env`, `configs/agent_env`, and
  launch/adapter scripts.
- Do not hide experiment semantics in launch scripts. Launchers orchestrate;
  profiles own configuration; env code owns environment behavior.

Common hook paths:

```text
--custom-generate-function-path examples.agent_env.<env>.rollout.generate
--custom-reward-post-process-path examples.agent_env.reward_post_process.post_process_rewards
--dynamic-sampling-filter-path examples.agent_env.reward_post_process.check_reward_nonzero_std
--rollout-sample-filter-path examples.agent_env.rollout.glm_style_pad_groups_filter
--group-rm
--custom-rm-path examples.agent_env.group_rm.group_reward
--rollout-function-path examples.agent_env.fully_async_rollout.generate_rollout_fully_async
```

## Configuration Ownership

- `configs/agent_env/runs/*.env`: select env, env config, model profile, train
  profile, topology profile, optional aux profile, and experiment naming only.
- `configs/agent_env/models/*.env`: model identity, model args, loss-mask
  family, dropout/model extras, and model compatibility defaults.
- `configs/agent_env/train/*.env`: env-specific training baseline, algorithm,
  sync/full-async mode, rollout sampling/filtering, batch sizes, token budgets,
  TP/CP, actor/rollout allocation, checkpointing, logging, and Slime flags.
- `configs/agent_env/topology/*.env`: node indexes, visible GPUs, and ports
  only. Keep algorithm, batch, token, actor/rollout, model, and aux semantics
  out of topology.
- `configs/agent_env/aux/*.env`: auxiliary inference endpoint only.
- `examples/agent_env/<env>/env_config.yaml`: environment semantics, reward
  fields, task/data settings, parser/interaction settings, and env server
  settings.

Train profile names should be:

```text
<env>_<algorithm>_<sync|fullasync>_<nodes>x<gpus>.env
```

Do not include model names or aux providers in train profile names.

## Change Discipline

- Locate the current owner of a setting before editing.
- Change the existing owner instead of adding a wrapper, override layer, copied
  YAML/env file, or one-off profile.
- Create a new profile only for a durable reusable baseline.
- Add config keys only when they are consumed by the launcher/adapter or Slime
  CLI and visible in resolved profiles.
- Remove dead compatibility paths and abandoned scripts instead of preserving
  them for vague safety.
- Prefer direct APIs over thin wrappers that only rename, format one path, or
  forward arguments. Keep wrappers only when they own validation, lifecycle,
  compatibility, metrics, retry, async, or cross-module boundary behavior.

## Rollout And Reward Rules

- Env wrappers own prompt rendering, parser/action semantics, backend reset/step
  calls, success/score interpretation, and env-specific metadata.
- Generic rollout code owns message/token ledger, env HTTP lease lifecycle,
  sample shape, common dumping, and Slime data contracts.
- Reward post-process owns final reward composition: env score, format reward,
  truncation penalty, and optional judge reward.
- Group RM should judge only when a judge is explicitly enabled. It should not
  silently become a second reward combiner.
- Dynamic sampling drops zero-variance reward groups after reward computation.
  It does not resample individual samples.
- GLM-style padding is for infra-discard recovery. Discard bad samples, then pad
  from valid samples; never train on discarded samples as real data.

## Training Pitfalls

- Disable model dropout for PPO/GRPO/RL training. Dropout corrupts old/current
  logprob comparisons and can make KL/clip metrics meaningless.
- In full-async training, use rollout-time logprobs as the old policy
  denominator, for example `USE_ROLLOUT_LOGPROBS=1`.
- Treat `train_rollout_logprob_abs_diff=0` under rollout-logprob mode as a
  metric wart if it compares rollout logprobs to themselves. Cross-check PPO KL,
  clipfrac, reward, length, and grad norm.
- Align `GLOBAL_BATCH_SIZE`, `ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT`, and
  full-async in-flight capacity unless deliberately testing staleness.
- `SGLANG_SERVER_CONCURRENCY` may limit request submission rather than SGLang's
  internal `max_running_requests`; verify the actual path before drawing
  throughput conclusions.
- Keep checkpointing sparse and capped during debugging.

## Review Checklist

- Is Slime core untouched?
- Is the run profile thin?
- Is each changed parameter in the right owner file?
- Are model, train, topology, aux, and env semantics separated?
- Are resolved profiles sufficient to audit the effective run?
- Did the change remove complexity rather than add a compensating patch?
