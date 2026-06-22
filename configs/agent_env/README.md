# Agent Env Run Profiles

This directory keeps experiment and topology configuration out of
`examples/agent_env/<env>`.

- `runs/*.env`: one launchable experiment manifest. It should only select the
  env, env config, model profile, topology profile, train profile, optional aux
  profile, and experiment naming.
- `models/*.env`: model identity, Megatron model-args script, model-family loss
  mask, and required compatibility entrypoint.
- `topology/*.env`: node indexes, ports, and Ray-visible GPU layout.
- `train/*.env`: algorithm, rollout sampling/filtering, batch/perf layout,
  logging, and checkpoint parameters.
- `aux/*.env`: auxiliary inference endpoint parameters.

Launch through:

```bash
bash scripts/mlf/launch_agentic_training.sh configs/agent_env/runs/tau2_qwen35_4b_grpo_m2p7_3train_1aux.env
```

The launcher writes resolved profiles under the run log directory before
starting services, so the exact effective parameters are auditable.

Keep run files thin. If a setting changes model family behavior, put it in a
model profile. If it changes Slime training behavior, put it in a train
profile. If it changes node layout or ports, put it in a topology profile. If
it changes auxiliary inference, put it in an aux profile. If it changes
environment semantics, put it in the env `env_config.yaml`.

Override rule:

- Values in profiles are written as `${VAR:-default}` so command-line
  environment overrides work for declared keys.
- A setting is considered train-controllable only if it appears in the selected
  run/topology/train/aux profile and then appears in `resolved_*.env`.
- For a new Slime or env-server knob, add it to the appropriate profile first;
  do not rely on an undeclared shell variable being implicitly forwarded.

`configs/nodes/agent_env_all.txt` is the single source of truth for current
cluster IPs. It is operational state, not an experiment hyperparameter. After
Arnold resets a trial, update only that file. Topology profiles select train
and aux nodes by zero-based `NODE_INDICES` / `AUX_NODE_INDICES`.

Keep API keys, W&B keys, and provider credentials out of these profiles. Source
them from NAS secret env files at launch time.
