# Agent Env Run Profiles

This directory keeps experiment and topology configuration out of
`examples/agent_env/<env>`.

- `runs/*.env`: one launchable experiment wiring file.
- `topology/*.env`: node lists, ports, and Ray-visible GPU layout.
- `train/*.env`: model, GRPO, rollout, logging, and checkpoint parameters.
- `aux/*.env`: auxiliary inference endpoint parameters.

Launch through:

```bash
bash examples/agent_env/scripts/launch_agent_env.sh configs/agent_env/runs/tau2_qwen35_4b_grpo_m2p7_3train_1aux.env
```

The launcher writes resolved profiles under the run log directory before
starting services, so the exact effective parameters are auditable.

Override rule:

- Values in profiles are written as `${VAR:-default}` so command-line
  environment overrides work for declared keys.
- A setting is considered train-controllable only if it appears in the selected
  run/topology/train/aux profile and then appears in `resolved_*.env`.
- For a new Slime or env-server knob, add it to the appropriate profile first;
  do not rely on an undeclared shell variable being implicitly forwarded.

Node IP files under `configs/nodes/` are operational state. They should be
updated when Arnold resets a trial, but they are not experiment
hyperparameters. For tau2, keep the training Ray nodes and the auxiliary
usersim node split:

- `configs/nodes/agent_env_tau2_train_3x8.txt`
- `configs/nodes/agent_env_tau2_aux_1x8.txt`

Keep API keys, W&B keys, and provider credentials out of these profiles. Source
them from NAS secret env files at launch time.
