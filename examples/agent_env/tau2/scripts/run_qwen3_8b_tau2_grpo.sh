#!/bin/bash
set -euo pipefail

export ENV_NAME=tau2
export CUSTOM_GENERATE_FUNCTION_PATH=examples.agent_env.tau2.rollout.generate
export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}/configs/tau2_launch.yaml}
export ENV_SERVER_URL_VAR=TAU2_ENV_SERVER_URL
export TAU2_DATA_DIR=${TAU2_DATA_DIR:-${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}/data/tau2/data}

exec bash "${REPO_DIR:-/mnt/bn/jixf-nas-lq/mlf/code/slime}/examples/agent_env/scripts/run_qwen3_8b_agent_env_grpo.sh"
