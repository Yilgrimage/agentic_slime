#!/bin/bash
set -euo pipefail

export ENV_NAME=alfworld
export CUSTOM_GENERATE_FUNCTION_PATH=examples.agent_env.alfworld.rollout.generate
export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${ALFWORLD_CONFIG:-${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}/configs/alfworld_launch.yaml}}
export ENV_SERVER_URL_VAR=ALFWORLD_ENV_SERVER_URL
export DATA_DIR=${DATA_DIR:-${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}/data/alfworld}
export PROMPT_DATA_SCRIPT=${PROMPT_DATA_SCRIPT:-${REPO_DIR:-/mnt/bn/jixf-nas-lq/mlf/code/slime}/examples/agent_env/alfworld/prompt_data.py}

exec bash "${REPO_DIR:-/mnt/bn/jixf-nas-lq/mlf/code/slime}/examples/agent_env/scripts/run_qwen3_8b_agent_env_grpo.sh"
