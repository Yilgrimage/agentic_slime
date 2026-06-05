#!/bin/bash
set -euo pipefail

export ENV_NAME=appworld
export CUSTOM_GENERATE_FUNCTION_PATH=examples.agent_env.appworld.rollout.generate
export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}/configs/appworld_launch.yaml}
export ENV_SERVER_URL_VAR=APPWORLD_ENV_SERVER_URL
export APPWORLD_ROOT=${APPWORLD_ROOT:-${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}/data/appworld}
export HOME=${APPWORLD_ROOT}

exec bash "${REPO_DIR:-/mnt/bn/jixf-nas-lq/mlf/code/slime}/examples/agent_env/scripts/run_qwen3_8b_agent_env_grpo.sh"
