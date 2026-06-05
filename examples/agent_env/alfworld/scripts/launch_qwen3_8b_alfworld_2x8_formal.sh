#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

NODES_FILE=${NODES_FILE:-configs/nodes/agent_env_2x8.txt}
ENV_POOL_SIZE=${ENV_POOL_SIZE:-64}
EXP_PROJECT=${EXP_PROJECT:-${PROJECT_NAME:-Qwen3-8B_alfworld_grpo}}
EXP_NAME=${EXP_NAME:-${RUN_NAME:-alfworld-qwen3-8b-grpo-2x8-formal-gbs128}}
WANDB_PROJECT=${WANDB_PROJECT:-${EXP_PROJECT}}
WANDB_ENTITY=${WANDB_ENTITY:-yilgrimage-bytedance}
WANDB_GROUP=${WANDB_GROUP:-${EXP_NAME}}
TOTAL_NUM_STEPS=${TOTAL_NUM_STEPS:-100}
RESUME_FROM=${RESUME_FROM:-}
EXTRA_LAUNCH_ARGS=()

usage() {
  cat <<'EOF'
Usage: launch_qwen3_8b_alfworld_2x8_formal.sh [--dry-run]

Launch a 2-node Qwen3-8B ALFWorld GRPO run through scripts/mlf/launch_agentic_training.sh.

Environment overrides:
  NODES_FILE              Node list, default configs/nodes/agent_env_2x8.txt
  ENV_POOL_SIZE           ALFWorld env processes per node, default 64
  EXP_PROJECT             Local/W&B project, default Qwen3-8B_alfworld_grpo
  EXP_NAME                Local exp dir and W&B run name
  WANDB_PROJECT           W&B project, default EXP_PROJECT
  WANDB_ENTITY            W&B team/entity, default yilgrimage-bytedance
  WANDB_GROUP             W&B group/run-name source, default EXP_NAME
  TOTAL_NUM_STEPS         Total rollout/train steps, default 100
  RESUME_FROM             Slime checkpoint dir to continue from
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) EXTRA_LAUNCH_ARGS+=(--dry-run); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

TRAIN_CMD=$(
  cat <<EOF
SLIME_ENV=/tmp/mlf-envs/slime \
EXP_PROJECT=${EXP_PROJECT} \
EXP_NAME=${EXP_NAME} \
LOAD_DIR=${RESUME_FROM} \
USE_EXISTING_AGENT_INFRA=1 \
SUBMIT_VIA_RAY_JOB=1 \
ENABLE_WANDB=1 \
WANDB_PROJECT=${WANDB_PROJECT} \
WANDB_ENTITY=${WANDB_ENTITY} \
WANDB_GROUP=${WANDB_GROUP} \
ENABLE_ALFWORLD_EVAL=0 \
NUM_STEPS=${TOTAL_NUM_STEPS} \
ROLLOUT_BATCH_SIZE=16 \
N_SAMPLES_PER_PROMPT=8 \
GLOBAL_BATCH_SIZE=128 \
ROLLOUT_MAX_CONTEXT_LEN=10240 \
ROLLOUT_MAX_RESPONSE_LEN=512 \
MAX_TOKENS_PER_GPU=20480 \
SGLANG_SERVER_CONCURRENCY=16 \
ACTOR_NUM_NODES=1 \
ACTOR_GPUS=8 \
ROLLOUT_GPUS=8 \
TP_SIZE=8 \
CP_SIZE=1 \
SAVE_INTERVAL=9999 \
bash examples/agent_env/alfworld/scripts/run_qwen3_8b_alfworld_grpo.sh
EOF
)

cd "${REPO_DIR}"
bash scripts/mlf/launch_agentic_training.sh \
  --env alfworld \
  --nodes "${NODES_FILE}" \
  --env-pool-size "${ENV_POOL_SIZE}" \
  --train-cmd "${TRAIN_CMD}" \
  ${EXTRA_LAUNCH_ARGS+"${EXTRA_LAUNCH_ARGS[@]}"}
