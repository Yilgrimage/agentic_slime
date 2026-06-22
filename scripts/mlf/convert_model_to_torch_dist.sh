#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
REPO_DIR=${REPO_DIR:-${MLF_NAS_ROOT}/code/slime}
MEGATRON_PATH=${MEGATRON_PATH:-${MLF_NAS_ROOT}/code/Megatron-LM}
SLIME_ENV=${SLIME_ENV:-${MLF_LOCAL_ENVS}/slime}

MODEL_BASENAME=${MODEL_BASENAME:-Qwen3.5-9B}
MODEL_ARGS_SCRIPT=${MODEL_ARGS_SCRIPT:-scripts/models/qwen3.5-9B.sh}
MODEL_DIR=${MODEL_DIR:-${MLF_NAS_ROOT}/models/${MODEL_BASENAME}}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${MLF_NAS_ROOT}/models/${MODEL_BASENAME}_torch_dist}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-0}
FORCE=${FORCE:-0}

usage() {
  cat <<EOF
Usage: $(basename "$0") [--force]

Environment overrides:
  MODEL_BASENAME       Default: Qwen3.5-9B
  MODEL_ARGS_SCRIPT    Default: scripts/models/qwen3.5-9B.sh
  MODEL_DIR            Default: \${MLF_NAS_ROOT}/models/\${MODEL_BASENAME}
  TORCH_DIST_DIR       Default: \${MLF_NAS_ROOT}/models/\${MODEL_BASENAME}_torch_dist
  NPROC_PER_NODE       Default: 8
  MASTER_PORT          Default: random free-ish port
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Missing HF model directory: ${MODEL_DIR}" >&2
  exit 1
fi

if [ -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ] && [ "${FORCE}" != "1" ]; then
  echo "torch_dist checkpoint already exists: ${TORCH_DIST_DIR}"
  exit 0
fi

if [ -e "${TORCH_DIST_DIR}" ] && [ "${FORCE}" = "1" ]; then
  rm -rf "${TORCH_DIST_DIR}"
fi
mkdir -p "$(dirname "${TORCH_DIST_DIR}")"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MEGATRON_PATH}:${REPO_DIR}:${SLIME_ENV}/lib/python3.12/site-packages"
export PATH="${SLIME_ENV}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-${SLIME_ENV}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${SLIME_ENV}/lib:${SLIME_ENV}/lib64:${LD_LIBRARY_PATH:-}"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1,${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1,${NO_PROXY:-}"

cd "${REPO_DIR}"
# shellcheck disable=SC1090
source "${MODEL_ARGS_SCRIPT}"

if [ "${MASTER_PORT}" = "0" ]; then
  MASTER_PORT=$((20000 + RANDOM % 20000))
fi

echo "Converting ${MODEL_DIR} -> ${TORCH_DIST_DIR} with ${NPROC_PER_NODE} process(es)"
torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}" \
  --save "${TORCH_DIST_DIR}"
