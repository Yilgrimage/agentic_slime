#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

ENV_NAME=${ENV_NAME:?Set ENV_NAME to tau2 or appworld}
CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:?Set CUSTOM_GENERATE_FUNCTION_PATH}
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:?Set CUSTOM_CONFIG_PATH}
ENV_SERVER_URL_VAR=${ENV_SERVER_URL_VAR:?Set ENV_SERVER_URL_VAR}

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
WANDB_SECRET_FILE=${WANDB_SECRET_FILE:-${MLF_NAS_ROOT}/secrets/wandb.env}
REPO_DIR=${REPO_DIR:-${MLF_NAS_ROOT}/code/slime}
MEGATRON_PATH=${MEGATRON_PATH:-${MLF_NAS_ROOT}/code/Megatron-LM}
SLIME_ENV=${SLIME_ENV:-${MLF_LOCAL_ENVS}/slime}
SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_ENV}/bin/python}
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT:-train.py}

if [ -f "${MLF_LOCAL_ROOT}/env.sh" ]; then
  source "${MLF_LOCAL_ROOT}/env.sh"
fi
if [ -f "${WANDB_SECRET_FILE}" ]; then
  case "$-" in
    *x*) _restore_xtrace=1; set +x ;;
    *) _restore_xtrace=0 ;;
  esac
  set -a
  # shellcheck disable=SC1090
  source "${WANDB_SECRET_FILE}"
  set +a
  if [ "${_restore_xtrace}" = "1" ]; then
    set -x
  fi
  unset _restore_xtrace
fi

ENV_SERVER_URL=${ENV_SERVER_URL:-${AGENT_ENV_ROUTER_URL:-}}
if [ -z "${ENV_SERVER_URL}" ]; then
  ENV_SERVER_URL=${!ENV_SERVER_URL_VAR:-}
fi
if [ -z "${ENV_SERVER_URL}" ]; then
  echo "Missing env server URL. Set AGENT_ENV_ROUTER_URL or ${ENV_SERVER_URL_VAR}." >&2
  exit 1
fi
export "${ENV_SERVER_URL_VAR}=${ENV_SERVER_URL}"

LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-${MLF_LOCAL_ROOT}/models/Qwen3-8B}
LOCAL_TORCH_DIST_DIR=${LOCAL_TORCH_DIST_DIR:-${MLF_LOCAL_ROOT}/models/Qwen3-8B_torch_dist}
MODEL_DIR=${MODEL_DIR:-$([ -d "${LOCAL_MODEL_DIR}" ] && echo "${LOCAL_MODEL_DIR}" || echo "${MLF_NAS_ROOT}/models/Qwen3-8B")}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-$([ -d "${LOCAL_TORCH_DIST_DIR}" ] && echo "${LOCAL_TORCH_DIST_DIR}" || echo "${MLF_NAS_ROOT}/models/Qwen3-8B_torch_dist")}

EXP_PROJECT=${EXP_PROJECT:-${PROJECT_NAME:-Qwen3-8B_${ENV_NAME}_grpo}}
EXP_NAME=${EXP_NAME:-${RUN_NAME:-qwen3-8b-${ENV_NAME}-grpo}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${MLF_LOCAL_ROOT}/outputs}
SAVE_DIR=${SAVE_DIR:-${OUTPUT_ROOT}/${EXP_PROJECT}/${EXP_NAME}}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${MLF_LOCAL_ROOT}/ray/${ENV_NAME}_${USER}}
DATA_DIR=${DATA_DIR:-${MLF_LOCAL_ROOT}/data/${ENV_NAME}}
PROMPT_NUM_TASKS=${PROMPT_NUM_TASKS:-}
DATA_PATH=${DATA_PATH:-}

export TMPDIR=${TMPDIR:-${MLF_LOCAL_ROOT}/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${MLF_LOCAL_ROOT}/cache/xdg}
export HF_HOME=${HF_HOME:-${MLF_LOCAL_ROOT}/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${MLF_LOCAL_ROOT}/cache/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${MLF_LOCAL_ROOT}/cache/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${MLF_LOCAL_ROOT}/cache/cuda}

mkdir -p "${MLF_LOCAL_ROOT}/configs" "${MLF_LOCAL_ROOT}/logs" "${DATA_DIR}" "${SAVE_DIR}" "${RAY_TEMP_DIR}" "${TMPDIR}" \
  "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

TRAIN_CUSTOM_CONFIG_PATH=${TRAIN_CUSTOM_CONFIG_PATH:-${MLF_LOCAL_ROOT}/configs/${ENV_NAME}_train_runtime.yaml}
"${SLIME_PYTHON}" - <<PYH
from pathlib import Path
import yaml

base = Path("${CUSTOM_CONFIG_PATH}")
target = Path("${TRAIN_CUSTOM_CONFIG_PATH}")
cfg = yaml.safe_load(base.read_text()) or {}
cfg["env_server_url"] = "${ENV_SERVER_URL}".rstrip("/")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(cfg, sort_keys=False))
PYH

if [ -z "${PROMPT_NUM_TASKS}" ]; then
  PROMPT_NUM_TASKS=$("${SLIME_PYTHON}" - <<PYH
import json
import urllib.request
base_url = "${ENV_SERVER_URL}".rstrip("/")
status = json.loads(urllib.request.urlopen(f"{base_url}/status", timeout=30).read().decode())
workers = status.get("workers") or []
worker_tasks = [int(w["num_tasks"]) for w in workers if w.get("ok") and w.get("num_tasks")]
if worker_tasks:
    print(min(worker_tasks))
else:
    raise SystemExit(f"{base_url}/status did not report worker num_tasks: {status}")
PYH
)
fi

DATA_PATH=${DATA_PATH:-${DATA_DIR}/train_${PROMPT_NUM_TASKS}.jsonl}
if [ ! -f "${DATA_PATH}" ]; then
  "${SLIME_PYTHON}" "${REPO_DIR}/examples/agent_env/scripts/prompt_data.py" \
    --output "${DATA_PATH}" \
    --split train \
    --num-tasks "${PROMPT_NUM_TASKS}"
fi

SLIME_CUDA_HOME=${SLIME_CUDA_HOME:-${SLIME_ENV}}
unset PYTHONPATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MEGATRON_PATH}:${REPO_DIR}:${SLIME_ENV}/lib/python3.12/site-packages"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379}
export CUDA_HOME="${SLIME_CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${SLIME_ENV}/nvvm/bin:${SLIME_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CPATH="${CUDA_HOME}/include:${SLIME_ENV}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:${SLIME_ENV}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${SLIME_ENV}/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${SLIME_ENV}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${SLIME_ENV}/lib:${SLIME_ENV}/lib64:${LD_LIBRARY_PATH:-}"

cd "${REPO_DIR}"
source scripts/models/qwen3-8B.sh

ROLLOUT_FUNCTION_PATH=${ROLLOUT_FUNCTION_PATH:-}
if [ -z "${ROLLOUT_FUNCTION_PATH}" ]; then
  if [ "${TRAIN_ENTRYPOINT}" = "train_async.py" ]; then
    ROLLOUT_FUNCTION_PATH=slime.rollout.fully_async_rollout.generate_rollout_fully_async
  else
    ROLLOUT_FUNCTION_PATH=slime.rollout.sglang_rollout.generate_rollout
  fi
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-9999}"
)
USE_KL_LOSS=${USE_KL_LOSS:-0}
LOAD_DIR=${LOAD_DIR:-${TORCH_DIST_DIR}}
if [ "${USE_KL_LOSS}" = "1" ] || [ "${KL_LOSS_COEF:-0.00}" != "0.00" ]; then
  CKPT_ARGS+=(--ref-load "${REF_LOAD_DIR:-${LOAD_DIR}}")
fi
CKPT_ARGS+=(--load "${LOAD_DIR}")

ROLLOUT_ARGS=(
   --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
   --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}"
   --custom-rollout-log-function-path "${CUSTOM_GENERATE_FUNCTION_PATH%.*}.log_rollout_data"
   --custom-eval-rollout-log-function-path "${CUSTOM_GENERATE_FUNCTION_PATH%.*}.log_eval_rollout_data"
   --custom-config-path "${TRAIN_CUSTOM_CONFIG_PATH}"
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-${NUM_STEPS:-100}}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-10240}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-512}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
   --balance-data
)

PERF_ARGS=(
   --colocate
   --tensor-model-parallel-size "${TP_SIZE:-4}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE:-1}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-12288}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
if [ "${USE_KL_LOSS}" = "1" ] || [ "${KL_LOSS_COEF:-0.00}" != "0.00" ]; then
  GRPO_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF:-0.00}" --kl-loss-type "${KL_LOSS_TYPE:-low_var_kl}")
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE:-1}"
   --rollout-gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-8}"
)

MISC_ARGS=(
   --num-steps "${NUM_STEPS:-100}"
   --log-interval 1
   --seed "${SEED:-42}"
   --ray-temp-dir "${RAY_TEMP_DIR}"
   --actor-num-nodes "${ACTOR_NUM_NODES:-2}"
   --actor-num-gpus-per-node "${ACTOR_GPUS:-4}"
   --rollout-num-gpus "${ROLLOUT_GPUS:-8}"
   --num-gpus-per-node "${NUM_GPUS:-4}"
)

WANDB_PROJECT=${WANDB_PROJECT:-${EXP_PROJECT}}
WANDB_GROUP=${WANDB_GROUP:-${EXP_NAME}}
WANDB_ARGS=()
if [ "${ENABLE_WANDB:-0}" = "1" ] || [ "${USE_WANDB:-0}" = "1" ]; then
  WANDB_ARGS=(
     --use-wandb
     --wandb-project "${WANDB_PROJECT}"
     --wandb-group "${WANDB_GROUP}"
     --wandb-dir "${WANDB_DIR:-${MLF_LOCAL_ROOT}/wandb/${EXP_PROJECT}/${EXP_NAME}}"
     --disable-wandb-random-suffix
  )
  if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
  fi
  if [ -n "${WANDB_BASE_URL:-}" ]; then
    WANDB_ARGS+=(--wandb-host "${WANDB_BASE_URL}")
  fi
  if [ -n "${WANDB_ENTITY:-${WANDB_TEAM:-}}" ]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY:-${WANDB_TEAM:-}}")
  fi
fi

if [ -z "${TRAIN_ENV_VARS_JSON:-}" ]; then
  TRAIN_ENV_VARS_JSON=$("${SLIME_PYTHON}" - <<PYH
import json, os
keys = [
    "PYTHONPATH", "PYTHONNOUSERSITE", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_HOME",
    "PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
    "LD_LIBRARY_PATH", "RAY_ADDRESS", "${ENV_SERVER_URL_VAR}", "TAU2_DATA_DIR", "APPWORLD_ROOT",
    "WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_ENTITY",
]
print(json.dumps({key: os.environ[key] for key in keys if key in os.environ and os.environ[key] != ""}))
PYH
)
fi

"${SLIME_PYTHON}" "${REPO_DIR}/${TRAIN_ENTRYPOINT}" \
   "${CKPT_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   --train-env-vars "${TRAIN_ENV_VARS_JSON}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}"
