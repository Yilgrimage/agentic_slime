#!/bin/bash

set -ex

export PYTHONUNBUFFERED=1

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
LOCAL_RUNTIME_ROOT=${LOCAL_RUNTIME_ROOT:-${MLF_LOCAL_ROOT}}
NAS_MLF_ROOT=${NAS_MLF_ROOT:-${MLF_NAS_ROOT}}

USER_SLIME_ENV=${SLIME_ENV:-}
USER_WEBSHOP_ENV=${WEBSHOP_ENV:-}
if [ -f "${MLF_LOCAL_ROOT}/env.sh" ]; then
  source "${MLF_LOCAL_ROOT}/env.sh"
fi

NAS_SLIME_ENV=${NAS_SLIME_ENV:-${MLF_NAS_ROOT}/envs/slime-official}
LOCAL_SLIME_ENV=${LOCAL_SLIME_ENV:-${MLF_LOCAL_ENVS}/slime-official}
if [ -n "${USER_SLIME_ENV}" ]; then
  SLIME_ENV="${USER_SLIME_ENV}"
elif [ -z "${SLIME_ENV:-}" ]; then
  if [ -x "${LOCAL_SLIME_ENV}/bin/python" ]; then
    SLIME_ENV="${LOCAL_SLIME_ENV}"
  else
    SLIME_ENV="${NAS_SLIME_ENV}"
  fi
fi

NAS_WEBSHOP_ENV=${NAS_WEBSHOP_ENV:-${MLF_NAS_ROOT}/envs/webshop}
LOCAL_WEBSHOP_ENV=${LOCAL_WEBSHOP_ENV:-${MLF_LOCAL_ENVS}/webshop}
if [ -n "${USER_WEBSHOP_ENV}" ]; then
  WEBSHOP_ENV="${USER_WEBSHOP_ENV}"
elif [ -z "${WEBSHOP_ENV:-}" ]; then
  if [ -x "${LOCAL_WEBSHOP_ENV}/bin/python" ]; then
    WEBSHOP_ENV="${LOCAL_WEBSHOP_ENV}"
  else
    WEBSHOP_ENV="${NAS_WEBSHOP_ENV}"
  fi
fi

REPO_DIR=${REPO_DIR:-${NAS_MLF_ROOT}/code/slime}
MEGATRON_PATH=${MEGATRON_PATH:-${NAS_MLF_ROOT}/code/Megatron-LM}
SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_ENV}/bin/python}
WEBSHOP_PYTHON=${WEBSHOP_PYTHON:-${WEBSHOP_ENV}/bin/python}

NAS_WEBSHOP_SRC=${NAS_WEBSHOP_SRC:-${NAS_MLF_ROOT}/code/WebShop}
LOCAL_WEBSHOP_SRC=${LOCAL_WEBSHOP_SRC:-${LOCAL_RUNTIME_ROOT}/code/WebShop}
WEBSHOP_LIB=${WEBSHOP_LIB:-${LOCAL_WEBSHOP_SRC}}
NAS_WEBSHOP_DATA=${NAS_WEBSHOP_DATA:-${NAS_MLF_ROOT}/data/webshop}
LOCAL_WEBSHOP_DATA=${LOCAL_WEBSHOP_DATA:-${LOCAL_RUNTIME_ROOT}/data/webshop}
WEBSHOP_DATA_DIR=${WEBSHOP_DATA_DIR:-${LOCAL_WEBSHOP_DATA}}
WEBSHOP_PROMPT_NUM_TASKS=${WEBSHOP_PROMPT_NUM_TASKS:-13}
DATA_PATH=${DATA_PATH:-${WEBSHOP_DATA_DIR}/train_${WEBSHOP_PROMPT_NUM_TASKS}.jsonl}
EVAL_VALID_PATH=${EVAL_VALID_PATH:-${WEBSHOP_DATA_DIR}/valid_${WEBSHOP_PROMPT_NUM_TASKS}.jsonl}

BASE_WEBSHOP_CONFIG=${BASE_WEBSHOP_CONFIG:-${REPO_DIR}/examples/webshop/webshop_smoke_config.yaml}
WEBSHOP_CONFIG=${WEBSHOP_CONFIG:-${LOCAL_RUNTIME_ROOT}/configs/webshop_smoke_config.yaml}
BASE_WEBSHOP_EVAL_CONFIG=${BASE_WEBSHOP_EVAL_CONFIG:-${REPO_DIR}/examples/webshop/webshop_eval_config.yaml}
WEBSHOP_EVAL_CONFIG=${WEBSHOP_EVAL_CONFIG:-${LOCAL_RUNTIME_ROOT}/configs/webshop_eval_config.yaml}
WEBSHOP_SERVER_HOST=${WEBSHOP_SERVER_HOST:-127.0.0.1}
WEBSHOP_SERVER_PORT=${WEBSHOP_SERVER_PORT:-18180}
WEBSHOP_ENV_SERVER_URL=${WEBSHOP_ENV_SERVER_URL:-http://${WEBSHOP_SERVER_HOST}:${WEBSHOP_SERVER_PORT}}
WEBSHOP_SERVER_LOG=${WEBSHOP_SERVER_LOG:-${LOCAL_RUNTIME_ROOT}/logs/webshop_env_server.log}

LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-${LOCAL_RUNTIME_ROOT}/models/Qwen3-8B}
LOCAL_TORCH_DIST_DIR=${LOCAL_TORCH_DIST_DIR:-${LOCAL_RUNTIME_ROOT}/models/Qwen3-8B_torch_dist}
MODEL_DIR=${MODEL_DIR:-$([ -d "${LOCAL_MODEL_DIR}" ] && echo "${LOCAL_MODEL_DIR}" || echo "${NAS_MLF_ROOT}/models/Qwen3-8B")}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-$([ -d "${LOCAL_TORCH_DIST_DIR}" ] && echo "${LOCAL_TORCH_DIST_DIR}" || echo "${NAS_MLF_ROOT}/models/Qwen3-8B_torch_dist")}
SAVE_DIR=${SAVE_DIR:-${LOCAL_RUNTIME_ROOT}/outputs/Qwen3-8B_webshop_slime_smoke}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${LOCAL_RUNTIME_ROOT}/ray/webshop_${USER}}

export TMPDIR=${TMPDIR:-${LOCAL_RUNTIME_ROOT}/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${LOCAL_RUNTIME_ROOT}/cache/xdg}
export HF_HOME=${HF_HOME:-${LOCAL_RUNTIME_ROOT}/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${LOCAL_RUNTIME_ROOT}/cache/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${LOCAL_RUNTIME_ROOT}/cache/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${LOCAL_RUNTIME_ROOT}/cache/cuda}

mkdir -p "${LOCAL_RUNTIME_ROOT}/configs" "${LOCAL_RUNTIME_ROOT}/logs" "${LOCAL_RUNTIME_ROOT}/code" "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

if [ ! -d "${WEBSHOP_LIB}" ] && [ -d "${NAS_WEBSHOP_SRC}" ]; then
  mkdir -p "$(dirname "${WEBSHOP_LIB}")"
  cp -a "${NAS_WEBSHOP_SRC}" "${WEBSHOP_LIB}"
fi

if [ ! -d "${WEBSHOP_DATA_DIR}" ] && [ -d "${NAS_WEBSHOP_DATA}" ]; then
  mkdir -p "$(dirname "${WEBSHOP_DATA_DIR}")"
  cp -a "${NAS_WEBSHOP_DATA}" "${WEBSHOP_DATA_DIR}"
fi

mkdir -p "$(dirname "${WEBSHOP_CONFIG}")"
sed "s|data_dir: /tmp/mlf-runtime/data/webshop|data_dir: ${WEBSHOP_DATA_DIR}|" "${BASE_WEBSHOP_CONFIG}" > "${WEBSHOP_CONFIG}"
"${SLIME_PYTHON}" - <<PYH
from pathlib import Path
import os
import yaml

path = Path("${WEBSHOP_CONFIG}")
cfg = yaml.safe_load(path.read_text()) or {}
server = cfg.setdefault("env_server", {})
overrides = {
    "pool_size": os.environ.get("WEBSHOP_ENV_POOL_SIZE"),
    "acquire_timeout_s": os.environ.get("WEBSHOP_ENV_ACQUIRE_TIMEOUT_S"),
    "lease_ttl_s": os.environ.get("WEBSHOP_ENV_LEASE_TTL_S"),
    "worker_request_timeout_s": os.environ.get("WEBSHOP_ENV_WORKER_REQUEST_TIMEOUT_S"),
}
for key, value in overrides.items():
    if value is None:
        continue
    server[key] = int(value) if key == "pool_size" else float(value)
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PYH

if [ ! -f "${DATA_PATH}" ] || [ ! -f "${EVAL_VALID_PATH}" ]; then
  "${SLIME_PYTHON}" "${REPO_DIR}/examples/webshop/make_prompt_data.py" \
    --output-dir "${WEBSHOP_DATA_DIR}" \
    --num-tasks "${WEBSHOP_PROMPT_NUM_TASKS}" \
    --splits train valid
fi

sed "s|path: /tmp/mlf-runtime/data/webshop/valid_100.jsonl|path: ${EVAL_VALID_PATH}|" \
  "${BASE_WEBSHOP_EVAL_CONFIG}" > "${WEBSHOP_EVAL_CONFIG}"

WEBSHOP_SERVER_PID=""
cleanup() {
  if [ -n "${WEBSHOP_SERVER_PID}" ] && kill -0 "${WEBSHOP_SERVER_PID}" 2>/dev/null; then
    kill "${WEBSHOP_SERVER_PID}" 2>/dev/null || true
    wait "${WEBSHOP_SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export WEBSHOP_ENV_SERVER_URL WEBSHOP_LIB WEBSHOP_DATA="${WEBSHOP_DATA_DIR}"
export JAVA_HOME="${WEBSHOP_ENV}/lib/jvm"
export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
PYTHONPATH="${WEBSHOP_LIB}" "${WEBSHOP_PYTHON}" "${REPO_DIR}/examples/webshop/env_server.py" \
  --host "${WEBSHOP_SERVER_HOST}" \
  --port "${WEBSHOP_SERVER_PORT}" \
  --config "${WEBSHOP_CONFIG}" \
  > "${WEBSHOP_SERVER_LOG}" 2>&1 &
WEBSHOP_SERVER_PID=$!

WEBSHOP_SERVER_READY=0
for _ in $(seq 1 180); do
  if "${SLIME_PYTHON}" - <<PYH 2>/dev/null
import json
import urllib.request
url = "${WEBSHOP_ENV_SERVER_URL}/health"
with urllib.request.urlopen(url, timeout=1) as resp:
    data = json.loads(resp.read().decode())
    raise SystemExit(0 if data.get("ok") else 1)
PYH
  then
    WEBSHOP_SERVER_READY=1
    break
  fi
  if ! kill -0 "${WEBSHOP_SERVER_PID}" 2>/dev/null; then
    echo "WebShop env server exited early. Log follows:"
    cat "${WEBSHOP_SERVER_LOG}" || true
    exit 1
  fi
  sleep 1
done

if [ "${WEBSHOP_SERVER_READY}" -ne 1 ]; then
  echo "WebShop env server did not become healthy at ${WEBSHOP_ENV_SERVER_URL}. Log follows:"
  cat "${WEBSHOP_SERVER_LOG}" || true
  exit 1
fi

SLIME_CUDA_HOME=${SLIME_CUDA_HOME:-${SLIME_ENV}}
unset PYTHONPATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MEGATRON_PATH}:${REPO_DIR}:${SLIME_ENV}/lib/python3.12/site-packages"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME="${SLIME_CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${SLIME_ENV}/nvvm/bin:${SLIME_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CPATH="${CUDA_HOME}/include:${SLIME_ENV}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:${SLIME_ENV}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${SLIME_ENV}/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${SLIME_ENV}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${SLIME_ENV}/lib:${SLIME_ENV}/lib64:${LD_LIBRARY_PATH:-}"

NUM_GPUS=${NUM_GPUS:-4}
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}
RAY_PORT=${RAY_PORT:-8265}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

mkdir -p "${SAVE_DIR}" "${RAY_TEMP_DIR}" "${TMPDIR}"

"${SLIME_PYTHON}" -m ray.scripts.scripts stop --force 2>/dev/null || true
pkill -u "${USER}" -f "sglang.launch_server" 2>/dev/null || true
pkill -u "${USER}" -f "sglang_router" 2>/dev/null || true
sleep 3

cd "${REPO_DIR}"
source scripts/models/qwen3-8B.sh

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${TORCH_DIST_DIR}"
   --save "${SAVE_DIR}"
   --save-interval 9999
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path examples.webshop.generate.generate
   --custom-rollout-log-function-path examples.webshop.rollout_logging.log_rollout_data
   --custom-eval-rollout-log-function-path examples.webshop.rollout_logging.log_eval_rollout_data
   --custom-config-path "${WEBSHOP_CONFIG}"
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-1}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-2}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-2}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-8192}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-512}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE:-4}"
   --balance-data
)

EVAL_ARGS=()
if [ "${ENABLE_WEBSHOP_EVAL:-0}" = "1" ]; then
  EVAL_ARGS=(
     --eval-function-path slime.rollout.sglang_rollout.generate_rollout
     --eval-interval "${EVAL_INTERVAL:-1}"
     --eval-config "${WEBSHOP_EVAL_CONFIG}"
     --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN:-512}}"
     --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
     --eval-temperature "${EVAL_TEMPERATURE:-0.0}"
     --eval-top-p "${EVAL_TOP_P:-1.0}"
     --eval-top-k "${EVAL_TOP_K:--1}"
  )
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-${ACTOR_GPUS}}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE:-1}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE:-1}"
   --rollout-gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-8}"
)

MISC_ARGS=(
   --num-steps "${NUM_STEPS:-1}"
   --log-interval 1
   --seed "${SEED:-42}"
   --ray-address "127.0.0.1:${RAY_PORT}"
   --ray-temp-dir "${RAY_TEMP_DIR}"
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${ACTOR_GPUS}"
   --rollout-num-gpus "${ROLLOUT_GPUS}"
)

"${SLIME_PYTHON}" train_async.py \
   "${CKPT_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
