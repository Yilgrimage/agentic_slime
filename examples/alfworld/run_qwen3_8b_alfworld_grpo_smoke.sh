#!/bin/bash

set -ex

export PYTHONUNBUFFERED=1

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
USER_SLIME_ENV=${SLIME_ENV:-}
USER_ALFWORLD_CONFIG=${ALFWORLD_CONFIG:-}
USER_ALFWORLD_SERVER_PORT=${ALFWORLD_SERVER_PORT:-}
USER_ALFWORLD_SERVER_LOG=${ALFWORLD_SERVER_LOG:-}
USER_SAVE_DIR=${SAVE_DIR:-}
USER_RAY_TEMP_DIR=${RAY_TEMP_DIR:-}
USER_RAY_PORT=${RAY_PORT:-}
USER_ALFWORLD_EVAL_CONFIG=${ALFWORLD_EVAL_CONFIG:-}
if [ -f "${MLF_LOCAL_ROOT}/env.sh" ]; then
  source "${MLF_LOCAL_ROOT}/env.sh"
fi
if [ -n "${USER_SLIME_ENV}" ]; then
  SLIME_ENV="${USER_SLIME_ENV}"
fi
if [ -n "${USER_ALFWORLD_CONFIG}" ]; then
  ALFWORLD_CONFIG="${USER_ALFWORLD_CONFIG}"
fi
if [ -n "${USER_ALFWORLD_SERVER_PORT}" ]; then
  ALFWORLD_SERVER_PORT="${USER_ALFWORLD_SERVER_PORT}"
fi
if [ -n "${USER_ALFWORLD_SERVER_LOG}" ]; then
  ALFWORLD_SERVER_LOG="${USER_ALFWORLD_SERVER_LOG}"
fi
if [ -n "${USER_SAVE_DIR}" ]; then
  SAVE_DIR="${USER_SAVE_DIR}"
fi
if [ -n "${USER_RAY_TEMP_DIR}" ]; then
  RAY_TEMP_DIR="${USER_RAY_TEMP_DIR}"
fi
if [ -n "${USER_RAY_PORT}" ]; then
  RAY_PORT="${USER_RAY_PORT}"
fi
NAS_SLIME_ENV=${NAS_SLIME_ENV:-${MLF_NAS_ROOT}/envs/slime-official}
LOCAL_SLIME_ENV=${LOCAL_SLIME_ENV:-${MLF_LOCAL_ENVS}/slime-official}
if [ -z "${SLIME_ENV:-}" ]; then
  if [ -x "${LOCAL_SLIME_ENV}/bin/python" ]; then
    SLIME_ENV="${LOCAL_SLIME_ENV}"
  else
    SLIME_ENV="${NAS_SLIME_ENV}"
  fi
fi
SLIME_CUDA_HOME=${SLIME_CUDA_HOME:-${SLIME_ENV}}
unset PYTHONPATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
export PYTHONNOUSERSITE=1
export CUDA_HOME="${SLIME_CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${SLIME_ENV}/nvvm/bin:${SLIME_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CPATH="${CUDA_HOME}/include:${SLIME_ENV}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:${SLIME_ENV}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${SLIME_ENV}/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${SLIME_ENV}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${SLIME_ENV}/lib:${SLIME_ENV}/lib64:${LD_LIBRARY_PATH:-}"

NAS_MLF_ROOT=${NAS_MLF_ROOT:-${MLF_NAS_ROOT}}
LOCAL_RUNTIME_ROOT=${LOCAL_RUNTIME_ROOT:-${MLF_LOCAL_ROOT}}

REPO_DIR=${REPO_DIR:-${NAS_MLF_ROOT}/code/slime}
MEGATRON_PATH=${MEGATRON_PATH:-${NAS_MLF_ROOT}/code/Megatron-LM}
PYTHON_BIN=${PYTHON_BIN:-${SLIME_ENV}/bin/python}

NAS_ALFWORLD_DATA=${NAS_ALFWORLD_DATA:-${NAS_MLF_ROOT}/data/alfworld}
LOCAL_ALFWORLD_DATA=${LOCAL_ALFWORLD_DATA:-${LOCAL_RUNTIME_ROOT}/data/alfworld}
ALFWORLD_DATA_DIR=${ALFWORLD_DATA_DIR:-${LOCAL_ALFWORLD_DATA}}
NAS_ALFWORLD_LIB=${NAS_ALFWORLD_LIB:-${NAS_MLF_ROOT}/pythonlibs/alfworld_text}
LOCAL_ALFWORLD_LIB=${LOCAL_ALFWORLD_LIB:-${LOCAL_RUNTIME_ROOT}/pythonlibs/alfworld_text}
ALFWORLD_LIB=${ALFWORLD_LIB:-${LOCAL_ALFWORLD_LIB}}
DATA_PATH=${DATA_PATH:-${ALFWORLD_DATA_DIR}/train_100.jsonl}
EVAL_VALID_SEEN_PATH=${EVAL_VALID_SEEN_PATH:-${ALFWORLD_DATA_DIR}/valid_seen_100.jsonl}
EVAL_VALID_UNSEEN_PATH=${EVAL_VALID_UNSEEN_PATH:-${ALFWORLD_DATA_DIR}/valid_unseen_100.jsonl}
BASE_ALFWORLD_CONFIG=${BASE_ALFWORLD_CONFIG:-${REPO_DIR}/examples/alfworld/alfworld_smoke_config.yaml}
ALFWORLD_CONFIG=${ALFWORLD_CONFIG:-${LOCAL_RUNTIME_ROOT}/configs/alfworld_smoke_config.yaml}
BASE_ALFWORLD_EVAL_CONFIG=${BASE_ALFWORLD_EVAL_CONFIG:-${REPO_DIR}/examples/alfworld/alfworld_eval_config.yaml}
ALFWORLD_EVAL_CONFIG=${ALFWORLD_EVAL_CONFIG:-${LOCAL_RUNTIME_ROOT}/configs/alfworld_eval_config.yaml}
ALFWORLD_SERVER_HOST=${ALFWORLD_SERVER_HOST:-127.0.0.1}
ALFWORLD_SERVER_PORT=${ALFWORLD_SERVER_PORT:-18080}
ALFWORLD_ENV_SERVER_URL=${ALFWORLD_ENV_SERVER_URL:-http://${ALFWORLD_SERVER_HOST}:${ALFWORLD_SERVER_PORT}}
ALFWORLD_SERVER_LOG=${ALFWORLD_SERVER_LOG:-${LOCAL_RUNTIME_ROOT}/logs/alfworld_env_server.log}

LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-${LOCAL_RUNTIME_ROOT}/models/Qwen3-8B}
LOCAL_TORCH_DIST_DIR=${LOCAL_TORCH_DIST_DIR:-${LOCAL_RUNTIME_ROOT}/models/Qwen3-8B_torch_dist}
if [ -d "${LOCAL_MODEL_DIR}" ]; then
  DEFAULT_MODEL_DIR="${LOCAL_MODEL_DIR}"
else
  DEFAULT_MODEL_DIR="${NAS_MLF_ROOT}/models/Qwen3-8B"
fi
if [ -d "${LOCAL_TORCH_DIST_DIR}" ]; then
  DEFAULT_TORCH_DIST_DIR="${LOCAL_TORCH_DIST_DIR}"
else
  DEFAULT_TORCH_DIST_DIR="${NAS_MLF_ROOT}/models/Qwen3-8B_torch_dist"
fi
MODEL_DIR=${MODEL_DIR:-${DEFAULT_MODEL_DIR}}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${DEFAULT_TORCH_DIST_DIR}}
SAVE_DIR=${SAVE_DIR:-${LOCAL_RUNTIME_ROOT}/outputs/Qwen3-8B_alfworld_slime_smoke}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${LOCAL_RUNTIME_ROOT}/ray/alfworld_${USER}}

export TMPDIR=${TMPDIR:-${LOCAL_RUNTIME_ROOT}/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${LOCAL_RUNTIME_ROOT}/cache/xdg}
export HF_HOME=${HF_HOME:-${LOCAL_RUNTIME_ROOT}/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${LOCAL_RUNTIME_ROOT}/cache/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${LOCAL_RUNTIME_ROOT}/cache/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${LOCAL_RUNTIME_ROOT}/cache/cuda}

mkdir -p "${LOCAL_RUNTIME_ROOT}/configs" "${LOCAL_RUNTIME_ROOT}/logs" "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

if [ ! -d "${ALFWORLD_DATA_DIR}" ] && [ -d "${NAS_ALFWORLD_DATA}" ]; then
  mkdir -p "$(dirname "${ALFWORLD_DATA_DIR}")"
  cp -a "${NAS_ALFWORLD_DATA}" "${ALFWORLD_DATA_DIR}"
fi

if [ ! -d "${ALFWORLD_LIB}" ] && [ -d "${NAS_ALFWORLD_LIB}" ]; then
  mkdir -p "$(dirname "${ALFWORLD_LIB}")"
  cp -a "${NAS_ALFWORLD_LIB}" "${ALFWORLD_LIB}"
fi

if [ "${ALFWORLD_CONFIG}" = "${LOCAL_RUNTIME_ROOT}/configs/alfworld_smoke_config.yaml" ]; then
  sed "s|^alfworld_data_dir:.*|alfworld_data_dir: ${ALFWORLD_DATA_DIR}|" "${BASE_ALFWORLD_CONFIG}" > "${ALFWORLD_CONFIG}"
fi

if [ ! -f "${DATA_PATH}" ] || [ ! -f "${EVAL_VALID_SEEN_PATH}" ] || [ ! -f "${EVAL_VALID_UNSEEN_PATH}" ]; then
  "${PYTHON_BIN}" "${REPO_DIR}/examples/alfworld/make_prompt_data.py" \
    --output-dir "${ALFWORLD_DATA_DIR}" \
    --num-tasks "${ALFWORLD_PROMPT_NUM_TASKS:-100}" \
    --splits train valid_seen valid_unseen
fi

if [ -n "${USER_ALFWORLD_EVAL_CONFIG}" ]; then
  ALFWORLD_EVAL_CONFIG="${USER_ALFWORLD_EVAL_CONFIG}"
elif [ "${ALFWORLD_EVAL_CONFIG}" = "${LOCAL_RUNTIME_ROOT}/configs/alfworld_eval_config.yaml" ]; then
  sed \
    -e "s|path: /tmp/mlf-runtime/alfworld/data/valid_seen_100.jsonl|path: ${EVAL_VALID_SEEN_PATH}|" \
    -e "s|path: /tmp/mlf-runtime/alfworld/data/valid_unseen_100.jsonl|path: ${EVAL_VALID_UNSEEN_PATH}|" \
    "${BASE_ALFWORLD_EVAL_CONFIG}" > "${ALFWORLD_EVAL_CONFIG}"
fi

ALFWORLD_SERVER_PID=""
cleanup() {
  if [ -n "${ALFWORLD_SERVER_PID}" ] && kill -0 "${ALFWORLD_SERVER_PID}" 2>/dev/null; then
    kill "${ALFWORLD_SERVER_PID}" 2>/dev/null || true
    wait "${ALFWORLD_SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export ALFWORLD_ENV_SERVER_URL
ALFWORLD_SERVER_PYTHONPATH="${ALFWORLD_LIB}:${SLIME_ENV}/lib/python3.12/site-packages:${REPO_DIR}"
PYTHONPATH="${ALFWORLD_SERVER_PYTHONPATH}" "${PYTHON_BIN}" "${REPO_DIR}/examples/alfworld/env_server.py" \
  --host "${ALFWORLD_SERVER_HOST}" \
  --port "${ALFWORLD_SERVER_PORT}" \
  --config "${ALFWORLD_CONFIG}" \
  > "${ALFWORLD_SERVER_LOG}" 2>&1 &
ALFWORLD_SERVER_PID=$!

for _ in $(seq 1 120); do
  if "${PYTHON_BIN}" - <<PYH
import json
import urllib.request
url = "${ALFWORLD_ENV_SERVER_URL}/health"
with urllib.request.urlopen(url, timeout=1) as resp:
    data = json.loads(resp.read().decode())
    raise SystemExit(0 if data.get("ok") else 1)
PYH
  then
    break
  fi
  if ! kill -0 "${ALFWORLD_SERVER_PID}" 2>/dev/null; then
    echo "ALFWorld env server exited early. Log follows:"
    cat "${ALFWORLD_SERVER_LOG}" || true
    exit 1
  fi
  sleep 1
done

NUM_GPUS=${NUM_GPUS:-4}
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}
RAY_PORT=${RAY_PORT:-8265}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

mkdir -p "${SAVE_DIR}" "${RAY_TEMP_DIR}" "${TMPDIR}"

"${PYTHON_BIN}" -m ray.scripts.scripts stop --force 2>/dev/null || true
pkill -u "${USER}" -f "sglang.launch_server" 2>/dev/null || true
pkill -u "${USER}" -f "sglang_router" 2>/dev/null || true
sleep 3

cd "${REPO_DIR}"
source scripts/models/qwen3-8B.sh

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "${NVLINK_COUNT}" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${TORCH_DIST_DIR}"
   --save "${SAVE_DIR}"
   --save-interval 9999
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path examples.alfworld.generate_with_alfworld.generate
   --custom-config-path "${ALFWORLD_CONFIG}"
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-1}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-2}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-2}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-512}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE:-4}"
   --balance-data
)

EVAL_ARGS=()
if [ "${ENABLE_ALFWORLD_EVAL:-1}" = "1" ]; then
  EVAL_ARGS=(
     --eval-interval "${EVAL_INTERVAL:-5}"
     --eval-config "${ALFWORLD_EVAL_CONFIG}"
     --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN:-384}}"
     --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
     --eval-temperature "${EVAL_TEMPERATURE:-0.0}"
     --eval-top-p "${EVAL_TOP_P:-1.0}"
     --eval-top-k "${EVAL_TOP_K:--1}"
  )
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${ACTOR_GPUS}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
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
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.45}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-4}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export PYTHONPATH="${MEGATRON_PATH}:${REPO_DIR}:${SLIME_ENV}/lib/python3.12/site-packages"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export RAY_ADDRESS=127.0.0.1:6379
"${PYTHON_BIN}" -m ray.scripts.scripts start --head \
   --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port="${RAY_PORT}" \
   --temp-dir "${RAY_TEMP_DIR}"


"${PYTHON_BIN}" train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
