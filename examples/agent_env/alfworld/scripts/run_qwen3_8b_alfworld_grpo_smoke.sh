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
ALFWORLD_TRAIN_NUM_TASKS=${ALFWORLD_TRAIN_NUM_TASKS:-${ALFWORLD_PROMPT_NUM_TASKS:-100}}
ALFWORLD_EVAL_NUM_TASKS=${ALFWORLD_EVAL_NUM_TASKS:-16}
NAS_ALFWORLD_LIB=${NAS_ALFWORLD_LIB:-${NAS_MLF_ROOT}/pythonlibs/alfworld_text}
LOCAL_ALFWORLD_LIB=${LOCAL_ALFWORLD_LIB:-${LOCAL_RUNTIME_ROOT}/pythonlibs/alfworld_text}
ALFWORLD_LIB=${ALFWORLD_LIB:-${LOCAL_ALFWORLD_LIB}}
DATA_PATH=${DATA_PATH:-${ALFWORLD_DATA_DIR}/train_${ALFWORLD_TRAIN_NUM_TASKS}.jsonl}
EVAL_VALID_SEEN_PATH=${EVAL_VALID_SEEN_PATH:-${ALFWORLD_DATA_DIR}/valid_seen_${ALFWORLD_EVAL_NUM_TASKS}.jsonl}
EVAL_VALID_UNSEEN_PATH=${EVAL_VALID_UNSEEN_PATH:-${ALFWORLD_DATA_DIR}/valid_unseen_${ALFWORLD_EVAL_NUM_TASKS}.jsonl}
BASE_ALFWORLD_CONFIG=${BASE_ALFWORLD_CONFIG:-${REPO_DIR}/examples/agent_env/alfworld/smoke_config.yaml}
ALFWORLD_CONFIG=${ALFWORLD_CONFIG:-${LOCAL_RUNTIME_ROOT}/configs/alfworld_smoke_config.yaml}
BASE_ALFWORLD_EVAL_CONFIG=${BASE_ALFWORLD_EVAL_CONFIG:-${REPO_DIR}/examples/agent_env/alfworld/eval_config.yaml}
ALFWORLD_EVAL_CONFIG=${ALFWORLD_EVAL_CONFIG:-${LOCAL_RUNTIME_ROOT}/configs/alfworld_eval_config.yaml}
ALFWORLD_SERVER_HOST=${ALFWORLD_SERVER_HOST:-127.0.0.1}
ALFWORLD_SERVER_PORT=${ALFWORLD_SERVER_PORT:-18080}
USE_EXISTING_AGENT_INFRA=${USE_EXISTING_AGENT_INFRA:-0}
if [ "${USE_EXISTING_AGENT_INFRA}" = "1" ]; then
  ALFWORLD_ENV_SERVER_URL=${ALFWORLD_ENV_SERVER_URL:-http://127.0.0.1:19000}
else
  ALFWORLD_ENV_SERVER_URL=${ALFWORLD_ENV_SERVER_URL:-http://${ALFWORLD_SERVER_HOST}:${ALFWORLD_SERVER_PORT}}
fi
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

if [ -z "${USER_ALFWORLD_CONFIG}" ]; then
  mkdir -p "$(dirname "${ALFWORLD_CONFIG}")"
  sed "s|^alfworld_data_dir:.*|alfworld_data_dir: ${ALFWORLD_DATA_DIR}|" "${BASE_ALFWORLD_CONFIG}" > "${ALFWORLD_CONFIG}"
fi

if [ ! -f "${DATA_PATH}" ]; then
  "${PYTHON_BIN}" "${REPO_DIR}/examples/agent_env/alfworld/prompt_data.py" \
    --output-dir "${ALFWORLD_DATA_DIR}" \
    --num-tasks "${ALFWORLD_TRAIN_NUM_TASKS}" \
    --splits train
fi

if [ ! -f "${EVAL_VALID_SEEN_PATH}" ] || [ ! -f "${EVAL_VALID_UNSEEN_PATH}" ]; then
  "${PYTHON_BIN}" "${REPO_DIR}/examples/agent_env/alfworld/prompt_data.py" \
    --output-dir "${ALFWORLD_DATA_DIR}" \
    --num-tasks "${ALFWORLD_EVAL_NUM_TASKS}" \
    --splits valid_seen valid_unseen
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
if [ "${USE_EXISTING_AGENT_INFRA}" != "1" ]; then
  PYTHONPATH="${ALFWORLD_SERVER_PYTHONPATH}" "${PYTHON_BIN}" "${REPO_DIR}/examples/agent_env/alfworld/server.py" \
    --host "${ALFWORLD_SERVER_HOST}" \
    --port "${ALFWORLD_SERVER_PORT}" \
    --config "${ALFWORLD_CONFIG}" \
    > "${ALFWORLD_SERVER_LOG}" 2>&1 &
  ALFWORLD_SERVER_PID=$!
fi

ALFWORLD_SERVER_READY=0
for _ in $(seq 1 120); do
  if "${PYTHON_BIN}" - <<PYH 2>/dev/null
import json
import urllib.request
url = "${ALFWORLD_ENV_SERVER_URL}/health"
with urllib.request.urlopen(url, timeout=1) as resp:
    data = json.loads(resp.read().decode())
    raise SystemExit(0 if data.get("ok") else 1)
PYH
  then
    ALFWORLD_SERVER_READY=1
    break
  fi
  if [ -n "${ALFWORLD_SERVER_PID}" ] && ! kill -0 "${ALFWORLD_SERVER_PID}" 2>/dev/null; then
    echo "ALFWorld env server exited early. Log follows:"
    cat "${ALFWORLD_SERVER_LOG}" || true
    exit 1
  fi
  sleep 1
done

if [ "${ALFWORLD_SERVER_READY}" -ne 1 ]; then
  echo "ALFWorld env server did not become healthy at ${ALFWORLD_ENV_SERVER_URL}. Log follows:"
  cat "${ALFWORLD_SERVER_LOG}" || true
  exit 1
fi

NUM_GPUS=${NUM_GPUS:-4}
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}
RAY_PORT=${RAY_PORT:-8265}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

mkdir -p "${SAVE_DIR}" "${RAY_TEMP_DIR}" "${TMPDIR}"

if [ "${USE_EXISTING_AGENT_INFRA}" != "1" ]; then
  "${PYTHON_BIN}" -m ray.scripts.scripts stop --force 2>/dev/null || true
  pkill -u "${USER}" -f "sglang.launch_server" 2>/dev/null || true
  pkill -u "${USER}" -f "sglang_router" 2>/dev/null || true
  sleep 3
fi

cd "${REPO_DIR}"
source scripts/models/qwen3-8B.sh

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "${NVLINK_COUNT}" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${TORCH_DIST_DIR}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-9999}"
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path examples.agent_env.alfworld.rollout.generate
   --custom-rollout-log-function-path examples.agent_env.alfworld.rollout.log_rollout_data
   --custom-eval-rollout-log-function-path examples.agent_env.alfworld.rollout.log_eval_rollout_data
   --custom-config-path "${ALFWORLD_CONFIG}"
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
if [ "${ENABLE_ALFWORLD_EVAL:-1}" = "1" ]; then
  EVAL_ARGS=(
     --eval-function-path "${ALFWORLD_EVAL_FUNCTION_PATH:-slime.rollout.sglang_rollout.generate_rollout}"
     --eval-interval "${EVAL_INTERVAL:-1}"
     --eval-config "${ALFWORLD_EVAL_CONFIG}"
     --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN:-384}}"
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
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.45}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-4}"
)

MISC_ARGS=(
   --num-steps "${NUM_STEPS:-1}"
   --log-interval 1
   --seed "${SEED:-42}"
   --ray-temp-dir "${RAY_TEMP_DIR}"
   --actor-num-nodes "${ACTOR_NUM_NODES:-1}"
   --actor-num-gpus-per-node "${ACTOR_GPUS}"
   --rollout-num-gpus "${ROLLOUT_GPUS}"
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
if [ "${USE_EXISTING_AGENT_INFRA}" != "1" ]; then
  "${PYTHON_BIN}" -m ray.scripts.scripts start --head \
     --node-ip-address "${MASTER_ADDR}" \
     --num-gpus "${NUM_GPUS}" \
     --disable-usage-stats \
     --dashboard-host=0.0.0.0 \
     --dashboard-port="${RAY_PORT}" \
     --temp-dir "${RAY_TEMP_DIR}"
fi

TRAIN_ENTRY=(
   "${PYTHON_BIN}" "${REPO_DIR}/train_async.py"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
)

if [ "${SUBMIT_VIA_RAY_JOB:-0}" = "1" ]; then
  RAY_JOB_ADDRESS=${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}
  RUNTIME_ENV_JSON=$("${PYTHON_BIN}" - <<PYH
import json
env = {
    "PYTHONPATH": "${PYTHONPATH}",
    "PYTHONNOUSERSITE": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUDA_HOME": "${CUDA_HOME}",
    "PATH": "${PATH}",
    "CPATH": "${CPATH}",
    "C_INCLUDE_PATH": "${C_INCLUDE_PATH}",
    "CPLUS_INCLUDE_PATH": "${CPLUS_INCLUDE_PATH}",
    "LIBRARY_PATH": "${LIBRARY_PATH}",
    "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}",
    "ALFWORLD_ENV_SERVER_URL": "${ALFWORLD_ENV_SERVER_URL}",
    "ALFWORLD_LIB": "${ALFWORLD_LIB}",
    "ALFWORLD_DATA": "${ALFWORLD_DATA_DIR}",
    "no_proxy": "127.0.0.1,localhost,10.136.98.20,10.136.98.214,10.136.101.70,10.136.101.154",
}
print(json.dumps({"env_vars": env}))
PYH
)
  "${SLIME_ENV}/bin/ray" job submit \
     --address="${RAY_JOB_ADDRESS}" \
     --runtime-env-json="${RUNTIME_ENV_JSON}" \
     -- "${TRAIN_ENTRY[@]}"
else
  "${TRAIN_ENTRY[@]}"
fi
