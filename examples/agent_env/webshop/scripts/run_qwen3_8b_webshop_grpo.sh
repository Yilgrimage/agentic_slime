#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
LOCAL_RUNTIME_ROOT=${LOCAL_RUNTIME_ROOT:-${MLF_LOCAL_ROOT}}
NAS_MLF_ROOT=${NAS_MLF_ROOT:-${MLF_NAS_ROOT}}
WANDB_SECRET_FILE=${WANDB_SECRET_FILE:-${MLF_NAS_ROOT}/secrets/wandb.env}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

USER_SLIME_ENV=${SLIME_ENV:-}
USER_WEBSHOP_ENV=${WEBSHOP_ENV:-}
if [ -f "${MLF_LOCAL_ROOT}/env.sh" ]; then
  source "${MLF_LOCAL_ROOT}/env.sh"
fi
if [ -f "${WANDB_SECRET_FILE}" ]; then
  # Expected keys: WANDB_API_KEY, optionally WANDB_BASE_URL/WANDB_ENTITY.
  # Keep this file out of git and chmod it to 600 on NAS.
  # shellcheck disable=SC1090
  case "$-" in
    *x*) _restore_xtrace=1; set +x ;;
    *) _restore_xtrace=0 ;;
  esac
  set -a
  source "${WANDB_SECRET_FILE}"
  set +a
  if [ "${_restore_xtrace}" = "1" ]; then
    set -x
  fi
  unset _restore_xtrace
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
WEBSHOP_DATA_SIZE=${WEBSHOP_DATA_SIZE:-small}
if [ "${WEBSHOP_DATA_SIZE}" = "full" ] || [ "${WEBSHOP_DATA_SIZE}" = "all" ]; then
  DEFAULT_WEBSHOP_NUM_PRODUCTS=100000
  DEFAULT_WEBSHOP_PRODUCT_FILE="${WEBSHOP_DATA_DIR}/data/items_shuffle.json"
  DEFAULT_WEBSHOP_ATTR_FILE="${WEBSHOP_DATA_DIR}/data/items_ins_v2.json"
  DEFAULT_WEBSHOP_ENV_POOL_SIZE=32
else
  DEFAULT_WEBSHOP_NUM_PRODUCTS=1000
  DEFAULT_WEBSHOP_PRODUCT_FILE="${WEBSHOP_DATA_DIR}/data/items_shuffle_1000.json"
  DEFAULT_WEBSHOP_ATTR_FILE="${WEBSHOP_DATA_DIR}/data/items_ins_v2_1000.json"
  DEFAULT_WEBSHOP_ENV_POOL_SIZE=32
fi
WEBSHOP_NUM_PRODUCTS=${WEBSHOP_NUM_PRODUCTS:-${DEFAULT_WEBSHOP_NUM_PRODUCTS}}
WEBSHOP_PRODUCT_FILE=${WEBSHOP_PRODUCT_FILE:-${DEFAULT_WEBSHOP_PRODUCT_FILE}}
WEBSHOP_ATTR_FILE=${WEBSHOP_ATTR_FILE:-${DEFAULT_WEBSHOP_ATTR_FILE}}
WEBSHOP_ENV_POOL_SIZE=${WEBSHOP_ENV_POOL_SIZE:-${DEFAULT_WEBSHOP_ENV_POOL_SIZE}}
WEBSHOP_PROMPT_NUM_TASKS=${WEBSHOP_PROMPT_NUM_TASKS:-}
USER_DATA_PATH=${DATA_PATH:-}
USER_EVAL_VALID_PATH=${EVAL_VALID_PATH:-}

BASE_WEBSHOP_CONFIG=${BASE_WEBSHOP_CONFIG:-${REPO_DIR}/examples/agent_env/webshop/train_config.yaml}
WEBSHOP_CONFIG=${WEBSHOP_CONFIG:-${LOCAL_RUNTIME_ROOT}/configs/webshop_train_config.yaml}
BASE_WEBSHOP_EVAL_CONFIG=${BASE_WEBSHOP_EVAL_CONFIG:-${REPO_DIR}/examples/agent_env/webshop/eval_config.yaml}
WEBSHOP_EVAL_CONFIG=${WEBSHOP_EVAL_CONFIG:-${LOCAL_RUNTIME_ROOT}/configs/webshop_eval_config.yaml}
WEBSHOP_SERVER_HOST=${WEBSHOP_SERVER_HOST:-127.0.0.1}
WEBSHOP_SERVER_PORT=${WEBSHOP_SERVER_PORT:-18180}
USE_EXISTING_AGENT_INFRA=${USE_EXISTING_AGENT_INFRA:-0}
if [ "${USE_EXISTING_AGENT_INFRA}" = "1" ]; then
  WEBSHOP_ENV_SERVER_URL=${WEBSHOP_ENV_SERVER_URL:-http://127.0.0.1:19000}
else
  WEBSHOP_ENV_SERVER_URL=${WEBSHOP_ENV_SERVER_URL:-http://${WEBSHOP_SERVER_HOST}:${WEBSHOP_SERVER_PORT}}
fi
WEBSHOP_SERVER_LOG=${WEBSHOP_SERVER_LOG:-${LOCAL_RUNTIME_ROOT}/logs/webshop_env_server.log}

LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-${LOCAL_RUNTIME_ROOT}/models/Qwen3-8B}
LOCAL_TORCH_DIST_DIR=${LOCAL_TORCH_DIST_DIR:-${LOCAL_RUNTIME_ROOT}/models/Qwen3-8B_torch_dist}
MODEL_DIR=${MODEL_DIR:-$([ -d "${LOCAL_MODEL_DIR}" ] && echo "${LOCAL_MODEL_DIR}" || echo "${NAS_MLF_ROOT}/models/Qwen3-8B")}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-$([ -d "${LOCAL_TORCH_DIST_DIR}" ] && echo "${LOCAL_TORCH_DIST_DIR}" || echo "${NAS_MLF_ROOT}/models/Qwen3-8B_torch_dist")}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${LOCAL_RUNTIME_ROOT}/ray/webshop_${USER}}

EXP_PROJECT=${EXP_PROJECT:-${PROJECT_NAME:-Qwen3-8B_webshop_grpo}}
EXP_NAME=${EXP_NAME:-${RUN_NAME:-${WANDB_GROUP:-qwen3-8b-webshop-grpo}}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${LOCAL_RUNTIME_ROOT}/outputs}
SAVE_DIR=${SAVE_DIR:-${OUTPUT_ROOT}/${EXP_PROJECT}/${EXP_NAME}}

export TMPDIR=${TMPDIR:-${LOCAL_RUNTIME_ROOT}/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${LOCAL_RUNTIME_ROOT}/cache/xdg}
export HF_HOME=${HF_HOME:-${LOCAL_RUNTIME_ROOT}/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${LOCAL_RUNTIME_ROOT}/cache/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${LOCAL_RUNTIME_ROOT}/cache/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${LOCAL_RUNTIME_ROOT}/cache/cuda}

mkdir -p "${LOCAL_RUNTIME_ROOT}/configs" "${LOCAL_RUNTIME_ROOT}/logs" "${LOCAL_RUNTIME_ROOT}/code" "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

if [ "${USE_EXISTING_AGENT_INFRA}" != "1" ] && [ ! -d "${WEBSHOP_LIB}" ] && [ -d "${NAS_WEBSHOP_SRC}" ]; then
  mkdir -p "$(dirname "${WEBSHOP_LIB}")"
  cp -a "${NAS_WEBSHOP_SRC}" "${WEBSHOP_LIB}"
fi

if [ "${USE_EXISTING_AGENT_INFRA}" != "1" ] && [ -d "${NAS_WEBSHOP_DATA}" ]; then
  mkdir -p "${WEBSHOP_DATA_DIR}"
  cp -a "${NAS_WEBSHOP_DATA}/." "${WEBSHOP_DATA_DIR}/"
fi

mkdir -p "$(dirname "${WEBSHOP_CONFIG}")"
sed "s|data_dir: /tmp/mlf-runtime/data/webshop|data_dir: ${WEBSHOP_DATA_DIR}|" "${BASE_WEBSHOP_CONFIG}" > "${WEBSHOP_CONFIG}"
"${SLIME_PYTHON}" - <<PYH
from pathlib import Path
import os
import yaml

path = Path("${WEBSHOP_CONFIG}")
cfg = yaml.safe_load(path.read_text()) or {}
webshop = cfg.setdefault("webshop", {})
webshop["data_dir"] = "${WEBSHOP_DATA_DIR}"
webshop["product_file"] = "${WEBSHOP_PRODUCT_FILE}"
webshop["attr_file"] = "${WEBSHOP_ATTR_FILE}"
webshop["num_products"] = int("${WEBSHOP_NUM_PRODUCTS}")
if os.environ.get("WANDB_PROJECT"):
    cfg["wandb_project"] = os.environ["WANDB_PROJECT"]
if os.environ.get("WANDB_GROUP"):
    cfg["wandb_group"] = os.environ["WANDB_GROUP"]
if os.environ.get("WANDB_ENTITY"):
    cfg["wandb_team"] = os.environ["WANDB_ENTITY"]
elif os.environ.get("WANDB_TEAM"):
    cfg["wandb_team"] = os.environ["WANDB_TEAM"]
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

config_value() {
  local path=$1
  local key=$2
  local default=$3
  "${SLIME_PYTHON}" - "${path}" "${key}" "${default}" <<'PY'
import sys
import yaml

path, key, default = sys.argv[1:4]
with open(path) as f:
    cfg = yaml.safe_load(f) or {}
value = cfg.get(key, default)
if value is None:
    value = default
print(value)
PY
}

WANDB_PROJECT=${WANDB_PROJECT:-${EXP_PROJECT}}
WANDB_GROUP=${WANDB_GROUP:-${EXP_NAME}}
WANDB_TEAM=${WANDB_TEAM:-$(config_value "${WEBSHOP_CONFIG}" wandb_team "")}

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
if [ "${USE_EXISTING_AGENT_INFRA}" != "1" ]; then
  PYTHONPATH="${WEBSHOP_LIB}" "${WEBSHOP_PYTHON}" "${REPO_DIR}/examples/agent_env/webshop/server.py" \
    --host "${WEBSHOP_SERVER_HOST}" \
    --port "${WEBSHOP_SERVER_PORT}" \
    --config "${WEBSHOP_CONFIG}" \
    > "${WEBSHOP_SERVER_LOG}" 2>&1 &
  WEBSHOP_SERVER_PID=$!
fi

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
  if [ -n "${WEBSHOP_SERVER_PID}" ] && ! kill -0 "${WEBSHOP_SERVER_PID}" 2>/dev/null; then
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

if [ -z "${WEBSHOP_PROMPT_NUM_TASKS}" ]; then
  WEBSHOP_PROMPT_NUM_TASKS=$("${SLIME_PYTHON}" - <<PYH
import json
import urllib.request
base_url = "${WEBSHOP_ENV_SERVER_URL}".rstrip("/")
data = json.loads(urllib.request.urlopen(f"{base_url}/health", timeout=5).read().decode())
num_tasks = data.get("num_tasks")
if not num_tasks:
    status = json.loads(urllib.request.urlopen(f"{base_url}/status", timeout=5).read().decode())
    workers = status.get("workers") or []
    worker_tasks = [int(w["num_tasks"]) for w in workers if w.get("num_tasks")]
    if worker_tasks:
        num_tasks = min(worker_tasks)
if not num_tasks:
    raise SystemExit("WebShop server did not report num_tasks")
print(int(num_tasks))
PYH
)
  DATA_PATH=${DATA_PATH:-${WEBSHOP_DATA_DIR}/train_${WEBSHOP_PROMPT_NUM_TASKS}.jsonl}
  EVAL_VALID_PATH=${EVAL_VALID_PATH:-${WEBSHOP_DATA_DIR}/valid_${WEBSHOP_PROMPT_NUM_TASKS}.jsonl}
fi

DATA_PATH=${USER_DATA_PATH:-${WEBSHOP_DATA_DIR}/train_${WEBSHOP_PROMPT_NUM_TASKS}.jsonl}
EVAL_VALID_PATH=${USER_EVAL_VALID_PATH:-${WEBSHOP_DATA_DIR}/valid_${WEBSHOP_PROMPT_NUM_TASKS}.jsonl}

sed "s|path: /tmp/mlf-runtime/data/webshop/valid_100.jsonl|path: ${EVAL_VALID_PATH}|" \
  "${BASE_WEBSHOP_EVAL_CONFIG}" > "${WEBSHOP_EVAL_CONFIG}"

if [ ! -f "${DATA_PATH}" ] || [ ! -f "${EVAL_VALID_PATH}" ]; then
  "${SLIME_PYTHON}" "${REPO_DIR}/examples/agent_env/webshop/prompt_data.py" \
    --output-dir "${WEBSHOP_DATA_DIR}" \
    --num-tasks "${WEBSHOP_PROMPT_NUM_TASKS}" \
    --splits train valid
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

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_GPUS=${ACTOR_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
RAY_PORT=${RAY_PORT:-8265}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

mkdir -p "${SAVE_DIR}" "${RAY_TEMP_DIR}" "${TMPDIR}"

if [ "${USE_EXISTING_AGENT_INFRA}" != "1" ]; then
  "${SLIME_PYTHON}" -m ray.scripts.scripts stop --force 2>/dev/null || true
  pkill -u "${USER}" -f "sglang.launch_server" 2>/dev/null || true
  pkill -u "${USER}" -f "sglang_router" 2>/dev/null || true
  sleep 3
fi

cd "${REPO_DIR}"
source scripts/models/qwen3-8B.sh

if [ "${USE_EXISTING_AGENT_INFRA}" != "1" ]; then
  "${SLIME_PYTHON}" -m ray.scripts.scripts start --head \
     --node-ip-address "${MASTER_ADDR}" \
     --num-gpus "${NUM_GPUS}" \
     --disable-usage-stats \
     --dashboard-host=0.0.0.0 \
     --dashboard-port="${RAY_PORT}" \
     --temp-dir "${RAY_TEMP_DIR}"
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-9999}"
)
USE_KL_LOSS=${USE_KL_LOSS:-1}
if [ -n "${LOAD_DIR:-}" ]; then
  REF_LOAD_DIR=${REF_LOAD_DIR:-${LOAD_DIR}}
else
  REF_LOAD_DIR=${REF_LOAD_DIR:-${TORCH_DIST_DIR}}
fi
if [ "${USE_KL_LOSS}" = "1" ] || [ "${KL_LOSS_COEF:-0.00}" != "0.00" ]; then
  CKPT_ARGS+=(--ref-load "${REF_LOAD_DIR}")
fi
if [ -n "${LOAD_DIR:-}" ]; then
  CKPT_ARGS+=(--load "${LOAD_DIR}")
fi

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path examples.agent_env.webshop.rollout.generate
   --custom-rollout-log-function-path examples.agent_env.webshop.rollout.log_rollout_data
   --custom-eval-rollout-log-function-path examples.agent_env.webshop.rollout.log_eval_rollout_data
   --custom-config-path "${WEBSHOP_CONFIG}"
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-${NUM_STEPS:-100}}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-10240}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-512}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
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
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-20480}"
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
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
if [ -n "${LOAD_DIR:-}" ] && [ "${RESUME_OVERRIDE_OPT_SCHEDULER:-1}" = "1" ]; then
  OPTIMIZER_ARGS+=(--override-opt-param-scheduler)
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE:-1}"
   --rollout-gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-16}"
)

MISC_ARGS=(
   --num-steps "${NUM_STEPS:-100}"
   --log-interval 1
   --seed "${SEED:-42}"
   --ray-temp-dir "${RAY_TEMP_DIR}"
   --actor-num-nodes "${ACTOR_NUM_NODES}"
   --actor-num-gpus-per-node "${ACTOR_GPUS}"
   --rollout-num-gpus "${ROLLOUT_GPUS}"
   --num-gpus-per-node "${NUM_GPUS}"
)

WANDB_ARGS=()
if [ "${ENABLE_WANDB:-0}" = "1" ] || [ "${USE_WANDB:-0}" = "1" ]; then
  WANDB_ARGS=(
     --use-wandb
     --wandb-project "${WANDB_PROJECT}"
     --wandb-group "${WANDB_GROUP}"
     --wandb-dir "${WANDB_DIR:-${LOCAL_RUNTIME_ROOT}/wandb/${EXP_PROJECT}/${EXP_NAME}}"
  )
  if [ "${WANDB_RANDOM_SUFFIX:-0}" != "1" ]; then
    WANDB_ARGS+=(--disable-wandb-random-suffix)
  fi
  if [ -n "${WANDB_BASE_URL:-}" ]; then
    WANDB_ARGS+=(--wandb-host "${WANDB_BASE_URL}")
  fi
  if [ -n "${WANDB_ENTITY:-${WANDB_TEAM}}" ]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY:-${WANDB_TEAM}}")
  fi
fi

if [ -z "${TRAIN_ENV_VARS_JSON:-}" ]; then
  TRAIN_ENV_VARS_JSON=$("${SLIME_PYTHON}" - <<'PYH'
import json
import os

keys = [
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_HOME",
    "PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "WEBSHOP_LIB",
    "WEBSHOP_DATA",
    "JAVA_HOME",
    "JVM_PATH",
]
print(json.dumps({key: os.environ[key] for key in keys if key in os.environ}))
PYH
)
fi
TRAIN_ENV_ARGS=(--train-env-vars "${TRAIN_ENV_VARS_JSON}")

TRAIN_ENTRY=(
   "${SLIME_PYTHON}" "${REPO_DIR}/train_async.py"
   "${CKPT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${EVAL_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${TRAIN_ENV_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

if [ "${SUBMIT_VIA_RAY_JOB:-0}" = "1" ]; then
  RAY_JOB_ADDRESS=${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}
  RUNTIME_ENV_JSON=$("${SLIME_PYTHON}" - <<PYH
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
    "WEBSHOP_ENV_SERVER_URL": "${WEBSHOP_ENV_SERVER_URL}",
    "WEBSHOP_LIB": "${WEBSHOP_LIB}",
    "WEBSHOP_DATA": "${WEBSHOP_DATA_DIR}",
    "JAVA_HOME": "${JAVA_HOME}",
    "JVM_PATH": "${JVM_PATH}",
    "WANDB_API_KEY": "${WANDB_API_KEY:-}",
    "WANDB_BASE_URL": "${WANDB_BASE_URL:-}",
    "WANDB_ENTITY": "${WANDB_ENTITY:-}",
    "no_proxy": "127.0.0.1,localhost,10.136.98.20,10.136.98.214,10.136.101.70,10.136.101.154",
}
env = {key: value for key, value in env.items() if value != ""}
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
