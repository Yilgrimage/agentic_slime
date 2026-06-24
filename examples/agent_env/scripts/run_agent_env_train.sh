#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1

ENV_ROUTER_URL_ARG=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --env-server-url)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --env-server-url" >&2
        exit 1
      fi
      ENV_ROUTER_URL_ARG=$2
      shift 2
      ;;
    *)
      echo "Unknown run_agent_env_train.sh argument: $1" >&2
      exit 1
      ;;
  esac
done

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
REPO_DIR=${REPO_DIR:-${MLF_NAS_ROOT}/code/slime}

resolve_repo_path() {
  local path=$1
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${REPO_DIR}" "${path}"
  fi
}

if [ -n "${TRAIN_PROFILE:-}" ]; then
  TRAIN_PROFILE_PATH=$(resolve_repo_path "${TRAIN_PROFILE}")
  [ -f "${TRAIN_PROFILE_PATH}" ] || { echo "Missing train profile: ${TRAIN_PROFILE_PATH}" >&2; exit 1; }
  set -a
  # shellcheck disable=SC1090
  source "${TRAIN_PROFILE_PATH}"
  set +a
fi

if [ -n "${RESUME_FROM:-}" ] && [ -z "${LOAD_DIR:-}" ]; then
  export LOAD_DIR="${RESUME_FROM}"
fi

ENV_NAME=${ENV_NAME:?Set ENV_NAME to alfworld, webshop, tau2, or appworld}
WANDB_SECRET_FILE=${WANDB_SECRET_FILE:-${MLF_NAS_ROOT}/secrets/wandb.env}
MEGATRON_PATH=${MEGATRON_PATH:-${MLF_NAS_ROOT}/code/Megatron-LM}
if [ -z "${SLIME_ENV:-}" ]; then
  if [ -x "${MLF_LOCAL_ENVS}/slime-official/bin/python" ]; then
    SLIME_ENV="${MLF_LOCAL_ENVS}/slime-official"
  elif [ -x "${MLF_LOCAL_ENVS}/slime/bin/python" ]; then
    SLIME_ENV="${MLF_LOCAL_ENVS}/slime"
  elif [ -x "${MLF_NAS_ROOT}/envs/slime-official/bin/python" ]; then
    SLIME_ENV="${MLF_NAS_ROOT}/envs/slime-official"
  else
    SLIME_ENV="${MLF_NAS_ROOT}/envs/slime"
  fi
fi
SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_ENV}/bin/python}
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT:-examples/agent_env/train_entrypoint.py}
AGENT_ENV_TRAIN_LOOP=${AGENT_ENV_TRAIN_LOOP:-async}

configure_env_defaults() {
  case "${ENV_NAME}" in
    alfworld)
      export CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-examples.agent_env.alfworld.rollout.generate}
      export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${ALFWORLD_CONFIG:-${ENV_CONFIG:-examples/agent_env/alfworld/env_config.yaml}}}
      export DATA_DIR=${DATA_DIR:-${MLF_LOCAL_ROOT}/data/alfworld}
      export PROMPT_DATA_SCRIPT=${PROMPT_DATA_SCRIPT:-${REPO_DIR}/examples/agent_env/alfworld/prompt_data.py}
      ;;
    webshop)
      export CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-examples.agent_env.webshop.rollout.generate}
      export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${WEBSHOP_CONFIG:-${ENV_CONFIG:-examples/agent_env/webshop/env_config.yaml}}}
      export DATA_DIR=${DATA_DIR:-${MLF_LOCAL_ROOT}/data/webshop}
      export PROMPT_DATA_SCRIPT=${PROMPT_DATA_SCRIPT:-${REPO_DIR}/examples/agent_env/webshop/prompt_data.py}
      ;;
    tau2)
      export LITELLM_LOCAL_MODEL_COST_MAP=${LITELLM_LOCAL_MODEL_COST_MAP:-True}
      export CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-examples.agent_env.tau2.rollout.generate}
      export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${ENV_CONFIG:-examples/agent_env/tau2/env_config.yaml}}
      export PROMPT_DATA_PYTHON=${PROMPT_DATA_PYTHON:-${TAU2_ENV:-${MLF_LOCAL_ENVS}/tau2}/bin/python}
      export PROMPT_USE_SERVER_NUM_TASKS=${PROMPT_USE_SERVER_NUM_TASKS:-0}
      configure_tau2_prompt_data
      ;;
    appworld)
      export CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-examples.agent_env.appworld.rollout.generate}
      export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${ENV_CONFIG:-examples/agent_env/appworld/env_config.yaml}}
      export APPWORLD_ROOT=${APPWORLD_ROOT:-${MLF_LOCAL_ROOT}/data/appworld}
      export HOME=${APPWORLD_ROOT}
      ;;
    *)
      echo "Unsupported ENV_NAME: ${ENV_NAME}" >&2
      exit 1
      ;;
  esac
}

configure_tau2_prompt_data() {
  eval "$("${PROMPT_DATA_PYTHON}" - <<'PY'
import os
import shlex
from pathlib import Path

import yaml

config_path = Path(os.environ["CUSTOM_CONFIG_PATH"])
cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
prompt = (cfg or {}).get("agent_prompt_data") or {}
local_root = os.environ.get("MLF_LOCAL_ROOT", "/tmp/mlf-runtime")
repo_dir = os.environ.get("REPO_DIR", "/mnt/bn/jixf-nas-lq/mlf/code/slime")

def expand(value):
    if value is None:
        return ""
    return os.path.expandvars(str(value))

def emit(name, value):
    if value is None or value == "":
        return
    if isinstance(value, bool):
        value = "1" if value else "0"
    elif isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    else:
        value = str(value)
    print(f"export {name}={shlex.quote(value)}")

source = prompt.get("source") or "official"
data_dir = expand(prompt.get("data_dir") or f"{local_root}/data/tau2/data")
domains = prompt.get("domains") or ["retail"]
if isinstance(domains, str):
    domains_csv = domains
else:
    domains_csv = ",".join(str(item) for item in domains)
num_tasks = prompt.get("num_tasks", "all")
if num_tasks is None:
    num_tasks = "all"
seed = prompt.get("seed", os.environ.get("SEED", "42"))
output_dir = expand(prompt.get("output_dir") or f"{local_root}/data/tau2/{source}_prompt")

args = [
    "--source", source,
    "--data-dir", data_dir,
    "--domains", domains_csv,
    "--split", str(prompt.get("split") or "train"),
    "--seed", str(seed),
]
if source == "areal_synthetic":
    args.extend(["--areal-root", expand(prompt.get("areal_root") or f"{local_root}/data/tau2/areal_synthetic")])
    if prompt.get("areal_input"):
        args.extend(["--areal-input", expand(prompt["areal_input"])])
    if prompt.get("task_file_dir"):
        args.extend(["--task-file-dir", expand(prompt["task_file_dir"])])
if prompt.get("domain_weights"):
    args.extend(["--domain-weights", str(prompt["domain_weights"])])
if prompt.get("task_sets"):
    args.extend(["--task-sets", str(prompt["task_sets"])])

emit("TAU2_DATA_SOURCE", source)
emit("TAU2_DATA_DIR", data_dir)
emit("PROMPT_DATA_SCRIPT", expand(prompt.get("script") or f"{repo_dir}/examples/agent_env/tau2/prompt_data.py"))
emit("PROMPT_DATA_PYTHON", expand(prompt.get("python") or os.environ.get("PROMPT_DATA_PYTHON")))
emit("PROMPT_NUM_TASKS", num_tasks)
emit("DATA_DIR", output_dir)
emit("PROMPT_DATA_EXTRA_ARGS", " ".join(shlex.quote(item) for item in args))
PY
)"
}

configure_env_defaults
CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:?Set CUSTOM_GENERATE_FUNCTION_PATH}
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:?Set CUSTOM_CONFIG_PATH}
export DYNAMIC_SAMPLING_FILTER_PATH=${DYNAMIC_SAMPLING_FILTER_PATH:-}

is_async_entrypoint() {
  case "${AGENT_ENV_TRAIN_LOOP:-}" in
    async|fullasync|full_async)
      return 0
      ;;
    sync)
      return 1
      ;;
  esac
  case "$(basename -- "${TRAIN_ENTRYPOINT}")" in
    train_async.py)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

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
export WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-300}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}
export NUM_STEPS=${NUM_STEPS:-${TOTAL_NUM_STEPS:-100}}
export NUM_ROLLOUT=${NUM_ROLLOUT:-${TOTAL_NUM_STEPS:-${NUM_STEPS}}}
export SAVE_INTERVAL=${SAVE_INTERVAL:-${TOTAL_NUM_STEPS:-${NUM_STEPS}}}
export AGENT_ENV_ROLLOUT_DUMP_N=${AGENT_ENV_ROLLOUT_DUMP_N:-${ROLLOUT_CASE_DUMP_N:-0}}
export AGENT_ENV_ROLLOUT_DUMP_DISCARD_N=${AGENT_ENV_ROLLOUT_DUMP_DISCARD_N:-${ROLLOUT_CASE_DUMP_DISCARD_N:-${AGENT_ENV_ROLLOUT_DUMP_N}}}
export AGENT_ENV_ROLLOUT_DUMP_TRACE=${AGENT_ENV_ROLLOUT_DUMP_TRACE:-${ROLLOUT_CASE_DUMP_TRACE:-both}}
export CUSTOM_RM_PATH=${CUSTOM_RM_PATH:-examples.agent_env.group_rm.group_reward}
export GROUP_RM=${GROUP_RM:-1}
export AGENT_ENV_JUDGE_MODE=${AGENT_ENV_JUDGE_MODE:-none}
export RM_TYPE=${RM_TYPE:-}
export RM_URL=${RM_URL:-}
export REWARD_KEY=${REWARD_KEY:-}
export EVAL_REWARD_KEY=${EVAL_REWARD_KEY:-}
export LOG_REWARD_CATEGORY=${LOG_REWARD_CATEGORY:-}

ENV_ROUTER_URL=${ENV_ROUTER_URL_ARG}
if [ -z "${ENV_ROUTER_URL}" ]; then
  echo "Missing env server URL. Pass --env-server-url to run_agent_env_train.sh." >&2
  exit 1
fi

MODEL_BASENAME=${MODEL_BASENAME:-Qwen3.5-9B}
MODEL_ARGS_SCRIPT=${MODEL_ARGS_SCRIPT:-scripts/models/qwen3.5-9B.sh}
MODEL_DIR=${MODEL_DIR:-${MLF_NAS_ROOT}/models/${MODEL_BASENAME}}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${MLF_NAS_ROOT}/models/${MODEL_BASENAME}_torch_dist}

EXP_PROJECT=${EXP_PROJECT:-${PROJECT_NAME:-${MODEL_BASENAME}_${ENV_NAME}_grpo}}
EXP_NAME=${EXP_NAME:-${RUN_NAME:-${MODEL_BASENAME}-${ENV_NAME}-grpo}}
RUN_ROOT=${RUN_ROOT:-${MLF_NAS_ROOT}/runs/${EXP_PROJECT}/${EXP_NAME}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}}
SAVE_DIR=${SAVE_DIR:-${RUN_ROOT}/checkpoints}
LOG_DIR=${LOG_DIR:-${RUN_ROOT}/logs}
WANDB_DIR=${WANDB_DIR:-${RUN_ROOT}/wandb}
RUN_USER=${USER:-$(id -un 2>/dev/null || echo unknown)}
export USER=${USER:-${RUN_USER}}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${MLF_LOCAL_ROOT}/ray/${ENV_NAME}_${RUN_USER}}
DATA_DIR=${DATA_DIR:-${MLF_LOCAL_ROOT}/data/${ENV_NAME}}
PROMPT_NUM_TASKS=${PROMPT_NUM_TASKS:-}
DATA_PATH=${DATA_PATH:-}
PROMPT_DATA_SCRIPT=${PROMPT_DATA_SCRIPT:-${REPO_DIR}/examples/agent_env/scripts/prompt_data.py}
PROMPT_DATA_PYTHON=${PROMPT_DATA_PYTHON:-${SLIME_PYTHON}}
PROMPT_USE_SERVER_NUM_TASKS=${PROMPT_USE_SERVER_NUM_TASKS:-1}

export TMPDIR=${TMPDIR:-${MLF_LOCAL_ROOT}/tmp}
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR:-},${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR:-},${NO_PROXY:-}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${MLF_LOCAL_ROOT}/cache/xdg}
export HF_HOME=${HF_HOME:-${MLF_LOCAL_ROOT}/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${MLF_LOCAL_ROOT}/cache/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${MLF_LOCAL_ROOT}/cache/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${MLF_LOCAL_ROOT}/cache/cuda}

mkdir -p "${MLF_LOCAL_ROOT}/logs" "${DATA_DIR}" "${SAVE_DIR}" "${LOG_DIR}" "${WANDB_DIR}" "${RAY_TEMP_DIR}" "${TMPDIR}" \
  "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

if [ -z "${PROMPT_NUM_TASKS}" ] && [ "${PROMPT_USE_SERVER_NUM_TASKS}" = "1" ]; then
  PROMPT_NUM_TASKS=$("${SLIME_PYTHON}" - <<PYH
import json
import urllib.request
base_url = "${ENV_ROUTER_URL}".rstrip("/")
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
PROMPT_NUM_TASKS=${PROMPT_NUM_TASKS:-all}

DATA_PATH=${DATA_PATH:-${DATA_DIR}/train_${PROMPT_NUM_TASKS}.jsonl}
if [ "${FORCE_PROMPT_DATA:-0}" = "1" ] || [ ! -f "${DATA_PATH}" ]; then
  read -r -a PROMPT_DATA_EXTRA_ARGS_ARRAY <<< "${PROMPT_DATA_EXTRA_ARGS:-}"
  PROMPT_NUM_TASK_ARGS=()
  if [ "${PROMPT_NUM_TASKS}" != "all" ]; then
    PROMPT_NUM_TASK_ARGS=(--num-tasks "${PROMPT_NUM_TASKS}")
  fi
  "${PROMPT_DATA_PYTHON}" "${PROMPT_DATA_SCRIPT}" \
    --output "${DATA_PATH}" \
    --split train \
    "${PROMPT_NUM_TASK_ARGS[@]}" \
    "${PROMPT_DATA_EXTRA_ARGS_ARRAY[@]}"
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
source "${MODEL_ARGS_SCRIPT}"
read -r -a MODEL_EXTRA_ARGS_ARRAY <<< "${MODEL_EXTRA_ARGS:-}"

ROLLOUT_FUNCTION_PATH=${ROLLOUT_FUNCTION_PATH:-}
if [ -z "${ROLLOUT_FUNCTION_PATH}" ]; then
  if is_async_entrypoint; then
    ROLLOUT_FUNCTION_PATH=slime.rollout.fully_async_rollout.generate_rollout_fully_async
  else
    ROLLOUT_FUNCTION_PATH=slime.rollout.sglang_rollout.generate_rollout
  fi
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-${NUM_STEPS}}"
)
case "${NO_SAVE_OPTIM:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    CKPT_ARGS+=(--no-save-optim)
    ;;
esac
case "${ASYNC_SAVE:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    CKPT_ARGS+=(--async-save)
    ;;
esac
USE_KL_LOSS=${USE_KL_LOSS:-0}
LOAD_DIR=${LOAD_DIR:-${TORCH_DIST_DIR}}
if [ ! -d "${LOAD_DIR}" ]; then
  cat >&2 <<EOF
Missing Megatron torch_dist checkpoint:
  LOAD_DIR=${LOAD_DIR}

Create it first, for example:
  cd ${REPO_DIR}
  source ${MODEL_ARGS_SCRIPT}
  PYTHONPATH=${MEGATRON_PATH}:${REPO_DIR} torchrun --nproc_per_node 8 tools/convert_hf_to_torch_dist.py \\
    "\${MODEL_ARGS[@]}" \\
    --hf-checkpoint ${MODEL_DIR} \\
    --save ${TORCH_DIST_DIR}
EOF
  exit 1
fi
if [ "${USE_KL_LOSS}" = "1" ] || [ "${KL_LOSS_COEF:-0.00}" != "0.00" ]; then
  CKPT_ARGS+=(--ref-load "${REF_LOAD_DIR:-${LOAD_DIR}}")
fi
CKPT_ARGS+=(--load "${LOAD_DIR}")

ROLLOUT_ARGS=(
   --env-server-url "${ENV_ROUTER_URL}"
   --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
   --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}"
   --custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH:-examples.agent_env.reward_post_process.post_process_rewards}"
   --custom-rollout-log-function-path "${CUSTOM_GENERATE_FUNCTION_PATH%.*}.log_rollout_data"
   --custom-eval-rollout-log-function-path "${CUSTOM_GENERATE_FUNCTION_PATH%.*}.log_eval_rollout_data"
   --custom-config-path "${CUSTOM_CONFIG_PATH}"
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
   --loss-mask-type "${LOSS_MASK_TYPE:-qwen3_5}"
   --balance-data
)
if [ -n "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
case "${GROUP_RM:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    ROLLOUT_ARGS+=(--group-rm)
    if [ -n "${CUSTOM_RM_PATH:-}" ]; then
      ROLLOUT_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")
    fi
    ;;
  *)
    if [ -n "${CUSTOM_RM_PATH:-}" ]; then
      ROLLOUT_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")
    fi
    ;;
esac
if [ -n "${RM_TYPE:-}" ]; then
  ROLLOUT_ARGS+=(--rm-type "${RM_TYPE}")
fi
if [ -n "${RM_URL:-}" ]; then
  ROLLOUT_ARGS+=(--rm-url "${RM_URL}")
fi
if [ -n "${REWARD_KEY:-}" ]; then
  ROLLOUT_ARGS+=(--reward-key "${REWARD_KEY}")
fi
if [ -n "${EVAL_REWARD_KEY:-}" ]; then
  ROLLOUT_ARGS+=(--eval-reward-key "${EVAL_REWARD_KEY}")
fi
if [ -n "${LOG_REWARD_CATEGORY:-}" ]; then
  ROLLOUT_ARGS+=(--log-reward-category "${LOG_REWARD_CATEGORY}")
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-4}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE:-1}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity "${RECOMPUTE_GRANULARITY:-full}"
   --recompute-method "${RECOMPUTE_METHOD:-uniform}"
   --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-12288}"
)
if [ -n "${LOG_PROBS_CHUNK_SIZE:-}" ] && [ "${LOG_PROBS_CHUNK_SIZE}" != "-1" ]; then
  PERF_ARGS+=(--log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}")
fi
COLOCATE=${COLOCATE:-}
if [ -z "${COLOCATE}" ]; then
  if is_async_entrypoint; then
    COLOCATE=0
  else
    COLOCATE=1
  fi
fi
if [ "${COLOCATE}" = "1" ]; then
  PERF_ARGS=(--colocate "${PERF_ARGS[@]}")
fi

GRPO_ARGS=(
   --advantage-estimator "${ADVANTAGE_ESTIMATOR:-grpo}"
   --entropy-coef "${ENTROPY_COEF:-0.00}"
   --eps-clip "${EPS_CLIP:-0.2}"
   --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
)
case "${REWARDS_NORMALIZATION:-1}" in
  0|false|FALSE|no|NO|off|OFF)
    GRPO_ARGS+=(--disable-rewards-normalization)
    ;;
esac
case "${GRPO_STD_NORMALIZATION:-1}" in
  0|false|FALSE|no|NO|off|OFF)
    GRPO_ARGS+=(--disable-grpo-std-normalization)
    ;;
esac
case "${NORMALIZE_ADVANTAGES:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    GRPO_ARGS+=(--normalize-advantages)
    ;;
esac
case "${USE_ROLLOUT_LOGPROBS:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    GRPO_ARGS+=(--use-rollout-logprobs)
    ;;
esac
if [ "${USE_KL_LOSS}" = "1" ] || [ "${KL_LOSS_COEF:-0.00}" != "0.00" ]; then
  GRPO_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF:-0.00}" --kl-loss-type "${KL_LOSS_TYPE:-low_var_kl}")
fi

DEBUG_ARGS=()
case "${SAVE_DEBUG_TRAIN_DATA:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    if [ -n "${SAVE_DEBUG_TRAIN_DATA_PATH:-}" ]; then
      DEBUG_TRAIN_DATA_PATH="${SAVE_DEBUG_TRAIN_DATA_PATH}"
    else
      DEBUG_TRAIN_DATA_PATH="${RUN_ROOT}/debug/train_data/"'{rollout_id}_{rank}.pt'
    fi
    DEBUG_ARGS+=(--save-debug-train-data "${DEBUG_TRAIN_DATA_PATH}")
    ;;
esac

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY:-0.1}"
   --adam-beta1 "${ADAM_BETA1:-0.9}"
   --adam-beta2 "${ADAM_BETA2:-0.98}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE:-1}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.55}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-8}"
)

MISC_ARGS=(
   --agent-env-train-loop "${AGENT_ENV_TRAIN_LOOP}"
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
if [ "${#WANDB_GROUP}" -gt 128 ]; then
  echo "WANDB_GROUP is longer than 128 chars; truncating for W&B: ${WANDB_GROUP}" >&2
  WANDB_GROUP="${WANDB_GROUP:0:128}"
fi
WANDB_ARGS=()
if [ "${ENABLE_WANDB:-0}" = "1" ] || [ "${USE_WANDB:-0}" = "1" ]; then
  WANDB_ARGS=(
     --use-wandb
     --wandb-project "${WANDB_PROJECT}"
     --wandb-group "${WANDB_GROUP}"
     --wandb-dir "${WANDB_DIR}"
     --disable-wandb-random-suffix
  )
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
    "LD_LIBRARY_PATH", "RAY_ADDRESS",
    "TAU2_DATA_DIR", "TAU2_AREAL_ROOT", "APPWORLD_ROOT", "RUN_ROOT", "LOG_DIR", "WANDB_DIR",
    "AGENT_ENV_ROLLOUT_DUMP_N", "AGENT_ENV_ROLLOUT_DUMP_DISCARD_N",
    "AGENT_ENV_ROLLOUT_DUMP_TRACE",
    "AGENT_ENV_GLM_PADDING_MIN_VALID_FRACTION", "AGENT_ENV_GLM_PADDING_MAX_SEEN_GROUPS",
    "AGENT_ENV_JUDGE_MODE",
    "AUX_ENDPOINT_PROVIDER", "AUX_ENDPOINT_MODEL", "AUX_ENDPOINT_BASE_URL", "AUX_ENDPOINT_API_KEY_PATH",
    "AUX_ENDPOINT_TIMEOUT_S", "AUX_ENDPOINT_MAX_TOKENS", "AUX_ENDPOINT_TEMPERATURE", "AUX_ENDPOINT_TOP_P",
    "AUX_ENDPOINT_ENABLE_THINKING", "AUX_ENDPOINT_SEPARATE_REASONING", "AUX_ENDPOINT_REASONING_EFFORT",
    "LITELLM_LOCAL_MODEL_COST_MAP",
    "WANDB_BASE_URL", "WANDB_ENTITY", "WANDB_HTTP_TIMEOUT", "WANDB_INIT_TIMEOUT",
]
print(json.dumps({key: os.environ[key] for key in keys if key in os.environ and os.environ[key] != ""}))
PYH
)
fi

echo "Launching ${ENV_NAME} GRPO with ${TRAIN_ENTRYPOINT}"
echo "Checkpoint options: save_interval=${SAVE_INTERVAL:-${NUM_STEPS}} no_save_optim=${NO_SAVE_OPTIM:-0} async_save=${ASYNC_SAVE:-0}"
"${SLIME_PYTHON}" "${REPO_DIR}/${TRAIN_ENTRYPOINT}" \
   "${CKPT_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${MODEL_EXTRA_ARGS_ARRAY[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   --train-env-vars "${TRAIN_ENV_VARS_JSON}" \
   "${GRPO_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}"
