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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}
ROOT_DIR=${ROOT_DIR:-$(cd "${REPO_DIR}/../.." && pwd -P)}
LOCAL_RUNTIME_DIR=${LOCAL_RUNTIME_DIR:-/tmp/server-ops-runtime}
LOCAL_ENVS_DIR=${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/utils/slime_runtime.sh"

resolve_repo_path() {
  local path=$1
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${REPO_DIR}" "${path}"
  fi
}

is_true_value() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_nonzero_number() {
  local value=${1:-0}
  awk -v value="${value}" 'BEGIN {
    if (value !~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) {
      exit 2
    }
    exit ((value + 0) != 0) ? 0 : 1
  }'
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

ENV_NAME=${ENV_NAME:?Set ENV_NAME to alfworld, webshop, tau2, appworld, or openclaw}
WANDB_SECRET_FILE=${WANDB_SECRET_FILE:-${ROOT_DIR}/secrets/wandb.env}
resolve_slime_runtime
WANDB_ENABLED=0
if [ "${ENABLE_WANDB:-0}" = "1" ] || [ "${USE_WANDB:-0}" = "1" ]; then
  resolve_wandb_runtime
  WANDB_ENABLED=1
else
  unset WANDB_RUNTIME_RESOLVED WANDB_ENV WANDB_PYTHON WANDB_PYTHONPATH
fi
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT:-examples/agent_env/train_entrypoint.py}
AGENT_ENV_TRAIN_LOOP=${AGENT_ENV_TRAIN_LOOP:-async}

configure_env_defaults() {
  export AGENT_ENV_DATA_DIR=${AGENT_ENV_DATA_DIR:-${LOCAL_RUNTIME_DIR}/data/${ENV_NAME}}
  case "${ENV_NAME}" in
    alfworld)
      export CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-examples.agent_env.alfworld.rollout.generate}
      export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${ALFWORLD_CONFIG:-${ENV_CONFIG:-examples/agent_env/alfworld/env_config.yaml}}}
      export DATA_DIR=${DATA_DIR:-${LOCAL_RUNTIME_DIR}/data/alfworld}
      export PROMPT_DATA_SCRIPT=${PROMPT_DATA_SCRIPT:-${REPO_DIR}/examples/agent_env/alfworld/prompt_data.py}
      export PROMPT_DATA_PYTHON=${PROMPT_DATA_PYTHON:-${ALFWORLD_ENV:-${LOCAL_ENVS_DIR}/alfworld}/bin/python}
      export PROMPT_DATA_CONFIG=${PROMPT_DATA_CONFIG:-${CUSTOM_CONFIG_PATH}}
      ;;
    webshop)
      export CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-examples.agent_env.webshop.rollout.generate}
      export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${WEBSHOP_CONFIG:-${ENV_CONFIG:-examples/agent_env/webshop/env_config.yaml}}}
      export DATA_DIR=${DATA_DIR:-${LOCAL_RUNTIME_DIR}/data/webshop}
      export PROMPT_DATA_SCRIPT=${PROMPT_DATA_SCRIPT:-${REPO_DIR}/examples/agent_env/webshop/prompt_data.py}
      ;;
    tau2)
      export LITELLM_LOCAL_MODEL_COST_MAP=${LITELLM_LOCAL_MODEL_COST_MAP:-True}
      export CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-examples.agent_env.tau2.rollout.generate}
      export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${ENV_CONFIG:-examples/agent_env/tau2/env_config.yaml}}
      export PROMPT_DATA_PYTHON=${PROMPT_DATA_PYTHON:-${TAU2_ENV:-${LOCAL_ENVS_DIR}/tau2}/bin/python}
      export PROMPT_USE_SERVER_NUM_TASKS=${PROMPT_USE_SERVER_NUM_TASKS:-0}
      configure_tau2_prompt_data
      ;;
    appworld)
      export CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-examples.agent_env.appworld.rollout.generate}
      export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${ENV_CONFIG:-examples/agent_env/appworld/env_config.yaml}}
      export APPWORLD_ROOT=${APPWORLD_ROOT:-${LOCAL_RUNTIME_DIR}/data/appworld}
      export HOME=${APPWORLD_ROOT}
      export PROMPT_DATA_SCRIPT=${PROMPT_DATA_SCRIPT:-${REPO_DIR}/examples/agent_env/appworld/prompt_data.py}
      export PROMPT_DATA_PYTHON=${PROMPT_DATA_PYTHON:-${APPWORLD_ENV:-${LOCAL_ENVS_DIR}/appworld}/bin/python}
      export PROMPT_DATA_CONFIG=${PROMPT_DATA_CONFIG:-${CUSTOM_CONFIG_PATH}}
      ;;
    openclaw)
      export CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-examples.agent_env.openclaw.rollout.generate}
      export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${ENV_CONFIG:-examples/agent_env/openclaw/env_config.yaml}}
      export PROMPT_DATA_SCRIPT=${PROMPT_DATA_SCRIPT:-${REPO_DIR}/examples/agent_env/openclaw/prompt_data.py}
      export PROMPT_DATA_CONFIG=${PROMPT_DATA_CONFIG:-${CUSTOM_CONFIG_PATH}}
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
local_root = os.environ.get("LOCAL_RUNTIME_DIR", "/tmp/server-ops-runtime")
agent_env_data_dir = os.environ.get("AGENT_ENV_DATA_DIR") or f"{local_root}/data/{os.environ.get('ENV_NAME', 'tau2')}"
repo_dir = os.environ["REPO_DIR"]

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
data_dir = expand(prompt.get("data_dir") or f"{agent_env_data_dir}/data")
domains = prompt.get("domains") or ["retail"]
if isinstance(domains, str):
    domains_csv = domains
else:
    domains_csv = ",".join(str(item) for item in domains)
num_tasks = prompt.get("num_tasks", "all")
if num_tasks is None:
    num_tasks = "all"
seed = prompt.get("seed", os.environ.get("SEED", "42"))
output_dir = expand(prompt.get("output_dir") or f"{agent_env_data_dir}/{source}_prompt")

args = [
    "--source", source,
    "--data-dir", data_dir,
    "--domains", domains_csv,
    "--split", str(prompt.get("split") or "train"),
    "--seed", str(seed),
]
if source == "areal_synthetic":
    args.extend(["--areal-root", expand(prompt.get("areal_root") or f"{agent_env_data_dir}/areal_synthetic")])
    if prompt.get("areal_input"):
        args.extend(["--areal-input", expand(prompt["areal_input"])])
    if prompt.get("task_file_dir"):
        args.extend(["--task-file-dir", expand(prompt["task_file_dir"])])
if prompt.get("domain_weights"):
    args.extend(["--domain-weights", str(prompt["domain_weights"])])
if prompt.get("task_sets"):
    args.extend(["--task-sets", str(prompt["task_sets"])])

emit("TAU2_DATA_SOURCE", source)
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
REWARD_PROFILE=${REWARD_PROFILE:?Set REWARD_PROFILE in resolved train profile}
REWARD_PROFILE=$(resolve_repo_path "${REWARD_PROFILE}")
[ -f "${REWARD_PROFILE}" ] || { echo "Missing reward profile: ${REWARD_PROFILE}" >&2; exit 1; }
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

if [ -f "${LOCAL_RUNTIME_DIR}/env.sh" ]; then
  source "${LOCAL_RUNTIME_DIR}/env.sh"
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
GROUP_RM=1
if [ "${NUM_ROLLOUT}" = "0" ] && [ -n "${EVAL_INTERVAL:-}" ] && [ -z "${LR_DECAY_ITERS:-}" ]; then
  # Eval-only still initializes the Megatron actor so dist checkpoints can be
  # loaded and synced into rollout engines. Megatron's scheduler requires a
  # positive decay step count even though no training step will run.
  export LR_DECAY_ITERS=1
fi
if [ "${NUM_ROLLOUT}" = "0" ] && [ -n "${EVAL_INTERVAL:-}" ]; then
  # Slime's eval rollout path asserts that group RM is disabled. Eval metrics
  # come from the env rollout result, not from training-time group RM.
  GROUP_RM=0
fi
export AGENT_ENV_ROLLOUT_DUMP_N=${AGENT_ENV_ROLLOUT_DUMP_N:-0}
export AGENT_ENV_ROLLOUT_DUMP_DISCARD_N=${AGENT_ENV_ROLLOUT_DUMP_DISCARD_N:-${AGENT_ENV_ROLLOUT_DUMP_N}}
export AGENT_ENV_ROLLOUT_DUMP_FORMAT_N=${AGENT_ENV_ROLLOUT_DUMP_FORMAT_N:-0}
export AGENT_ENV_ROLLOUT_DUMP_TOTAL_N=${AGENT_ENV_ROLLOUT_DUMP_TOTAL_N:-0}
export AGENT_ENV_ROLLOUT_DUMP_TRACE=${AGENT_ENV_ROLLOUT_DUMP_TRACE:-both}
export AGENT_ENV_CREDIT_DUMP_N=${AGENT_ENV_CREDIT_DUMP_N:-0}
export AGENT_ENV_CREDIT_DUMP_TOTAL_N=${AGENT_ENV_CREDIT_DUMP_TOTAL_N:-0}
export AGENT_ENV_REWARD_GROUP_DUMP_N=${AGENT_ENV_REWARD_GROUP_DUMP_N:-0}
export AGENT_ENV_REWARD_GROUP_DUMP_TOTAL_N=${AGENT_ENV_REWARD_GROUP_DUMP_TOTAL_N:-0}
CUSTOM_RM_PATH=examples.agent_env.group_rm.group_reward

ENV_ROUTER_URL=${ENV_ROUTER_URL_ARG}
if [ -z "${ENV_ROUTER_URL}" ]; then
  echo "Missing env server URL. Pass --env-server-url to run_agent_env_train.sh." >&2
  exit 1
fi

MODEL_BASENAME=${MODEL_BASENAME:-Qwen3.5-9B}
MODEL_ARGS_SCRIPT=${MODEL_ARGS_SCRIPT:-scripts/models/qwen3.5-9B.sh}
MODEL_DIR=${MODEL_DIR:-${ROOT_DIR}/models/${MODEL_BASENAME}}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${ROOT_DIR}/models/${MODEL_BASENAME}_torch_dist}

EXP_PROJECT=${EXP_PROJECT:-${PROJECT_NAME:-${MODEL_BASENAME}_${ENV_NAME}_grpo}}
EXP_NAME=${EXP_NAME:-${RUN_NAME:-${MODEL_BASENAME}-${ENV_NAME}-grpo}}
RUN_ROOT=${RUN_ROOT:-${ROOT_DIR}/runs/${EXP_PROJECT}/${EXP_NAME}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}}
SAVE_DIR=${SAVE_DIR:-${RUN_ROOT}/checkpoints}
LOG_DIR=${LOG_DIR:-${RUN_ROOT}/logs}
WANDB_DIR=${WANDB_DIR:-${RUN_ROOT}/wandb}
RUN_USER=${USER:-$(id -un 2>/dev/null || echo unknown)}
export USER=${USER:-${RUN_USER}}
DATA_DIR=${DATA_DIR:-${LOCAL_RUNTIME_DIR}/data/${ENV_NAME}}
PROMPT_NUM_TASKS=${PROMPT_NUM_TASKS:-}
DATA_PATH=${DATA_PATH:-}
PROMPT_DATA_SCRIPT=${PROMPT_DATA_SCRIPT:-${REPO_DIR}/examples/agent_env/scripts/prompt_data.py}
PROMPT_DATA_PYTHON=${PROMPT_DATA_PYTHON:-${SLIME_PYTHON}}
PROMPT_USE_SERVER_NUM_TASKS=${PROMPT_USE_SERVER_NUM_TASKS:-1}

validate_prompt_data() {
  local path="$1"
  "${SLIME_PYTHON}" - "${path}" "${ENV_NAME}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
env_name = sys.argv[2]

if not path.exists():
    raise SystemExit(f"prompt data does not exist: {path}")

rows = 0
def prompt_text(prompt):
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], dict):
        content = prompt[0].get("content")
        if isinstance(content, str):
            return content.strip()
    return ""

with path.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        rows += 1
        row = json.loads(line)
        prompt = row.get("prompt")
        if not prompt_text(prompt):
            raise SystemExit(f"{path}:{line_no} has empty/invalid prompt")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise SystemExit(f"{path}:{line_no} has missing metadata object")
        if "task_index" not in metadata:
            raise SystemExit(f"{path}:{line_no} metadata is missing task_index")
        if env_name in {"alfworld", "webshop", "appworld", "openclaw", "tau2"}:
            if metadata.get("task_id") in (None, "", []):
                raise SystemExit(f"{path}:{line_no} {env_name} metadata is missing task_id")
        if env_name == "appworld":
            for key in ("task_id", "dataset_name"):
                if metadata.get(key) in (None, "", []):
                    raise SystemExit(f"{path}:{line_no} appworld metadata is missing {key}")
        elif env_name == "openclaw":
            if not isinstance(metadata.get("task"), dict):
                raise SystemExit(f"{path}:{line_no} openclaw metadata is missing task object")
        elif env_name == "tau2":
            if not isinstance(metadata.get("task_ref"), dict) and (
                metadata.get("domain") in (None, "", []) or metadata.get("task_set") in (None, "", [])
            ):
                raise SystemExit(f"{path}:{line_no} tau2 metadata needs task_ref or domain/task_set")

if rows <= 0:
    raise SystemExit(f"prompt data has no rows: {path}")
PY
}

eval_requested() {
  [ -n "${EVAL_INTERVAL:-}" ] || [ -n "${EVAL_CONFIG:-}" ] || [ -n "${EVAL_PROMPT_DATA:-}" ]
}

upper_name() {
  printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_' | tr -c 'A-Z0-9_' '_'
}

eval_prompt_env_var() {
  local split="$1"
  printf '%s_%s_PROMPT_DATA\n' "$(upper_name "${ENV_NAME}")" "$(upper_name "${split}")"
}

default_eval_splits() {
  case "${ENV_NAME}" in
    alfworld)
      printf '%s\n' "valid_seen valid_unseen"
      ;;
    webshop)
      printf '%s\n' "valid"
      ;;
    appworld)
      printf '%s\n' "dev"
      ;;
    tau2)
      printf '%s\n' "dev"
      ;;
    openclaw)
      printf '%s\n' "eval"
      ;;
    *)
      printf '%s\n' ""
      ;;
  esac
}

prepare_native_eval() {
  eval_requested || return 0

  local default_eval_config="${REPO_DIR}/examples/agent_env/${ENV_NAME}/eval_config.yaml"
  if [ -z "${EVAL_CONFIG:-}" ] && [ -z "${EVAL_PROMPT_DATA:-}" ] && [ -f "${default_eval_config}" ]; then
    export EVAL_CONFIG="${default_eval_config}"
  fi
  if [ -n "${EVAL_CONFIG:-}" ]; then
    export EVAL_CONFIG="$(resolve_repo_path "${EVAL_CONFIG}")"
  fi
  if [ -z "${EVAL_CONFIG:-}" ] && [ -z "${EVAL_PROMPT_DATA:-}" ]; then
    echo "Native eval requested but ${ENV_NAME} has no eval_config.yaml and EVAL_PROMPT_DATA is empty." >&2
    exit 1
  fi

  # Full-async rollout functions are training-only; Slime's stock rollout owns
  # native eval and calls the env-specific custom generate function per sample.
  export EVAL_FUNCTION_PATH=${EVAL_FUNCTION_PATH:-slime.rollout.sglang_rollout.generate_rollout}

  if [ -n "${EVAL_PROMPT_DATA:-}" ] && [ -z "${EVAL_CONFIG:-}" ]; then
    return 0
  fi

  local eval_splits="${EVAL_SPLITS:-$(default_eval_splits)}"
  if [ -z "${eval_splits}" ]; then
    echo "Native eval requested but EVAL_SPLITS is empty for ENV_NAME=${ENV_NAME}." >&2
    exit 1
  fi

  local eval_dir="${EVAL_DATA_DIR:-${RUN_ROOT}/prompt_data/eval}"
  mkdir -p "${eval_dir}"

  local eval_script="${EVAL_PROMPT_DATA_SCRIPT:-${PROMPT_DATA_SCRIPT}}"
  local eval_python="${EVAL_PROMPT_DATA_PYTHON:-${PROMPT_DATA_PYTHON}}"
  local eval_config="${EVAL_PROMPT_DATA_CONFIG:-${PROMPT_DATA_CONFIG:-}}"
  local eval_extra="${EVAL_PROMPT_DATA_EXTRA_ARGS:-}"
  read -r -a EVAL_PROMPT_DATA_EXTRA_ARGS_ARRAY <<< "${eval_extra}"

  local eval_config_args=()
  if [ -n "${eval_config}" ]; then
    eval_config_args=(--config "${eval_config}")
  fi

  local split path var count count_args
  for split in ${eval_splits}; do
    path="${eval_dir}/${split}.jsonl"
    count="${EVAL_PROMPT_NUM_TASKS:-}"
    count_args=()
    if [ -z "${count}" ] && [ "${ENV_NAME}" = "webshop" ]; then
      count="${PROMPT_NUM_TASKS}"
    fi
    if [ -n "${count}" ] && [ "${count}" != "all" ]; then
      count_args=(--num-tasks "${count}")
    elif [ "${ENV_NAME}" = "webshop" ]; then
      echo "WebShop native eval needs EVAL_PROMPT_NUM_TASKS or a concrete PROMPT_NUM_TASKS." >&2
      exit 1
    fi

    PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${eval_python}" "${eval_script}" \
      --output "${path}" \
      --split "${split}" \
      "${eval_config_args[@]}" \
      "${count_args[@]}" \
      "${EVAL_PROMPT_DATA_EXTRA_ARGS_ARRAY[@]}"
    validate_prompt_data "${path}"
    var="$(eval_prompt_env_var "${split}")"
    export "${var}=${path}"
    export "$(upper_name "${ENV_NAME}")_EVAL_PROMPT_DATA=${path}"
    export "$(upper_name "${ENV_NAME}")_EVAL_DATASET_NAME=${ENV_NAME}-${split}"
  done
}

export TMPDIR=${TMPDIR:-${LOCAL_RUNTIME_DIR}/tmp}
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR:-},${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR:-},${NO_PROXY:-}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${LOCAL_RUNTIME_DIR}/cache/xdg}
export HF_HOME=${HF_HOME:-${LOCAL_RUNTIME_DIR}/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${LOCAL_RUNTIME_DIR}/cache/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${LOCAL_RUNTIME_DIR}/cache/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${LOCAL_RUNTIME_DIR}/cache/cuda}

mkdir -p "${LOCAL_RUNTIME_DIR}/logs" "${DATA_DIR}" "${SAVE_DIR}" "${LOG_DIR}" "${WANDB_DIR}" "${TMPDIR}" \
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
read -r -a PROMPT_DATA_EXTRA_ARGS_ARRAY <<< "${PROMPT_DATA_EXTRA_ARGS:-}"
PROMPT_DATA_CONFIG_ARGS=()
if [ -n "${PROMPT_DATA_CONFIG:-}" ]; then
  PROMPT_DATA_CONFIG_ARGS=(--config "${PROMPT_DATA_CONFIG}")
fi
PROMPT_NUM_TASK_ARGS=()
if [ "${PROMPT_NUM_TASKS}" != "all" ]; then
  PROMPT_NUM_TASK_ARGS=(--num-tasks "${PROMPT_NUM_TASKS}")
fi
PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PROMPT_DATA_PYTHON}" "${PROMPT_DATA_SCRIPT}" \
  --output "${DATA_PATH}" \
  --split train \
  "${PROMPT_DATA_CONFIG_ARGS[@]}" \
  "${PROMPT_NUM_TASK_ARGS[@]}" \
  "${PROMPT_DATA_EXTRA_ARGS_ARRAY[@]}"
validate_prompt_data "${DATA_PATH}"
prepare_native_eval

resolve_slime_cuda_home
SLIME_SITE_PACKAGES=$(python_site_packages "${SLIME_PYTHON}")
unset PYTHONPATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
export PYTHONNOUSERSITE=1
if [ "${WANDB_ENABLED}" = "1" ] && [ -n "${WANDB_PYTHONPATH:-}" ]; then
  export PYTHONPATH="${WANDB_PYTHONPATH}:${MEGATRON_PATH}:${REPO_DIR}:${SLIME_SITE_PACKAGES}"
else
  export PYTHONPATH="${MEGATRON_PATH}:${REPO_DIR}:${SLIME_SITE_PACKAGES}"
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379}
export CUDA_HOME="${SLIME_CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${SLIME_ENV}/nvvm/bin:${SLIME_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
join_colon_paths() {
  local result=""
  local path
  for path in "$@"; do
    [ -n "${path}" ] || continue
    if [ -n "${result}" ]; then
      result="${result}:${path}"
    else
      result="${path}"
    fi
  done
  printf '%s\n' "${result}"
}

SLIME_INCLUDE_PATH="${SLIME_ENV}/include"
if [ "$(readlink -f "${SLIME_INCLUDE_PATH}" 2>/dev/null || printf '%s\n' "${SLIME_INCLUDE_PATH}")" = "/usr/include" ]; then
  SLIME_INCLUDE_PATH=""
fi
export CPATH="$(join_colon_paths "${CUDA_HOME}/include" "${SLIME_INCLUDE_PATH}" "${CPATH:-}")"
export C_INCLUDE_PATH="$(join_colon_paths "${CUDA_HOME}/include" "${SLIME_INCLUDE_PATH}" "${C_INCLUDE_PATH:-}")"
export CPLUS_INCLUDE_PATH="$(join_colon_paths "${CUDA_HOME}/include" "${SLIME_INCLUDE_PATH}" "${CPLUS_INCLUDE_PATH:-}")"
export LIBRARY_PATH="$(build_slime_library_path "${LIBRARY_PATH:-}")"
export LD_LIBRARY_PATH="$(build_slime_library_path "${LD_LIBRARY_PATH:-}")"

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
case "${NO_LOAD_OPTIM:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    CKPT_ARGS+=(--no-load-optim)
    ;;
esac
case "${NO_LOAD_RNG:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    CKPT_ARGS+=(--no-load-rng)
    ;;
esac
case "${FINETUNE:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    CKPT_ARGS+=(--finetune)
    ;;
esac
case "${ASYNC_SAVE:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    CKPT_ARGS+=(--async-save)
    ;;
esac
case "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    CKPT_ARGS+=(--use-checkpoint-opt-param-scheduler)
    ;;
esac
case "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    CKPT_ARGS+=(--override-opt-param-scheduler)
    ;;
esac
USE_KL_LOSS=${USE_KL_LOSS:-0}
KL_LOSS_ENABLED=0
if is_true_value "${USE_KL_LOSS}"; then
  KL_LOSS_ENABLED=1
elif is_nonzero_number "${KL_LOSS_COEF:-0.00}"; then
  KL_LOSS_ENABLED=1
else
  kl_loss_status=$?
  if [ "${kl_loss_status}" -eq 2 ]; then
    echo "Invalid KL_LOSS_COEF: ${KL_LOSS_COEF:-0.00}" >&2
    exit 1
  fi
fi
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
if [ "${KL_LOSS_ENABLED}" = "1" ]; then
  CKPT_ARGS+=(--ref-load "${REF_LOAD_DIR:-${LOAD_DIR}}")
fi
CKPT_ARGS+=(--load "${LOAD_DIR}")

resolve_megatron_role_config() {
  [ -n "${MEGATRON_CONFIG_PATH:-}" ] || return 0

  local source_path resolved_path
  source_path=$(resolve_repo_path "${MEGATRON_CONFIG_PATH}")
  [ -f "${source_path}" ] || { echo "Missing Megatron role config: ${source_path}" >&2; exit 1; }

  ACTOR_SAVE_DIR=${ACTOR_SAVE_DIR:-${SAVE_DIR}/actor}
  CRITIC_SAVE_DIR=${CRITIC_SAVE_DIR:-${SAVE_DIR}/critic}
  resolved_path="${LOG_DIR}/resolved_megatron_config.yaml"
  export MEGATRON_CONFIG_SOURCE="${source_path}"
  export MEGATRON_CONFIG_RESOLVED="${resolved_path}"
  export ACTOR_SAVE_DIR CRITIC_SAVE_DIR

  "${SLIME_PYTHON}" - <<'PY'
import os
from pathlib import Path

import yaml

source = Path(os.environ["MEGATRON_CONFIG_SOURCE"])
target = Path(os.environ["MEGATRON_CONFIG_RESOLVED"])
repo_dir = Path(os.environ["REPO_DIR"])

cfg = yaml.safe_load(source.read_text()) or {}
entries = cfg.get("megatron")
if not isinstance(entries, list):
    raise SystemExit(f"{source} must contain a top-level megatron list")

def role_entry(role):
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("role") == role]
    if len(matches) > 1:
        raise SystemExit(f"{source} has multiple megatron entries for role={role}")
    if matches:
        return matches[0]
    entry = {"name": "default", "role": role, "overrides": {}}
    entries.append(entry)
    return entry

def canonical(path):
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if not os.path.isabs(expanded):
        expanded = str(repo_dir / expanded)
    return os.path.normpath(os.path.abspath(expanded))

role_saves = {
    "actor": os.environ["ACTOR_SAVE_DIR"],
    "critic": os.environ["CRITIC_SAVE_DIR"],
}
for role, save_dir in role_saves.items():
    entry = role_entry(role)
    overrides = entry.get("overrides")
    if overrides is None:
        overrides = entry.get("args")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise SystemExit(f"{source} megatron role={role} overrides must be a mapping")
    overrides["save"] = save_dir
    entry["overrides"] = overrides
    entry.pop("args", None)

actor_save = canonical(role_saves["actor"])
critic_save = canonical(role_saves["critic"])
common = os.path.commonpath([actor_save, critic_save])
if common == actor_save or common == critic_save:
    raise SystemExit(
        "Actor and critic checkpoint directories must not overlap: "
        f"actor={actor_save} critic={critic_save}"
    )

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

  MEGATRON_CONFIG_PATH="${resolved_path}"
  export MEGATRON_CONFIG_PATH
  echo "Megatron role config: source=${source_path} resolved=${MEGATRON_CONFIG_PATH}"
  echo "Role checkpoint dirs: actor=${ACTOR_SAVE_DIR} critic=${CRITIC_SAVE_DIR}"
}

resolve_megatron_role_config

ROLLOUT_ARGS=(
   --env-server-url "${ENV_ROUTER_URL}"
   --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
   --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}"
   --custom-reward-post-process-path examples.agent_env.reward_post_process.post_process_rewards
   --custom-rollout-log-function-path "${CUSTOM_GENERATE_FUNCTION_PATH%.*}.log_rollout_data"
   --custom-eval-rollout-log-function-path "${CUSTOM_GENERATE_FUNCTION_PATH%.*}.log_eval_rollout_data"
   --custom-config-path "${CUSTOM_CONFIG_PATH}"
   --agent-env-reward-profile "${REWARD_PROFILE}"
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
AGENT_ENV_ARGS=()
if [ -n "${AGENT_ENV_CONFIG_OVERRIDES_JSON:-}" ]; then
  AGENT_ENV_ARGS+=(--agent-env-config-overrides "${AGENT_ENV_CONFIG_OVERRIDES_JSON}")
fi
case "${APPLY_CHAT_TEMPLATE:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    ROLLOUT_ARGS+=(--apply-chat-template)
    ;;
esac
if [ -n "${APPLY_CHAT_TEMPLATE_KWARGS:-}" ]; then
  ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi
if [ -n "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [ -n "${ROLLOUT_SAMPLE_FILTER_PATH:-}" ]; then
  ROLLOUT_ARGS+=(--rollout-sample-filter-path "${ROLLOUT_SAMPLE_FILTER_PATH}")
fi
if [ "${GROUP_RM}" = "1" ]; then
  ROLLOUT_ARGS+=(--group-rm)
fi
ROLLOUT_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")

EVAL_ARGS=()
if [ -n "${EVAL_CONFIG:-}" ]; then
  EVAL_ARGS+=(--eval-config "${EVAL_CONFIG}")
fi
if [ -n "${EVAL_PROMPT_DATA:-}" ]; then
  read -r -a EVAL_PROMPT_DATA_ARRAY <<< "${EVAL_PROMPT_DATA}"
  EVAL_ARGS+=(--eval-prompt-data "${EVAL_PROMPT_DATA_ARRAY[@]}")
fi
if [ -n "${EVAL_INTERVAL:-}" ]; then
  EVAL_ARGS+=(--eval-interval "${EVAL_INTERVAL}")
fi
if [ -n "${EVAL_FUNCTION_PATH:-}" ]; then
  EVAL_ARGS+=(--eval-function-path "${EVAL_FUNCTION_PATH}")
fi
if [ -n "${EVAL_MAX_RESPONSE_LEN:-}" ]; then
  EVAL_ARGS+=(--eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}")
fi
if [ -n "${EVAL_TEMPERATURE:-}" ]; then
  EVAL_ARGS+=(--eval-temperature "${EVAL_TEMPERATURE}")
fi
if [ -n "${EVAL_TOP_P:-}" ]; then
  EVAL_ARGS+=(--eval-top-p "${EVAL_TOP_P}")
fi
if [ -n "${EVAL_TOP_K:-}" ]; then
  EVAL_ARGS+=(--eval-top-k "${EVAL_TOP_K}")
fi
case "${SKIP_EVAL_BEFORE_TRAIN:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    EVAL_ARGS+=(--skip-eval-before-train)
    ;;
esac

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
   --value-clip "${VALUE_CLIP:-0.2}"
   --gamma "${GAMMA:-1.0}"
   --lambd "${LAMBD:-1.0}"
   --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS:-0}"
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
if [ "${KL_LOSS_ENABLED}" = "1" ]; then
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
if [ -n "${LR_DECAY_ITERS:-}" ]; then
  OPTIMIZER_ARGS+=(--lr-decay-iters "${LR_DECAY_ITERS}")
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE:-1}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.55}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-8}"
)

START_ARGS=()
if [ -n "${START_ROLLOUT_ID:-}" ]; then
  START_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi

MISC_ARGS=(
   --agent-env-train-loop "${AGENT_ENV_TRAIN_LOOP}"
   --num-steps "${NUM_STEPS:-100}"
   --log-interval 1
   --seed "${SEED:-42}"
   --actor-num-nodes "${ACTOR_NUM_NODES:-2}"
   --actor-num-gpus-per-node "${ACTOR_GPUS:-4}"
   --rollout-num-gpus "${ROLLOUT_GPUS:-8}"
   --num-gpus-per-node "${NUM_GPUS:-4}"
)
if [ -n "${MEGATRON_CONFIG_PATH:-}" ]; then
  MISC_ARGS+=(--megatron-config-path "${MEGATRON_CONFIG_PATH}")
fi

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
    "ROOT_DIR", "LOCAL_RUNTIME_DIR", "LOCAL_ENVS_DIR",
    "PYTHONPATH", "PYTHONNOUSERSITE", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_HOME",
    "PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
    "LD_LIBRARY_PATH", "RAY_ADDRESS",
    "SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
    "AGENT_ENV_DATA_DIR", "APPWORLD_ROOT", "RUN_ROOT", "LOG_DIR", "WANDB_DIR",
    "WANDB_RUNTIME", "WANDB_RUNTIME_RESOLVED", "WANDB_ENV", "WANDB_PYTHON", "WANDB_PYTHONPATH",
    "AGENT_ENV_ROLLOUT_DUMP_N", "AGENT_ENV_ROLLOUT_DUMP_DISCARD_N",
    "AGENT_ENV_ROLLOUT_DUMP_FORMAT_N",
    "AGENT_ENV_ROLLOUT_DUMP_TOTAL_N",
    "AGENT_ENV_ROLLOUT_DUMP_TRACE",
    "AGENT_ENV_MAX_CHECKPOINTS",
    "AGENT_ENV_ASYNC_MAX_INFLIGHT_GROUPS",
    "AGENT_ENV_GLM_PADDING_MIN_VALID_FRACTION", "AGENT_ENV_GLM_PADDING_MAX_SEEN_GROUPS",
    "AGENT_ENV_DYNAMIC_DROP_ZERO_STD",
    "AGENT_ENV_ROPD_TEACHER_INDEX_PATH", "AGENT_ENV_LUFFY_TEACHER_INDEX_PATH",
    "AGENT_ENV_ROPD_DUMP_N", "AGENT_ENV_ROPD_DUMP_TOTAL_N", "AGENT_ENV_ROPD_DUMP_DIR",
    "AGENT_ENV_CREDIT_DUMP_N", "AGENT_ENV_CREDIT_DUMP_TOTAL_N", "AGENT_ENV_CREDIT_DUMP_DIR",
    "AGENT_ENV_REWARD_GROUP_DUMP_N", "AGENT_ENV_REWARD_GROUP_DUMP_TOTAL_N", "AGENT_ENV_REWARD_GROUP_DUMP_DIR",
    "AGENT_ENV_ADVANTAGE_DUMP_N", "AGENT_ENV_ADVANTAGE_DUMP_TOTAL_N", "AGENT_ENV_ADVANTAGE_DUMP_DIR",
    "AGENT_ENV_ADVANTAGE_DUMP_TOKEN_IDS",
    "SAVE_DEBUG_TRAIN_DATA",
    "AUX_ENDPOINT_PROVIDER", "AUX_ENDPOINT_MODEL", "AUX_ENDPOINT_BASE_URL", "AUX_ENDPOINT_API_KEY_PATH",
    "AUX_ENDPOINT_TIMEOUT_S", "AUX_ENDPOINT_MAX_TOKENS", "AUX_ENDPOINT_TEMPERATURE", "AUX_ENDPOINT_TOP_P",
    "AUX_ENDPOINT_ENABLE_THINKING", "AUX_ENDPOINT_SEPARATE_REASONING", "AUX_ENDPOINT_REASONING_EFFORT",
    "LITELLM_LOCAL_MODEL_COST_MAP",
    "WANDB_BASE_URL", "WANDB_ENTITY", "WANDB_HTTP_TIMEOUT", "WANDB_INIT_TIMEOUT",
]
for key in os.environ.get("AUX_TRAIN_ENV_KEYS", "").replace(",", " ").split():
    if key and key not in keys:
        keys.append(key)
print(json.dumps({key: os.environ[key] for key in keys if key in os.environ and os.environ[key] != ""}))
PYH
)
fi

echo "Launching ${ENV_NAME} ${ADVANTAGE_ESTIMATOR:-grpo} with ${TRAIN_ENTRYPOINT}"
echo "Checkpoint options: save_interval=${SAVE_INTERVAL:-${NUM_STEPS}} no_save_optim=${NO_SAVE_OPTIM:-0} no_load_optim=${NO_LOAD_OPTIM:-0} async_save=${ASYNC_SAVE:-0}"
"${SLIME_PYTHON}" "${REPO_DIR}/${TRAIN_ENTRYPOINT}" \
   "${CKPT_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${MODEL_EXTRA_ARGS_ARRAY[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${AGENT_ENV_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   --train-env-vars "${TRAIN_ENV_VARS_JSON}" \
   "${GRPO_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${START_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}"
