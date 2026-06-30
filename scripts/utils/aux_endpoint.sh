#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}
ROOT_DIR=${ROOT_DIR:-$(cd "${REPO_DIR}/../.." && pwd -P)}
OPS_SCRIPTS_DIR=${OPS_SCRIPTS_DIR:-${ROOT_DIR}/scripts}
LOCAL_ENVS_DIR=${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}
LOG_DIR=${LOG_DIR:-/tmp/server-ops-runtime/logs}
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/slime_runtime.sh"
AUX_PYTHON=${AUX_PYTHON:-}
AUX_SERVE_PYTHON=${AUX_SERVE_PYTHON:-}
AUX_HELPER_PYTHON=${AUX_HELPER_PYTHON:-}

CMD=start
AUX_CONFIG=${AUX_CONFIG:-}
AUX_SPEC=${AUX_SPEC:-}
AUX_ENV_FILE=${AUX_ENV_FILE:-${LOG_DIR}/aux_endpoint.env}
DEFAULT_AUX_NODES_FILE=configs/nodes/agent_env_all.txt
AUX_NODES_FILE=${AUX_NODES_FILE:-${DEFAULT_AUX_NODES_FILE}}
AUX_NODE_INDICES=${AUX_NODE_INDICES:-}
AUX_NODE=${AUX_NODE:-}
AUX_PORT=${AUX_PORT:-}
AUX_GPUS=${AUX_GPUS:-}
AUX_TP=${AUX_TP:-}
AUX_PP=${AUX_PP:-}
AUX_SESSION=${AUX_SESSION:-}
AUX_MEM_FRACTION=${AUX_MEM_FRACTION:-}
AUX_GPU_MEMORY_UTILIZATION=${AUX_GPU_MEMORY_UTILIZATION:-}
AUX_MAX_MODEL_LEN=${AUX_MAX_MODEL_LEN:-}
AUX_REASONING_PARSER=${AUX_REASONING_PARSER:-}
AUX_TOOL_CALL_PARSER=${AUX_TOOL_CALL_PARSER:-}
AUX_SERVED_MODEL_NAME=${AUX_SERVED_MODEL_NAME:-}
AUX_VLLM_BIN=${AUX_VLLM_BIN:-}
AUX_VLLM_COMPILATION_CONFIG=${AUX_VLLM_COMPILATION_CONFIG:-}
AUX_BASE_URL=${AUX_BASE_URL:-}
AUX_API_KEY_PATH=${AUX_API_KEY_PATH:-}
AUX_TIMEOUT_S=${AUX_TIMEOUT_S:-}
AUX_MAX_TOKENS=${AUX_MAX_TOKENS:-}
AUX_TEMPERATURE=${AUX_TEMPERATURE:-}
AUX_TOP_P=${AUX_TOP_P:-}
AUX_ENABLE_THINKING=${AUX_ENABLE_THINKING:-}
AUX_SEPARATE_REASONING=${AUX_SEPARATE_REASONING:-}
AUX_REASONING_EFFORT=${AUX_REASONING_EFFORT:-}
AUX_EXTRA_SERVER_ARGS=${AUX_EXTRA_SERVER_ARGS:-}
BENCH_ON_AUX_START=${BENCH_ON_AUX_START:-}
GPU_MEMORY_CLEAR_THRESHOLD_MIB=${GPU_MEMORY_CLEAR_THRESHOLD_MIB:-1024}
GPU_MEMORY_CLEAR_TIMEOUT_S=${GPU_MEMORY_CLEAR_TIMEOUT_S:-120}
DRY_RUN=0

SSH_USER=${SSH_USER:-tiger}
SSH_PORT=${SSH_PORT:-10413}
if [ -z "${SSH_KEY:-}" ]; then
  if [ -f "${ROOT_DIR}/secrets/byte_id_rsa" ]; then
    SSH_KEY="${ROOT_DIR}/secrets/byte_id_rsa"
  else
    SSH_KEY="/home/${SSH_USER}/.ssh/byte_id_rsa"
  fi
fi
SSH_IPV6=${SSH_IPV6:-1}

usage() {
  cat <<'EOF'
Usage: aux_endpoint.sh [start|print|stop] --spec provider/model [options]

Starts or describes an OpenAI-compatible auxiliary model endpoint.

Providers:
  sglang/<model-or-path>         Start SGLang on an aux node.
  local/<model-or-path>          Backward-compatible alias for sglang.
  vllm/<model-or-path>           Start vLLM on an aux node.
  deepseek/<model>               Use DeepSeek API, default base https://api.deepseek.com.
  ark/<model>                    Use ByteDance Ark API, default base https://ark-cn-beijing.bytedance.net/api/v3.
  openai/<model>                 Use a generic OpenAI-compatible API.
  aicolate/<model>               Use an internal OpenAI-compatible API via AUX_BASE_URL.

Options:
  --config PATH                  Source an aux endpoint profile before parsing options.
  --env-file PATH                File to write endpoint shell variables.
  --nodes FILE                   Aux node list for local provider.
  --node HOST                    Aux node for local provider.
  --node-index INDEX             Select aux node by zero-based index from --nodes.
  --port PORT                    Local aux SGLang port.
  --gpus CSV                     Local aux CUDA_VISIBLE_DEVICES.
  --tp N                         Local aux tensor parallel size.
  --pp N                         Local aux pipeline parallel size.
  --max-model-len N              Local vLLM max model length.
  --gpu-memory-utilization X     Local vLLM GPU memory utilization.
  --served-model-name NAME       Model name exposed by the local endpoint.
  --model-path PATH              Override local model path.
  --base-url URL                 External API base URL.
  --api-key-path PATH            External API key path. Raw keys should not be passed.
  --dry-run
EOF
}

load_config_if_present() {
  local config=$1
  [ -n "${config}" ] || return 0
  if [[ "${config}" != /* ]]; then
    config="${REPO_DIR}/${config}"
  fi
  [ -f "${config}" ] || { echo "Missing aux config: ${config}" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "${config}"
}

for ((i = 1; i <= $#; i++)); do
  if [ "${!i}" = "--config" ]; then
    j=$((i + 1))
    if [ "${j}" -gt "$#" ]; then
      echo "--config requires a path" >&2
      exit 1
    fi
    AUX_CONFIG=${!j}
    break
  fi
done
load_config_if_present "${AUX_CONFIG}"
set_slime_runtime_defaults

while [ $# -gt 0 ]; do
  case "$1" in
    start|print|stop) CMD=$1; shift ;;
    --config) AUX_CONFIG=$2; shift 2 ;;
    --spec) AUX_SPEC=$2; shift 2 ;;
    --env-file) AUX_ENV_FILE=$2; shift 2 ;;
    --nodes) AUX_NODES_FILE=$2; shift 2 ;;
    --node) AUX_NODE=$2; shift 2 ;;
    --node-index|--node-indices|--nodes-index) AUX_NODE_INDICES=$2; shift 2 ;;
    --port) AUX_PORT=$2; shift 2 ;;
    --gpus) AUX_GPUS=$2; shift 2 ;;
    --tp) AUX_TP=$2; shift 2 ;;
    --pp) AUX_PP=$2; shift 2 ;;
    --max-model-len) AUX_MAX_MODEL_LEN=$2; shift 2 ;;
    --gpu-memory-utilization) AUX_GPU_MEMORY_UTILIZATION=$2; shift 2 ;;
    --served-model-name) AUX_SERVED_MODEL_NAME=$2; shift 2 ;;
    --model-path) AUX_MODEL_PATH=$2; shift 2 ;;
    --base-url) AUX_BASE_URL=$2; shift 2 ;;
    --api-key-path) AUX_API_KEY_PATH=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

default_aux_serve_python() {
  if [ -n "${SLIME_ENV:-}" ]; then
    printf '%s\n' "${SLIME_PYTHON:-${SLIME_ENV}/bin/python}"
    return 0
  fi

  case "${SLIME_RUNTIME}" in
    image)
      printf '%s\n' "${SLIME_IMAGE_PYTHON}"
      ;;
    pack|conda_pack|conda-pack)
      printf '%s\n' "${SLIME_PACK_PATH}/bin/python"
      ;;
    auto)
      if [ -x "${SLIME_IMAGE_PYTHON}" ] && slime_python_imports_slime "${SLIME_IMAGE_PYTHON}"; then
        printf '%s\n' "${SLIME_IMAGE_PYTHON}"
      else
        printf '%s\n' "${SLIME_PACK_PATH}/bin/python"
      fi
      ;;
    *)
      echo "Unsupported SLIME_RUNTIME=${SLIME_RUNTIME}; expected auto, image, or pack" >&2
      exit 1
      ;;
  esac
}

apply_aux_defaults() {
  AUX_SPEC=${AUX_SPEC:-local/Qwen3.5-122B-A10B}
  if [ -z "${AUX_NODE}" ] && [ -z "${AUX_NODE_INDICES}" ] && [ "${AUX_NODES_FILE}" = "${DEFAULT_AUX_NODES_FILE}" ]; then
    AUX_NODE_INDICES=3
  fi
  if [ -z "${AUX_PYTHON}" ]; then
    AUX_PYTHON=$(default_aux_serve_python)
  fi
  AUX_SERVE_PYTHON=${AUX_SERVE_PYTHON:-${AUX_PYTHON}}
  if [ -z "${AUX_HELPER_PYTHON}" ]; then
    if [ -x "${AUX_PYTHON}" ]; then
      AUX_HELPER_PYTHON="${AUX_PYTHON}"
    elif command -v python3 >/dev/null 2>&1; then
      AUX_HELPER_PYTHON=$(command -v python3)
    else
      AUX_HELPER_PYTHON=python
    fi
  fi
  AUX_PORT=${AUX_PORT:-18080}
  AUX_GPUS=${AUX_GPUS:-0,1,2,3,4,5,6,7}
  AUX_PP=${AUX_PP:-1}
  AUX_SESSION=${AUX_SESSION:-agent_env_aux_endpoint}
  AUX_MEM_FRACTION=${AUX_MEM_FRACTION:-0.65}
  AUX_GPU_MEMORY_UTILIZATION=${AUX_GPU_MEMORY_UTILIZATION:-0.90}
  AUX_MAX_MODEL_LEN=${AUX_MAX_MODEL_LEN:-4096}
  AUX_SERVED_MODEL_NAME=${AUX_SERVED_MODEL_NAME:-aux-model}
  AUX_VLLM_COMPILATION_CONFIG=${AUX_VLLM_COMPILATION_CONFIG:-'{"mode":3}'}
  AUX_TIMEOUT_S=${AUX_TIMEOUT_S:-120}
  AUX_MAX_TOKENS=${AUX_MAX_TOKENS:-512}
  AUX_TEMPERATURE=${AUX_TEMPERATURE:-0.0}
  AUX_TOP_P=${AUX_TOP_P:-1.0}
  AUX_ENABLE_THINKING=${AUX_ENABLE_THINKING:-0}
  AUX_SEPARATE_REASONING=${AUX_SEPARATE_REASONING:-1}
  BENCH_ON_AUX_START=${BENCH_ON_AUX_START:-1}
}

apply_aux_defaults

if [ "${CMD}" = "print" ]; then
  DRY_RUN=1
fi

resolve_path() {
  local path=$1
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${REPO_DIR}" "${path}"
  fi
}

http_host() {
  local node=$1
  if [[ "${node}" == *:* ]]; then
    printf '[%s]' "${node}"
  else
    printf '%s' "${node}"
  fi
}

is_current_node() {
  local node=$1
  [ "${node}" = "this" ] && return 0
  [ "${node}" = "$(hostname)" ] && return 0
  hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${node}" && return 0
  ip addr 2>/dev/null | grep -Fq "${node}" && return 0
  return 1
}

ssh_node() {
  local host=$1
  local remote_cmd=$2
  local encoded wrapped_cmd
  local args=()
  if [ "${SSH_IPV6}" = "1" ]; then
    args+=("-6")
  fi
  args+=(
    "-n"
    "-o" "BatchMode=yes"
    "-o" "StrictHostKeyChecking=no"
    "-o" "UserKnownHostsFile=/dev/null"
    "-o" "GlobalKnownHostsFile=/dev/null"
    "-o" "CheckHostIP=no"
    "-o" "IdentitiesOnly=yes"
    "-i" "${SSH_KEY}"
    "-p" "${SSH_PORT}"
    "${SSH_USER}@${host}"
  )
  encoded=$(printf '%s' "${remote_cmd}" | base64 | tr -d '\n')
  wrapped_cmd=$(printf "printf %%s '%s' | base64 -d | bash" "${encoded}")
  ssh "${args[@]}" "${wrapped_cmd}"
}

remote_query() {
  local host=$1
  local command=$2
  local attempt
  for attempt in 1 2 3; do
    if ssh_node "${host}" "${command}"; then
      return 0
    fi
    echo "Retry ${attempt}/3 failed for query on ${host}" >&2
    sleep 5
  done
  return 1
}

remote_first_ip() {
  local host=$1
  remote_query "${host}" "hostname -I | tr ' ' '\\n' | grep -m1 ."
}

remote_start_tmux() {
  local host=$1
  local session=$2
  local command=$3
  local remote_cmd
  remote_cmd=$(printf 'tmux kill-session -t %q 2>/dev/null || true; tmux new-session -d -s %q %q' "${session}" "${session}" "${command}")
  if [ "${DRY_RUN}" = "1" ]; then
    printf '+ ssh %s %q\n' "${host}" "${remote_cmd}"
    return 0
  fi
  remote_query "${host}" "${remote_cmd}"
}

health_url() {
  local base=$1
  if [[ "${base}" == */v1 ]]; then
    printf '%s/models' "${base}"
  else
    printf '%s/v1/models' "${base%/}"
  fi
}

wait_openai_models() {
  local host=$1
  local url=$2
  local py_code
  py_code="import json,urllib.request,time,sys; url='${url}'; ok=False
for _ in range(300):
    try:
        data=json.loads(urllib.request.urlopen(url,timeout=2).read().decode())
        if 'data' in data:
            print('ready', url); ok=True; break
    except Exception:
        pass
    time.sleep(2)
sys.exit(0 if ok else 1)"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ wait ${host} ${url}"
    return 0
  fi
  if is_current_node "${host}"; then
    "${AUX_HELPER_PYTHON}" -c "${py_code}"
  else
    remote_query "${host}" "$(printf '%q ' "${AUX_SERVE_PYTHON}" -c "${py_code}")"
  fi
}

infer_tool_call_parser_from_model() {
  local model_path=$1
  local engine=${2:-sglang}
  "${AUX_HELPER_PYTHON}" - <<PY
import json
from pathlib import Path

model_path = Path("${model_path}")
engine = "${engine}"
model_name = model_path.name.lower()
path = model_path / "tokenizer_config.json"
template = ""
try:
    template = str((json.loads(path.read_text()).get("chat_template") or ""))
except Exception:
    template = ""

if "minimax" in model_name and "m2" in model_name:
    print("minimax_m2" if engine == "vllm" else "minimax-m2")
elif "<function=" in template and "<parameter=" in template:
    print("qwen3_coder")
elif "<tool_call>" in template and '"name"' in template and '"arguments"' in template:
    print("qwen")
else:
    print("qwen")
PY
}

infer_reasoning_parser_from_model() {
  local model_path=$1
  local engine=${2:-sglang}
  "${AUX_HELPER_PYTHON}" - <<PY
from pathlib import Path

engine = "${engine}"
model_name = Path("${model_path}").name.lower()
if "minimax" in model_name and "m2" in model_name:
    print("minimax_m2" if engine == "vllm" else "minimax-append-think")
else:
    print("qwen3")
PY
}

infer_tp_from_model() {
  local model_path=$1
  "${AUX_HELPER_PYTHON}" - <<PY
from pathlib import Path

model_name = Path("${model_path}").name.lower()
if "minimax" in model_name and "m2" in model_name:
    print("4")
else:
    print("8")
PY
}

require_aux_serve_python() {
  local node=$1
  [ "${DRY_RUN}" = "0" ] || return 0
  if is_current_node "${node}"; then
    [ -x "${AUX_SERVE_PYTHON}" ] || {
      echo "Missing aux serve python: ${AUX_SERVE_PYTHON}" >&2
      exit 1
    }
  else
    remote_query "${node}" "$(printf 'test -x %q || { echo %q >&2; exit 1; }' \
      "${AUX_SERVE_PYTHON}" "Missing aux serve python: ${AUX_SERVE_PYTHON}")"
  fi
}

write_env_file() {
  local provider=$1
  local model=$2
  local base_url=$3
  local api_key_path=$4
  local started_local=$5
  local nodes_file=${6:-}
  local node_indices=${7:-}
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ write ${AUX_ENV_FILE}"
    return 0
  fi
  mkdir -p "$(dirname "${AUX_ENV_FILE}")"
  {
    printf 'AUX_ENDPOINT_PROVIDER=%q\n' "${provider}"
    printf 'AUX_ENDPOINT_MODEL=%q\n' "${model}"
    printf 'AUX_ENDPOINT_BASE_URL=%q\n' "${base_url}"
    printf 'AUX_ENDPOINT_API_KEY_PATH=%q\n' "${api_key_path}"
    printf 'AUX_ENDPOINT_TIMEOUT_S=%q\n' "${AUX_TIMEOUT_S}"
    printf 'AUX_ENDPOINT_MAX_TOKENS=%q\n' "${AUX_MAX_TOKENS}"
    printf 'AUX_ENDPOINT_TEMPERATURE=%q\n' "${AUX_TEMPERATURE}"
    printf 'AUX_ENDPOINT_TOP_P=%q\n' "${AUX_TOP_P}"
    printf 'AUX_ENDPOINT_ENABLE_THINKING=%q\n' "${AUX_ENABLE_THINKING}"
    printf 'AUX_ENDPOINT_SEPARATE_REASONING=%q\n' "${AUX_SEPARATE_REASONING}"
    printf 'AUX_ENDPOINT_REASONING_EFFORT=%q\n' "${AUX_REASONING_EFFORT}"
    printf 'AUX_ENDPOINT_STARTED_LOCAL=%q\n' "${started_local}"
    printf 'AUX_ENDPOINT_NODES_FILE=%q\n' "${nodes_file}"
    printf 'AUX_ENDPOINT_NODE_INDICES=%q\n' "${node_indices}"
    printf 'AUX_ENDPOINT_SESSION=%q\n' "${AUX_SESSION}"
  } > "${AUX_ENV_FILE}"
  chmod 600 "${AUX_ENV_FILE}"
  echo "aux endpoint env: ${AUX_ENV_FILE}"
  echo "aux endpoint: provider=${provider} model=${model} base_url=${base_url}"
}

split_spec() {
  if [[ "${AUX_SPEC}" == */* ]]; then
    AUX_PROVIDER=${AUX_SPEC%%/*}
    AUX_MODEL_NAME=${AUX_SPEC#*/}
  else
    AUX_PROVIDER=local
    AUX_MODEL_NAME=${AUX_SPEC}
  fi
  AUX_PROVIDER=$(printf '%s' "${AUX_PROVIDER}" | tr '[:upper:]' '[:lower:]')
  if [ "${AUX_PROVIDER}" = "local" ]; then
    AUX_PROVIDER=sglang
  fi
  if [ -z "${AUX_MODEL_NAME}" ]; then
    echo "Invalid AUX_SPEC=${AUX_SPEC}; expected provider/model" >&2
    exit 1
  fi
}

start_external_endpoint() {
  local provider=$1
  local model=$2
  local base_url api_key_path
  case "${provider}" in
    deepseek)
      base_url=${AUX_BASE_URL:-https://api.deepseek.com}
      api_key_path=${AUX_API_KEY_PATH:-${ROOT_DIR}/secrets/deepseek_api_key}
      ;;
    ark)
      base_url=${AUX_BASE_URL:-https://ark-cn-beijing.bytedance.net/api/v3}
      api_key_path=${AUX_API_KEY_PATH:-${ROOT_DIR}/secrets/ark_api_key}
      ;;
    openai)
      base_url=${AUX_BASE_URL:-}
      api_key_path=${AUX_API_KEY_PATH:-${ROOT_DIR}/secrets/openai_api_key}
      ;;
    aicolate)
      base_url=${AUX_BASE_URL:-${AICOLATE_BASE_URL:-}}
      api_key_path=${AUX_API_KEY_PATH:-${ROOT_DIR}/secrets/aicolate_api_key}
      ;;
    *)
      echo "Unsupported external provider: ${provider}" >&2
      exit 1
      ;;
  esac
  [ -n "${base_url}" ] || { echo "AUX_BASE_URL is required for provider=${provider}" >&2; exit 1; }
  [ -n "${api_key_path}" ] || { echo "AUX_API_KEY_PATH is required for provider=${provider}" >&2; exit 1; }
  if [ "${DRY_RUN}" = "0" ]; then
    local key_path
    IFS=',' read -ra key_paths <<< "${api_key_path}"
    for key_path in "${key_paths[@]}"; do
      key_path=$(printf '%s' "${key_path}" | xargs)
      if [ -n "${key_path}" ] && [ ! -f "${key_path}" ]; then
        echo "API key path does not exist: ${key_path}" >&2
        exit 1
      fi
    done
  fi
  write_env_file "${provider}" "${model}" "${base_url}" "${api_key_path}" 0 ""
}

resolve_local_model_path() {
  local model_path=${AUX_MODEL_PATH:-}
  if [ -z "${model_path}" ]; then
    if [[ "${AUX_MODEL_NAME}" = /* ]]; then
      model_path=${AUX_MODEL_NAME}
    else
      model_path="${ROOT_DIR}/models/${AUX_MODEL_NAME}"
    fi
  fi
  printf '%s\n' "${model_path}"
}

resolve_aux_node() {
  local nodes_file=$1
  if [ -z "${AUX_NODE}" ]; then
    [ -f "${nodes_file}" ] || { echo "Missing aux nodes file: ${nodes_file}" >&2; exit 1; }
    AUX_NODE=$(selected_aux_nodes "${nodes_file}" "${AUX_NODE_INDICES}" | head -n 1)
    [ -n "${AUX_NODE}" ] || {
      echo "No aux node selected from ${nodes_file} with AUX_NODE_INDICES=${AUX_NODE_INDICES}" >&2
      exit 1
    }
  fi
  printf '%s\n' "${AUX_NODE}"
}

selected_aux_nodes() {
  local nodes_file=$1
  local selector=${2:-}
  if [ -z "${selector}" ]; then
    awk 'NF && $1 !~ /^#/ {print $1}' "${nodes_file}"
  else
    awk -v sel="${selector}" '
      BEGIN {
        n = split(sel, parts, ",")
        for (i = 1; i <= n; i++) wanted[parts[i]] = 1
        idx = 0
      }
      NF && $1 !~ /^#/ {
        if (idx in wanted) print $1
        idx++
      }
    ' "${nodes_file}"
  fi
}

stop_bench_for_aux_node() {
  local nodes_file=$1
  [ "${DRY_RUN}" = "0" ] || return 0
  [ "${BENCH_ON_AUX_START}" = "1" ] || return 0
  [ -f "${OPS_SCRIPTS_DIR}/run_bench.sh" ] || return 0
  local selected_count
  selected_count=$(selected_aux_nodes "${nodes_file}" "${AUX_NODE_INDICES}" | wc -l | tr -d ' ')
  if [ -z "${AUX_NODE_INDICES}" ] && [ "${selected_count}" -ne 1 ]; then
    echo "Refusing to stop GPU bench for aux without a single selected node: ${nodes_file}" >&2
    return 1
  fi
  echo "Stopping GPU bench for aux node"
  if [ -n "${AUX_NODE_INDICES}" ]; then
    bash "${OPS_SCRIPTS_DIR}/run_bench.sh" stop --nodes "${nodes_file}" --node "${AUX_NODE_INDICES}" || true
  else
    bash "${OPS_SCRIPTS_DIR}/run_bench.sh" stop --nodes "${nodes_file}" || true
  fi
}

cleanup_aux_runtime_cmd() {
  cat <<EOF
tmux kill-session -t ${AUX_SESSION@Q} 2>/dev/null || true
pkill -f '[m]lf_torch_bench.py' 2>/dev/null || true
pkill -f '[v]llm serve' 2>/dev/null || true
pkill -f '[V]LLM::Worker' 2>/dev/null || true
pkill -f '[E]ngineCore' 2>/dev/null || true
pkill -f '[A]PIServer' 2>/dev/null || true
pkill -f '[s]glang.launch_server' 2>/dev/null || true
sleep 2
if command -v nvidia-smi >/dev/null 2>&1; then
  threshold=${GPU_MEMORY_CLEAR_THRESHOLD_MIB}
  deadline=\$((\$(date +%s) + ${GPU_MEMORY_CLEAR_TIMEOUT_S}))
  while true; do
    over=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk -v t="\$threshold" 'BEGIN { over=0 } { gsub(/[[:space:]]/, "", \$1); if (\$1 + 0 > t) over=1 } END { print over }')
    [ "\${over:-0}" = "0" ] && break
    [ "\$(date +%s)" -ge "\$deadline" ] && break
    sleep 2
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
fi
EOF
}

cleanup_aux_runtime() {
  local node=$1
  local cmd
  cmd=$(cleanup_aux_runtime_cmd)
  if [ "${DRY_RUN}" = "1" ]; then
    printf '+ cleanup aux runtime on %s\n' "${node}"
    return 0
  fi
  if is_current_node "${node}"; then
    bash -lc "${cmd}" || true
  else
    remote_query "${node}" "${cmd}" || true
  fi
}

start_sglang_endpoint() {
  local nodes_file model_path node node_addr base_url tool_parser reasoning_parser tp log serve_cmd
  nodes_file=$(resolve_path "${AUX_NODES_FILE}")
  model_path=$(resolve_local_model_path)
  node=$(resolve_aux_node "${nodes_file}")
  require_aux_serve_python "${node}"
  stop_bench_for_aux_node "${nodes_file}"
  cleanup_aux_runtime "${node}"
  if [ "${DRY_RUN}" = "1" ]; then
    node_addr=${node}
  elif is_current_node "${node}"; then
    node_addr=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  else
    node_addr=$(remote_first_ip "${node}")
  fi
  base_url="http://$(http_host "${node_addr}"):${AUX_PORT}/v1"
  tool_parser="${AUX_TOOL_CALL_PARSER:-$(infer_tool_call_parser_from_model "${model_path}" sglang)}"
  reasoning_parser="${AUX_REASONING_PARSER:-$(infer_reasoning_parser_from_model "${model_path}" sglang)}"
  tp="${AUX_TP:-$(infer_tp_from_model "${model_path}")}"
  log="${LOG_DIR}/aux_endpoint.log"
  serve_cmd=$(printf 'cd %q && mkdir -p %q && export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q no_proxy=%q NO_PROXY=%q && %q -m sglang.launch_server --model-path %q --served-model-name %q --host 0.0.0.0 --port %q --tp-size %q --mem-fraction-static %q --reasoning-parser %q --tool-call-parser %q --trust-remote-code > %q 2>&1' \
    "${REPO_DIR}" "${LOG_DIR}" "${AUX_GPUS}" "${REPO_DIR}" \
    "${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}" "${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}" \
    "${AUX_SERVE_PYTHON}" "${model_path}" "${AUX_SERVED_MODEL_NAME}" "${AUX_PORT}" "${tp}" "${AUX_MEM_FRACTION}" \
    "${reasoning_parser}" "${tool_parser}" "${log}")
  if [ -n "${AUX_EXTRA_SERVER_ARGS}" ]; then
    serve_cmd="${serve_cmd} ${AUX_EXTRA_SERVER_ARGS}"
  fi
  if is_current_node "${node}"; then
    if [ "${DRY_RUN}" = "1" ]; then
      printf '+ tmux new-session -d -s %q %q\n' "${AUX_SESSION}" "${serve_cmd}"
      wait_openai_models "${node}" "$(health_url "${base_url}")"
      write_env_file "sglang" "${AUX_SERVED_MODEL_NAME}" "${base_url}" "" 1 "${nodes_file}" "${AUX_NODE_INDICES}"
      echo "aux endpoint log: ${node}:${log}"
      return 0
    fi
    tmux kill-session -t "${AUX_SESSION}" 2>/dev/null || true
    tmux new-session -d -s "${AUX_SESSION}" "${serve_cmd}"
  else
    remote_start_tmux "${node}" "${AUX_SESSION}" "${serve_cmd}"
  fi
  wait_openai_models "${node}" "$(health_url "${base_url}")"
  write_env_file "sglang" "${AUX_SERVED_MODEL_NAME}" "${base_url}" "" 1 "${nodes_file}" "${AUX_NODE_INDICES}"
  echo "aux endpoint log: ${node}:${log}"
}

start_vllm_endpoint() {
  local nodes_file model_path node node_addr base_url tool_parser reasoning_parser tp log serve_cmd vllm_bin
  nodes_file=$(resolve_path "${AUX_NODES_FILE}")
  model_path=$(resolve_local_model_path)
  node=$(resolve_aux_node "${nodes_file}")
  require_aux_serve_python "${node}"
  stop_bench_for_aux_node "${nodes_file}"
  cleanup_aux_runtime "${node}"
  if [ "${DRY_RUN}" = "1" ]; then
    node_addr=${node}
  elif is_current_node "${node}"; then
    node_addr=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  else
    node_addr=$(remote_first_ip "${node}")
  fi
  base_url="http://$(http_host "${node_addr}"):${AUX_PORT}/v1"
  tool_parser="${AUX_TOOL_CALL_PARSER:-$(infer_tool_call_parser_from_model "${model_path}" vllm)}"
  reasoning_parser="${AUX_REASONING_PARSER:-$(infer_reasoning_parser_from_model "${model_path}" vllm)}"
  tp="${AUX_TP:-$(infer_tp_from_model "${model_path}")}"
  log="${LOG_DIR}/aux_endpoint.log"
  vllm_bin="${AUX_VLLM_BIN:-$(dirname "${AUX_SERVE_PYTHON}")/vllm}"
  serve_cmd=$(printf 'cd %q && mkdir -p %q && export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=%q VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=%q no_proxy=%q NO_PROXY=%q && %q serve %q --served-model-name %q --trust-remote-code --tensor-parallel-size %q --pipeline-parallel-size %q --host 0.0.0.0 --port %q --max-model-len %q --gpu-memory-utilization %q --enable-auto-tool-choice --tool-call-parser %q --reasoning-parser %q --compilation-config %q > %q 2>&1' \
    "${REPO_DIR}" "${LOG_DIR}" "${AUX_GPUS}" "${REPO_DIR}" \
    "${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}" "${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}" \
    "${vllm_bin}" "${model_path}" "${AUX_SERVED_MODEL_NAME}" "${tp}" "${AUX_PP}" "${AUX_PORT}" \
    "${AUX_MAX_MODEL_LEN}" "${AUX_GPU_MEMORY_UTILIZATION}" "${tool_parser}" "${reasoning_parser}" "${AUX_VLLM_COMPILATION_CONFIG}" "${log}")
  if [ -n "${AUX_EXTRA_SERVER_ARGS}" ]; then
    serve_cmd="${serve_cmd} ${AUX_EXTRA_SERVER_ARGS}"
  fi
  if is_current_node "${node}"; then
    if [ "${DRY_RUN}" = "1" ]; then
      printf '+ tmux new-session -d -s %q %q\n' "${AUX_SESSION}" "${serve_cmd}"
      wait_openai_models "${node}" "$(health_url "${base_url}")"
      write_env_file "vllm" "${AUX_SERVED_MODEL_NAME}" "${base_url}" "" 1 "${nodes_file}" "${AUX_NODE_INDICES}"
      echo "aux endpoint log: ${node}:${log}"
      return 0
    fi
    tmux kill-session -t "${AUX_SESSION}" 2>/dev/null || true
    tmux new-session -d -s "${AUX_SESSION}" "${serve_cmd}"
  else
    remote_start_tmux "${node}" "${AUX_SESSION}" "${serve_cmd}"
  fi
  wait_openai_models "${node}" "$(health_url "${base_url}")"
  write_env_file "vllm" "${AUX_SERVED_MODEL_NAME}" "${base_url}" "" 1 "${nodes_file}" "${AUX_NODE_INDICES}"
  echo "aux endpoint log: ${node}:${log}"
}

stop_local_endpoint() {
  if [ -f "${AUX_ENV_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${AUX_ENV_FILE}"
  fi
  local nodes_file=${AUX_ENDPOINT_NODES_FILE:-}
  local node_indices=${AUX_ENDPOINT_NODE_INDICES:-${AUX_NODE_INDICES:-}}
  local session=${AUX_ENDPOINT_SESSION:-${AUX_SESSION}}
  [ -z "${nodes_file}" ] || nodes_file=$(resolve_path "${nodes_file}")
  if [ -z "${nodes_file}" ] || [ ! -f "${nodes_file}" ]; then
    nodes_file=$(resolve_path "${AUX_NODES_FILE}")
  fi
  [ -f "${nodes_file}" ] || return 0
  local selected_count
  selected_count=$(selected_aux_nodes "${nodes_file}" "${node_indices}" | wc -l | tr -d ' ')
  if [ -z "${node_indices}" ] && [ "${selected_count}" -ne 1 ]; then
    echo "Refusing to stop aux endpoint across multiple nodes without AUX_ENDPOINT_NODE_INDICES" >&2
    return 1
  fi
  while IFS= read -r node; do
    [ -z "${node}" ] && continue
    if is_current_node "${node}"; then
      AUX_SESSION="${session}" cleanup_aux_runtime "${node}"
    else
      AUX_SESSION="${session}" cleanup_aux_runtime "${node}"
    fi
  done < <(selected_aux_nodes "${nodes_file}" "${node_indices}")
}

split_spec

case "${CMD}" in
  start|print)
    case "${AUX_PROVIDER}" in
      sglang) start_sglang_endpoint ;;
      vllm) start_vllm_endpoint ;;
      deepseek|ark|openai|aicolate) start_external_endpoint "${AUX_PROVIDER}" "${AUX_MODEL_NAME}" ;;
      *) echo "Unsupported AUX provider: ${AUX_PROVIDER}" >&2; exit 1 ;;
    esac
    ;;
  stop)
    stop_local_endpoint
    ;;
esac
