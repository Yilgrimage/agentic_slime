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
LOCAL_RUNTIME_DIR=${LOCAL_RUNTIME_DIR:-/tmp/server-ops-runtime}
WANDB_SECRET_FILE=${WANDB_SECRET_FILE:-${ROOT_DIR}/secrets/wandb.env}
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/slime_runtime.sh"

RUN_PROFILE=${RUN_PROFILE:-}
INTERNAL_ROLE=
RESOLVED_CONFIG=
DRY_RUN=0

ENV_NAME=${ENV_NAME:-}
ENV_CONFIG=${ENV_CONFIG:-}
REWARD_PROFILE=${REWARD_PROFILE:-}
MODEL_PROFILE=${MODEL_PROFILE:-}
TRAIN_PROFILE=${TRAIN_PROFILE:-}
TRAIN_ADAPTER=${TRAIN_ADAPTER:-examples/agent_env/scripts/run_agent_env_train.sh}
TOPOLOGY_PROFILE=${TOPOLOGY_PROFILE:-}
AUX_PROFILE=${AUX_PROFILE:-}

NODES_FILE=${NODES_FILE:-}
AUX_NODES_FILE=${AUX_NODES_FILE:-}
NODE_INDICES=${NODE_INDICES:-}
AUX_NODE_INDICES=${AUX_NODE_INDICES:-}
ENV_PORT=${ENV_PORT:-}
ROUTER_PORT=${ROUTER_PORT:-}
RAY_PORT=${RAY_PORT:-6379}
RAY_CUDA_VISIBLE_DEVICES=${RAY_CUDA_VISIBLE_DEVICES:-}
NUM_GPUS_PER_NODE_FOR_RAY=${NUM_GPUS_PER_NODE_FOR_RAY:-}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-}
RAY_START_MAX_ATTEMPTS=${RAY_START_MAX_ATTEMPTS:-2}
RAY_HEAD_START_TIMEOUT_S=${RAY_HEAD_START_TIMEOUT_S:-90}

EXP_PROJECT=${EXP_PROJECT:-}
EXP_NAME=${EXP_NAME:-}
RUN_ROOT=${RUN_ROOT:-}
LOG_DIR=${LOG_DIR:-}
WANDB_DIR=${WANDB_DIR:-}
SAVE_DIR=${SAVE_DIR:-}

RESET_TRAIN_RUNTIME_ON_START=${RESET_TRAIN_RUNTIME_ON_START:-1}
BENCH_ON_TRAIN_EXIT=${BENCH_ON_TRAIN_EXIT:-1}
BENCH_ON_LAUNCH_FAILURE=${BENCH_ON_LAUNCH_FAILURE:-1}

SSH_USER=${SSH_USER:-tiger}
SSH_PORT=${SSH_PORT:-10413}
if [ -z "${SSH_KEY:-}" ]; then
  if [ -f "${ROOT_DIR}/secrets/byte_id_rsa" ]; then
    SSH_KEY="${ROOT_DIR}/secrets/byte_id_rsa"
  else
    SSH_KEY="/home/${SSH_USER}/.ssh/byte_id_rsa"
  fi
fi
SSH_KEY=${SSH_KEY/#\~/${HOME}}
SSH_IPV6=${SSH_IPV6:-1}
SSH_JUMP=${SSH_JUMP:-}

RESOLVED_TRAIN_EXTRA_KEYS=(
  ROOT_DIR LOCAL_RUNTIME_DIR LOCAL_ENVS_DIR REPO_DIR RUN_ROOT LOG_DIR WANDB_DIR SAVE_DIR
  ENV_CONFIG REWARD_PROFILE CUSTOM_CONFIG_PATH MODEL_DIR TORCH_DIST_DIR LOAD_DIR REF_LOAD_DIR
  AGENT_ENV_DATA_DIR
  NO_LOAD_OPTIM NO_LOAD_RNG FINETUNE
  SOCKET_IFNAME NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME
  SLIME_RUNTIME SLIME_ENV SLIME_PYTHON SLIME_PACK_NAME SLIME_PACK_PATH SLIME_IMAGE_ENV SLIME_IMAGE_PYTHON
  MEGATRON_PATH MEGATRON_IMAGE_PATH WANDB_SECRET_FILE
  WANDB_RUNTIME WANDB_PACK_NAME WANDB_PACK_PATH
  EVAL_CONFIG EVAL_INTERVAL EVAL_FUNCTION_PATH EVAL_PROMPT_DATA EVAL_MAX_RESPONSE_LEN
  EVAL_TEMPERATURE EVAL_TOP_P EVAL_TOP_K SKIP_EVAL_BEFORE_TRAIN
  LR_DECAY_ITERS
  AGENT_ENV_ROLLOUT_DUMP_N AGENT_ENV_ROLLOUT_DUMP_DISCARD_N AGENT_ENV_ROLLOUT_DUMP_FORMAT_N
  AGENT_ENV_ROLLOUT_DUMP_TRACE
  AGENT_ENV_ROPD_DUMP_N AGENT_ENV_ROPD_DUMP_DIR
)

RESOLVED_LAUNCH_KEYS=(
  ROOT_DIR REPO_DIR OPS_SCRIPTS_DIR LOCAL_ENVS_DIR LOCAL_RUNTIME_DIR WANDB_SECRET_FILE
  SLIME_RUNTIME SLIME_ENV SLIME_PYTHON SLIME_PACK_NAME SLIME_PACK_PATH SLIME_IMAGE_ENV SLIME_IMAGE_PYTHON
  MEGATRON_PATH MEGATRON_IMAGE_PATH
  WANDB_RUNTIME WANDB_PACK_NAME WANDB_PACK_PATH
  ENV_NAME ENV_CONFIG REWARD_PROFILE AGENT_ENV_DATA_DIR MODEL_PROFILE TRAIN_PROFILE TRAIN_ADAPTER RESOLVED_TRAIN_ENV
  NODES_FILE NODE_INDICES AUX_NODES_FILE AUX_NODE_INDICES AUX_ENV_FILE ENV_PORT ROUTER_PORT RAY_PORT
  RAY_CUDA_VISIBLE_DEVICES NUM_GPUS_PER_NODE_FOR_RAY RAY_MIN_WORKER_PORT RAY_MAX_WORKER_PORT
  RAY_START_MAX_ATTEMPTS RAY_HEAD_START_TIMEOUT_S
  SOCKET_IFNAME NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME
  RUN_ROOT LOG_DIR WANDB_DIR SAVE_DIR EXP_PROJECT EXP_NAME
  RESET_TRAIN_RUNTIME_ON_START BENCH_ON_TRAIN_EXIT BENCH_ON_LAUNCH_FAILURE
  SSH_USER SSH_PORT SSH_KEY SSH_IPV6 SSH_JUMP
  AUX_ENDPOINT_PROVIDER AUX_ENDPOINT_MODEL AUX_ENDPOINT_BASE_URL AUX_ENDPOINT_API_KEY_PATH
  AUX_ENDPOINT_TIMEOUT_S AUX_ENDPOINT_MAX_TOKENS AUX_ENDPOINT_TEMPERATURE AUX_ENDPOINT_TOP_P
  AUX_ENDPOINT_ENABLE_THINKING AUX_ENDPOINT_SEPARATE_REASONING AUX_ENDPOINT_REASONING_EFFORT
  AUX_ENDPOINT_STARTED_LOCAL AUX_ENDPOINT_NODES_FILE AUX_ENDPOINT_SESSION
)

AUX_PROFILE_APPEND_KEYS=(
  ROOT_DIR OPS_SCRIPTS_DIR LOCAL_ENVS_DIR REPO_DIR LOG_DIR AUX_ENV_FILE AUX_NODES_FILE AUX_NODE_INDICES
  SLIME_RUNTIME SLIME_ENV SLIME_PYTHON SLIME_PACK_NAME SLIME_PACK_PATH SLIME_IMAGE_ENV SLIME_IMAGE_PYTHON
)

usage() {
  cat <<'EOF'
Usage:
  launch_agentic_training.sh <run-profile.env> [--dry-run]
  launch_agentic_training.sh --internal-role head --resolved RUN_ROOT/logs/resolved_launch.env
  launch_agentic_training.sh --internal-role worker --resolved RUN_ROOT/logs/resolved_launch.env --head-address HOST

The public entrypoint resolves profiles once; internal roles consume resolved_launch.env.
EOF
}

if [ $# -gt 0 ] && [[ "${1}" != -* ]]; then
  RUN_PROFILE=$1
  shift
fi

HEAD_ADDRESS=${HEAD_ADDRESS:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --internal-role) INTERNAL_ROLE=$2; shift 2 ;;
    --resolved) RESOLVED_CONFIG=$2; shift 2 ;;
    --head-address) HEAD_ADDRESS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

configure_internal_no_proxy() {
  local additions existing host_ips
  additions="localhost,127.0.0.1,0.0.0.0,::1"
  if [ -n "${HEAD_ADDRESS:-}" ]; then
    additions="${additions},${HEAD_ADDRESS}"
  fi
  host_ips="$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//' || true)"
  if [ -n "${host_ips}" ]; then
    additions="${additions},${host_ips}"
  fi
  existing="${no_proxy:-${NO_PROXY:-}}"
  if [ -n "${existing}" ]; then
    export no_proxy="${additions},${existing}"
  else
    export no_proxy="${additions}"
  fi
  export NO_PROXY="${no_proxy}"
}

configure_internal_no_proxy

resolve_path() {
  local path=${1:-}
  [ -n "${path}" ] || return 0
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${REPO_DIR}" "${path}"
  fi
}

quote_assign() {
  local key=$1 value=${2-}
  printf '%s=%q\n' "${key}" "${value}"
}

quote_assign_vars() {
  local key
  for key in "$@"; do
    quote_assign "${key}" "${!key-}"
  done
}

quote_export() {
  local key=$1 value=${2-}
  printf 'export %s=%q\n' "${key}" "${value}"
}

quote_export_vars() {
  local key
  for key in "$@"; do
    quote_export "${key}" "${!key-}"
  done
}

source_env_file() {
  local path=$1
  local label=${2:-profile}
  path=$(resolve_path "${path}")
  [ -f "${path}" ] || { echo "Missing ${label}: ${path}" >&2; exit 1; }
  set -a
  # shellcheck disable=SC1090
  source "${path}"
  set +a
}

profile_keys() {
  local path
  for path in "$@"; do
    [ -n "${path}" ] || continue
    path=$(resolve_path "${path}")
    [ -f "${path}" ] || continue
    sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "${path}"
  done
}

skip_resolved_train_key() {
  case "$1" in
    RUN_PROFILE|MODEL_PROFILE|TRAIN_PROFILE|TOPOLOGY_PROFILE|AUX_PROFILE|RESOLVED_*|RESOLVED_CONFIG)
      return 0 ;;
    NODES_FILE|NODE_INDICES|AUX_NODES_FILE|AUX_NODE_INDICES|ENV_PORT|ROUTER_PORT|RAY_PORT|RAY_*|NUM_GPUS_PER_NODE_FOR_RAY)
      return 0 ;;
    SSH_*|OPS_SCRIPTS_DIR|BENCH_*|RESET_TRAIN_RUNTIME_ON_START|DRY_RUN|INTERNAL_ROLE|HEAD_ADDRESS)
      return 0 ;;
    *)
      return 1 ;;
  esac
}

resolved_train_env_keys() {
  local path key
  for path in "$@"; do
    [ -n "${path}" ] || continue
    path=$(resolve_path "${path}")
    [ -f "${path}" ] || continue
    while read -r key; do
      [ -n "${key}" ] || continue
      skip_resolved_train_key "${key}" && continue
      printf '%s\n' "${key}"
    done < <(profile_keys "${path}")
  done
}

write_resolved_train_env() {
  local target=$1
  shift
  local profile_paths=("$@")
  local key value seen_keys=" "
  mkdir -p "$(dirname "${target}")"
  {
    printf '# Generated by %s\n' "$(basename "$0")"
    while read -r key; do
      printf '%s=%q\n' "${key}" "${!key-}"
      seen_keys="${seen_keys}${key} "
    done < <(resolved_train_env_keys "${profile_paths[@]}" | awk 'NF && !seen[$0]++')
    while read -r key; do
      case "${seen_keys}" in
        *" ${key} "*) continue ;;
      esac
      [[ -v ${key} ]] || continue
      value=${!key}
      [ -n "${value}" ] || continue
      printf '%s=%q\n' "${key}" "${value}"
    done < <(printf '%s\n' "${RESOLVED_TRAIN_EXTRA_KEYS[@]}")
  } > "${target}"
}

write_resolved_profile() {
  local target=$1
  shift
  local profile_paths=("$@")
  mkdir -p "$(dirname "${target}")"
  {
    printf '# Generated by %s\n' "$(basename "$0")"
    profile_keys "${profile_paths[@]}" | awk 'NF && !seen[$0]++' | while read -r key; do
      printf '%s=%q\n' "${key}" "${!key-}"
    done
  } > "${target}"
}

valid_env_key() {
  [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}

append_aux_train_env_keys() {
  local target=$1
  local key
  [ -n "${AUX_TRAIN_ENV_KEYS:-}" ] || return 0
  {
    printf '\n# Aux profile declared train-visible environment.\n'
    quote_assign AUX_TRAIN_ENV_KEYS "${AUX_TRAIN_ENV_KEYS}"
    for key in ${AUX_TRAIN_ENV_KEYS//,/ }; do
      [ -n "${key}" ] || continue
      if ! valid_env_key "${key}"; then
        echo "Invalid AUX_TRAIN_ENV_KEYS entry: ${key}" >&2
        exit 1
      fi
      [[ -v ${key} ]] || continue
      [ -n "${!key}" ] || continue
      quote_assign "${key}" "${!key}"
    done
  } >> "${target}"
}

write_named_env() {
  local target=$1
  shift
  mkdir -p "$(dirname "${target}")"
  {
    printf '# Generated by %s\n' "$(basename "$0")"
    quote_assign_vars "$@"
  } > "${target}"
}

read_nodes_from() {
  local nodes_file=${1:-}
  local selector=${2:-}
  if [ -z "${nodes_file}" ]; then
    echo "this"
  elif [ -z "${selector}" ]; then
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

read_nodes() {
  read_nodes_from "${NODES_FILE}" "${NODE_INDICES}"
}

first_node() {
  read_nodes | head -n 1
}

node_count() {
  read_nodes | wc -l | tr -d ' '
}

is_current_node() {
  local node=$1
  [ "${node}" = "this" ] && return 0
  [ "${node}" = "$(hostname)" ] && return 0
  hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${node}" && return 0
  ip addr 2>/dev/null | grep -Fq "${node}" && return 0
  return 1
}

http_host() {
  local node=$1
  if [[ "${node}" == *:* ]]; then
    printf '[%s]' "${node}"
  else
    printf '%s' "${node}"
  fi
}

safe_label() {
  local value=$1
  value=${value//:/_}
  value=${value//./_}
  value=${value//\//_}
  printf '%s\n' "${value}"
}

local_node_label() {
  local ip
  ip=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 . || true)
  safe_label "${ip:-$(hostname)}"
}

role_log_path() {
  local name=$1
  local stem=${name%.*}
  local ext=
  [ "${stem}" = "${name}" ] || ext=".${name##*.}"
  if [ "${INTERNAL_ROLE:-}" = "worker" ]; then
    printf '%s/%s_%s%s\n' "${LOG_DIR}" "${stem}" "$(local_node_label)" "${ext}"
  else
    printf '%s/%s\n' "${LOG_DIR}" "${name}"
  fi
}

ssh_node() {
  local host=$1
  local remote_cmd=$2
  local encoded wrapped
  local args=()
  [ "${SSH_IPV6}" = "1" ] && args+=("-6")
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
  )
  [ -z "${SSH_JUMP}" ] || args+=("-J" "${SSH_JUMP}")
  args+=("${SSH_USER}@${host}")
  encoded=$(printf '%s' "${remote_cmd}" | base64 | tr -d '\n')
  wrapped=$(printf "printf %%s '%s' | base64 -d | bash" "${encoded}")
  if [ "${DRY_RUN}" = "1" ]; then
    printf '+ ssh %s %q\n' "${host}" "${remote_cmd}"
    return 0
  fi
  ssh "${args[@]}" "${wrapped}"
}

remote_query() {
  local host=$1
  local cmd=$2
  local attempt
  for attempt in 1 2 3; do
    if ssh_node "${host}" "${cmd}"; then
      return 0
    fi
    echo "Retry ${attempt}/3 failed on ${host}" >&2
    sleep 5
  done
  return 1
}

remote_first_ip() {
  local host=$1
  remote_query "${host}" "hostname -I | tr ' ' '\n' | grep -m1 ."
}

tmux_start_local() {
  local session=$1
  local script=$2
  local log=${3:-}
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ tmux ${session}"
    printf '%s\n' "${script}"
    return 0
  fi
  tmux kill-session -t "${session}" 2>/dev/null || true
  local tmp="/tmp/${session}.sh"
  printf '%s\n' "${script}" > "${tmp}"
  chmod +x "${tmp}"
  if [ -n "${log}" ]; then
    mkdir -p "$(dirname "${log}")"
    tmux new-session -d -s "${session}" "bash ${tmp} > ${log} 2>&1"
  else
    tmux new-session -d -s "${session}" "bash ${tmp}"
  fi
}

tmux_start_remote() {
  local host=$1
  local session=$2
  local script=$3
  local log=${4:-}
  local payload remote_cmd
  payload=$(printf '%s\n' "${script}" | base64 | tr -d '\n')
  remote_cmd=$(cat <<EOF
set -euo pipefail
tmp=/tmp/${session}.sh
printf %s '${payload}' | base64 -d > "\${tmp}"
chmod +x "\${tmp}"
tmux kill-session -t ${session@Q} 2>/dev/null || true
EOF
)
  if [ -n "${log}" ]; then
    remote_cmd+=$(printf '\nmkdir -p %q\ntmux new-session -d -s %q "bash %q > %q 2>&1"\n' "$(dirname "${log}")" "${session}" "/tmp/${session}.sh" "${log}")
  else
    remote_cmd+=$(printf '\ntmux new-session -d -s %q "bash %q"\n' "${session}" "/tmp/${session}.sh")
  fi
  echo "Starting ${session} on ${host}"
  remote_query "${host}" "${remote_cmd}"
}

tmux_session_active() {
  local session=$1
  tmux has-session -t "${session}" 2>/dev/null
}

run_bench_nodes() {
  local action=$1
  local nodes_file=$2
  local selector=${3:-}
  [ -n "${nodes_file}" ] || return 0
  local cmd
  cmd=$(printf 'ROOT_DIR=%q LOCAL_ENVS_DIR=%q BENCH_PYTHON=%q SSH_KEY=%q SSH_IPV6=%q bash %q %q --nodes %q' \
    "${ROOT_DIR}" "${LOCAL_ENVS_DIR}" "${BENCH_PYTHON:-${SLIME_PYTHON:-${SLIME_IMAGE_PYTHON:-/usr/bin/python}}}" "${SSH_KEY}" "${SSH_IPV6}" "${OPS_SCRIPTS_DIR}/run_bench.sh" "${action}" "${nodes_file}")
  [ -z "${selector}" ] || cmd+=$(printf ' --node %q' "${selector}")
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ ${cmd}"
  else
    bash -lc "${cmd}" || true
  fi
}

reset_runtime_cmd() {
  local ray_stop_python
  ray_stop_python=${SLIME_PYTHON:-${SLIME_IMAGE_PYTHON:-/usr/bin/python}}
  cat <<EOF
set +e
for session in agent_env_ray_head agent_env_ray_worker agent_env_${ENV_NAME}_env agent_env_${ENV_NAME}_router agent_env_${ENV_NAME}_train agent_env_multi_head agent_env_multi_worker; do
  tmux kill-session -t "\${session}" 2>/dev/null || true
done
tmux ls 2>/dev/null | awk -F: '/^agent_env_.*_(env|router|train):/ {print \$1}' | while read -r session; do
  tmux kill-session -t "\${session}" 2>/dev/null || true
done
if [ -x "${ray_stop_python}" ]; then
  "${ray_stop_python}" -m ray.scripts.scripts stop --force >/tmp/server_ops_ray_stop.log 2>&1 || true
elif command -v ray >/dev/null 2>&1; then
  ray stop --force >/tmp/server_ops_ray_stop.log 2>&1 || true
fi
pkill -f '[e]xamples/agent_env/.*/server.py' 2>/dev/null || true
pkill -f '[e]xamples/agent_env/router.py' 2>/dev/null || true
pkill -f '[s]glang.launch_server' 2>/dev/null || true
pkill -f '[r]ay::SGLangEngine' 2>/dev/null || true
pkill -f '[s]glang::scheduler' 2>/dev/null || true
pkill -f '[s]glang::detokenizer' 2>/dev/null || true
pkill -f '[t]rain_entrypoint.py' 2>/dev/null || true
pkill -f '[s]lime/ray/train' 2>/dev/null || true
pkill -f '[t]rain_async.py' 2>/dev/null || true
pkill -f '[t]rain_async_compat.py' 2>/dev/null || true
pkill -f '[r]un_agent_env_train.sh' 2>/dev/null || true
pkill -f '[r]aylet|[g]cs_server|[p]lasma_store|[d]ashboard_agent|[d]ashboard.py' 2>/dev/null || true
if [ "${RESET_TRAIN_RUNTIME_ON_START}" = "force" ]; then
pkill -f '[r]ay::' 2>/dev/null || true
fi
sleep 2
if ! pgrep -u "\$(id -u)" -f '[r]aylet|[g]cs_server|[p]lasma_store|[d]ashboard_agent|[d]ashboard.py|[r]ay.scripts.scripts start' >/dev/null 2>&1; then
  find /tmp/ray -maxdepth 1 -mindepth 1 \( -name 'session_*' -o -name 'session_latest' \) -exec rm -rf -- {} + 2>/dev/null || true
fi
EOF
}

cleanup_ray_tmp_after_stop() {
  sleep 2
  if ! pgrep -u "$(id -u)" -f '[r]aylet|[g]cs_server|[p]lasma_store|[d]ashboard_agent|[d]ashboard.py|[r]ay.scripts.scripts start' >/dev/null 2>&1; then
    find /tmp/ray -maxdepth 1 -mindepth 1 \( -name 'session_*' -o -name 'session_latest' \) -exec rm -rf -- {} + 2>/dev/null || true
  fi
}

reset_runtime_on_nodes() {
  case "${RESET_TRAIN_RUNTIME_ON_START}" in
    1|true|TRUE|yes|YES|on|ON|force) ;;
    *) return 0 ;;
  esac
  local node cmd
  cmd=$(reset_runtime_cmd)
  for node in $(read_nodes); do
    if [ "${DRY_RUN}" = "1" ]; then
      echo "Dry-run: would reset training runtime on ${node}"
    else
      echo "Resetting training runtime on ${node}"
    fi
    if is_current_node "${node}"; then
      [ "${DRY_RUN}" = "1" ] && echo "+ local reset" || bash -lc "${cmd}" || true
    else
      ssh_node "${node}" "${cmd}" || true
    fi
  done
}

task_env_path() {
  case "${ENV_NAME}" in
    alfworld|webshop|tau2|appworld|openclaw) printf '%s/%s\n' "${LOCAL_ENVS_DIR}" "${ENV_NAME}" ;;
    *) return 1 ;;
  esac
}

task_env_python() {
  case "${ENV_NAME}" in
    alfworld|webshop|tau2|appworld|openclaw) printf '%s/bin/python\n' "$(task_env_path)" ;;
    *) return 1 ;;
  esac
}

require_runtime() {
  [ "${DRY_RUN}" = "1" ] && return 0
  local env_python
  [ -x "${SLIME_PYTHON}" ] || { echo "Missing slime python: ${SLIME_PYTHON}" >&2; exit 1; }
  case "${ENV_NAME}" in
    webshop|alfworld|tau2|appworld|openclaw)
      env_python=$(task_env_python)
      [ -x "${env_python}" ] || { echo "Missing ${ENV_NAME} env python: ${env_python}" >&2; exit 1; }
      ;;
    *) echo "Unsupported env: ${ENV_NAME}" >&2; exit 1 ;;
  esac
}

load_aux_endpoint_env() {
  [ -n "${AUX_ENV_FILE:-}" ] || return 0
  [ -f "${AUX_ENV_FILE}" ] || return 0
  [ "${DRY_RUN:-0}" = "0" ] || return 0
  # shellcheck disable=SC1090
  source "${AUX_ENV_FILE}"
}

server_runtime_exports() {
  quote_export_vars ROOT_DIR LOCAL_RUNTIME_DIR LOCAL_ENVS_DIR REPO_DIR
  quote_export AGENT_ENV_DATA_DIR "${AGENT_ENV_DATA_DIR:-${LOCAL_RUNTIME_DIR}/data/${ENV_NAME}}"
  case "${ENV_NAME}" in
    webshop)
      quote_export WEBSHOP_LIB "${WEBSHOP_LIB:-${LOCAL_RUNTIME_DIR}/code/WebShop}"
      ;;
    alfworld)
      quote_export ALFWORLD_LIB "${ALFWORLD_LIB:-${LOCAL_RUNTIME_DIR}/data/alfworld/pythonlibs/alfworld_text}"
      ;;
    appworld)
      quote_export APPWORLD_ROOT "${LOCAL_RUNTIME_DIR}/data/appworld"
      ;;
    tau2|openclaw) ;;
  esac
  quote_export_vars \
    AUX_ENDPOINT_PROVIDER AUX_ENDPOINT_MODEL AUX_ENDPOINT_BASE_URL AUX_ENDPOINT_API_KEY_PATH \
    AUX_ENDPOINT_TIMEOUT_S AUX_ENDPOINT_MAX_TOKENS AUX_ENDPOINT_TEMPERATURE AUX_ENDPOINT_TOP_P \
    AUX_ENDPOINT_ENABLE_THINKING AUX_ENDPOINT_SEPARATE_REASONING AUX_ENDPOINT_REASONING_EFFORT
}

start_env_server() {
  require_runtime
  local config="${ENV_CONFIG:?Set ENV_CONFIG}"
  [ -f "${config}" ] || { echo "Missing env config: ${config}" >&2; exit 1; }
  local script runtime_env env_python pythonpath extra_exports webshop_lib java_home jvm_path
  runtime_env=$(server_runtime_exports)
  env_python=$(task_env_python)
  pythonpath=${REPO_DIR}
  extra_exports=
  case "${ENV_NAME}" in
    webshop)
      webshop_lib=${WEBSHOP_LIB:-${LOCAL_RUNTIME_DIR}/code/WebShop}
      java_home=${WEBSHOP_JAVA_HOME:-$(task_env_path)/lib/jvm}
      jvm_path=${WEBSHOP_JVM_PATH:-${java_home}/lib/server/libjvm.so}
      pythonpath="${REPO_DIR}:${webshop_lib}"
      extra_exports=$(printf '%s\n%s\n%s\n' \
        "$(quote_export WEBSHOP_LIB "${webshop_lib}")" \
        "$(quote_export JAVA_HOME "${java_home}")" \
        "$(quote_export JVM_PATH "${jvm_path}")")
      ;;
    alfworld|openclaw) ;;
    tau2) extra_exports=$(quote_export LITELLM_LOCAL_MODEL_COST_MAP True) ;;
    appworld)
      extra_exports=$(quote_export HOME "${LOCAL_RUNTIME_DIR}/data/appworld")
      ;;
    *) echo "Unsupported env: ${ENV_NAME}" >&2; exit 1 ;;
  esac
  script=$(printf 'cd %q\n%s\n%s\nexport PYTHONNOUSERSITE=1 PYTHONPATH=%q\n%q %q --host 0.0.0.0 --port %q --config %q\n' \
    "${REPO_DIR}" "${runtime_env}" "${extra_exports}" "${pythonpath}" "${env_python}" "examples/agent_env/${ENV_NAME}/server.py" "${ENV_PORT}" "${config}")
  tmux_start_local "agent_env_${ENV_NAME}_env" "${script}" "$(role_log_path "${ENV_NAME}_env_server.log")"
}

ray_start_script() {
  local role=$1 node_ip=$2 head_addr=${3:-}
  local cmd=("${SLIME_PYTHON}" -m ray.scripts.scripts start)
  case "${role}" in
    head)
      cmd+=(--head --node-ip-address "${node_ip}" --port "${RAY_PORT}")
      cmd+=(--dashboard-host=0.0.0.0 --dashboard-port=8265)
      ;;
    worker)
      cmd+=(--address "${head_addr}:${RAY_PORT}" --node-ip-address "${node_ip}")
      ;;
    *) echo "Unsupported Ray role: ${role}" >&2; return 1 ;;
  esac
  cmd+=(--num-gpus "${NUM_GPUS_PER_NODE_FOR_RAY}" --min-worker-port "${RAY_MIN_WORKER_PORT}" --max-worker-port "${RAY_MAX_WORKER_PORT}")
  cmd+=(--disable-usage-stats --block)
  {
    printf 'export PYTHONNOUSERSITE=1 RAY_DISABLE_DOCKER_CPU_WARNING=1\n'
    printf '[ ! -f %q ] || { set -a; source %q; set +a; }\n' "${WANDB_SECRET_FILE}" "${WANDB_SECRET_FILE}"
    quote_export CUDA_VISIBLE_DEVICES "${RAY_CUDA_VISIBLE_DEVICES}"
    quote_export_vars SOCKET_IFNAME NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME
    printf 'mkdir -p %q\n' "${LOG_DIR}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  }
}

start_ray_head() {
  local node_ip=${HEAD_ADDRESS:-}
  [ -n "${node_ip}" ] || node_ip=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  local script attempt
  script=$(ray_start_script head "${node_ip}")
  for attempt in $(seq 1 "${RAY_START_MAX_ATTEMPTS}"); do
    echo "Starting Ray head attempt ${attempt}/${RAY_START_MAX_ATTEMPTS}"
    [ "${DRY_RUN}" = "1" ] || "${SLIME_PYTHON}" -m ray.scripts.scripts stop --force || true
    [ "${DRY_RUN}" = "1" ] || cleanup_ray_tmp_after_stop
    tmux_start_local agent_env_ray_head "${script}" "${LOG_DIR}/ray_head.log"
    if wait_ray_head_ready; then
      return 0
    fi
    tmux kill-session -t agent_env_ray_head 2>/dev/null || true
    sleep 5
  done
  echo "Ray head failed after ${RAY_START_MAX_ATTEMPTS} attempts" >&2
  return 1
}

start_ray_worker() {
  local head_addr=$1
  local node_ip
  node_ip=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  [ "${DRY_RUN}" = "1" ] || "${SLIME_PYTHON}" -m ray.scripts.scripts stop --force || true
  [ "${DRY_RUN}" = "1" ] || cleanup_ray_tmp_after_stop
  local script
  script=$(ray_start_script worker "${node_ip}" "${head_addr}")
  tmux_start_local agent_env_ray_worker "${script}" "$(role_log_path "ray_worker.log")"
}

http_wait_command() {
  local url=$1
  cat <<EOF
${SLIME_PYTHON@Q} - <<'PY'
import json, sys, time, urllib.request
url = ${url@Q}
for _ in range(300):
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=2).read().decode())
        if data.get("ok"):
            print("ready", url)
            sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
print("Timed out waiting for", url, file=sys.stderr)
sys.exit(1)
PY
EOF
}

wait_http() {
  local url=$1 cmd
  cmd=$(http_wait_command "${url}")
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ wait_http ${url}"
    return 0
  fi
  bash -lc "${cmd}"
}

remote_wait_http() {
  local host=$1 url=$2
  echo "Waiting ${host} ${url}"
  remote_query "${host}" "$(http_wait_command "${url}")"
}

ray_alive_nodes() {
  timeout 20 "${SLIME_PYTHON}" - <<PY
import ray
try:
    ray.init(address="127.0.0.1:${RAY_PORT}", ignore_reinit_error=True, logging_level="ERROR")
    print(sum(1 for n in ray.nodes() if n.get("Alive")))
finally:
    ray.shutdown()
PY
}

wait_ray_head_ready() {
  local deadline=$((SECONDS + RAY_HEAD_START_TIMEOUT_S))
  local alive
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ wait_ray_head_ready"
    return 0
  fi
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    alive=$(ray_alive_nodes 2>/dev/null | tail -n 1 || true)
    if [ "${alive:-0}" -ge 1 ] 2>/dev/null; then
      echo "Ray head ready: alive=${alive}"
      return 0
    fi
    if ! tmux_session_active agent_env_ray_head; then
      echo "Ray head tmux exited before ready" >&2
      tail -n 80 "${LOG_DIR}/ray_head.log" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for Ray head" >&2
  tail -n 80 "${LOG_DIR}/ray_head.log" >&2 || true
  return 1
}

wait_ray_nodes() {
  local expected=$1
  local alive
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ wait_ray_nodes ${expected}"
    return 0
  fi
  for _ in $(seq 1 180); do
    alive=$(ray_alive_nodes 2>/dev/null | tail -n 1 || true)
    if [ "${alive:-0}" -ge "${expected}" ] 2>/dev/null; then
      echo "Ray ready: alive=${alive}"
      return 0
    fi
    echo "Ray not ready: alive=${alive:-unknown}/${expected}"
    sleep 5
  done
  echo "Timed out waiting for Ray nodes" >&2
  return 1
}

start_router() {
  local workers_csv=$1
  local script
  script=$(printf 'cd %q\nexport PYTHONNOUSERSITE=1\nmkdir -p %q\n%q examples/agent_env/router.py --host 0.0.0.0 --port %q --workers %q\n' \
    "${REPO_DIR}" "${LOG_DIR}" "${SLIME_PYTHON}" "${ROUTER_PORT}" "${workers_csv}")
  tmux_start_local "agent_env_${ENV_NAME}_router" "${script}" "${LOG_DIR}/${ENV_NAME}_router.log"
}

write_train_driver() {
  local router_url=$1
  local driver="${LOG_DIR}/${ENV_NAME}_train_driver.sh"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    quote_export_vars ROOT_DIR LOCAL_RUNTIME_DIR LOCAL_ENVS_DIR REPO_DIR ENV_CONFIG
    # The adapter sources this final train env contract through its existing
    # TRAIN_PROFILE entrypoint; it is not a reusable source profile.
    quote_export TRAIN_PROFILE "${RESOLVED_TRAIN_ENV}"
    quote_export CUSTOM_CONFIG_PATH "${ENV_CONFIG}"
    quote_export_vars \
      SLIME_RUNTIME SLIME_ENV SLIME_PYTHON SLIME_PACK_NAME SLIME_PACK_PATH SLIME_IMAGE_ENV SLIME_IMAGE_PYTHON \
      MEGATRON_PATH MEGATRON_IMAGE_PATH \
      WANDB_RUNTIME WANDB_PACK_NAME WANDB_PACK_PATH \
      RUN_ROOT LOG_DIR WANDB_DIR SAVE_DIR AUX_ENV_FILE
    quote_export CUDA_VISIBLE_DEVICES "${RAY_CUDA_VISIBLE_DEVICES}"
    quote_export RAY_CUDA_VISIBLE_DEVICES "${RAY_CUDA_VISIBLE_DEVICES}"
    quote_export_vars RAY_PORT SOCKET_IFNAME NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME
    quote_export RAY_ADDRESS "127.0.0.1:${RAY_PORT}"
    printf 'if [ -n "${AUX_ENV_FILE:-}" ] && [ -f "${AUX_ENV_FILE}" ]; then\n'
    printf '  set -a\n'
    printf '  source "${AUX_ENV_FILE}"\n'
    printf '  set +a\n'
    printf 'fi\n'
    printf 'cd %q\n' "${REPO_DIR}"
    printf 'bash %q --env-server-url %q\n' "${TRAIN_ADAPTER}" "${router_url}"
  } > "${driver}"
  chmod +x "${driver}"
  printf '%s\n' "${driver}"
}

start_train_driver() {
  local router_url=$1
  local driver status_file train_log wrapper bench_log
  driver=$(write_train_driver "${router_url}")
  status_file="${LOG_DIR}/${ENV_NAME}_train_status.env"
  train_log="${LOG_DIR}/${ENV_NAME}_train.log"
  wrapper="${LOG_DIR}/${ENV_NAME}_train_tmux.sh"
  bench_log="${LOG_DIR}/bench_on_train_exit.log"
  {
    printf '#!/usr/bin/env bash\nset +e\n'
    quote_assign DRIVER "${driver}"
    quote_assign TRAIN_LOG "${train_log}"
    quote_assign STATUS_FILE "${status_file}"
    quote_assign BENCH_LOG "${bench_log}"
    quote_assign BENCH_PYTHON "${BENCH_PYTHON:-${SLIME_PYTHON}}"
    quote_assign RUN_BENCH "${OPS_SCRIPTS_DIR}/run_bench.sh"
    quote_assign_vars \
      BENCH_ON_TRAIN_EXIT ROOT_DIR LOCAL_ENVS_DIR NODES_FILE NODE_INDICES \
      SSH_USER SSH_PORT SSH_KEY SSH_IPV6 SSH_JUMP
    cat <<'EOF'
finish() {
  local code=$?
  printf "exit_code=%s\nend_time=%s\n" "${code}" "$(date -Is)" > "${STATUS_FILE}"
  if [ "${BENCH_ON_TRAIN_EXIT}" = "1" ] && [ -n "${NODES_FILE}" ] && [ -f "${RUN_BENCH}" ]; then
    {
      echo "[$(date -Is)] train exited code=${code}; running run_bench start --nodes ${NODES_FILE} --node ${NODE_INDICES}"
      bench_args=(start --nodes "${NODES_FILE}")
      [ -z "${NODE_INDICES}" ] || bench_args+=(--node "${NODE_INDICES}")
      ROOT_DIR="${ROOT_DIR}" LOCAL_ENVS_DIR="${LOCAL_ENVS_DIR}" BENCH_PYTHON="${BENCH_PYTHON}" SSH_USER="${SSH_USER}" SSH_PORT="${SSH_PORT}" \
        SSH_KEY="${SSH_KEY}" SSH_IPV6="${SSH_IPV6}" SSH_JUMP="${SSH_JUMP}" \
        bash "${RUN_BENCH}" "${bench_args[@]}"
    } >> "${BENCH_LOG}" 2>&1 || true
  fi
  exit "${code}"
}

trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf "state=running\nstart_time=%s\n" "$(date -Is)" > "${STATUS_FILE}"
bash "${DRIVER}" > "${TRAIN_LOG}" 2>&1
EOF
    printf 'code=$?\n'
    printf 'exit "$code"\n'
  } > "${wrapper}"
  chmod +x "${wrapper}"
  tmux_start_local "agent_env_${ENV_NAME}_train" "bash ${wrapper}"
  echo "Training submitted in tmux: agent_env_${ENV_NAME}_train"
  echo "Training log: ${train_log}"
  echo "Train status: ${status_file}"
}

run_worker() {
  source_env_file "${RESOLVED_CONFIG}" resolved
  resolve_slime_runtime
  start_env_server
  wait_http "http://127.0.0.1:${ENV_PORT}/health"
  start_ray_worker "${HEAD_ADDRESS:?worker needs --head-address}"
}

run_head() {
  source_env_file "${RESOLVED_CONFIG}" resolved
  resolve_slime_runtime
  mkdir -p "${LOG_DIR}"
  HEAD_ORCHESTRATION_COMPLETE=0
  head_failure_guard() {
    local code=$?
    if [ "${code}" -ne 0 ] \
      && [ "${HEAD_ORCHESTRATION_COMPLETE}" != "1" ] \
      && [ "${BENCH_ON_LAUNCH_FAILURE}" = "1" ]; then
      echo "Head orchestration failed with exit_code=${code}; starting bench on ${NODES_FILE}" >&2
      run_bench_nodes start "${NODES_FILE}" "${NODE_INDICES}"
    fi
    exit "${code}"
  }
  trap head_failure_guard EXIT
  local head_addr=${HEAD_ADDRESS:-}
  [ -n "${head_addr}" ] || head_addr=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  local head_http="http://$(http_host "${head_addr}"):${ENV_PORT}"
  local env_urls=("${head_http}")
  start_ray_head
  start_env_server
  wait_http "http://127.0.0.1:${ENV_PORT}/health"

  local node node_addr worker_script
  for node in $(read_nodes | tail -n +2); do
    if [ "${DRY_RUN}" = "1" ]; then
      node_addr="${node}"
    else
      node_addr=$(remote_first_ip "${node}")
    fi
    env_urls+=("http://$(http_host "${node_addr}"):${ENV_PORT}")
    worker_script=$(printf 'cd %q\nROOT_DIR=%q OPS_SCRIPTS_DIR=%q LOCAL_ENVS_DIR=%q LOCAL_RUNTIME_DIR=%q SSH_KEY=%q SSH_IPV6=%q bash scripts/utils/launch_agentic_training.sh --internal-role worker --resolved %q --head-address %q\n' \
      "${REPO_DIR}" "${ROOT_DIR}" "${OPS_SCRIPTS_DIR}" "${LOCAL_ENVS_DIR}" "${LOCAL_RUNTIME_DIR}" "${SSH_KEY}" "${SSH_IPV6}" "${RESOLVED_CONFIG}" "${head_addr}")
    tmux_start_remote "${node}" agent_env_multi_worker "${worker_script}" "${LOG_DIR}/multi_worker_$(safe_label "${node}").log"
  done
  for node in $(read_nodes | tail -n +2); do
    remote_wait_http "${node}" "http://127.0.0.1:${ENV_PORT}/health"
  done
  wait_ray_nodes "$(node_count)"

  local workers_csv
  workers_csv=$(IFS=,; echo "${env_urls[*]}")
  start_router "${workers_csv}"
  wait_http "http://127.0.0.1:${ROUTER_PORT}/health"
  start_train_driver "http://$(http_host "${head_addr}"):${ROUTER_PORT}"
  HEAD_ORCHESTRATION_COMPLETE=1
  trap - EXIT
}

write_resolved_launch_config() {
  load_aux_endpoint_env
  write_named_env "${RESOLVED_LAUNCH_CONFIG}" "${RESOLVED_LAUNCH_KEYS[@]}"
  # RESOLVED_CONFIG is used by internal roles; keep it self-referential.
  printf 'RESOLVED_CONFIG=%q\n' "${RESOLVED_LAUNCH_CONFIG}" >> "${RESOLVED_LAUNCH_CONFIG}"
}

load_run_profiles() {
  [ -n "${RUN_PROFILE}" ] || { echo "Missing run profile" >&2; usage >&2; exit 1; }
  RUN_PROFILE_PATH=$(resolve_path "${RUN_PROFILE}")
  source_env_file "${RUN_PROFILE_PATH}" run
  TOPOLOGY_PROFILE_PATH=$(resolve_path "${TOPOLOGY_PROFILE:?Set TOPOLOGY_PROFILE in run profile}")
  source_env_file "${TOPOLOGY_PROFILE_PATH}" topology
  MODEL_PROFILE_PATH=$(resolve_path "${MODEL_PROFILE:?Set MODEL_PROFILE in run profile}")
  source_env_file "${MODEL_PROFILE_PATH}" model
  TRAIN_PROFILE_PATH=$(resolve_path "${TRAIN_PROFILE:?Set TRAIN_PROFILE in run profile}")
  source_env_file "${TRAIN_PROFILE_PATH}" train
}

resolve_socket_ifnames() {
  SOCKET_IFNAME=${SOCKET_IFNAME:-${AGENT_ENV_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}}
  NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}
  GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}
  TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}
}

resolve_run_defaults() {
  ENV_NAME=${ENV_NAME:?Set ENV_NAME in run profile}
  set_slime_runtime_defaults
  set_wandb_runtime_defaults
  ENV_CONFIG=$(resolve_path "${ENV_CONFIG:?Set ENV_CONFIG in run profile}")
  REWARD_PROFILE=$(resolve_path "${REWARD_PROFILE:?Set REWARD_PROFILE in run profile}")
  TRAIN_ADAPTER=$(resolve_path "${TRAIN_ADAPTER}")
  NODES_FILE=$(resolve_path "${NODES_FILE}")
  if [ -n "${AUX_PROFILE:-}" ]; then
    AUX_NODES_FILE=${AUX_NODES_FILE:-${NODES_FILE}}
  fi
  [ -z "${AUX_NODES_FILE}" ] || AUX_NODES_FILE=$(resolve_path "${AUX_NODES_FILE}")
  [ -z "${NODES_FILE}" ] || [ -f "${NODES_FILE}" ] || { echo "Missing nodes file: ${NODES_FILE}" >&2; exit 1; }
  if [ "$(node_count)" -eq 0 ]; then
    echo "No training nodes selected from ${NODES_FILE} with NODE_INDICES=${NODE_INDICES}" >&2
    exit 1
  fi
  ENV_PORT=${ENV_PORT:-18180}
  ROUTER_PORT=${ROUTER_PORT:-19000}
  RAY_CUDA_VISIBLE_DEVICES=${RAY_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
  NUM_GPUS_PER_NODE_FOR_RAY=${NUM_GPUS_PER_NODE_FOR_RAY:-8}
  RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-20000}
  RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-29999}
  resolve_socket_ifnames
  EXP_PROJECT=${EXP_PROJECT:-${PROJECT_NAME:-${MODEL_BASENAME:-model}_${ENV_NAME}_grpo}}
  EXP_NAME=${EXP_NAME:-${RUN_NAME:-${MODEL_BASENAME:-model}-${ENV_NAME}-grpo}}
  RUN_ROOT=${RUN_ROOT:-${ROOT_DIR}/runs/${EXP_PROJECT}/${EXP_NAME}}
  LOG_DIR=${LOG_DIR:-${RUN_ROOT}/logs}
  WANDB_DIR=${WANDB_DIR:-${RUN_ROOT}/wandb}
  SAVE_DIR=${SAVE_DIR:-${RUN_ROOT}/checkpoints}
  AUX_ENV_FILE=${AUX_ENV_FILE:-${LOG_DIR}/aux_endpoint.env}
  RESOLVED_TRAIN_ENV=${LOG_DIR}/resolved_train.env
  RESOLVED_LAUNCH_CONFIG=${LOG_DIR}/resolved_launch.env
  mkdir -p "${LOG_DIR}" "${WANDB_DIR}" "${SAVE_DIR}"
}

write_resolved_aux_profile() {
  if [ -n "${AUX_PROFILE:-}" ]; then
    local aux_path
    aux_path=$(resolve_path "${AUX_PROFILE}")
    source_env_file "${aux_path}" aux
    RESOLVED_AUX_PROFILE=${LOG_DIR}/resolved_aux_profile.env
    write_resolved_profile "${RESOLVED_AUX_PROFILE}" "${aux_path}"
    {
      printf '\n'
      quote_assign_vars "${AUX_PROFILE_APPEND_KEYS[@]}"
    } >> "${RESOLVED_AUX_PROFILE}"
  else
    RESOLVED_AUX_PROFILE=
  fi
}

aux_profile_uses_local_gpu() {
  local spec=${AUX_SPEC:-}
  local provider
  if [[ "${spec}" == */* ]]; then
    provider=${spec%%/*}
  else
    provider=local
  fi
  provider=$(printf '%s' "${provider}" | tr '[:upper:]' '[:lower:]')
  case "${provider}" in
    local|sglang|vllm) return 0 ;;
    *) return 1 ;;
  esac
}

prepare_run() {
  load_run_profiles
  resolve_run_defaults
  write_resolved_aux_profile
  write_resolved_train_env "${RESOLVED_TRAIN_ENV}" "${RUN_PROFILE_PATH}" "${MODEL_PROFILE_PATH}" "${TRAIN_PROFILE_PATH}"
  append_aux_train_env_keys "${RESOLVED_TRAIN_ENV}"
}

start_aux_endpoint() {
  [ -n "${RESOLVED_AUX_PROFILE:-}" ] || return 0
  local aux_nodes_arg=()
  local aux_index_arg=()
  [ -z "${AUX_NODES_FILE:-}" ] || aux_nodes_arg=(--nodes "${AUX_NODES_FILE}")
  [ -z "${AUX_NODE_INDICES:-}" ] || aux_index_arg=(--node-index "${AUX_NODE_INDICES}")
  if aux_profile_uses_local_gpu; then
    run_bench_nodes stop "${AUX_NODES_FILE:-}" "${AUX_NODE_INDICES:-}"
  fi
  local dry=()
  [ "${DRY_RUN}" = "0" ] || dry=(--dry-run)
  bash "${REPO_DIR}/scripts/utils/aux_endpoint.sh" start \
    --config "${RESOLVED_AUX_PROFILE}" \
    --env-file "${AUX_ENV_FILE}" \
    "${aux_nodes_arg[@]}" \
    "${aux_index_arg[@]}" \
    "${dry[@]}"
}

submit_head() {
  local head head_addr script
  head=$(first_node)
  if [ "${DRY_RUN}" = "1" ]; then
    head_addr="${head}"
  else
    head_addr=$(remote_first_ip "${head}")
  fi
  script=$(printf 'cd %q\nROOT_DIR=%q OPS_SCRIPTS_DIR=%q LOCAL_ENVS_DIR=%q LOCAL_RUNTIME_DIR=%q SSH_KEY=%q SSH_IPV6=%q SSH_JUMP= bash scripts/utils/launch_agentic_training.sh --internal-role head --resolved %q --head-address %q\n' \
    "${REPO_DIR}" "${ROOT_DIR}" "${OPS_SCRIPTS_DIR}" "${LOCAL_ENVS_DIR}" "${LOCAL_RUNTIME_DIR}" "${SSH_KEY}" "${SSH_IPV6}" "${RESOLVED_LAUNCH_CONFIG}" "${head_addr}")
  if is_current_node "${head}"; then
    RESOLVED_CONFIG="${RESOLVED_LAUNCH_CONFIG}" HEAD_ADDRESS="${head_addr}" run_head
  else
    if [ "${DRY_RUN}" = "1" ]; then
      echo "Dry-run: would submit head orchestration on ${head}"
      printf '%s\n' "${script}"
      return 0
    fi
    tmux_start_remote "${head}" agent_env_multi_head "${script}" "${LOG_DIR}/multi_head.log"
    echo "Head orchestration submitted."
    echo "Head log: ${head}:${LOG_DIR}/multi_head.log"
  fi
}

LAUNCH_BENCH_GUARD_ACTIVE=0
LAUNCH_COMPLETED=0
HEAD_ORCHESTRATION_COMPLETE=0

bench_on_launch_exit() {
  local code=$?
  if [ "${code}" -ne 0 ] \
    && [ "${LAUNCH_BENCH_GUARD_ACTIVE}" = "1" ] \
    && [ "${LAUNCH_COMPLETED}" != "1" ] \
    && [ "${BENCH_ON_LAUNCH_FAILURE}" = "1" ]; then
    echo "Launch failed with exit_code=${code}; starting bench on ${NODES_FILE}" >&2
    run_bench_nodes start "${NODES_FILE}" "${NODE_INDICES}"
  fi
  exit "${code}"
}

stop_bench_nodes_quiescent() {
  run_bench_nodes stop "${NODES_FILE}" "${NODE_INDICES}"
  if [ "${DRY_RUN}" != "1" ]; then
    sleep 20
  fi
  run_bench_nodes stop "${NODES_FILE}" "${NODE_INDICES}"
}

main_launch() {
  prepare_run
  LAUNCH_BENCH_GUARD_ACTIVE=1
  LAUNCH_COMPLETED=0
  trap bench_on_launch_exit EXIT
  start_aux_endpoint
  write_resolved_launch_config
  reset_runtime_on_nodes
  stop_bench_nodes_quiescent
  submit_head
  run_bench_nodes stop "${NODES_FILE}" "${NODE_INDICES}"
  LAUNCH_COMPLETED=1
  LAUNCH_BENCH_GUARD_ACTIVE=0
  trap - EXIT
}

case "${INTERNAL_ROLE}" in
  "")
    main_launch
    ;;
  head)
    [ -n "${RESOLVED_CONFIG}" ] || { echo "--resolved is required for internal head" >&2; exit 1; }
    run_head
    ;;
  worker)
    [ -n "${RESOLVED_CONFIG}" ] || { echo "--resolved is required for internal worker" >&2; exit 1; }
    run_worker
    ;;
  *)
    echo "Unknown internal role: ${INTERNAL_ROLE}" >&2
    exit 1
    ;;
esac
