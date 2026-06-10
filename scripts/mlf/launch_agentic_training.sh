#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
REPO_DIR=${REPO_DIR:-${MLF_NAS_ROOT}/code/slime}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
WANDB_SECRET_FILE=${WANDB_SECRET_FILE:-${MLF_NAS_ROOT}/secrets/wandb.env}
LOG_DIR=${LOG_DIR:-${MLF_LOCAL_ROOT}/logs}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1,${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1,${NO_PROXY:-}"
if [ -f "${WANDB_SECRET_FILE}" ]; then
  # Expected keys: WANDB_API_KEY, optionally WANDB_BASE_URL/WANDB_ENTITY.
  # Keep this file out of git and chmod it to 600 on NAS.
  # shellcheck disable=SC1090
  source "${WANDB_SECRET_FILE}"
fi

ENV_NAME=webshop
DATA_SIZE=${WEBSHOP_DATA_SIZE:-full}
NODES_FILE=""
ROLE=auto
ORCHESTRATOR=${ORCHESTRATOR:-head}
NO_REMOTE_WORKERS=0
NO_ROUTER=0
ENV_PORT=18180
ROUTER_PORT=19000
RAY_PORT=6379
ENV_POOL_SIZE=""
TRAIN_CMD=""
AUX_NODE=${AUX_NODE:-}
AUX_NODES_FILE=${AUX_NODES_FILE:-configs/nodes/agent_env_tau2_aux_1x8.txt}
AUX_PORT=${AUX_PORT:-18080}
AUX_MODEL=${AUX_MODEL:-${MLF_NAS_ROOT}/models/Qwen3.5-9B}
AUX_GPUS=${AUX_GPUS:-0,1,2,3,4,5,6,7}
AUX_TP=${AUX_TP:-8}
AUX_MEM_FRACTION=${AUX_MEM_FRACTION:-0.65}
AUX_REASONING_PARSER=${AUX_REASONING_PARSER:-qwen3}
AUX_TOOL_CALL_PARSER=${AUX_TOOL_CALL_PARSER:-}
TAU2_USER_MODEL=${TAU2_USER_MODEL:-tau2-user-sim}
TAU2_USER_MODEL_API_KEY=${TAU2_USER_MODEL_API_KEY:-dummy}
TAU2_USER_MODEL_BASE_URL=${TAU2_USER_MODEL_BASE_URL:-}
TAU2_DATA_SOURCE=${TAU2_DATA_SOURCE:-areal_synthetic}
TAU2_DOMAINS=${TAU2_DOMAINS:-retail,airline,telecom}
TAU2_DOMAIN_WEIGHTS=${TAU2_DOMAIN_WEIGHTS:-}
TAU2_TASK_SETS=${TAU2_TASK_SETS:-}
TAU2_PROMPT_NUM_TASKS=${TAU2_PROMPT_NUM_TASKS:-all}
TAU2_PROMPT_SPLIT=${TAU2_PROMPT_SPLIT:-train}
TAU2_PROMPT_SEED=${TAU2_PROMPT_SEED:-42}
TAU2_AREAL_ROOT=${TAU2_AREAL_ROOT:-${MLF_LOCAL_ROOT}/data/tau2/areal_synthetic}
TAU2_AREAL_INPUT=${TAU2_AREAL_INPUT:-tau2_rl_train.jsonl}
TAU2_TASK_FILE_DIR=${TAU2_TASK_FILE_DIR:-}
BENCH_ON_TRAIN_EXIT=${BENCH_ON_TRAIN_EXIT:-1}
BENCH_CMD=${BENCH_CMD:-}
BENCH_GPUS=${BENCH_GPUS:-${RAY_CUDA_VISIBLE_DEVICES:-}}
AUX_BENCH_ON_TRAIN_EXIT=${AUX_BENCH_ON_TRAIN_EXIT:-1}
AUX_BENCH_GPUS=${AUX_BENCH_GPUS:-${AUX_GPUS}}
AUX_BENCH_CMD=${AUX_BENCH_CMD:-}
AUX_BENCH_NODES_FILE=${AUX_BENCH_NODES_FILE:-}
STARTED_AUX_SERVER=0
DRY_RUN=0
HEAD_ADDRESS=${HEAD_ADDRESS:-}
ROUTER_WORKERS=${ROUTER_WORKERS:-}
SSH_USER=${SSH_USER:-tiger}
SSH_PORT=${SSH_PORT:-10413}
SSH_KEY=${SSH_KEY:-~/.ssh/byte_id_rsa}
SSH_KEY=${SSH_KEY/#\~/${HOME}}
SSH_JUMP=${SSH_JUMP-jump-proxy-arnold-hl.byted.org}
SSH_CONFIG=${SSH_CONFIG:-}
SSH_IPV6=${SSH_IPV6:-1}
RAY_CUDA_VISIBLE_DEVICES=${RAY_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}

usage() {
  cat <<'EOF'
Usage: launch_agentic_training.sh [options]

Start agentic training infrastructure. Runtime preparation is intentionally
separate; run prepare_agentic_runtime.sh first.

Options:
  --env NAME           webshop, alfworld, tau2, or appworld
  --nodes FILE         Node list. Use a one-line file or omit for single-node.
  --role ROLE          auto, head, worker
  --orchestrator MODE  head or local. head submits one tmux on node0; local SSHes to every node.
  --no-remote-workers  Internal option: head starts local services only
  --no-router          Internal option: head does not start router/train
  --env-port PORT      Per-node env server port
  --router-port PORT   Head env router port
  --ray-port PORT      Ray head port
  --env-pool-size N    Env worker pool size
  --data-size SIZE     WebShop data size: small or full
  --train-cmd CMD      Command to execute on head after infra is ready
  --aux-node HOST
                       Start a dedicated OpenAI-compatible aux SGLang server on HOST.
                       Alternatively set TAU2_USER_MODEL_BASE_URL to use an existing API.
  --aux-port PORT
  --aux-gpus CSV
  --aux-model PATH
  --no-bench-on-exit   Do not restart GPU bench after the train driver exits
  RAY_CUDA_VISIBLE_DEVICES can restrict Ray/training to a GPU subset, e.g. 0,1,2,3.
  --head-address HOST  Internal option for worker role
  --router-workers CSV Internal option: explicit env server worker URLs
  --dry-run            Print actions only
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_NAME=$2; shift 2 ;;
    --nodes) NODES_FILE=$2; shift 2 ;;
    --role) ROLE=$2; shift 2 ;;
    --orchestrator) ORCHESTRATOR=$2; shift 2 ;;
    --no-remote-workers) NO_REMOTE_WORKERS=1; shift ;;
    --no-router) NO_ROUTER=1; shift ;;
    --env-port) ENV_PORT=$2; shift 2 ;;
    --router-port) ROUTER_PORT=$2; shift 2 ;;
    --ray-port) RAY_PORT=$2; shift 2 ;;
    --env-pool-size) ENV_POOL_SIZE=$2; shift 2 ;;
    --data-size) DATA_SIZE=$2; shift 2 ;;
    --train-cmd) TRAIN_CMD=$2; shift 2 ;;
    --aux-node) AUX_NODE=$2; shift 2 ;;
    --aux-port) AUX_PORT=$2; shift 2 ;;
    --aux-gpus) AUX_GPUS=$2; shift 2 ;;
    --aux-model) AUX_MODEL=$2; shift 2 ;;
    --no-bench-on-exit) BENCH_ON_TRAIN_EXIT=0; shift ;;
    --head-address) HEAD_ADDRESS=$2; shift 2 ;;
    --router-workers) ROUTER_WORKERS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

SLIME_ENV=${SLIME_ENV:-${MLF_LOCAL_ENVS}/slime}
SLIME_PYTHON="${SLIME_ENV}/bin/python"

read_nodes() {
  if [ -z "${NODES_FILE}" ]; then
    echo "this"
    return
  fi
  awk 'NF && $1 !~ /^#/ {print $1}' "${NODES_FILE}"
}

first_node() {
  read_nodes | head -n 1
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

fill_ssh_args() {
  local host=$1
  SSH_ARGS=()
  if [ -n "${SSH_CONFIG}" ]; then
    SSH_ARGS+=("-F" "${SSH_CONFIG}")
  fi
  if [ "${SSH_IPV6}" = "1" ]; then
    SSH_ARGS+=("-6")
  fi
  SSH_ARGS+=(
    "-o" "BatchMode=yes"
    "-o" "StrictHostKeyChecking=no"
    "-o" "UserKnownHostsFile=/dev/null"
    "-o" "GlobalKnownHostsFile=/dev/null"
    "-o" "CheckHostIP=no"
    "-o" "IdentitiesOnly=yes"
    "-i" "${SSH_KEY}"
    "-p" "${SSH_PORT}"
  )
  if [ -n "${SSH_JUMP}" ]; then
    SSH_ARGS+=("-J" "${SSH_JUMP}")
  fi
  SSH_ARGS+=("${SSH_USER}@${host}")
}

print_ssh_cmd() {
  printf 'ssh '
  printf '%q ' "${SSH_ARGS[@]}"
}

debug_ssh_cmd() {
  [ "${DEBUG_SSH:-0}" = "1" ] || return 0
  printf '+ '
  print_ssh_cmd
  printf '%q\n' "$1"
}

run_cmd() {
  echo "+ $*"
  if [ "${DRY_RUN}" -eq 0 ]; then
    "$@"
  fi
}

require_runtime() {
  [ "${DRY_RUN}" -eq 0 ] || return 0
  [ -x "${SLIME_PYTHON}" ] || { echo "Missing slime env: ${SLIME_PYTHON}" >&2; exit 1; }
  case "${ENV_NAME}" in
    webshop)
      [ -x "${MLF_LOCAL_ENVS}/webshop/bin/python" ] || { echo "Missing WebShop env" >&2; exit 1; }
      [ -d "${MLF_LOCAL_ROOT}/code/WebShop" ] || { echo "Missing WebShop source mirror" >&2; exit 1; }
      [ -d "${MLF_LOCAL_ROOT}/data/webshop" ] || { echo "Missing WebShop data" >&2; exit 1; }
      ;;
    alfworld)
      [ -x "${MLF_LOCAL_ENVS}/alfworld/bin/python" ] || { echo "Missing ALFWorld env" >&2; exit 1; }
      [ -d "${MLF_LOCAL_ROOT}/data/alfworld" ] || { echo "Missing ALFWorld data" >&2; exit 1; }
      ;;
    tau2)
      [ -x "${MLF_LOCAL_ENVS}/tau2/bin/python" ] || { echo "Missing tau2 env" >&2; exit 1; }
      [ -d "${MLF_LOCAL_ROOT}/data/tau2" ] || { echo "Missing tau2 data" >&2; exit 1; }
      ;;
    appworld)
      [ -x "${MLF_LOCAL_ENVS}/appworld/bin/python" ] || { echo "Missing AppWorld env" >&2; exit 1; }
      [ -d "${MLF_LOCAL_ROOT}/data/appworld" ] || { echo "Missing AppWorld data" >&2; exit 1; }
      ;;
    *) echo "Unsupported env: ${ENV_NAME}" >&2; exit 1 ;;
  esac
}

ensure_log_dir() {
  mkdir -p "${LOG_DIR}"
}

infer_tool_call_parser_from_model() {
  local model_path=$1
  "${SLIME_PYTHON}" - <<PY
import json
from pathlib import Path

path = Path("${model_path}") / "tokenizer_config.json"
template = ""
try:
    template = str((json.loads(path.read_text()).get("chat_template") or ""))
except Exception:
    template = ""

if "<function=" in template and "<parameter=" in template:
    print("qwen3_coder")
elif "<tool_call>" in template and '"name"' in template and '"arguments"' in template:
    print("qwen")
else:
    print("qwen")
PY
}

write_webshop_config() {
  local config="${MLF_LOCAL_ROOT}/configs/webshop_launch.yaml"
  local pool_size="${ENV_POOL_SIZE:-32}"
  local product_file attr_file num_products
  if [ "${DATA_SIZE}" = "full" ] || [ "${DATA_SIZE}" = "all" ]; then
    product_file="${MLF_LOCAL_ROOT}/data/webshop/data/items_shuffle.json"
    attr_file="${MLF_LOCAL_ROOT}/data/webshop/data/items_ins_v2.json"
    num_products=100000
  else
    product_file="${MLF_LOCAL_ROOT}/data/webshop/data/items_shuffle_1000.json"
    attr_file="${MLF_LOCAL_ROOT}/data/webshop/data/items_ins_v2_1000.json"
    num_products=1000
  fi
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${config}"
    return
  fi
  mkdir -p "${MLF_LOCAL_ROOT}/configs"
  "${SLIME_PYTHON}" - <<PY
from pathlib import Path
import yaml
base = Path("${REPO_DIR}/examples/agent_env/webshop/train_config.yaml")
cfg = yaml.safe_load(base.read_text()) or {}
cfg.setdefault("webshop", {})
cfg["webshop"]["data_dir"] = "${MLF_LOCAL_ROOT}/data/webshop"
cfg["webshop"]["product_file"] = "${product_file}"
cfg["webshop"]["attr_file"] = "${attr_file}"
cfg["webshop"]["num_products"] = ${num_products}
cfg.setdefault("env_server", {})
cfg["env_server"]["pool_size"] = ${pool_size}
cfg["env_server"]["worker_start_timeout_s"] = 600
Path("${config}").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${config}"
}

write_alfworld_config() {
  local config="${MLF_LOCAL_ROOT}/configs/alfworld_launch.yaml"
  local pool_size="${ENV_POOL_SIZE:-32}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${config}"
    return
  fi
  mkdir -p "${MLF_LOCAL_ROOT}/configs"
  "${SLIME_PYTHON}" - <<PY
from pathlib import Path
import yaml

base = Path("${REPO_DIR}/examples/agent_env/alfworld/train_config.yaml")
cfg = yaml.safe_load(base.read_text()) or {}
cfg.setdefault("alfworld", {})
cfg["alfworld"]["data_dir"] = "${MLF_LOCAL_ROOT}/data/alfworld"
cfg.setdefault("env_server", {})
cfg["env_server"]["pool_size"] = int("${pool_size}")
cfg["env_server"]["worker_start_timeout_s"] = 600
Path("${config}").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${config}"
}

write_tau2_config() {
  local config="${MLF_LOCAL_ROOT}/configs/tau2_launch.yaml"
  local pool_size="${ENV_POOL_SIZE:-8}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${config}"
    return
  fi
  mkdir -p "${MLF_LOCAL_ROOT}/configs"
  TAU2_CONFIG_PATH="${config}" \
  TAU2_CONFIG_POOL_SIZE="${pool_size}" \
  TAU2_CONFIG_REPO_DIR="${REPO_DIR}" \
  TAU2_CONFIG_LOCAL_ROOT="${MLF_LOCAL_ROOT}" \
  TAU2_CONFIG_LOCAL_ENVS="${MLF_LOCAL_ENVS}" \
  TAU2_CONFIG_USER_MODEL_BASE_URL="${TAU2_USER_MODEL_BASE_URL}" \
  TAU2_CONFIG_USER_MODEL="${TAU2_USER_MODEL}" \
  TAU2_CONFIG_DATA_SOURCE="${TAU2_DATA_SOURCE}" \
  TAU2_CONFIG_DOMAINS="${TAU2_DOMAINS}" \
  TAU2_CONFIG_DOMAIN_WEIGHTS="${TAU2_DOMAIN_WEIGHTS}" \
  TAU2_CONFIG_TASK_SETS="${TAU2_TASK_SETS}" \
  TAU2_CONFIG_PROMPT_NUM_TASKS="${TAU2_PROMPT_NUM_TASKS}" \
  TAU2_CONFIG_PROMPT_SPLIT="${TAU2_PROMPT_SPLIT}" \
  TAU2_CONFIG_PROMPT_SEED="${TAU2_PROMPT_SEED}" \
  TAU2_CONFIG_AREAL_ROOT="${TAU2_AREAL_ROOT}" \
  TAU2_CONFIG_AREAL_INPUT="${TAU2_AREAL_INPUT}" \
  TAU2_CONFIG_TASK_FILE_DIR="${TAU2_TASK_FILE_DIR}" \
  "${SLIME_PYTHON}" - <<'PY'
import os
from pathlib import Path
import yaml

def scalar(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

base = Path(scalar("TAU2_CONFIG_REPO_DIR")) / "examples/agent_env/tau2/train_config.yaml"
cfg = yaml.safe_load(base.read_text()) or {}
cfg.setdefault("tau2", {})
local_root = scalar("TAU2_CONFIG_LOCAL_ROOT")
cfg["tau2"]["data_dir"] = f"{local_root}/data/tau2/data"
if scalar("TAU2_CONFIG_USER_MODEL_BASE_URL"):
    cfg["tau2"]["user_model_base_url"] = scalar("TAU2_CONFIG_USER_MODEL_BASE_URL")
cfg["tau2"]["user_model"] = scalar("TAU2_CONFIG_USER_MODEL")
prompt_data = cfg.setdefault("agent_prompt_data", {})
prompt_data["script"] = f"{scalar('TAU2_CONFIG_REPO_DIR')}/examples/agent_env/tau2/prompt_data.py"
prompt_data["python"] = f"{scalar('TAU2_CONFIG_LOCAL_ENVS')}/tau2/bin/python"
prompt_data["source"] = scalar("TAU2_CONFIG_DATA_SOURCE", "official")
prompt_data["data_dir"] = f"{local_root}/data/tau2/data"
prompt_data["areal_root"] = scalar("TAU2_CONFIG_AREAL_ROOT")
prompt_data["areal_input"] = scalar("TAU2_CONFIG_AREAL_INPUT", "tau2_rl_train.jsonl")
prompt_data["task_file_dir"] = scalar("TAU2_CONFIG_TASK_FILE_DIR")
prompt_data["domains"] = csv_list(scalar("TAU2_CONFIG_DOMAINS", "retail"))
prompt_data["task_sets"] = scalar("TAU2_CONFIG_TASK_SETS")
prompt_data["domain_weights"] = scalar("TAU2_CONFIG_DOMAIN_WEIGHTS")
prompt_data["split"] = scalar("TAU2_CONFIG_PROMPT_SPLIT", "train")
prompt_data["num_tasks"] = scalar("TAU2_CONFIG_PROMPT_NUM_TASKS", "all")
prompt_data["seed"] = int(scalar("TAU2_CONFIG_PROMPT_SEED", "42"))
prompt_data["output_dir"] = f"{local_root}/data/tau2/{prompt_data['source']}_prompt"
cfg.setdefault("env_server", {})
cfg["env_server"]["pool_size"] = int(scalar("TAU2_CONFIG_POOL_SIZE", "8"))
cfg["env_server"]["worker_start_timeout_s"] = 600
Path(scalar("TAU2_CONFIG_PATH")).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${config}"
}

sglang_health_url() {
  local base=$1
  if [[ "${base}" == */v1 ]]; then
    printf '%s/models' "${base}"
  else
    printf '%s/v1/models' "${base%/}"
  fi
}

wait_aux_http() {
  local url=$1
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "+ wait_aux_http ${url}"
    return 0
  fi
  for _ in $(seq 1 300); do
    if "${SLIME_PYTHON}" - <<PY 2>/dev/null
import json, urllib.request
data = json.loads(urllib.request.urlopen("${url}", timeout=2).read().decode())
raise SystemExit(0 if "data" in data else 1)
PY
    then
      echo "ready: ${url}"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

remote_wait_aux_http() {
  local host=$1
  local url=$2
  echo "Waiting ${host} ${url}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "+ remote_wait_aux_http ${host} ${url}"
    return 0
  fi
  local remote_cmd py_code
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
  remote_cmd=$(printf '%q ' bash -lc "$(printf '%q ' "${SLIME_PYTHON}" -c "${py_code}")")
  fill_ssh_args "${host}"
  ssh "${SSH_ARGS[@]}" "${remote_cmd}"
}

start_aux_server() {
  [ "${ENV_NAME}" = "tau2" ] || return 0
  if [ -n "${TAU2_USER_MODEL_BASE_URL}" ]; then
    export TAU2_USER_MODEL_BASE_URL TAU2_USER_MODEL TAU2_USER_MODEL_API_KEY
    return 0
  fi
  if [ -z "${AUX_NODE}" ] && [ -n "${AUX_NODES_FILE}" ]; then
    local aux_nodes_file="${AUX_NODES_FILE}"
    if [[ "${aux_nodes_file}" != /* ]]; then
      aux_nodes_file="${REPO_DIR}/${aux_nodes_file}"
    fi
    if [ -f "${aux_nodes_file}" ]; then
      AUX_NODE=$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${aux_nodes_file}")
    fi
  fi
  if [ -z "${AUX_NODE}" ]; then
    echo "tau2 aux service requires TAU2_USER_MODEL_BASE_URL or --aux-node HOST." >&2
    echo "For a dedicated local service, pass --aux-node <host> and reserve its GPUs outside Ray/training." >&2
    exit 1
  fi
  local node_addr="${AUX_NODE}"
  ensure_log_dir
  AUX_BENCH_NODES_FILE=${AUX_BENCH_NODES_FILE:-${LOG_DIR}/tau2_aux_nodes.txt}
  [ "${DRY_RUN}" -eq 1 ] || printf '%s\n' "${AUX_NODE}" > "${AUX_BENCH_NODES_FILE}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    node_addr="${AUX_NODE}"
  elif [ "${AUX_NODE}" = "this" ]; then
    node_addr=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  elif is_current_node "${AUX_NODE}"; then
    node_addr=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  else
    node_addr=$(remote_first_ip "${AUX_NODE}")
  fi
  export TAU2_USER_MODEL_BASE_URL="http://$(http_host "${node_addr}"):${AUX_PORT}/v1"
  export TAU2_USER_MODEL TAU2_USER_MODEL_API_KEY
  local aux_tool_call_parser="${AUX_TOOL_CALL_PARSER:-$(infer_tool_call_parser_from_model "${AUX_MODEL}")}"
  echo "tau2 aux tool-call parser: ${aux_tool_call_parser} (model: ${AUX_MODEL})"

  local session="mlf_tau2_aux"
  local log="${LOG_DIR}/tau2_aux.log"
  local serve_cmd
  serve_cmd=$(printf 'cd %q && mkdir -p %q && export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q no_proxy=%q NO_PROXY=%q && %q -m sglang.launch_server --model-path %q --served-model-name %q --host 0.0.0.0 --port %q --tp-size %q --mem-fraction-static %q --reasoning-parser %q --tool-call-parser %q --trust-remote-code > %q 2>&1' \
    "${REPO_DIR}" \
    "${LOG_DIR}" \
    "${AUX_GPUS}" \
    "${REPO_DIR}:${SLIME_ENV}/lib/python3.12/site-packages" \
    "${no_proxy}" \
    "${NO_PROXY}" \
    "${SLIME_PYTHON}" \
    "${AUX_MODEL}" \
    "${TAU2_USER_MODEL}" \
    "${AUX_PORT}" \
    "${AUX_TP}" \
    "${AUX_MEM_FRACTION}" \
    "${AUX_REASONING_PARSER}" \
    "${aux_tool_call_parser}" \
    "${log}")
  if is_current_node "${AUX_NODE}" || [ "${AUX_NODE}" = "this" ]; then
    run_cmd tmux kill-session -t "${session}" 2>/dev/null || true
    run_cmd tmux new-session -d -s "${session}" "${serve_cmd}"
    wait_aux_http "$(sglang_health_url "${TAU2_USER_MODEL_BASE_URL}")"
  else
    remote_start_tmux "${AUX_NODE}" "${session}" "${log}" "${serve_cmd}"
    remote_wait_aux_http "${AUX_NODE}" "http://127.0.0.1:${AUX_PORT}/v1/models"
  fi
  STARTED_AUX_SERVER=1
  echo "tau2 aux API: ${TAU2_USER_MODEL_BASE_URL}"
  echo "tau2 aux log: ${AUX_NODE}:${log}"
}

write_appworld_config() {
  local config="${MLF_LOCAL_ROOT}/configs/appworld_launch.yaml"
  local pool_size="${ENV_POOL_SIZE:-4}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${config}"
    return
  fi
  mkdir -p "${MLF_LOCAL_ROOT}/configs"
  "${SLIME_PYTHON}" - <<PY
from pathlib import Path
import yaml
base = Path("${REPO_DIR}/examples/agent_env/appworld/train_config.yaml")
cfg = yaml.safe_load(base.read_text()) or {}
cfg.setdefault("appworld", {})
cfg["appworld"]["root"] = "${MLF_LOCAL_ROOT}/data/appworld"
cfg.setdefault("env_server", {})
cfg["env_server"]["pool_size"] = ${pool_size}
cfg["env_server"]["worker_start_timeout_s"] = 600
Path("${config}").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${config}"
}

start_env_server() {
  require_runtime
  ensure_log_dir
  case "${ENV_NAME}" in
    webshop)
      local config
      config=$(write_webshop_config)
      run_cmd tmux kill-session -t "mlf_${ENV_NAME}_env" 2>/dev/null || true
      run_cmd tmux new-session -d -s "mlf_${ENV_NAME}_env" \
        "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 WEBSHOP_LIB=${MLF_LOCAL_ROOT}/code/WebShop WEBSHOP_DATA=${MLF_LOCAL_ROOT}/data/webshop JAVA_HOME=${MLF_LOCAL_ENVS}/webshop/lib/jvm JVM_PATH=${MLF_LOCAL_ENVS}/webshop/lib/jvm/lib/server/libjvm.so PYTHONPATH=${REPO_DIR}:${MLF_LOCAL_ROOT}/code/WebShop && ${MLF_LOCAL_ENVS}/webshop/bin/python examples/agent_env/webshop/server.py --host 0.0.0.0 --port ${ENV_PORT} --config ${config} > ${LOG_DIR}/${ENV_NAME}_env_server.log 2>&1"
      ;;
    alfworld)
      local config
      config=$(write_alfworld_config)
      run_cmd tmux kill-session -t "mlf_${ENV_NAME}_env" 2>/dev/null || true
      run_cmd tmux new-session -d -s "mlf_${ENV_NAME}_env" \
        "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 ALFWORLD_DATA=${MLF_LOCAL_ROOT}/data/alfworld ALFWORLD_LIB=${MLF_NAS_ROOT}/code/alfworld PYTHONPATH=${REPO_DIR} && ${MLF_LOCAL_ENVS}/alfworld/bin/python examples/agent_env/alfworld/server.py --host 0.0.0.0 --port ${ENV_PORT} --config ${config} > ${LOG_DIR}/${ENV_NAME}_env_server.log 2>&1"
      ;;
    tau2)
      local config
      config=$(write_tau2_config)
      run_cmd tmux kill-session -t "mlf_${ENV_NAME}_env" 2>/dev/null || true
      run_cmd tmux new-session -d -s "mlf_${ENV_NAME}_env" \
        "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 LITELLM_LOCAL_MODEL_COST_MAP=True TAU2_DATA_DIR=${MLF_LOCAL_ROOT}/data/tau2/data PYTHONPATH=${REPO_DIR} && ${MLF_LOCAL_ENVS}/tau2/bin/python examples/agent_env/tau2/server.py --host 0.0.0.0 --port ${ENV_PORT} --config ${config} > ${LOG_DIR}/${ENV_NAME}_env_server.log 2>&1"
      ;;
    appworld)
      local config
      config=$(write_appworld_config)
      run_cmd tmux kill-session -t "mlf_${ENV_NAME}_env" 2>/dev/null || true
      run_cmd tmux new-session -d -s "mlf_${ENV_NAME}_env" \
        "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 HOME=${MLF_LOCAL_ROOT}/data/appworld APPWORLD_ROOT=${MLF_LOCAL_ROOT}/data/appworld PYTHONPATH=${REPO_DIR} && ${MLF_LOCAL_ENVS}/appworld/bin/python examples/agent_env/appworld/server.py --host 0.0.0.0 --port ${ENV_PORT} --config ${config} > ${LOG_DIR}/${ENV_NAME}_env_server.log 2>&1"
      ;;
  esac
}

start_ray_head() {
  local node_ip=${HEAD_ADDRESS:-}
  if [ -z "${node_ip}" ]; then
    node_ip=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  fi
  run_cmd tmux kill-session -t mlf_ray_head 2>/dev/null || true
  run_cmd "${SLIME_PYTHON}" -m ray.scripts.scripts stop --force
  run_cmd tmux new-session -d -s mlf_ray_head \
    "export PYTHONNOUSERSITE=1 RAY_DISABLE_DOCKER_CPU_WARNING=1; if [ -n '${RAY_CUDA_VISIBLE_DEVICES}' ]; then export CUDA_VISIBLE_DEVICES='${RAY_CUDA_VISIBLE_DEVICES}'; fi; mkdir -p ${LOG_DIR}; ${SLIME_PYTHON} -m ray.scripts.scripts start --head --node-ip-address ${node_ip} --port ${RAY_PORT} --num-gpus ${NUM_GPUS_PER_NODE_FOR_RAY:-${NUM_GPUS:-4}} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --block > ${LOG_DIR}/ray_head.log 2>&1"
}

start_ray_worker() {
  local head=$1
  local node_ip
  node_ip=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  run_cmd tmux kill-session -t mlf_ray_worker 2>/dev/null || true
  run_cmd "${SLIME_PYTHON}" -m ray.scripts.scripts stop --force
  run_cmd tmux new-session -d -s mlf_ray_worker \
    "export PYTHONNOUSERSITE=1 RAY_DISABLE_DOCKER_CPU_WARNING=1; if [ -n '${RAY_CUDA_VISIBLE_DEVICES}' ]; then export CUDA_VISIBLE_DEVICES='${RAY_CUDA_VISIBLE_DEVICES}'; fi; mkdir -p ${LOG_DIR}; ${SLIME_PYTHON} -m ray.scripts.scripts start --address ${head}:${RAY_PORT} --node-ip-address ${node_ip} --num-gpus ${NUM_GPUS_PER_NODE_FOR_RAY:-${NUM_GPUS:-4}} --disable-usage-stats --block > ${LOG_DIR}/ray_worker.log 2>&1"
}

wait_http() {
  local url=$1
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "+ wait_http ${url}"
    return 0
  fi
  for _ in $(seq 1 300); do
    if "${SLIME_PYTHON}" - <<PY 2>/dev/null
import json, urllib.request
data = json.loads(urllib.request.urlopen("${url}", timeout=2).read().decode())
raise SystemExit(0 if data.get("ok") else 1)
PY
    then
      echo "ready: ${url}"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

remote_wait_http() {
  local host=$1
  local url=$2
  echo "Waiting ${host} ${url}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "+ remote_wait_http ${host} ${url}"
    return 0
  fi
  local remote_cmd py_code
  py_code="import json,urllib.request,time,sys; url='${url}'; ok=False
for _ in range(300):
    try:
        data=json.loads(urllib.request.urlopen(url,timeout=2).read().decode())
        if data.get('ok'):
            print('ready', url); ok=True; break
    except Exception:
        pass
    time.sleep(2)
sys.exit(0 if ok else 1)"
  remote_cmd=$(printf '%q ' bash -lc "$(printf '%q ' "${SLIME_PYTHON}" -c "${py_code}")")
  fill_ssh_args "${host}"
  ssh "${SSH_ARGS[@]}" "${remote_cmd}"
}

remote_ray_alive_nodes() {
  local host=$1
  local remote_cmd py_code
  py_code="import ray; ray.init(address='127.0.0.1:${RAY_PORT}', ignore_reinit_error=True); print(sum(1 for n in ray.nodes() if n.get('Alive'))); ray.shutdown()"
  remote_cmd=$(printf '%q ' bash -lc "$(printf '%q ' "${SLIME_PYTHON}" -c "${py_code}")")
  fill_ssh_args "${host}"
  ssh "${SSH_ARGS[@]}" "${remote_cmd}"
}

local_ray_alive_nodes() {
  local py_code
  py_code="import ray; ray.init(address='127.0.0.1:${RAY_PORT}', ignore_reinit_error=True); print(sum(1 for n in ray.nodes() if n.get('Alive'))); ray.shutdown()"
  "${SLIME_PYTHON}" -c "${py_code}"
}

wait_ray_nodes() {
  local head=$1
  local expected=$2
  echo "Waiting Ray nodes on ${head}: expected=${expected}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "+ wait_ray_nodes ${head} ${expected}"
    return 0
  fi
  local alive
  for _ in $(seq 1 180); do
    alive=$(remote_ray_alive_nodes "${head}" 2>/dev/null | tail -n 1 || true)
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

wait_local_ray_nodes() {
  local expected=$1
  echo "Waiting local Ray nodes: expected=${expected}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "+ wait_local_ray_nodes ${expected}"
    return 0
  fi
  local alive
  for _ in $(seq 1 180); do
    alive=$(local_ray_alive_nodes 2>/dev/null | tail -n 1 || true)
    if [ "${alive:-0}" -ge "${expected}" ] 2>/dev/null; then
      echo "Ray ready: alive=${alive}"
      return 0
    fi
    echo "Ray not ready: alive=${alive:-unknown}/${expected}"
    sleep 5
  done
  echo "Timed out waiting for local Ray nodes" >&2
  return 1
}

start_router() {
  local workers=()
  local node
  local joined
  if [ -n "${ROUTER_WORKERS}" ]; then
    joined="${ROUTER_WORKERS}"
  else
    for node in $(read_nodes); do
      [ "${node}" = "this" ] && node="127.0.0.1"
      workers+=("http://$(http_host "${node}"):${ENV_PORT}")
    done
    joined=$(IFS=,; echo "${workers[*]}")
  fi
  run_cmd tmux kill-session -t "mlf_${ENV_NAME}_router" 2>/dev/null || true
  run_cmd tmux new-session -d -s "mlf_${ENV_NAME}_router" \
    "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 && mkdir -p ${LOG_DIR} && ${SLIME_PYTHON} examples/agent_env/router.py --host 0.0.0.0 --port ${ROUTER_PORT} --workers ${joined} > ${LOG_DIR}/${ENV_NAME}_router.log 2>&1"
}

start_train_driver() {
  if [ -z "${TRAIN_CMD}" ]; then
    return 0
  fi
  ensure_log_dir
  local session="mlf_${ENV_NAME}_train"
  local train_log="${LOG_DIR}/${ENV_NAME}_train.log"
  local status_file="${LOG_DIR}/${ENV_NAME}_train_status.env"
  local bench_log="${LOG_DIR}/${ENV_NAME}_bench_on_exit.log"
  local bench_cmd="${BENCH_CMD}"
  if [ -z "${bench_cmd}" ] && [ "${BENCH_ON_TRAIN_EXIT}" = "1" ]; then
    bench_cmd=$(printf 'if [ -f %q ]; then bash %q start' "${MLF_NAS_ROOT}/bash/run_bench.sh" "${MLF_NAS_ROOT}/bash/run_bench.sh")
    if [ -n "${NODES_FILE}" ]; then
      bench_cmd+=" $(printf '%q %q' --nodes "${NODES_FILE}")"
    fi
    if [ -n "${BENCH_GPUS}" ]; then
      bench_cmd+=" $(printf '%q %q' --gpus "${BENCH_GPUS}")"
    fi
    bench_cmd+='; else echo "missing run_bench.sh"; fi'
  fi
  local aux_bench_cmd="${AUX_BENCH_CMD}"
  if [ -z "${aux_bench_cmd}" ] && [ "${BENCH_ON_TRAIN_EXIT}" = "1" ] && [ "${AUX_BENCH_ON_TRAIN_EXIT}" = "1" ] && [ "${STARTED_AUX_SERVER}" = "1" ]; then
    local aux_stop_cmd
    if is_current_node "${AUX_NODE}" || [ "${AUX_NODE}" = "this" ]; then
      aux_stop_cmd='tmux kill-session -t mlf_tau2_aux 2>/dev/null || true'
    else
      local aux_ssh=(ssh)
      if [ "${SSH_IPV6}" = "1" ]; then
        aux_ssh+=("-6")
      fi
      aux_ssh+=(
        "-o" "BatchMode=yes"
        "-o" "StrictHostKeyChecking=no"
        "-o" "UserKnownHostsFile=/dev/null"
        "-o" "GlobalKnownHostsFile=/dev/null"
        "-o" "CheckHostIP=no"
        "-o" "IdentitiesOnly=yes"
        "-i" "${SSH_KEY}"
        "-p" "${SSH_PORT}"
      )
      if [ -n "${SSH_JUMP}" ]; then
        aux_ssh+=("-J" "${SSH_JUMP}")
      fi
      aux_ssh+=("${SSH_USER}@${AUX_NODE}")
      aux_stop_cmd="$(printf '%q ' "${aux_ssh[@]}") $(printf '%q' 'tmux kill-session -t mlf_tau2_aux 2>/dev/null || true')"
    fi
    aux_bench_cmd=$(printf 'if [ -f %q ]; then %s; bash %q start --nodes %q --gpus %q; else echo "missing run_bench.sh"; fi' \
      "${MLF_NAS_ROOT}/bash/run_bench.sh" \
      "${aux_stop_cmd}" \
      "${MLF_NAS_ROOT}/bash/run_bench.sh" \
      "${AUX_BENCH_NODES_FILE}" \
      "${AUX_BENCH_GPUS}")
  fi
  if [ -n "${aux_bench_cmd}" ]; then
    bench_cmd="${bench_cmd:-:}; ${aux_bench_cmd}"
  fi
  local cmd
  cmd=$(printf 'cd %q && mkdir -p %q && on_exit(){ code=$?; printf "exit_code=%%s\nend_time=%%s\n" "$code" "$(date -Is)" > %q; if [ %q = 1 ]; then echo "[bench-on-exit] train exited with code $code at $(date -Is)" >> %q; bash -lc %q >> %q 2>&1 || true; fi; exit "$code"; }; trap on_exit EXIT; printf "state=running\nstart_time=%%s\n" "$(date -Is)" > %q; AGENT_ENV_ROUTER_URL=%q WEBSHOP_ENV_SERVER_URL=%q ALFWORLD_ENV_SERVER_URL=%q TAU2_ENV_SERVER_URL=%q APPWORLD_ENV_SERVER_URL=%q WEBSHOP_DATA_SIZE=%q bash -lc %q > %q 2>&1' \
    "${REPO_DIR}" \
    "${LOG_DIR}" \
    "${status_file}" \
    "${BENCH_ON_TRAIN_EXIT}" \
    "${bench_log}" \
    "${bench_cmd:-:}" \
    "${bench_log}" \
    "${status_file}" \
    "${AGENT_ENV_ROUTER_URL}" \
    "${WEBSHOP_ENV_SERVER_URL}" \
    "${ALFWORLD_ENV_SERVER_URL}" \
    "${TAU2_ENV_SERVER_URL}" \
    "${APPWORLD_ENV_SERVER_URL}" \
    "${WEBSHOP_DATA_SIZE:-${DATA_SIZE}}" \
    "${TRAIN_CMD}" \
    "${train_log}")
  run_cmd tmux kill-session -t "${session}" 2>/dev/null || true
  run_cmd tmux new-session -d -s "${session}" "${cmd}"
  echo "Training submitted in tmux: ${session}"
  echo "Training log: ${train_log}"
  echo "Train status: ${status_file}"
  if [ "${BENCH_ON_TRAIN_EXIT}" = "1" ]; then
    echo "Bench-on-exit log: ${bench_log}"
  fi
}

remote_worker() {
  local host=$1
  local head=$2
  local remote_cmd
  remote_cmd=$(
    printf 'cd %q && MLF_NAS_ROOT=%q MLF_LOCAL_ENVS=%q MLF_LOCAL_ROOT=%q LOG_DIR=%q REPO_DIR=%q bash scripts/mlf/launch_agentic_training.sh ' \
      "${REPO_DIR}" "${MLF_NAS_ROOT}" "${MLF_LOCAL_ENVS}" "${MLF_LOCAL_ROOT}" "${LOG_DIR}" "${REPO_DIR}"
    printf '%q ' --role worker --env "${ENV_NAME}" --env-port "${ENV_PORT}" --ray-port "${RAY_PORT}" --data-size "${DATA_SIZE}" --head-address "${head}"
    [ -z "${ENV_POOL_SIZE}" ] || printf '%q ' --env-pool-size "${ENV_POOL_SIZE}"
    [ "${DRY_RUN}" -eq 0 ] || printf '%q ' --dry-run
  )
  echo "Starting worker ${host}"
  fill_ssh_args "${host}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '+ '
    print_ssh_cmd
    printf '%q\n' "${remote_cmd}"
    return
  fi
  ssh "${SSH_ARGS[@]}" "${remote_cmd}"
}

remote_cmd_prefix() {
  printf 'cd %q && MLF_NAS_ROOT=%q MLF_LOCAL_ENVS=%q MLF_LOCAL_ROOT=%q LOG_DIR=%q REPO_DIR=%q RAY_CUDA_VISIBLE_DEVICES=%q NUM_GPUS_PER_NODE_FOR_RAY=%q AUX_NODE=%q AUX_PORT=%q AUX_MODEL=%q AUX_GPUS=%q AUX_TP=%q AUX_MEM_FRACTION=%q AUX_REASONING_PARSER=%q AUX_TOOL_CALL_PARSER=%q TAU2_USER_MODEL=%q TAU2_USER_MODEL_API_KEY=%q TAU2_USER_MODEL_BASE_URL=%q bash scripts/mlf/launch_agentic_training.sh ' \
    "${REPO_DIR}" "${MLF_NAS_ROOT}" "${MLF_LOCAL_ENVS}" "${MLF_LOCAL_ROOT}" "${LOG_DIR}" "${REPO_DIR}" "${RAY_CUDA_VISIBLE_DEVICES}" "${NUM_GPUS_PER_NODE_FOR_RAY:-${NUM_GPUS:-4}}" \
    "${AUX_NODE}" "${AUX_PORT}" "${AUX_MODEL}" "${AUX_GPUS}" "${AUX_TP}" "${AUX_MEM_FRACTION}" "${AUX_REASONING_PARSER}" "${AUX_TOOL_CALL_PARSER}" "${TAU2_USER_MODEL}" "${TAU2_USER_MODEL_API_KEY}" "${TAU2_USER_MODEL_BASE_URL}"
}

remote_start_tmux() {
  local host=$1
  local session=$2
  local log=$3
  local command=$4
  local remote_cmd
  local attempt
  remote_cmd=$(printf 'mkdir -p %q; tmux kill-session -t %q 2>/dev/null || true; tmux new-session -d -s %q %q' "$(dirname "${log}")" "${session}" "${session}" "${command}")
  echo "Starting ${session} on ${host}"
  fill_ssh_args "${host}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '+ '
    print_ssh_cmd
    printf '%q\n' "${remote_cmd}"
    return
  fi
  debug_ssh_cmd "${remote_cmd}"
  for attempt in 1 2 3; do
    if ssh "${SSH_ARGS[@]}" "${remote_cmd}"; then
      return 0
    fi
    echo "Retry ${attempt}/3 failed for ${session} on ${host}" >&2
    sleep 5
  done
  echo "Failed to start ${session} on ${host}; log hint: ${log}" >&2
  return 1
}

remote_start_router() {
  local head=$1
  local workers_csv=$2
  local cmd
  cmd="$(
    remote_cmd_prefix
    printf '%q ' --role router --env "${ENV_NAME}" --nodes "${NODES_FILE}" --env-port "${ENV_PORT}" --router-port "${ROUTER_PORT}" --ray-port "${RAY_PORT}" --data-size "${DATA_SIZE}"
    printf '%q ' --router-workers "${workers_csv}"
    [ -z "${ENV_POOL_SIZE}" ] || printf '%q ' --env-pool-size "${ENV_POOL_SIZE}"
    [ -z "${TRAIN_CMD}" ] || printf '%q ' --train-cmd "${TRAIN_CMD}"
  ) > ${LOG_DIR}/multi_router.log 2>&1"
  remote_start_tmux "${head}" mlf_multi_router "${LOG_DIR}/multi_router.log" "${cmd}"
}

remote_query() {
  local host=$1
  local command=$2
  local attempt
  fill_ssh_args "${host}"
  for attempt in 1 2 3; do
    if ssh "${SSH_ARGS[@]}" "${command}"; then
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

local_orchestrator() {
  if [ -z "${NODES_FILE}" ]; then
    echo "--orchestrator local requires --nodes FILE" >&2
    exit 1
  fi
  local head head_addr node cmd node_addr
  local env_worker_urls=()
  head=$(first_node)
  if [ -n "${HEAD_ADDRESS}" ]; then
    head_addr="${HEAD_ADDRESS}"
  elif [ "${DRY_RUN}" -eq 1 ]; then
    head_addr="${head}"
  else
    head_addr=$(remote_first_ip "${head}")
  fi
  echo "Local orchestrator: head=${head} head_addr=${head_addr}"
  env_worker_urls+=("http://$(http_host "${head_addr}"):${ENV_PORT}")

  cmd="$(
    remote_cmd_prefix
    printf '%q ' --role head --no-remote-workers --no-router --env "${ENV_NAME}" --nodes "${NODES_FILE}" --env-port "${ENV_PORT}" --router-port "${ROUTER_PORT}" --ray-port "${RAY_PORT}" --data-size "${DATA_SIZE}"
    [ -z "${ENV_POOL_SIZE}" ] || printf '%q ' --env-pool-size "${ENV_POOL_SIZE}"
    [ -z "${TRAIN_CMD}" ] || printf '%q ' --train-cmd "${TRAIN_CMD}"
  ) > ${LOG_DIR}/multi_head.log 2>&1"
  remote_start_tmux "${head}" mlf_multi_head "${LOG_DIR}/multi_head.log" "${cmd}"

  # Give Ray head a short lead before workers join. Env prewarming continues in
  # parallel on every node.
  if [ "${DRY_RUN}" -eq 0 ]; then
    sleep 20
  fi

  for node in $(read_nodes | tail -n +2); do
    if [ "${DRY_RUN}" -eq 1 ]; then
      node_addr="${node}"
    else
      node_addr=$(remote_first_ip "${node}")
    fi
    env_worker_urls+=("http://$(http_host "${node_addr}"):${ENV_PORT}")
    cmd="$(
      remote_cmd_prefix
      printf '%q ' --role worker --env "${ENV_NAME}" --env-port "${ENV_PORT}" --ray-port "${RAY_PORT}" --data-size "${DATA_SIZE}" --head-address "${head_addr}"
      [ -z "${ENV_POOL_SIZE}" ] || printf '%q ' --env-pool-size "${ENV_POOL_SIZE}"
    ) > ${LOG_DIR}/multi_worker.log 2>&1"
    remote_start_tmux "${node}" mlf_multi_worker "${LOG_DIR}/multi_worker.log" "${cmd}"
  done

  for node in $(read_nodes); do
    remote_wait_http "${node}" "http://127.0.0.1:${ENV_PORT}/health"
  done
  wait_ray_nodes "${head}" "$(read_nodes | wc -l | tr -d ' ')"
  local workers_csv
  workers_csv=$(IFS=,; echo "${env_worker_urls[*]}")
  remote_start_router "${head}" "${workers_csv}"
  remote_wait_http "${head}" "http://127.0.0.1:${ROUTER_PORT}/health"

  echo "Multi-node launch submitted."
  echo "Head log: ${head}:${LOG_DIR}/multi_head.log"
  echo "Worker log: <worker>:${LOG_DIR}/multi_worker.log"
}

run_worker() {
  local head_addr=${HEAD_ADDRESS:-${HEAD_NODE:-}}
  if [ -z "${head_addr}" ]; then
    echo "Worker role requires HEAD_NODE or --head-address support from caller" >&2
    exit 1
  fi
  start_env_server
  wait_http "http://127.0.0.1:${ENV_PORT}/health"
  start_ray_worker "${head_addr}"
}

run_head() {
  local head expected_nodes
  local env_worker_urls=()
  if [ -n "${HEAD_ADDRESS}" ]; then
    head="${HEAD_ADDRESS}"
  else
    head=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  fi
  start_aux_server
  start_ray_head
  start_env_server
  wait_http "http://127.0.0.1:${ENV_PORT}/health"
  env_worker_urls+=("http://$(http_host "${head}"):${ENV_PORT}")
  if [ "${NO_ROUTER}" -eq 1 ]; then
    echo "Head local services are ready."
    return
  fi
  if [ -n "${NODES_FILE}" ] && [ "${NO_REMOTE_WORKERS}" -eq 0 ]; then
    local node cmd node_addr
    for node in $(read_nodes | tail -n +2); do
      cmd="$(
        remote_cmd_prefix
        printf '%q ' --role worker --env "${ENV_NAME}" --env-port "${ENV_PORT}" --ray-port "${RAY_PORT}" --data-size "${DATA_SIZE}" --head-address "${head}"
        [ -z "${ENV_POOL_SIZE}" ] || printf '%q ' --env-pool-size "${ENV_POOL_SIZE}"
      ) > ${LOG_DIR}/multi_worker.log 2>&1"
      remote_start_tmux "${node}" mlf_multi_worker "${LOG_DIR}/multi_worker.log" "${cmd}"
    done
    for node in $(read_nodes | tail -n +2); do
      remote_wait_http "${node}" "http://127.0.0.1:${ENV_PORT}/health"
      if [ "${DRY_RUN}" -eq 1 ]; then
        node_addr="${node}"
      else
        node_addr=$(remote_first_ip "${node}")
      fi
      env_worker_urls+=("http://$(http_host "${node_addr}"):${ENV_PORT}")
    done
    expected_nodes=$(read_nodes | wc -l | tr -d ' ')
    wait_local_ray_nodes "${expected_nodes}"
  fi
  ROUTER_WORKERS=$(IFS=,; echo "${env_worker_urls[*]}")
  start_router
  wait_http "http://127.0.0.1:${ROUTER_PORT}/health"
  if [ -n "${TRAIN_CMD}" ]; then
    export AGENT_ENV_ROUTER_URL="http://$(http_host "${head}"):${ROUTER_PORT}"
    export WEBSHOP_ENV_SERVER_URL="${AGENT_ENV_ROUTER_URL}"
    export ALFWORLD_ENV_SERVER_URL="${AGENT_ENV_ROUTER_URL}"
    export TAU2_ENV_SERVER_URL="${AGENT_ENV_ROUTER_URL}"
    export APPWORLD_ENV_SERVER_URL="${AGENT_ENV_ROUTER_URL}"
    export WEBSHOP_DATA_SIZE="${DATA_SIZE}"
    echo "+ submit train driver: ${TRAIN_CMD}"
    [ "${DRY_RUN}" -eq 1 ] || start_train_driver
  else
    echo "Infra is ready. Router: http://127.0.0.1:${ROUTER_PORT}"
  fi
}

if [ "${ROLE}" = "auto" ]; then
  if [ "${ORCHESTRATOR}" = "local" ]; then
    local_orchestrator
    exit 0
  fi
  head=$(first_node)
  if [ -z "${NODES_FILE}" ] || is_current_node "${head}" || [ "${head}" = "this" ]; then
    ROLE=head
    # Once running on the cluster head, worker nodes are reached directly.
    # Keeping the local-machine jump proxy here breaks node-to-node SSH.
    SSH_JUMP=${INTERNAL_SSH_JUMP:-}
  else
    # Delegate orchestration to the first node.
    # From the head, workers are reached directly with the node-local key; the
    # local machine should only maintain this one SSH connection long enough to
    # submit the tmux session.
    remote_cmd=$(
      printf 'cd %q && MLF_NAS_ROOT=%q MLF_LOCAL_ENVS=%q MLF_LOCAL_ROOT=%q LOG_DIR=%q REPO_DIR=%q RAY_CUDA_VISIBLE_DEVICES=%q NUM_GPUS_PER_NODE_FOR_RAY=%q AUX_NODE=%q AUX_PORT=%q AUX_MODEL=%q AUX_GPUS=%q AUX_TP=%q AUX_MEM_FRACTION=%q AUX_REASONING_PARSER=%q AUX_TOOL_CALL_PARSER=%q TAU2_USER_MODEL=%q TAU2_USER_MODEL_API_KEY=%q TAU2_USER_MODEL_BASE_URL=%q SSH_JUMP= SSH_KEY=%q SSH_IPV6=1 bash scripts/mlf/launch_agentic_training.sh ' \
        "${REPO_DIR}" "${MLF_NAS_ROOT}" "${MLF_LOCAL_ENVS}" "${MLF_LOCAL_ROOT}" "${LOG_DIR}" "${REPO_DIR}" "${RAY_CUDA_VISIBLE_DEVICES}" "${NUM_GPUS_PER_NODE_FOR_RAY:-${NUM_GPUS:-4}}" \
        "${AUX_NODE}" "${AUX_PORT}" "${AUX_MODEL}" "${AUX_GPUS}" "${AUX_TP}" "${AUX_MEM_FRACTION}" "${AUX_REASONING_PARSER}" "${AUX_TOOL_CALL_PARSER}" "${TAU2_USER_MODEL}" "${TAU2_USER_MODEL_API_KEY}" "${TAU2_USER_MODEL_BASE_URL}" "/home/${SSH_USER}/.ssh/byte_id_rsa"
      printf '%q ' --role head --env "${ENV_NAME}" --nodes "${NODES_FILE}" --env-port "${ENV_PORT}" --router-port "${ROUTER_PORT}" --ray-port "${RAY_PORT}" --data-size "${DATA_SIZE}"
      [ -z "${ENV_POOL_SIZE}" ] || printf '%q ' --env-pool-size "${ENV_POOL_SIZE}"
      [ -z "${TRAIN_CMD}" ] || printf '%q ' --train-cmd "${TRAIN_CMD}"
    )
    remote_start_tmux "${head}" mlf_multi_head "${LOG_DIR}/multi_head.log" "${remote_cmd} > ${LOG_DIR}/multi_head.log 2>&1"
    echo "Head orchestration submitted."
    echo "Head log: ${head}:${LOG_DIR}/multi_head.log"
    exit 0
  fi
fi

case "${ROLE}" in
  head) run_head ;;
  worker) run_worker ;;
  router)
    start_router
    wait_http "http://127.0.0.1:${ROUTER_PORT}/health"
    if [ -n "${TRAIN_CMD}" ]; then
      export AGENT_ENV_ROUTER_URL="http://$(http_host "$(hostname -I | tr ' ' '\n' | grep -m1 .)"):${ROUTER_PORT}"
      export WEBSHOP_ENV_SERVER_URL="${AGENT_ENV_ROUTER_URL}"
      export ALFWORLD_ENV_SERVER_URL="${AGENT_ENV_ROUTER_URL}"
      export TAU2_ENV_SERVER_URL="${AGENT_ENV_ROUTER_URL}"
      export APPWORLD_ENV_SERVER_URL="${AGENT_ENV_ROUTER_URL}"
      export WEBSHOP_DATA_SIZE="${DATA_SIZE}"
      echo "+ submit train driver: ${TRAIN_CMD}"
      [ "${DRY_RUN}" -eq 1 ] || start_train_driver
    else
      echo "Router is ready. Router: http://127.0.0.1:${ROUTER_PORT}"
    fi
    ;;
  *) echo "Unsupported role: ${ROLE}" >&2; exit 1 ;;
esac
