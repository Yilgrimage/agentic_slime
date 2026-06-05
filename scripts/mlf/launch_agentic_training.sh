#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
REPO_DIR=${REPO_DIR:-${MLF_NAS_ROOT}/code/slime}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
WANDB_SECRET_FILE=${WANDB_SECRET_FILE:-${MLF_NAS_ROOT}/secrets/wandb.env}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
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
    --head-address) HEAD_ADDRESS=$2; shift 2 ;;
    --router-workers) ROUTER_WORKERS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

SLIME_PYTHON="${MLF_LOCAL_ENVS}/slime/bin/python"

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
    "-o" "StrictHostKeyChecking=no"
    "-o" "UserKnownHostsFile=/dev/null"
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

base = Path("${REPO_DIR}/examples/agent_env/alfworld/train_config.yaml")
text = base.read_text()
updates = {
    "alfworld_data_dir": "${MLF_LOCAL_ROOT}/data/alfworld",
    "alfworld_server_pool_size": "${pool_size}",
    "alfworld_server_worker_start_timeout_s": "600",
}
lines = text.splitlines()
seen = set()
out = []
for line in lines:
    key = line.split(":", 1)[0].strip() if ":" in line and not line.startswith((" ", "\t")) else None
    if key in updates:
        out.append(f"{key}: {updates[key]}")
        seen.add(key)
    else:
        out.append(line)
for key, value in updates.items():
    if key not in seen:
        out.append(f"{key}: {value}")
Path("${config}").write_text("\\n".join(out) + "\\n")
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
  "${SLIME_PYTHON}" - <<PY
from pathlib import Path
import yaml
base = Path("${REPO_DIR}/examples/agent_env/tau2/train_config.yaml")
cfg = yaml.safe_load(base.read_text()) or {}
cfg.setdefault("tau2", {})
cfg["tau2"]["data_dir"] = "${MLF_LOCAL_ROOT}/data/tau2/data"
cfg.setdefault("env_server", {})
cfg["env_server"]["pool_size"] = ${pool_size}
cfg["env_server"]["worker_start_timeout_s"] = 600
Path("${config}").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  echo "${config}"
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
  mkdir -p "${MLF_LOCAL_ROOT}/logs"
  case "${ENV_NAME}" in
    webshop)
      local config
      config=$(write_webshop_config)
      run_cmd tmux kill-session -t "mlf_${ENV_NAME}_env" 2>/dev/null || true
      run_cmd tmux new-session -d -s "mlf_${ENV_NAME}_env" \
        "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 WEBSHOP_LIB=${MLF_LOCAL_ROOT}/code/WebShop WEBSHOP_DATA=${MLF_LOCAL_ROOT}/data/webshop JAVA_HOME=${MLF_LOCAL_ENVS}/webshop/lib/jvm JVM_PATH=${MLF_LOCAL_ENVS}/webshop/lib/jvm/lib/server/libjvm.so PYTHONPATH=${REPO_DIR}:${MLF_LOCAL_ROOT}/code/WebShop && ${MLF_LOCAL_ENVS}/webshop/bin/python examples/agent_env/webshop/server.py --host 0.0.0.0 --port ${ENV_PORT} --config ${config} > ${MLF_LOCAL_ROOT}/logs/${ENV_NAME}_env_server.log 2>&1"
      ;;
    alfworld)
      local config
      config=$(write_alfworld_config)
      run_cmd tmux kill-session -t "mlf_${ENV_NAME}_env" 2>/dev/null || true
      run_cmd tmux new-session -d -s "mlf_${ENV_NAME}_env" \
        "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 ALFWORLD_DATA=${MLF_LOCAL_ROOT}/data/alfworld ALFWORLD_LIB=${MLF_NAS_ROOT}/code/alfworld PYTHONPATH=${REPO_DIR} && ${MLF_LOCAL_ENVS}/alfworld/bin/python examples/agent_env/alfworld/server.py --host 0.0.0.0 --port ${ENV_PORT} --config ${config} > ${MLF_LOCAL_ROOT}/logs/${ENV_NAME}_env_server.log 2>&1"
      ;;
    tau2)
      local config
      config=$(write_tau2_config)
      run_cmd tmux kill-session -t "mlf_${ENV_NAME}_env" 2>/dev/null || true
      run_cmd tmux new-session -d -s "mlf_${ENV_NAME}_env" \
        "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 TAU2_DATA_DIR=${MLF_LOCAL_ROOT}/data/tau2/data PYTHONPATH=${REPO_DIR} && ${MLF_LOCAL_ENVS}/tau2/bin/python examples/agent_env/tau2/server.py --host 0.0.0.0 --port ${ENV_PORT} --config ${config} > ${MLF_LOCAL_ROOT}/logs/${ENV_NAME}_env_server.log 2>&1"
      ;;
    appworld)
      local config
      config=$(write_appworld_config)
      run_cmd tmux kill-session -t "mlf_${ENV_NAME}_env" 2>/dev/null || true
      run_cmd tmux new-session -d -s "mlf_${ENV_NAME}_env" \
        "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 HOME=${MLF_LOCAL_ROOT}/data/appworld APPWORLD_ROOT=${MLF_LOCAL_ROOT}/data/appworld PYTHONPATH=${REPO_DIR} && ${MLF_LOCAL_ENVS}/appworld/bin/python examples/agent_env/appworld/server.py --host 0.0.0.0 --port ${ENV_PORT} --config ${config} > ${MLF_LOCAL_ROOT}/logs/${ENV_NAME}_env_server.log 2>&1"
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
    "export PYTHONNOUSERSITE=1 RAY_DISABLE_DOCKER_CPU_WARNING=1; if [ -n '${RAY_CUDA_VISIBLE_DEVICES}' ]; then export CUDA_VISIBLE_DEVICES='${RAY_CUDA_VISIBLE_DEVICES}'; fi; ${SLIME_PYTHON} -m ray.scripts.scripts start --head --node-ip-address ${node_ip} --port ${RAY_PORT} --num-gpus ${NUM_GPUS_PER_NODE_FOR_RAY:-${NUM_GPUS:-4}} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --block > /tmp/mlf_ray_head.log 2>&1"
}

start_ray_worker() {
  local head=$1
  local node_ip
  node_ip=$(hostname -I | tr ' ' '\n' | grep -m1 .)
  run_cmd tmux kill-session -t mlf_ray_worker 2>/dev/null || true
  run_cmd "${SLIME_PYTHON}" -m ray.scripts.scripts stop --force
  run_cmd tmux new-session -d -s mlf_ray_worker \
    "export PYTHONNOUSERSITE=1 RAY_DISABLE_DOCKER_CPU_WARNING=1; if [ -n '${RAY_CUDA_VISIBLE_DEVICES}' ]; then export CUDA_VISIBLE_DEVICES='${RAY_CUDA_VISIBLE_DEVICES}'; fi; ${SLIME_PYTHON} -m ray.scripts.scripts start --address ${head}:${RAY_PORT} --node-ip-address ${node_ip} --num-gpus ${NUM_GPUS_PER_NODE_FOR_RAY:-${NUM_GPUS:-4}} --disable-usage-stats --block > /tmp/mlf_ray_worker.log 2>&1"
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
    "cd ${REPO_DIR} && export PYTHONNOUSERSITE=1 && ${SLIME_PYTHON} examples/agent_env/router.py --host 0.0.0.0 --port ${ROUTER_PORT} --workers ${joined} > ${MLF_LOCAL_ROOT}/logs/${ENV_NAME}_router.log 2>&1"
}

remote_worker() {
  local host=$1
  local head=$2
  local remote_cmd
  remote_cmd=$(
    printf 'cd %q && MLF_NAS_ROOT=%q MLF_LOCAL_ENVS=%q MLF_LOCAL_ROOT=%q REPO_DIR=%q bash scripts/mlf/launch_agentic_training.sh ' \
      "${REPO_DIR}" "${MLF_NAS_ROOT}" "${MLF_LOCAL_ENVS}" "${MLF_LOCAL_ROOT}" "${REPO_DIR}"
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
  printf 'cd %q && MLF_NAS_ROOT=%q MLF_LOCAL_ENVS=%q MLF_LOCAL_ROOT=%q REPO_DIR=%q RAY_CUDA_VISIBLE_DEVICES=%q NUM_GPUS_PER_NODE_FOR_RAY=%q bash scripts/mlf/launch_agentic_training.sh ' \
    "${REPO_DIR}" "${MLF_NAS_ROOT}" "${MLF_LOCAL_ENVS}" "${MLF_LOCAL_ROOT}" "${REPO_DIR}" "${RAY_CUDA_VISIBLE_DEVICES}" "${NUM_GPUS_PER_NODE_FOR_RAY:-${NUM_GPUS:-4}}"
}

remote_start_tmux() {
  local host=$1
  local session=$2
  local log=$3
  local command=$4
  local remote_cmd
  local attempt
  remote_cmd=$(printf 'tmux kill-session -t %q 2>/dev/null || true; tmux new-session -d -s %q %q; tmux ls 2>/dev/null | grep %q' "${session}" "${session}" "${command}" "${session}")
  echo "Starting ${session} on ${host}"
  fill_ssh_args "${host}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '+ '
    print_ssh_cmd
    printf '%q\n' "${remote_cmd}"
    return
  fi
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
  ) > /tmp/mlf_multi_router.log 2>&1"
  remote_start_tmux "${head}" mlf_multi_router /tmp/mlf_multi_router.log "${cmd}"
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
  ) > /tmp/mlf_multi_head.log 2>&1"
  remote_start_tmux "${head}" mlf_multi_head /tmp/mlf_multi_head.log "${cmd}"

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
    ) > /tmp/mlf_multi_worker.log 2>&1"
    remote_start_tmux "${node}" mlf_multi_worker /tmp/mlf_multi_worker.log "${cmd}"
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
  echo "Head log: ${head}:/tmp/mlf_multi_head.log"
  echo "Worker log: <worker>:/tmp/mlf_multi_worker.log"
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
      ) > /tmp/mlf_multi_worker.log 2>&1"
      remote_start_tmux "${node}" mlf_multi_worker /tmp/mlf_multi_worker.log "${cmd}"
    done
    for node in $(read_nodes | tail -n +2); do
      remote_wait_http "${node}" "http://127.0.0.1:${ENV_PORT}/health"
      node_addr=$(remote_first_ip "${node}")
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
    echo "+ ${TRAIN_CMD}"
    if [ "${DRY_RUN}" -eq 0 ]; then
      bash -lc "${TRAIN_CMD}"
    fi
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
  else
    # Delegate orchestration to the first node.
    # From the head, workers are reached directly with the node-local key; the
    # local machine should only maintain this one SSH connection long enough to
    # submit the tmux session.
    remote_cmd=$(
      printf 'cd %q && MLF_NAS_ROOT=%q MLF_LOCAL_ENVS=%q MLF_LOCAL_ROOT=%q REPO_DIR=%q RAY_CUDA_VISIBLE_DEVICES=%q NUM_GPUS_PER_NODE_FOR_RAY=%q SSH_JUMP= SSH_KEY=%q SSH_IPV6=1 bash scripts/mlf/launch_agentic_training.sh ' \
        "${REPO_DIR}" "${MLF_NAS_ROOT}" "${MLF_LOCAL_ENVS}" "${MLF_LOCAL_ROOT}" "${REPO_DIR}" "${RAY_CUDA_VISIBLE_DEVICES}" "${NUM_GPUS_PER_NODE_FOR_RAY:-${NUM_GPUS:-4}}" "/home/${SSH_USER}/.ssh/byte_id_rsa"
      printf '%q ' --role head --env "${ENV_NAME}" --nodes "${NODES_FILE}" --env-port "${ENV_PORT}" --router-port "${ROUTER_PORT}" --ray-port "${RAY_PORT}" --data-size "${DATA_SIZE}"
      [ -z "${ENV_POOL_SIZE}" ] || printf '%q ' --env-pool-size "${ENV_POOL_SIZE}"
      [ -z "${TRAIN_CMD}" ] || printf '%q ' --train-cmd "${TRAIN_CMD}"
    )
    remote_start_tmux "${head}" mlf_multi_head /tmp/mlf_multi_head.log "${remote_cmd} > /tmp/mlf_multi_head.log 2>&1"
    echo "Head orchestration submitted."
    echo "Head log: ${head}:/tmp/mlf_multi_head.log"
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
      echo "+ ${TRAIN_CMD}"
      [ "${DRY_RUN}" -eq 1 ] || bash -lc "${TRAIN_CMD}"
    else
      echo "Router is ready. Router: http://127.0.0.1:${ROUTER_PORT}"
    fi
    ;;
  *) echo "Unsupported role: ${ROLE}" >&2; exit 1 ;;
esac
