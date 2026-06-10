#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
REPO_DIR=${REPO_DIR:-${MLF_NAS_ROOT}/code/slime}
NODES_FILE=""
LOCAL_ONLY=1
ALL_NODES=0
ORCHESTRATOR=${ORCHESTRATOR:-head}
ENVS=${ENVS:-slime,alfworld,webshop}
DATASETS=${DATASETS:-alfworld,webshop}
MODELS=${MODELS:-none}
SOURCES=${SOURCES:-webshop}
FORCE=0
CHECK_HASH=1
DRY_RUN=0
POLL_INTERVAL=${POLL_INTERVAL:-15}
SSH_CONNECT_TIMEOUT=${SSH_CONNECT_TIMEOUT:-10}
SSH_SERVER_ALIVE_INTERVAL=${SSH_SERVER_ALIVE_INTERVAL:-15}
SSH_SERVER_ALIVE_COUNT_MAX=${SSH_SERVER_ALIVE_COUNT_MAX:-2}
SSH_USER=${SSH_USER:-tiger}
SSH_PORT=${SSH_PORT:-10413}
SSH_KEY=${SSH_KEY:-~/.ssh/byte_id_rsa}
SSH_KEY=${SSH_KEY/#\~/${HOME}}
SSH_JUMP=${SSH_JUMP-jump-proxy-arnold-hl.byted.org}
SSH_CONFIG=${SSH_CONFIG:-}
SSH_IPV6=${SSH_IPV6:-1}

usage() {
  cat <<'EOF'
Usage: prepare_agentic_runtime.sh [options]

Prepare local or multi-node runtime assets. This script does not start Ray,
env servers, or training.

Options:
  --local-only         Prepare only the current node (default)
  --all-nodes         Prepare every node listed by --nodes
  --orchestrator MODE  head or local. head submits one tmux on node0; local SSHes to every node.
  --nodes FILE        Node list, one host/IP per line; comments allowed
  --envs LIST         Comma list: slime,alfworld,webshop,tau2,appworld,none
  --data LIST         Comma list: alfworld,webshop,tau2,appworld,none
  --models none       Kept for compatibility. Model copying is disabled; training reads models from NAS.
  --sources LIST      Comma list: webshop,tau2,appworld,none
  --force             Reinstall/copy even if local stamp matches
  --no-check-hash     Use existence checks only
  --dry-run           Print actions only
  -h, --help

SSH can be overridden with SSH_USER, SSH_PORT, SSH_KEY, SSH_JUMP, SSH_IPV6.
Use SSH_CONFIG for a local SSH config, e.g. one that maps the jump host name to
its IP while keeping the Kerberos host alias stable.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --local-only) LOCAL_ONLY=1; ALL_NODES=0; shift ;;
    --all-nodes) ALL_NODES=1; LOCAL_ONLY=0; shift ;;
    --orchestrator) ORCHESTRATOR=$2; shift 2 ;;
    --nodes) NODES_FILE=$2; shift 2 ;;
    --envs) ENVS=$2; shift 2 ;;
    --data) DATASETS=$2; shift 2 ;;
    --models) MODELS=$2; shift 2 ;;
    --sources) SOURCES=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    --no-check-hash) CHECK_HASH=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

materialize_args=(
  --envs "${ENVS}"
  --data "${DATASETS}"
  --models "${MODELS}"
  --sources "${SOURCES}"
)
[ "${FORCE}" -eq 0 ] || materialize_args+=(--force)
[ "${CHECK_HASH}" -eq 1 ] || materialize_args+=(--no-check-hash)

if [ "${MODELS}" != "none" ]; then
  echo "Model materialization is disabled. Use --models none and read model checkpoints from ${MLF_NAS_ROOT}/models." >&2
  exit 2
fi

run_local() {
  cd "${REPO_DIR}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '+ bash scripts/mlf/materialize_node_runtime.sh '
    printf '%q ' "${materialize_args[@]}"
    printf '\n'
    return
  fi
  bash scripts/mlf/materialize_node_runtime.sh "${materialize_args[@]}"
}

read_nodes() {
  local file=$1
  awk 'NF && $1 !~ /^#/ {print $1}' "${file}"
}

ssh_base() {
  local host=$1
  local args=()
  if [ -n "${SSH_CONFIG}" ]; then
    args+=("-F" "${SSH_CONFIG}")
  fi
  if [ "${SSH_IPV6}" = "1" ]; then
    args+=("-6")
  fi
  args+=(
    "-o" "StrictHostKeyChecking=no"
    "-o" "UserKnownHostsFile=/dev/null"
    "-o" "IdentitiesOnly=yes"
    "-i" "${SSH_KEY}"
    "-p" "${SSH_PORT}"
  )
  if [ -n "${SSH_JUMP}" ]; then
    args+=("-J" "${SSH_JUMP}")
  fi
  args+=("${SSH_USER}@${host}")
  printf '%q ' "${args[@]}"
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

run_remote() {
  local host=$1
  local remote_cmd
  remote_cmd=$(
    printf 'cd %q && MLF_NAS_ROOT=%q REPO_DIR=%q bash scripts/mlf/materialize_node_runtime.sh ' \
      "${REPO_DIR}" "${MLF_NAS_ROOT}" "${REPO_DIR}"
    printf '%q ' "${materialize_args[@]}"
  )
  echo "Preparing ${host}"
  fill_ssh_args "${host}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '+ ssh '
    printf '%q ' "${SSH_ARGS[@]}"
    printf '%q\n' "${remote_cmd}"
    return
  fi
  ssh "${SSH_ARGS[@]}" "${remote_cmd}"
}

remote_prepare_command() {
  printf 'cd %q && MLF_NAS_ROOT=%q REPO_DIR=%q bash scripts/mlf/materialize_node_runtime.sh ' \
    "${REPO_DIR}" "${MLF_NAS_ROOT}" "${REPO_DIR}"
  printf '%q ' "${materialize_args[@]}"
}

start_remote_prepare() {
  local host=$1
  local materialize_cmd
  local remote_cmd
  materialize_cmd=$(remote_prepare_command)
  remote_cmd=$(printf 'rm -f /tmp/mlf_prepare.exit; tmux kill-session -t mlf_node_prepare 2>/dev/null || true; tmux new-session -d -s mlf_node_prepare %q; tmux ls 2>/dev/null | grep mlf_node_prepare' "bash -lc '${materialize_cmd} > /tmp/mlf_prepare.log 2>&1; echo \$? > /tmp/mlf_prepare.exit'")
  echo "Submitting prepare ${host}"
  if is_current_node "${host}"; then
    if [ "${DRY_RUN}" -eq 1 ]; then
      printf '+ %q\n' "${remote_cmd}"
      return
    fi
    bash -lc "${remote_cmd}"
    return
  fi
  fill_ssh_args "${host}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '+ ssh '
    printf '%q ' "${SSH_ARGS[@]}"
    printf '%q\n' "${remote_cmd}"
    return
  fi
  ssh "${SSH_ARGS[@]}" "${remote_cmd}"
}

remote_prepare_status() {
  local host=$1
  local remote_cmd
  remote_cmd='if tmux has-session -t mlf_node_prepare 2>/dev/null; then echo RUNNING; elif [ -f /tmp/mlf_prepare.exit ]; then code=$(cat /tmp/mlf_prepare.exit); echo EXIT:${code}; tail -n 8 /tmp/mlf_prepare.log 2>/dev/null || true; else echo MISSING; tail -n 8 /tmp/mlf_prepare.log 2>/dev/null || true; fi'
  if is_current_node "${host}"; then
    bash -lc "${remote_cmd}"
    return
  fi
  fill_ssh_args "${host}"
  ssh "${SSH_ARGS[@]}" "${remote_cmd}"
}

wait_remote_prepares() {
  local hosts=("$@")
  local pending
  local host
  local status
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "+ wait_remote_prepares ${hosts[*]}"
    return 0
  fi
  while true; do
    pending=0
    for host in "${hosts[@]}"; do
      status=$(remote_prepare_status "${host}" || true)
      case "${status}" in
        RUNNING*)
          echo "${host}: RUNNING"
          pending=1
          ;;
        EXIT:0*)
          echo "${host}: DONE"
          ;;
        EXIT:*)
          echo "${host}: FAILED"
          echo "${status}"
          return 1
          ;;
        *)
          echo "${host}: UNKNOWN"
          echo "${status}"
          pending=1
          ;;
      esac
    done
    [ "${pending}" -eq 1 ] || break
    sleep "${POLL_INTERVAL}"
  done
}

read_nodes_array() {
  local file=$1
  local line
  hosts=()
  while IFS= read -r line; do
    [ -n "${line}" ] || continue
    case "${line}" in
      \#*) continue ;;
    esac
    hosts+=("${line}")
  done < <(read_nodes "${file}")
}

first_node() {
  read_nodes "${NODES_FILE}" | head -n 1
}

is_current_node() {
  local node=$1
  [ "${node}" = "this" ] && return 0
  [ "${node}" = "$(hostname)" ] && return 0
  hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${node}" && return 0
  ip addr 2>/dev/null | grep -Fq "${node}" && return 0
  return 1
}

submit_head_prepare() {
  local head=$1
  local remote_cmd
  remote_cmd=$(
    printf 'cd %q && MLF_NAS_ROOT=%q REPO_DIR=%q SSH_JUMP= SSH_KEY=%q SSH_IPV6=1 bash scripts/mlf/prepare_agentic_runtime.sh ' \
      "${REPO_DIR}" "${MLF_NAS_ROOT}" "${REPO_DIR}" "/home/${SSH_USER}/.ssh/byte_id_rsa"
    printf '%q ' --all-nodes --orchestrator local --nodes "${NODES_FILE}" --envs "${ENVS}" --data "${DATASETS}" --models "${MODELS}" --sources "${SOURCES}"
    [ "${FORCE}" -eq 0 ] || printf '%q ' --force
    [ "${CHECK_HASH}" -eq 1 ] || printf '%q ' --no-check-hash
  )
  local wrapped
  wrapped=$(printf 'rm -f /tmp/mlf_prepare_head.exit; tmux kill-session -t mlf_prepare_head 2>/dev/null || true; tmux new-session -d -s mlf_prepare_head %q; tmux ls 2>/dev/null | grep mlf_prepare_head' "bash -lc '${remote_cmd} > /tmp/mlf_prepare_head.log 2>&1; echo \$? > /tmp/mlf_prepare_head.exit'")
  echo "Submitting head prepare ${head}"
  fill_ssh_args "${head}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '+ ssh '
    printf '%q ' "${SSH_ARGS[@]}"
    printf '%q\n' "${wrapped}"
    return
  fi
  ssh "${SSH_ARGS[@]}" "${wrapped}"
  echo "Head prepare submitted."
  echo "Head log: ${head}:/tmp/mlf_prepare_head.log"
}

if [ "${LOCAL_ONLY}" -eq 1 ]; then
  run_local
  exit 0
fi

if [ "${ALL_NODES}" -eq 1 ]; then
  if [ -z "${NODES_FILE}" ]; then
    echo "--all-nodes requires --nodes FILE" >&2
    exit 1
  fi
  if [ ! -f "${NODES_FILE}" ]; then
    echo "Missing nodes file: ${NODES_FILE}" >&2
    exit 1
  fi
  if [ "${ORCHESTRATOR}" = "head" ]; then
    head=$(first_node)
    if ! is_current_node "${head}"; then
      submit_head_prepare "${head}"
      exit 0
    fi
    # We are already on the head. Use the node-local key for direct worker SSH.
    SSH_JUMP=${SSH_JUMP_ON_HEAD:-}
    SSH_KEY=${SSH_KEY_ON_HEAD:-/home/${SSH_USER}/.ssh/byte_id_rsa}
  fi
  hosts=()
  read_nodes_array "${NODES_FILE}"
  for host in "${hosts[@]}"; do
    start_remote_prepare "${host}"
  done
  wait_remote_prepares "${hosts[@]}"
  exit 0
fi

usage >&2
exit 1
