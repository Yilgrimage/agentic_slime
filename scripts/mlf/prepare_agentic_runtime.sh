#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
REPO_DIR=${REPO_DIR:-${MLF_NAS_ROOT}/code/slime}
NODES_FILE=""
LOCAL_ONLY=1
ALL_NODES=0
ENVS=${ENVS:-slime,alfworld,webshop}
DATASETS=${DATASETS:-alfworld,webshop}
MODELS=${MODELS:-qwen3-8b}
SOURCES=${SOURCES:-webshop}
FORCE=0
CHECK_HASH=1
SERIAL=0
POLL_INTERVAL=${POLL_INTERVAL:-15}
SSH_CONNECT_TIMEOUT=${SSH_CONNECT_TIMEOUT:-10}
SSH_SERVER_ALIVE_INTERVAL=${SSH_SERVER_ALIVE_INTERVAL:-15}
SSH_SERVER_ALIVE_COUNT_MAX=${SSH_SERVER_ALIVE_COUNT_MAX:-2}
SSH_USER=${SSH_USER:-tiger}
SSH_PORT=${SSH_PORT:-10413}
SSH_KEY=${SSH_KEY:-~/.ssh/byte_id_rsa}
SSH_KEY=${SSH_KEY/#\~/${HOME}}
SSH_JUMP=${SSH_JUMP:-jump-proxy-arnold-hl.byted.org}
SSH_IPV6=${SSH_IPV6:-1}

usage() {
  cat <<'EOF'
Usage: prepare_agentic_runtime.sh [options]

Prepare local or multi-node runtime assets. This script does not start Ray,
env servers, or training.

Options:
  --local-only         Prepare only the current node (default)
  --all-nodes         Prepare every node listed by --nodes
  --nodes FILE        Node list, one host/IP per line; comments allowed
  --envs LIST         Comma list: slime,alfworld,webshop,none
  --data LIST         Comma list: alfworld,webshop,none
  --models LIST       Comma list: qwen3-8b,none
  --sources LIST      Comma list: webshop,none
  --force             Reinstall/copy even if local stamp matches
  --no-check-hash     Use existence checks only
  --serial            Prepare nodes one by one instead of parallel tmux jobs
  -h, --help

SSH can be overridden with SSH_USER, SSH_PORT, SSH_KEY, SSH_JUMP, SSH_IPV6.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --local-only) LOCAL_ONLY=1; ALL_NODES=0; shift ;;
    --all-nodes) ALL_NODES=1; LOCAL_ONLY=0; shift ;;
    --nodes) NODES_FILE=$2; shift 2 ;;
    --envs) ENVS=$2; shift 2 ;;
    --data) DATASETS=$2; shift 2 ;;
    --models) MODELS=$2; shift 2 ;;
    --sources) SOURCES=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    --no-check-hash) CHECK_HASH=0; shift ;;
    --serial) SERIAL=1; shift ;;
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

run_local() {
  cd "${REPO_DIR}"
  bash scripts/mlf/materialize_node_runtime.sh "${materialize_args[@]}"
}

read_nodes() {
  local file=$1
  awk 'NF && $1 !~ /^#/ {print $1}' "${file}"
}

ssh_base() {
  local host=$1
  local args=()
  if [ "${SSH_IPV6}" = "1" ]; then
    args+=("-6")
  fi
  args+=(
    "-o" "GSSAPIAuthentication=yes"
    "-o" "GSSAPIDelegateCredentials=yes"
    "-o" "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
    "-o" "ServerAliveInterval=${SSH_SERVER_ALIVE_INTERVAL}"
    "-o" "ServerAliveCountMax=${SSH_SERVER_ALIVE_COUNT_MAX}"
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

run_remote() {
  local host=$1
  local remote_cmd
  remote_cmd=$(
    printf 'cd %q && MLF_NAS_ROOT=%q REPO_DIR=%q bash scripts/mlf/materialize_node_runtime.sh ' \
      "${REPO_DIR}" "${MLF_NAS_ROOT}" "${REPO_DIR}"
    printf '%q ' "${materialize_args[@]}"
  )
  echo "Preparing ${host}"
  # shellcheck disable=SC2046
  ssh $(ssh_base "${host}") "${remote_cmd}"
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
  remote_cmd=$(printf 'rm -f /tmp/mlf_prepare.exit; tmux kill-session -t mlf_prepare 2>/dev/null || true; tmux new-session -d -s mlf_prepare %q; tmux ls 2>/dev/null | grep mlf_prepare' "bash -lc '${materialize_cmd} > /tmp/mlf_prepare.log 2>&1; echo \$? > /tmp/mlf_prepare.exit'")
  echo "Submitting prepare ${host}"
  # shellcheck disable=SC2046
  ssh $(ssh_base "${host}") "${remote_cmd}"
}

remote_prepare_status() {
  local host=$1
  local remote_cmd
  remote_cmd='if tmux has-session -t mlf_prepare 2>/dev/null; then echo RUNNING; elif [ -f /tmp/mlf_prepare.exit ]; then code=$(cat /tmp/mlf_prepare.exit); echo EXIT:${code}; tail -n 8 /tmp/mlf_prepare.log 2>/dev/null || true; else echo MISSING; tail -n 8 /tmp/mlf_prepare.log 2>/dev/null || true; fi'
  # shellcheck disable=SC2046
  ssh $(ssh_base "${host}") "${remote_cmd}"
}

wait_remote_prepares() {
  local hosts=("$@")
  local pending
  local host
  local status
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
  if [ "${SERIAL}" -eq 1 ]; then
    for host in $(read_nodes "${NODES_FILE}"); do
      run_remote "${host}"
    done
  else
    hosts=()
    read_nodes_array "${NODES_FILE}"
    for host in "${hosts[@]}"; do
      start_remote_prepare "${host}"
    done
    wait_remote_prepares "${hosts[@]}"
  fi
  exit 0
fi

usage >&2
exit 1
