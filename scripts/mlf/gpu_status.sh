#!/usr/bin/env bash
set -euo pipefail

NODES_FILE=""
WATCH_INTERVAL=0
SSH_USER=${SSH_USER:-tiger}
SSH_PORT=${SSH_PORT:-10413}
SSH_KEY=${SSH_KEY:-~/.ssh/byte_id_rsa}
SSH_KEY=${SSH_KEY/#\~/${HOME}}
SSH_JUMP=${SSH_JUMP:-jump-proxy-arnold-hl.byted.org}
SSH_IPV6=${SSH_IPV6:-1}

usage() {
  cat <<'EOF'
Usage: gpu_status.sh [options]

Show GPU utilization and GPU processes for one or more nodes.

Options:
  --nodes FILE       Node list, one host/IP per line; comments allowed
  --watch SEC        Repeat every SEC seconds
  -h, --help

SSH can be overridden with SSH_USER, SSH_PORT, SSH_KEY, SSH_JUMP, SSH_IPV6.
Set SSH_JUMP=10.169.30.207 if DNS for the jump proxy is flaky.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --nodes) NODES_FILE=$2; shift 2 ;;
    --watch) WATCH_INTERVAL=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

read_nodes() {
  if [ -n "${NODES_FILE}" ]; then
    awk 'NF && $1 !~ /^#/ {print $1}' "${NODES_FILE}"
  else
    echo "this"
  fi
}

ssh_args() {
  local host=$1
  local args=()
  if [ "${SSH_IPV6}" = "1" ]; then
    args+=("-6")
  fi
  args+=("-o" "IdentitiesOnly=yes" "-i" "${SSH_KEY}" "-p" "${SSH_PORT}")
  if [ -n "${SSH_JUMP}" ]; then
    args+=("-J" "${SSH_JUMP}")
  fi
  args+=("${SSH_USER}@${host}")
  printf '%q ' "${args[@]}"
}

remote_status_cmd='
set -e
echo "HOST=$(hostname)"
date "+TIME=%F %T"
echo "GPU:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits |
  awk -F, '"'"'{gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3); gsub(/^ +| +$/, "", $4); gsub(/^ +| +$/, "", $5); gsub(/^ +| +$/, "", $6); gsub(/^ +| +$/, "", $7); printf("  gpu=%s util=%s%% mem=%s/%s MiB power=%s W temp=%s C name=%s\n", $1,$3,$4,$5,$6,$7,$2)}'"'"'
echo "PROCS:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null |
  awk -F, '"'"'NF {gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3); printf("  pid=%s mem=%s MiB cmd=%s\n", $1,$3,$2)}'"'"'
'

show_once() {
  local node
  for node in $(read_nodes); do
    echo "===== ${node} ====="
    if [ "${node}" = "this" ]; then
      bash -lc "${remote_status_cmd}" || true
    else
      # shellcheck disable=SC2046
      ssh $(ssh_args "${node}") "${remote_status_cmd}" || true
    fi
  done
}

if [ "${WATCH_INTERVAL}" = "0" ]; then
  show_once
else
  while true; do
    clear 2>/dev/null || true
    show_once
    sleep "${WATCH_INTERVAL}"
  done
fi
