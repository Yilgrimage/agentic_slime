#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
case "${ACTION}" in
  start|stop|status|restart|-h|--help|help)
    shift || true
    ;;
  *)
    ACTION="start"
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
LOCAL_ENVS_DIR="${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}"

SESSION="${RUN_BENCH_SESSION:-torch_bench}"
PY="${RUN_BENCH_SCRIPT:-/tmp/server_ops_torch_bench.py}"
PY_PATTERN="${RUN_BENCH_SCRIPT_PATTERN:-${PY}}"
LOG="${RUN_BENCH_LOG:-/tmp/server_ops_torch_bench.log}"
NODES_FILE=""
NODE_SELECTOR=""
SSH_USER="${SSH_USER:-$(id -un 2>/dev/null || echo tiger)}"
SSH_PORT="${SSH_PORT:-10413}"
if [ -z "${SSH_KEY:-}" ]; then
  if [ -f "${ROOT_DIR}/secrets/byte_id_rsa" ]; then
    SSH_KEY="${ROOT_DIR}/secrets/byte_id_rsa"
  else
    SSH_KEY="~/.ssh/byte_id_rsa"
  fi
fi
SSH_KEY="${SSH_KEY/#\~/${HOME}}"
SSH_JUMP="${SSH_JUMP:-}"
SSH_IPV6="${SSH_IPV6:-1}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-${ROOT_DIR}/scripts/run_bench.sh}"
KILL_TMUX_SERVER="${RUN_BENCH_KILL_TMUX_SERVER:-0}"
KILL_VLLM="${RUN_BENCH_KILL_VLLM:-0}"
READY_TIMEOUT_SECONDS="${RUN_BENCH_READY_TIMEOUT_SECONDS:-90}"
BENCH_PYTHON_EXPLICIT=0
if [ -n "${BENCH_PYTHON:-}" ]; then
  BENCH_PYTHON_EXPLICIT=1
fi
if [ -z "${BENCH_PYTHON:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    BENCH_PYTHON="$(command -v python3)"
  else
    BENCH_PYTHON="python"
  fi
fi

usage() {
  cat <<'EOF'
Usage:
  run_bench.sh [start|stop|status|restart]
  run_bench.sh [start|stop|status|restart] --nodes nodes.txt [--node 0,1]

Without --nodes, this controls local PyTorch CUDA stress.
With --nodes, it SSHes to selected nodes and runs this same script there.
It never starts vLLM/SGLang. On start, each selected node replaces only the
torch_bench keepalive session by default. Set RUN_BENCH_KILL_TMUX_SERVER=1 to
wipe the full tmux server before starting bench. Set RUN_BENCH_KILL_VLLM=1 to
also clean old vLLM bench/server processes.

Set BENCH_PYTHON to a Python executable with torch installed when system Python
does not provide CUDA torch.

For isolated tests, override RUN_BENCH_SESSION, RUN_BENCH_SCRIPT, and
RUN_BENCH_LOG.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --nodes)
      NODES_FILE="$2"
      shift 2
      ;;
    --node|--nodes-index|--node-index)
      NODE_SELECTOR="$2"
      shift 2
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

stop_local() {
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  pkill -f "${PY_PATTERN}" 2>/dev/null || true
  if [ "${KILL_VLLM}" = "1" ]; then
    tmux kill-session -t "vllm_bench" 2>/dev/null || true
    tmux kill-session -t "vllm_server" 2>/dev/null || true
    pkill -f "vllm bench serve" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
  fi
  wait_local_cleanup
}

cleanup_before_start() {
  if [ "${KILL_TMUX_SERVER}" = "1" ]; then
    tmux kill-server 2>/dev/null || true
  else
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
  fi
  pkill -f "${PY_PATTERN}" 2>/dev/null || true
  if [ "${KILL_VLLM}" = "1" ]; then
    tmux kill-session -t "vllm_bench" 2>/dev/null || true
    tmux kill-session -t "vllm_server" 2>/dev/null || true
    pkill -f "vllm bench serve" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
  fi
  wait_local_cleanup
}

wait_local_cleanup() {
  local deadline
  deadline=$((SECONDS + 60))
  while pgrep -f "${PY_PATTERN}" >/dev/null 2>&1; do
    [ "${SECONDS}" -ge "${deadline}" ] && break
    pkill -f "${PY_PATTERN}" 2>/dev/null || true
    sleep 1
  done
}

status_local() {
  tmux ls 2>/dev/null | grep -E "^${SESSION}:" || true
  pgrep -af "${PY_PATTERN}" || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
}

write_bench_worker() {
  cat > "${PY}" <<'PY'
import multiprocessing as mp
import os
import signal
import time

import torch


def worker(rank: int):
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    n = 8192
    a = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
    reserve = []
    free, total = torch.cuda.mem_get_info(rank)
    target = min(int(total * 0.90), max(0, free - 4 * 1024**3))
    need = max(0, target - torch.cuda.memory_allocated(rank))
    chunk = 1024**3
    while need > 0:
        size = min(chunk, need)
        reserve.append(torch.empty((size,), dtype=torch.uint8, device="cuda"))
        need -= size
    print(
        f"worker {rank} pid={os.getpid()} gpu={torch.cuda.get_device_name(rank)} "
        f"allocated_gb={torch.cuda.memory_allocated(rank) / 1e9:.1f}",
        flush=True,
    )
    while True:
        a = torch.mm(a, b)


if __name__ == "__main__":
    n_gpu = torch.cuda.device_count()
    assert n_gpu > 0, "no CUDA GPU visible"

    def spawn(rank: int):
        p = mp.Process(target=worker, args=(rank,), daemon=False)
        p.start()
        return p

    procs = {rank: spawn(rank) for rank in range(n_gpu)}

    def stop(*_):
        for p in procs.values():
            p.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        for rank, p in list(procs.items()):
            if not p.is_alive():
                print(
                    f"worker {rank} exited exitcode={p.exitcode}; restarting",
                    flush=True,
                )
                procs[rank] = spawn(rank)
        time.sleep(5)
PY
}

wait_bench_ready() {
  local expected ready deadline
  expected=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  [ "${expected:-0}" -gt 0 ] || expected=1
  deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    ready=$(grep -c '^worker ' "${LOG}" 2>/dev/null || true)
    if [ "${ready:-0}" -ge "${expected}" ]; then
      return 0
    fi
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  echo "[WARN] torch bench did not report ${expected} ready workers within ${READY_TIMEOUT_SECONDS}s" >&2
  tail -n 80 "${LOG}" >&2 || true
  return 1
}

start_local() {
  cleanup_before_start
  write_bench_worker
  : > "${LOG}"
  local cmd
  cmd=$(printf '%q -u %q >> %q 2>&1' "${BENCH_PYTHON}" "${PY}" "${LOG}")
  tmux new-session -d -s "${SESSION}" "${cmd}"
  wait_bench_ready || {
    status_local
    return 1
  }
  status_local
}

run_local() {
  case "${ACTION}" in
    stop)
      stop_local
      status_local
      ;;
    status)
      status_local
      ;;
    start|restart)
      start_local
      ;;
    *)
      echo "Unknown action: ${ACTION}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

read_nodes() {
  awk 'NF && $1 !~ /^#/ {print $1}' "${NODES_FILE}"
}

selected_nodes() {
  if [ -z "${NODES_FILE}" ]; then
    echo "this"
    return
  fi
  if [ -z "${NODE_SELECTOR}" ]; then
    read_nodes
    return
  fi
  awk -v sel="${NODE_SELECTOR}" '
    BEGIN {
      n = split(sel, parts, ",")
      for (i = 1; i <= n; i++) wanted[parts[i]] = 1
      idx = 0
    }
    NF && $1 !~ /^#/ {
      if (idx in wanted) print $1
      idx++
    }
  ' "${NODES_FILE}"
}

is_current_node() {
  local node=$1
  [ "${node}" = "this" ] && return 0
  [ "${node}" = "$(hostname)" ] && return 0
  hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${node}" && return 0
  ip addr 2>/dev/null | grep -Fq "${node}" && return 0
  return 1
}

run_one() {
  local node=$1
  local cmd bench_python_env=""
  if [ "${BENCH_PYTHON_EXPLICIT}" = "1" ]; then
    bench_python_env=$(printf ' BENCH_PYTHON=%q' "${BENCH_PYTHON}")
  fi
  cmd=$(printf 'ROOT_DIR=%q LOCAL_ENVS_DIR=%q RUN_BENCH_SESSION=%q RUN_BENCH_SCRIPT=%q RUN_BENCH_SCRIPT_PATTERN=%q RUN_BENCH_LOG=%q RUN_BENCH_READY_TIMEOUT_SECONDS=%q RUN_BENCH_KILL_TMUX_SERVER=%q RUN_BENCH_KILL_VLLM=%q%s bash %q %q' \
    "${ROOT_DIR}" "${LOCAL_ENVS_DIR}" "${SESSION}" "${PY}" "${PY_PATTERN}" "${LOG}" "${READY_TIMEOUT_SECONDS}" "${KILL_TMUX_SERVER}" "${KILL_VLLM}" "${bench_python_env}" "${REMOTE_SCRIPT}" "${ACTION}")
  if is_current_node "${node}"; then
    run_local
  else
    ssh_run "${node}" "${cmd}"
  fi
}

case "${ACTION}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  start|stop|status|restart) ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage >&2
    exit 1
    ;;
esac

if [ -z "${NODES_FILE}" ]; then
  run_local
  exit 0
fi

tmpdir=$(mktemp -d /tmp/server_ops_torch_bench_control.XXXXXX)
trap 'rm -rf "${tmpdir}"' EXIT

ssh_run() {
  local host=$1
  local cmd=$2
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
  )
  if [ -n "${SSH_JUMP}" ]; then
    args+=("-J" "${SSH_JUMP}")
  fi
  local attempt
  for attempt in 1 2 3; do
    if ssh "${args[@]}" "${SSH_USER}@${host}" "${cmd}"; then
      return 0
    fi
    sleep 3
  done
  return 1
}

pids=()
labels=()
idx=0
for node in $(selected_nodes); do
  echo "===== ${node} ${ACTION} ====="
  (run_one "${node}") > "${tmpdir}/${idx}.log" 2>&1 &
  pids+=("$!")
  labels+=("${node}")
  idx=$((idx + 1))
done

if [ "${idx}" -eq 0 ]; then
  echo "No selected nodes from ${NODES_FILE}" >&2
  exit 1
fi

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    failed=1
  fi
  echo "===== ${labels[$idx]} output ====="
  cat "${tmpdir}/${idx}.log"
done

exit "${failed}"
