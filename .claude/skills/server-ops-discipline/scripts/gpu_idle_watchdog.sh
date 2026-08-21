#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname -- "${SCRIPT_PATH}")"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
LOCAL_ENVS_DIR="${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}"
RUN_BENCH="${RUN_BENCH:-${ROOT_DIR}/scripts/run_bench.sh}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
WATCHDOG_GPU_INDICES="${GPU_IDLE_WATCHDOG_GPU_INDICES:-${BENCH_CUDA_VISIBLE_DEVICES}}"
SESSION_NAME="${GPU_IDLE_WATCHDOG_SESSION:-gpu_idle_watchdog}"
PID_FILE="${GPU_IDLE_WATCHDOG_PID_FILE:-/tmp/server_ops_gpu_idle_watchdog.pid}"
HOST_LABEL="$(hostname 2>/dev/null | tr -c 'A-Za-z0-9_.-' '_' || true)"
HOST_LABEL="${HOST_LABEL:-unknown}"
LOG_FILE="${GPU_IDLE_WATCHDOG_LOG:-/tmp/server_ops_gpu_idle_watchdog_${HOST_LABEL}.log}"
INTERVAL_SECONDS="${GPU_IDLE_WATCHDOG_INTERVAL_SECONDS:-10}"
IDLE_WINDOW_SECONDS="${GPU_IDLE_WATCHDOG_IDLE_WINDOW_SECONDS:-1800}"
IDLE_UTIL_THRESHOLD="${GPU_IDLE_WATCHDOG_IDLE_UTIL_THRESHOLD:-35}"
IDLE_MEM_THRESHOLD_MIB="${GPU_IDLE_WATCHDOG_IDLE_MEM_THRESHOLD_MIB:-0}"
WATCHDOG_PROCESS_REGEX='^[[:space:]]*[0-9]+[[:space:]]+bash[[:space:]]+/.*gpu_idle_watchdog[.]sh run([[:space:]]|$)'

usage() {
  cat <<'EOF'
Usage: gpu_idle_watchdog.sh [start|stop|status|restart|run]

Local GPU keepalive watchdog. It samples GPU utilization continuously and starts
{ROOT_DIR}/scripts/run_bench.sh after utilization stays below the idle threshold
for the configured idle window. An optional memory threshold can additionally
gate the trigger, but is disabled by default so a hung process with resident
model memory is still protected.
The start action keeps the watchdog itself in a tmux session so SSH logout does
not leave it as a fragile orphan process.

Environment:
  GPU_IDLE_WATCHDOG_INTERVAL_SECONDS     default 10
  GPU_IDLE_WATCHDOG_IDLE_WINDOW_SECONDS  default 1800
  GPU_IDLE_WATCHDOG_IDLE_UTIL_THRESHOLD  default 35
  GPU_IDLE_WATCHDOG_IDLE_MEM_THRESHOLD_MIB default 0 (disabled)
  GPU_IDLE_WATCHDOG_GPU_INDICES          optional physical GPU indexes to sample
                                         (defaults to BENCH_CUDA_VISIBLE_DEVICES)
  GPU_IDLE_WATCHDOG_LOG                  default /tmp/server_ops_gpu_idle_watchdog_$HOSTNAME.log
  BENCH_PYTHON                           optional Python executable passed to run_bench
  BENCH_CUDA_VISIBLE_DEVICES             optional GPU list passed to run_bench
  LOCAL_ENVS_DIR                         default /tmp/server-ops-envs, passed to run_bench
EOF
}

log() {
  echo "[$(date -Is)] $*"
}

watchdog_running() {
  [ -f "${PID_FILE}" ] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  [ -n "${pid}" ] || return 1
  kill -0 "${pid}" 2>/dev/null
}

watchdog_tmux_running() {
  tmux has-session -t "${SESSION_NAME}" 2>/dev/null
}

orphan_watchdog_pids() {
  local expected=
  if [ -f "${PID_FILE}" ]; then
    expected="$(cat "${PID_FILE}" 2>/dev/null || true)"
  fi
  ps -eo pid=,args= 2>/dev/null \
    | awk -v self="$$" -v expected="${expected}" -v regex="${WATCHDOG_PROCESS_REGEX}" '
        $0 ~ regex {
          pid = $1
          if (pid != self && pid != expected) print pid
        }'
}

stop_orphan_watchdogs() {
  local pids pid
  pids="$(orphan_watchdog_pids | tr '\n' ' ' || true)"
  [ -n "${pids//[[:space:]]/}" ] || return 0
  log "stopping orphan watchdog pid(s): ${pids}"
  for pid in ${pids}; do
    kill "${pid}" 2>/dev/null || true
  done
  sleep 1
  for pid in ${pids}; do
    kill -9 "${pid}" 2>/dev/null || true
  done
}

gpu_max_util() {
  nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v selected="${WATCHDOG_GPU_INDICES}" '
      BEGIN { max = -1 }
      {
        gsub(/[[:space:]]/, "", $1)
        gsub(/[[:space:]]/, "", $2)
        if (selected != "") {
          wanted = 0
          n = split(selected, indexes, ",")
          for (i = 1; i <= n; i++) {
            gsub(/[[:space:]]/, "", indexes[i])
            if ($1 == indexes[i]) wanted = 1
          }
          if (!wanted) next
        }
        if ($2 ~ /^[0-9]+$/ && $2 > max) max = $2
      }
      END {
        if (max < 0) exit 1
        print max
      }'
}

gpu_max_mem_mib() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v selected="${WATCHDOG_GPU_INDICES}" '
      BEGIN { max = -1 }
      {
        gsub(/[[:space:]]/, "", $1)
        gsub(/[[:space:]]/, "", $2)
        if (selected != "") {
          wanted = 0
          n = split(selected, indexes, ",")
          for (i = 1; i <= n; i++) {
            gsub(/[[:space:]]/, "", indexes[i])
            if ($1 == indexes[i]) wanted = 1
          }
          if (!wanted) next
        }
        if ($2 ~ /^[0-9]+$/ && $2 > max) max = $2
      }
      END {
        if (max < 0) exit 1
        print max
      }'
}

run_loop() {
  mkdir -p "$(dirname "${LOG_FILE}")"
  echo "$$" > "${PID_FILE}"
  cleanup_exit() {
    local code=$?
    rm -f "${PID_FILE}"
    log "watchdog exiting code=${code}"
    exit "${code}"
  }
  trap cleanup_exit EXIT
  trap 'exit 0' INT TERM
  log "watchdog started managed_gpus=${WATCHDOG_GPU_INDICES:-all} interval=${INTERVAL_SECONDS}s idle_window=${IDLE_WINDOW_SECONDS}s idle_threshold=${IDLE_UTIL_THRESHOLD}% idle_mem_threshold=${IDLE_MEM_THRESHOLD_MIB}MiB run_bench=${RUN_BENCH}"

  local now util mem_mib idle_since=0 idle_elapsed=0
  while true; do
    if util="$(gpu_max_util 2>/dev/null)" && mem_mib="$(gpu_max_mem_mib 2>/dev/null)"; then
      now=$(date +%s)
      if [ "${util}" -lt "${IDLE_UTIL_THRESHOLD}" ] \
        && { [ "${IDLE_MEM_THRESHOLD_MIB}" -le 0 ] || [ "${mem_mib}" -lt "${IDLE_MEM_THRESHOLD_MIB}" ]; }; then
        if [ "${idle_since}" -eq 0 ]; then
          idle_since="${now}"
        fi
        idle_elapsed=$((now - idle_since))
      else
        idle_since=0
        idle_elapsed=0
      fi
      log "max_gpu_util=${util}% max_gpu_mem=${mem_mib}MiB idle_elapsed=${idle_elapsed}/${IDLE_WINDOW_SECONDS}s idle_threshold=<${IDLE_UTIL_THRESHOLD}% idle_mem_threshold=<${IDLE_MEM_THRESHOLD_MIB}MiB"

      if [ "${idle_since}" -gt 0 ] && [ "${idle_elapsed}" -ge "${IDLE_WINDOW_SECONDS}" ]; then
        log "GPU utilization risk detected; starting bench"
        if [ -x "${RUN_BENCH}" ] || [ -f "${RUN_BENCH}" ]; then
          BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}" RUN_BENCH_KILL_TMUX_SERVER=0 bash "${RUN_BENCH}" start >> "${LOG_FILE}" 2>&1 || log "run_bench start failed"
        else
          log "missing run_bench script: ${RUN_BENCH}"
        fi
        idle_since=0
        idle_elapsed=0
      fi
    else
      log "failed to read nvidia-smi util/memory; no sample recorded"
    fi

    sleep "${INTERVAL_SECONDS}"
  done
}

start_watchdog() {
  stop_orphan_watchdogs
  if watchdog_running; then
    log "watchdog already running pid=$(cat "${PID_FILE}")"
    return 0
  fi
  tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  mkdir -p "$(dirname "${LOG_FILE}")"
  local cmd
  cmd=$(printf 'ROOT_DIR=%q LOCAL_ENVS_DIR=%q BENCH_PYTHON=%q BENCH_CUDA_VISIBLE_DEVICES=%q GPU_IDLE_WATCHDOG_GPU_INDICES=%q RUN_BENCH=%q GPU_IDLE_WATCHDOG_SESSION=%q GPU_IDLE_WATCHDOG_PID_FILE=%q GPU_IDLE_WATCHDOG_LOG=%q GPU_IDLE_WATCHDOG_INTERVAL_SECONDS=%q GPU_IDLE_WATCHDOG_IDLE_WINDOW_SECONDS=%q GPU_IDLE_WATCHDOG_IDLE_UTIL_THRESHOLD=%q GPU_IDLE_WATCHDOG_IDLE_MEM_THRESHOLD_MIB=%q bash %q run >> %q 2>&1' \
    "${ROOT_DIR}" "${LOCAL_ENVS_DIR}" "${BENCH_PYTHON:-}" "${BENCH_CUDA_VISIBLE_DEVICES}" "${WATCHDOG_GPU_INDICES}" "${RUN_BENCH}" "${SESSION_NAME}" "${PID_FILE}" "${LOG_FILE}" \
    "${INTERVAL_SECONDS}" "${IDLE_WINDOW_SECONDS}" "${IDLE_UTIL_THRESHOLD}" "${IDLE_MEM_THRESHOLD_MIB}" "${SCRIPT_PATH}" "${LOG_FILE}")
  tmux new-session -d -s "${SESSION_NAME}" "${cmd}"
  local deadline
  deadline=$((SECONDS + 5))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    watchdog_running && break
    sleep 1
  done
  if watchdog_running; then
    log "watchdog started pid=$(cat "${PID_FILE}") session=${SESSION_NAME} log=${LOG_FILE}"
  else
    log "watchdog failed to start; see ${LOG_FILE}"
    return 1
  fi
}

stop_watchdog() {
  if watchdog_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    kill "${pid}" 2>/dev/null || true
    sleep 1
  fi
  tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
  stop_orphan_watchdogs
  rm -f "${PID_FILE}"
  log "watchdog stopped"
}

status_watchdog() {
  local orphan_pids
  orphan_pids="$(orphan_watchdog_pids | tr '\n' ' ' || true)"
  if watchdog_running; then
    log "watchdog running pid=$(cat "${PID_FILE}") tmux=$(watchdog_tmux_running && echo yes || echo no) orphan_pids=${orphan_pids:-none} session=${SESSION_NAME} log=${LOG_FILE}"
    tail -n 20 "${LOG_FILE}" 2>/dev/null || true
  else
    log "watchdog not running tmux=$(watchdog_tmux_running && echo yes || echo no) orphan_pids=${orphan_pids:-none} session=${SESSION_NAME}"
    tail -n 20 "${LOG_FILE}" 2>/dev/null || true
  fi
}

case "${ACTION}" in
  start) start_watchdog ;;
  stop) stop_watchdog ;;
  restart)
    stop_watchdog
    start_watchdog
    ;;
  status) status_watchdog ;;
  run) run_loop ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
