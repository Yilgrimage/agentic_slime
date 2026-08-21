#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
REPO_DIR="${REPO_DIR:-${ROOT_DIR}/code/slime}"
RUNTIMES=none
RECREATE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: build_runtime_packs.sh --runtime LIST [options]

Build runtime packs from public sources and the selected agentic-slime checkout.
No existing pack or cluster-specific NAS path is required.

Options:
  --runtime LIST  slime,alfworld,appworld,tau2,webshop,wandb
  --repo DIR      Agentic-slime checkout (default ROOT_DIR/code/slime)
  --recreate      Remove each selected build prefix before rebuilding
  --dry-run       Print the commands without running them
  -h, --help

Set ROOT_DIR to any writable workspace root. High-churn caches should use
LOCAL_RUNTIME_DIR on local disk; final artifacts are written to ROOT_DIR/packs.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --runtime|--runtimes) RUNTIMES=$2; shift 2 ;;
    --repo) REPO_DIR=$2; shift 2 ;;
    --recreate) RECREATE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[ "${RUNTIMES}" != none ] || { usage >&2; exit 2; }
[ -d "${REPO_DIR}/slime" ] || { echo "Invalid agentic-slime checkout: ${REPO_DIR}" >&2; exit 2; }
if [ "${DRY_RUN}" -eq 0 ] && [ "${ALLOW_DIRTY_PACK_BUILD:-0}" != "1" ] && \
   [ -n "$(git -C "${REPO_DIR}" status --short)" ]; then
  echo "Refusing to build packs from a dirty agentic-slime checkout: ${REPO_DIR}" >&2
  echo "Commit the handoff source first, or set ALLOW_DIRTY_PACK_BUILD=1 for an explicitly non-reproducible build." >&2
  exit 2
fi

run_command() {
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

build_one() {
  local runtime=$1
  local script
  case "${runtime}" in
    slime)
      run_command env \
        ROOT_DIR="${ROOT_DIR}" \
        REPO_DIR="${REPO_DIR}" \
        SLIME_DIR="${REPO_DIR}" \
        SLIME_RECREATE="${RECREATE}" \
        bash "${REPO_DIR}/build_conda.sh"
      run_command env \
        ROOT_DIR="${ROOT_DIR}" \
        REPO_DIR="${REPO_DIR}" \
        SLIME_SOURCE="${REPO_DIR}" \
        SLIME_ENV_PREFIX="${SLIME_ENV_PREFIX:-${ROOT_DIR}/envs/slime-build}" \
        SLIME_BASE_REVISION="${SLIME_BASE_REVISION:-slime-$(git -C "${REPO_DIR}" rev-parse --short HEAD)}" \
        SLIME_REFRESH_BASE=1 \
        SGLANG_SOURCE="${SGLANG_SOURCE:-${ROOT_DIR}/code/sglang}" \
        MEGATRON_SOURCE="${MEGATRON_SOURCE:-${ROOT_DIR}/code/Megatron-LM}" \
        bash "${REPO_DIR}/scripts/utils/publish_slime_pack.sh"
      ;;
    alfworld|appworld|tau2|webshop)
      script="${REPO_DIR}/scripts/utils/build_${runtime}_env.sh"
      [ -f "${script}" ] || { echo "Missing runtime builder: ${script}" >&2; return 1; }
      case "${runtime}" in
        alfworld) recreate_var=ALFWORLD_RECREATE ;;
        appworld) recreate_var=APPWORLD_RECREATE ;;
        tau2) recreate_var=TAU2_RECREATE ;;
        webshop) recreate_var=WEBSHOP_RECREATE ;;
      esac
      run_command env ROOT_DIR="${ROOT_DIR}" REPO_DIR="${REPO_DIR}" "${recreate_var}=${RECREATE}" bash "${script}"
      ;;
    wandb)
      run_command env \
        ROOT_DIR="${ROOT_DIR}" \
        WANDB_RECREATE="${RECREATE}" \
        bash "${SCRIPT_DIR}/build_wandb_env.sh"
      ;;
    *) echo "Unsupported runtime: ${runtime}" >&2; return 1 ;;
  esac
}

printf '%s\n' "${RUNTIMES}" | tr ', ' '\n' | awk 'NF' | while IFS= read -r runtime; do
  build_one "${runtime}"
done
