#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MLF_NAS_ROOT="${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}"
SGLANG_REPO="${SGLANG_REPO:-${MLF_NAS_ROOT}/code/sglang}"
ACTION="${1:-apply}"

PATCHES=(
  "${REPO_DIR}/scripts/mlf/patches/sglang/qwen35_moe_text_config_dict.patch"
  "${REPO_DIR}/scripts/mlf/patches/sglang/qwen35_moe_text_norm_topk.patch"
)

usage() {
  echo "Usage: $(basename "$0") [apply|check|reverse]" >&2
}

case "${ACTION}" in
  apply|check|reverse) ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

if [ ! -d "${SGLANG_REPO}/.git" ]; then
  echo "SGLANG_REPO is not a git repo: ${SGLANG_REPO}" >&2
  exit 1
fi

cd "${SGLANG_REPO}"

for patch in "${PATCHES[@]}"; do
  if [ ! -f "${patch}" ]; then
    echo "Missing patch: ${patch}" >&2
    exit 1
  fi

  name="$(basename "${patch}")"
  case "${ACTION}" in
    check)
      if git apply --check "${patch}" >/dev/null 2>&1; then
        echo "[check] ${name}: can apply"
      elif git apply --reverse --check "${patch}" >/dev/null 2>&1; then
        echo "[check] ${name}: already applied"
      else
        echo "[check] ${name}: cannot apply cleanly" >&2
        exit 1
      fi
      ;;
    apply)
      if git apply --check "${patch}" >/dev/null 2>&1; then
        git apply "${patch}"
        echo "[apply] ${name}: applied"
      elif git apply --reverse --check "${patch}" >/dev/null 2>&1; then
        echo "[apply] ${name}: already applied"
      else
        echo "[apply] ${name}: cannot apply cleanly" >&2
        exit 1
      fi
      ;;
    reverse)
      if git apply --reverse --check "${patch}" >/dev/null 2>&1; then
        git apply --reverse "${patch}"
        echo "[reverse] ${name}: reverted"
      elif git apply --check "${patch}" >/dev/null 2>&1; then
        echo "[reverse] ${name}: already absent"
      else
        echo "[reverse] ${name}: cannot reverse cleanly" >&2
        exit 1
      fi
      ;;
  esac
done
