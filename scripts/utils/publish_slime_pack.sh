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
ARCHIVE_DIR=${ARCHIVE_DIR:-${ROOT_DIR}/envs/archives}
PACK_DIR=${PACK_DIR:-${ROOT_DIR}/packs}
CURRENT_FILE=${CURRENT_FILE:-${ARCHIVE_DIR}/slime-official.current}
MEGATRON_SOURCE=${MEGATRON_SOURCE:-${ROOT_DIR}/code/Megatron-LM}

mkdir -p "${PACK_DIR}"

if [ ! -f "${CURRENT_FILE}" ]; then
  echo "Missing slime current revision file: ${CURRENT_FILE}" >&2
  exit 1
fi

REVISION=$(cat "${CURRENT_FILE}")
SRC="${ARCHIVE_DIR}/${REVISION}.tar.gz"
if [ ! -f "${SRC}" ]; then
  echo "Missing slime archive: ${SRC}" >&2
  exit 1
fi

PACK_REVISION="${REVISION}"

if [ ! -d "${MEGATRON_SOURCE}/megatron" ]; then
  echo "Missing Megatron-LM source: ${MEGATRON_SOURCE}" >&2
  exit 1
fi

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/slime-pack.XXXXXX")
TMP_PACK="${PACK_DIR}/.slime.tar.gz.tmp.$$"
cleanup() {
  rm -rf "${WORK_DIR}"
  rm -f "${TMP_PACK}"
}
trap cleanup EXIT

ENV_DIR="${WORK_DIR}/env"
mkdir -p "${ENV_DIR}/src/Megatron-LM"
tar -xzf "${SRC}" -C "${ENV_DIR}"
tar -C "${MEGATRON_SOURCE}" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.mypy_cache' \
  --exclude='.pytest_cache' \
  --exclude='build' \
  --exclude='dist' \
  -cf - . | tar -C "${ENV_DIR}/src/Megatron-LM" -xf -

MEGATRON_REVISION=$(git -C "${MEGATRON_SOURCE}" rev-parse --short HEAD 2>/dev/null || echo unknown)
PACK_REVISION="${REVISION}+megatron-${MEGATRON_REVISION}"
tar -C "${ENV_DIR}" -czf "${TMP_PACK}" .
mv -f "${TMP_PACK}" "${PACK_DIR}/slime.tar.gz"

if [ -f "${SRC}.sha256" ]; then
  cp -f "${SRC}.sha256" "${PACK_DIR}/slime.tar.gz.base.sha256"
fi
sha256sum "${PACK_DIR}/slime.tar.gz" > "${PACK_DIR}/slime.tar.gz.sha256"
printf "%s\n" "${PACK_REVISION}" > "${PACK_DIR}/slime.revision"
{
  printf "base_revision=%s\n" "${REVISION}"
  printf "include_megatron=1\n"
  printf "megatron_source=%s\n" "${MEGATRON_SOURCE}"
  printf "pack_revision=%s\n" "${PACK_REVISION}"
} > "${PACK_DIR}/slime.pack_note"

echo "SLIME_PACK=${PACK_DIR}/slime.tar.gz"
echo "SLIME_REVISION=${PACK_REVISION}"
