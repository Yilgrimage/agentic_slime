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

cp -f "${SRC}" "${PACK_DIR}/slime.tar.gz"
if [ -f "${SRC}.sha256" ]; then
  cp -f "${SRC}.sha256" "${PACK_DIR}/slime.tar.gz.source.sha256"
fi
sha256sum "${PACK_DIR}/slime.tar.gz" > "${PACK_DIR}/slime.tar.gz.sha256"
printf "%s\n" "${REVISION}" > "${PACK_DIR}/slime.revision"

echo "SLIME_PACK=${PACK_DIR}/slime.tar.gz"
echo "SLIME_REVISION=${REVISION}"
