#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
ARCHIVE_DIR=${ARCHIVE_DIR:-${MLF_NAS_ROOT}/envs/archives}
PACK_DIR=${PACK_DIR:-${MLF_NAS_ROOT}/packs}
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
