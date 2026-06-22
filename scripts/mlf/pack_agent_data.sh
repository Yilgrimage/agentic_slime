#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
PACK_DIR=${PACK_DIR:-${MLF_NAS_ROOT}/packs}
DATASETS=${DATASETS:-alfworld webshop tau2 appworld}

mkdir -p "${PACK_DIR}"

pack_dataset() {
  local name=$1
  local src="${MLF_NAS_ROOT}/data/${name}"
  local out="${PACK_DIR}/${name}-data.tar.gz"
  if [ ! -d "${src}" ]; then
    echo "Missing data directory: ${src}" >&2
    exit 1
  fi
  tar -C "${MLF_NAS_ROOT}/data" \
    --exclude='*.parts' \
    --exclude='*.part' \
    -czf "${out}" "${name}"
  sha256sum "${out}" > "${out}.sha256"
  {
    echo "name=${name}"
    echo "source=${src}"
    echo "archive=${out}"
    echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "bytes=$(stat -c %s "${out}")"
  } > "${PACK_DIR}/${name}-data.manifest"
  echo "${name}_DATA_PACK=${out}"
}

for dataset in ${DATASETS}; do
  pack_dataset "${dataset}"
done
