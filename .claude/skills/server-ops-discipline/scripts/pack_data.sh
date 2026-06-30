#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
PACK_DIR=${PACK_DIR:-${ROOT_DIR}/packs}
DATASETS=${DATASETS:-none}
VALIDATE_DATA=${VALIDATE_DATA:-1}
VALIDATE_DATA_LOAD=${VALIDATE_DATA_LOAD:-0}
DRY_RUN=0

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/dataset_utils.sh"

usage() {
  cat <<'EOF'
Usage: pack_data.sh [options]

Validate and archive prepared shared datasets from ROOT_DIR/data into ROOT_DIR/packs.
This script does not download data. Run prepare_data.sh first when shared data is
missing or incomplete.

Options:
  --data LIST          Comma/space list of dataset names, or none
  --no-validate-data   Skip dataset layout validation before packing
  --validate-data-load Run supported env load smoke tests before packing
  --dry-run            Validate and print pack actions without writing archives
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATASETS=$2; shift 2 ;;
    --no-validate-data) VALIDATE_DATA=0; shift ;;
    --validate-data-load) VALIDATE_DATA_LOAD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "${PACK_DIR}"

pack_dataset() {
  local name=$1
  local src="${ROOT_DIR}/data/${name}"
  local out="${PACK_DIR}/${name}-data.tar.gz"
  if [ ! -d "${src}" ]; then
    echo "Missing data directory: ${src}" >&2
    exit 1
  fi
  validate_dataset "${name}" "${src}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "+ tar -C ${ROOT_DIR}/data --exclude='*.parts' --exclude='*.part' -czf ${out} ${name}"
    return
  fi
  tar -C "${ROOT_DIR}/data" \
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

if [ "${DATASETS}" = "none" ]; then
  echo "No datasets selected. Use --data name1,name2 or set DATASETS."
  exit 0
fi

for dataset in $(ops_list_items "${DATASETS}"); do
  pack_dataset "${dataset}"
done
