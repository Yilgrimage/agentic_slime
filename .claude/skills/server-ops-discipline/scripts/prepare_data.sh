#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
LOCAL_ENVS_DIR="${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}"
DATASETS=${DATASETS:-none}
VALIDATE_DATA=${VALIDATE_DATA:-1}
VALIDATE_DATA_LOAD=${VALIDATE_DATA_LOAD:-0}
FORCE=0

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/dataset_utils.sh"

usage() {
  cat <<'EOF'
Usage: prepare_data.sh [options]

Prepare shared datasets under ROOT_DIR/data. This is the only server-ops entrypoint
that may download or construct dataset contents. Node-local materialization should
only unpack/copy already prepared data.

Options:
  --data LIST          Comma/space list: alfworld,webshop,tau2,appworld,mcp_server
  --force              Rebuild supported datasets even if validation already passes
  --no-validate-data   Skip final layout validation
  --validate-data-load Run supported env load smoke tests after preparation
  -h, --help

Optional dataset inputs:
  WEBSHOP_DATA_SOURCE_DIR   Existing WebShop checkout/data root with data/ and search_engine/
  TAU2_DATA_SOURCE_DIR      tau2 official data root, usually code/tau2-bench/data
  TAU2_AREAL_SOURCE_DIR     AReaL synthetic tau2 root with tau2_rl_train.jsonl
  TAU2_DOWNLOAD_AREAL=1     Download AReaL synthetic tau2 data if no local source exists
  APPWORLD_DATA_SOURCE_DIR  Existing AppWorld root containing data/datasets, data/base_dbs, and data/tasks
  MCP_SERVER_DATA_SOURCE_DIR
                            Private/NAS MCP task data tree copied into ROOT_DIR/data/mcp_server
  MCP_SERVER_IPR_PRODUCT_CHECK_SOURCE_FILE
                            Optional single JSONL copied to mcp_server/ipr_product_check/
  MCP_SERVER_ROPD_SMOKE_SOURCE_FILE
                            Optional teacher JSONL copied to mcp_server/ropd_smoke/
  MCP_SERVER_REQUIRED_FILES Relative files that must exist after MCP data preparation

tau2 preparation also generates portable AReaL task files and prompt JSONL under
ROOT_DIR/data/tau2. Pack and materialize that data on every node that can host
an env server.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATASETS=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    --no-validate-data) VALIDATE_DATA=0; shift ;;
    --validate-data-load) VALIDATE_DATA_LOAD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ "${DATASETS}" = "none" ]; then
  echo "No datasets selected. Use --data alfworld,webshop,tau2,appworld,mcp_server or set DATASETS."
  exit 0
fi

mkdir -p "${ROOT_DIR}/data"

for dataset in $(ops_list_items "${DATASETS}"); do
  prepare_dataset "${dataset}"
done
