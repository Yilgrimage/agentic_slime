#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
ROOT_ARG=""
ASSUME_YES=0

usage() {
  cat <<'EOF'
Usage: configure_root.sh [ROOT_DIR] [--yes] [--config FILE]

Write the local, non-git server-ops root config. ROOT_DIR should be the shared
cluster workspace root that contains scripts/, code/, data/, models/, packs/,
runs/, and secrets/ as applicable. If models/ or runs/ must live under another
NAS quota tree, keep these names as symlinks under ROOT_DIR.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG_FILE=$2; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [ -n "${ROOT_ARG}" ]; then
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 2
      fi
      ROOT_ARG=$1
      shift
      ;;
  esac
done

if [ -z "${ROOT_ARG}" ]; then
  printf 'ROOT_DIR: '
  IFS= read -r ROOT_ARG
fi

if [ -z "${ROOT_ARG}" ] || [[ "${ROOT_ARG}" != /* ]]; then
  echo "ROOT_DIR must be an absolute path" >&2
  exit 2
fi
ROOT_ARG=${ROOT_ARG%/}
echo "ROOT_DIR=${ROOT_ARG}"
if [ "${ASSUME_YES}" != "1" ]; then
  printf 'Use this ROOT_DIR for server ops on this machine? [y/N] '
  IFS= read -r answer
  case "${answer}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted"; exit 1 ;;
  esac
fi

mkdir -p "$(dirname "${CONFIG_FILE}")"
umask 077
cat > "${CONFIG_FILE}" <<EOF
# Local machine config for jingyuan server ops. Do not commit this file.
export ROOT_DIR=${ROOT_ARG@Q}
EOF

echo "Wrote ${CONFIG_FILE}"
