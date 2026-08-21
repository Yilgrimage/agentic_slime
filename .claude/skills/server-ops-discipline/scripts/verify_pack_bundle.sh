#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
PACK_DIR="${PACK_DIR:-${ROOT_DIR}/packs}"
RUNTIMES=none
EXTRACT=1

usage() {
  cat <<'EOF'
Usage: verify_pack_bundle.sh --runtime LIST [--pack-dir DIR] [--no-extract]

Verify each runtime tarball, portable checksum, revision, manifest, extraction,
conda-unpack, and basic imports. LIST accepts slime,alfworld,appworld,tau2,
webshop,wandb separated by commas or spaces.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --runtime|--runtimes) RUNTIMES=$2; shift 2 ;;
    --pack-dir) PACK_DIR=$2; shift 2 ;;
    --no-extract) EXTRACT=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[ "${RUNTIMES}" != none ] || { usage >&2; exit 2; }

verify_one() {
  local name=$1
  local pack="${PACK_DIR}/${name}.tar.gz"
  local hash="${pack}.sha256"
  local revision="${PACK_DIR}/${name}.revision"
  local manifest="${PACK_DIR}/${name}.manifest.txt"
  local tmp
  local python

  for file in "${pack}" "${hash}" "${revision}" "${manifest}"; do
    [ -f "${file}" ] || { echo "Missing pack bundle member: ${file}" >&2; return 1; }
  done
  (cd "${PACK_DIR}" && sha256sum -c "${name}.tar.gz.sha256")
  tar -tzf "${pack}" >/dev/null
  [ "${EXTRACT}" -eq 1 ] || return 0

  tmp=$(mktemp -d "${TMPDIR:-/tmp}/${name}-pack-verify.XXXXXX")
  (
    trap 'rm -rf "${tmp}"' EXIT
    tar -xzf "${pack}" -C "${tmp}"
    [ -x "${tmp}/bin/conda-unpack" ] || { echo "Missing conda-unpack in ${name}" >&2; exit 1; }
    "${tmp}/bin/conda-unpack"
    python="${tmp}/bin/python"
    case "${name}" in
      slime)
        PYTHONPATH="${tmp}/src/slime:${tmp}/src/Megatron-LM:${tmp}/src/sglang/python" \
          PYTHONNOUSERSITE=1 "${python}" -c 'import slime, sglang, torch; print("slime_pack_imports_ok", torch.__version__)'
        [ -d "${tmp}/src/Megatron-LM/megatron" ]
        ;;
      alfworld) PYTHONNOUSERSITE=1 "${python}" -c 'import alfworld, textworld; print("alfworld_pack_imports_ok")' ;;
      appworld) PYTHONNOUSERSITE=1 "${python}" -c 'import appworld; print("appworld_pack_imports_ok")' ;;
      tau2) PYTHONNOUSERSITE=1 "${python}" -c 'import tau2; print("tau2_pack_imports_ok")' ;;
      webshop) PYTHONNOUSERSITE=1 "${python}" -c 'import flask, pyserini, spacy; print("webshop_pack_imports_ok")' ;;
      wandb) PYTHONNOUSERSITE=1 "${python}" -c 'import wandb; print("wandb_pack_imports_ok", wandb.__version__)' ;;
      *) echo "Unsupported runtime: ${name}" >&2; exit 1 ;;
    esac
  )
  rm -rf "${tmp}"
}

printf '%s\n' "${RUNTIMES}" | tr ', ' '\n' | awk 'NF' | while IFS= read -r runtime; do
  verify_one "${runtime}"
done
