#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
MICROMAMBA="${MICROMAMBA:-${ROOT_DIR}/tools/micromamba/bin/micromamba}"
MICROMAMBA_VERSION="${MICROMAMBA_VERSION:-latest}"

case "$(uname -m)" in
  x86_64|amd64) platform=linux-64 ;;
  aarch64|arm64) platform=linux-aarch64 ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

if [ -x "${MICROMAMBA}" ]; then
  echo "MICROMAMBA=${MICROMAMBA}"
  "${MICROMAMBA}" --version
  exit 0
fi

for command_name in curl tar bzip2; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Missing required command: ${command_name}" >&2
    exit 1
  }
done

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/micromamba-bootstrap.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
url="https://micro.mamba.pm/api/micromamba/${platform}/${MICROMAMBA_VERSION}"
curl -fL --retry 3 "${url}" | tar -xj -C "${tmp_dir}" bin/micromamba
mkdir -p "$(dirname "${MICROMAMBA}")"
install -m 0755 "${tmp_dir}/bin/micromamba" "${MICROMAMBA}"

echo "MICROMAMBA=${MICROMAMBA}"
"${MICROMAMBA}" --version
