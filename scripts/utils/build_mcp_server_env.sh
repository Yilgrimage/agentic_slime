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
LOCAL_RUNTIME_DIR=${LOCAL_RUNTIME_DIR:-/tmp/server-ops-runtime}
MICROMAMBA=${MICROMAMBA:-${ROOT_DIR}/tools/micromamba/bin/micromamba}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-${ROOT_DIR}/tools/micromamba/root}
CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-${LOCAL_RUNTIME_DIR}/mcp_server/conda-pkgs}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${ROOT_DIR}/envs/pip-cache}
ENV_PREFIX=${MCP_SERVER_ENV_PREFIX:-${LOCAL_RUNTIME_DIR}/envs/mcp_server}
PACK_DIR=${PACK_DIR:-${ROOT_DIR}/packs}
REVISION=${MCP_SERVER_REVISION:-mcp_server-psm-$(date -u +%Y%m%d)}

MCP_VERSION=${MCP_VERSION:-1.28.0}
BYTEDANCE_MCP_VERSION=${BYTEDANCE_MCP_VERSION:-0.2.45}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}

export MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS PIP_CACHE_DIR PYTHONNOUSERSITE=1
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M || true

mkdir -p "${PACK_DIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "$(dirname "${ENV_PREFIX}")"

if [ ! -x "${MICROMAMBA}" ]; then
  echo "Missing micromamba: ${MICROMAMBA}" >&2
  exit 1
fi

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  "${MICROMAMBA}" create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip -c conda-forge
fi

export PATH="${ENV_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "mcp==${MCP_VERSION}" \
  "bytedance-mcp==${BYTEDANCE_MCP_VERSION}" \
  PyYAML \
  conda-pack

python -m pip check
python - <<'PY'
import importlib.metadata as metadata

for module in ("mcp", "bytedance.mcp", "bytedance.mcp.mcp_client", "yaml", "anyio"):
    __import__(module, fromlist=["*"])

print("mcp_server_env_imports_ok", {
    "mcp": metadata.version("mcp"),
    "bytedance-mcp": metadata.version("bytedance-mcp"),
    "pyyaml": metadata.version("PyYAML"),
    "conda-pack": metadata.version("conda-pack"),
})
PY

tmp_pack="${PACK_DIR}/mcp_server.tar.gz.tmp"
tmp_sha="${PACK_DIR}/mcp_server.tar.gz.sha256.tmp"
tmp_revision="${PACK_DIR}/mcp_server.revision.tmp"

conda-pack -p "${ENV_PREFIX}" -o "${tmp_pack}" --force
sha256sum "${tmp_pack}" > "${tmp_sha}"
printf "%s\n" "${REVISION}" > "${tmp_revision}"

mv "${tmp_pack}" "${PACK_DIR}/mcp_server.tar.gz"
mv "${tmp_sha}" "${PACK_DIR}/mcp_server.tar.gz.sha256"
mv "${tmp_revision}" "${PACK_DIR}/mcp_server.revision"

echo "MCP_SERVER_ENV=${ENV_PREFIX}"
echo "MCP_SERVER_PACK=${PACK_DIR}/mcp_server.tar.gz"
echo "MCP_SERVER_REVISION=${REVISION}"
