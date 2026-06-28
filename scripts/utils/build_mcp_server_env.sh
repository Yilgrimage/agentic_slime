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
OPENAI_VERSION=${OPENAI_VERSION:-1.99.1}
ANTHROPIC_VERSION=${ANTHROPIC_VERSION:-0.96.0}
VOLCENGINE_PYTHON_SDK_VERSION=${VOLCENGINE_PYTHON_SDK_VERSION:-5.0.26}
PILLOW_VERSION=${PILLOW_VERSION:-11.3.0}
REQUESTS_VERSION=${REQUESTS_VERSION:-2.32.5}
REGEX_VERSION=${REGEX_VERSION:-2026.2.28}
TIKTOKEN_VERSION=${TIKTOKEN_VERSION:-0.12.0}
OMEGACONF_VERSION=${OMEGACONF_VERSION:-2.3.0}
MAMMOTH_VERSION=${MAMMOTH_VERSION:-1.12.0}
MARKDOWNIFY_VERSION=${MARKDOWNIFY_VERSION:-1.2.2}
OPENPYXL_VERSION=${OPENPYXL_VERSION:-3.1.5}
PDFMINER_SIX_VERSION=${PDFMINER_SIX_VERSION:-20260107}
PYTHON_PPTX_VERSION=${PYTHON_PPTX_VERSION:-1.0.2}
MARKITDOWN_VERSION=${MARKITDOWN_VERSION:-0.1.6}
AIOHTTP_VERSION=${AIOHTTP_VERSION:-3.13.3}
SYMPY_VERSION=${SYMPY_VERSION:-1.14.0}
VALLEYDANCE_ROOT=${VALLEYDANCE_ROOT:-}

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
  "openai==${OPENAI_VERSION}" \
  "anthropic==${ANTHROPIC_VERSION}" \
  "volcengine-python-sdk==${VOLCENGINE_PYTHON_SDK_VERSION}" \
  "pillow==${PILLOW_VERSION}" \
  "requests==${REQUESTS_VERSION}" \
  "regex==${REGEX_VERSION}" \
  "tiktoken==${TIKTOKEN_VERSION}" \
  "omegaconf==${OMEGACONF_VERSION}" \
  "mammoth==${MAMMOTH_VERSION}" \
  "markdownify==${MARKDOWNIFY_VERSION}" \
  "openpyxl==${OPENPYXL_VERSION}" \
  "pdfminer.six==${PDFMINER_SIX_VERSION}" \
  "python-pptx==${PYTHON_PPTX_VERSION}" \
  "markitdown==${MARKITDOWN_VERSION}" \
  "aiohttp==${AIOHTTP_VERSION}" \
  "sympy==${SYMPY_VERSION}" \
  PyYAML \
  conda-pack

python -m pip check
python - <<'PY'
import importlib.metadata as metadata

for module in (
    "mcp",
    "bytedance.mcp",
    "bytedance.mcp.mcp_client",
    "yaml",
    "anyio",
    "openai",
    "anthropic",
    "volcenginesdkarkruntime",
    "PIL",
    "requests",
    "regex",
    "tiktoken",
    "omegaconf",
    "mammoth",
    "markdownify",
    "openpyxl",
    "pdfminer",
    "pptx",
    "markitdown",
    "aiohttp",
    "sympy",
):
    __import__(module, fromlist=["*"])

print("mcp_server_env_imports_ok", {
    "mcp": metadata.version("mcp"),
    "bytedance-mcp": metadata.version("bytedance-mcp"),
    "openai": metadata.version("openai"),
    "volcengine-python-sdk": metadata.version("volcengine-python-sdk"),
    "pillow": metadata.version("pillow"),
    "markitdown": metadata.version("markitdown"),
    "pyyaml": metadata.version("PyYAML"),
    "conda-pack": metadata.version("conda-pack"),
})
PY

if [ -n "${VALLEYDANCE_ROOT}" ] && [ -d "${VALLEYDANCE_ROOT}/Mini-Agent" ]; then
  VALLEYDANCE_ROOT="${VALLEYDANCE_ROOT}" python - <<'PY'
import importlib.util
import os
from pathlib import Path

utils_path = Path(os.environ["VALLEYDANCE_ROOT"]) / "Mini-Agent" / "mini_agent" / "llm" / "utils.py"
if utils_path.exists():
    spec = importlib.util.spec_from_file_location("mini_agent_llm_utils_check", utils_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    print("valleydance_lightweight_import_ok", utils_path)
PY
fi

tmp_pack="${PACK_DIR}/mcp_server.tmp.tar.gz"
tmp_sha="${PACK_DIR}/mcp_server.tar.gz.sha256.tmp"
tmp_revision="${PACK_DIR}/mcp_server.revision.tmp"
archive_dir="${PACK_DIR}/archive"
archive_stamp="$(date -u +%Y%m%dT%H%M%SZ)"

conda-pack -p "${ENV_PREFIX}" -o "${tmp_pack}" --force
sha256sum "${tmp_pack}" | awk -v pack="${PACK_DIR}/mcp_server.tar.gz" '{print $1 "  " pack}' > "${tmp_sha}"
printf "%s\n" "${REVISION}" > "${tmp_revision}"

mkdir -p "${archive_dir}"
if [ -f "${PACK_DIR}/mcp_server.tar.gz" ]; then
  mv "${PACK_DIR}/mcp_server.tar.gz" "${archive_dir}/mcp_server.${archive_stamp}.tar.gz"
fi
if [ -f "${PACK_DIR}/mcp_server.tar.gz.sha256" ]; then
  mv "${PACK_DIR}/mcp_server.tar.gz.sha256" "${archive_dir}/mcp_server.${archive_stamp}.tar.gz.sha256"
fi
if [ -f "${PACK_DIR}/mcp_server.revision" ]; then
  mv "${PACK_DIR}/mcp_server.revision" "${archive_dir}/mcp_server.${archive_stamp}.revision"
fi

mv "${tmp_pack}" "${PACK_DIR}/mcp_server.tar.gz"
mv "${tmp_sha}" "${PACK_DIR}/mcp_server.tar.gz.sha256"
mv "${tmp_revision}" "${PACK_DIR}/mcp_server.revision"

echo "MCP_SERVER_ENV=${ENV_PREFIX}"
echo "MCP_SERVER_PACK=${PACK_DIR}/mcp_server.tar.gz"
echo "MCP_SERVER_REVISION=${REVISION}"
