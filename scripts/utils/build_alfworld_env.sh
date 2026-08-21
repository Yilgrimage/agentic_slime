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
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runtime_pack_common.sh"
runtime_pack_configure_pip
LOCAL_RUNTIME_DIR=${LOCAL_RUNTIME_DIR:-/tmp/server-ops-runtime}
MICROMAMBA=${MICROMAMBA:-${ROOT_DIR}/tools/micromamba/bin/micromamba}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-${ROOT_DIR}/tools/micromamba/root}
CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-${LOCAL_RUNTIME_DIR}/alfworld/conda-pkgs}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${ROOT_DIR}/envs/pip-cache}
ENV_PREFIX=${ALFWORLD_ENV_PREFIX:-${ROOT_DIR}/envs/alfworld}
PACK_DIR=${PACK_DIR:-${ROOT_DIR}/packs}
ALFWORLD_VERSION=${ALFWORLD_VERSION:-0.4.2}
ALFWORLD_RECREATE=${ALFWORLD_RECREATE:-0}
REVISION=${ALFWORLD_REVISION:-alfworld-${ALFWORLD_VERSION}}

export MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS PIP_CACHE_DIR PYTHONNOUSERSITE=1
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M || true

mkdir -p "${PACK_DIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "$(dirname "${ENV_PREFIX}")"
runtime_pack_prepare_prefix "${ENV_PREFIX}" "${ALFWORLD_RECREATE}"

if [ ! -x "${MICROMAMBA}" ]; then
  echo "Missing micromamba: ${MICROMAMBA}" >&2
  exit 1
fi

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  "${MICROMAMBA}" create -y -p "${ENV_PREFIX}" python=3.10 pip -c conda-forge
fi

export PATH="${ENV_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "alfworld==${ALFWORLD_VERSION}" PyYAML conda-pack

python - <<'PY'
import alfworld
import textworld
import yaml

print("alfworld_env_imports_ok", {
    "alfworld": getattr(alfworld, "__file__", "unknown"),
    "textworld": getattr(textworld, "__version__", "unknown"),
    "yaml": getattr(yaml, "__version__", "unknown"),
})
PY

runtime_pack_publish \
  alfworld "${ENV_PREFIX}" "${REVISION}" \
  "builder_repo=$(runtime_pack_git_revision "${REPO_DIR}")" \
  "alfworld_version=${ALFWORLD_VERSION}"
