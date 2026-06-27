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
CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-${LOCAL_RUNTIME_DIR}/appworld/conda-pkgs}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${ROOT_DIR}/envs/pip-cache}
ENV_PREFIX=${APPWORLD_ENV_PREFIX:-${ROOT_DIR}/envs/appworld}
APPWORLD_LIB=${APPWORLD_LIB:-${ROOT_DIR}/code/appworld}
APPWORLD_ROOT=${APPWORLD_ROOT:-${ROOT_DIR}/data/appworld}
PACK_DIR=${PACK_DIR:-${ROOT_DIR}/packs}
REVISION=${APPWORLD_REVISION:-appworld-$(date +%Y%m%d)}

export MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS PIP_CACHE_DIR PYTHONNOUSERSITE=1 APPWORLD_ROOT
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M || true

mkdir -p "${PACK_DIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "$(dirname "${ENV_PREFIX}")" "$(dirname "${APPWORLD_LIB}")" "${APPWORLD_ROOT}"

if [ ! -x "${MICROMAMBA}" ]; then
  echo "Missing micromamba: ${MICROMAMBA}" >&2
  exit 1
fi

if [ ! -d "${APPWORLD_LIB}/.git" ]; then
  git clone --depth 1 https://github.com/StonyBrookNLP/appworld.git "${APPWORLD_LIB}"
else
  git -C "${APPWORLD_LIB}" fetch --depth 1 origin main
  git -C "${APPWORLD_LIB}" reset --hard origin/main
fi

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  "${MICROMAMBA}" create -y -p "${ENV_PREFIX}" python=3.11 pip -c conda-forge
fi

export PATH="${ENV_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

python -m pip install --upgrade pip setuptools wheel
python -m pip install appworld conda-pack

# AppWorld's CLI treats --root as the project root and stores benchmark data
# under ${APPWORLD_ROOT}/data. Its install command also writes tests under
# $HOME/.appworld, so use APPWORLD_ROOT as HOME while installing task assets.
HOME="${APPWORLD_ROOT}" APPWORLD_ROOT="${APPWORLD_ROOT}" appworld install --root "${APPWORLD_ROOT}"
HOME="${APPWORLD_ROOT}" APPWORLD_ROOT="${APPWORLD_ROOT}" appworld download data --root "${APPWORLD_ROOT}"

python - <<'PY'
import importlib.metadata as metadata
import os
from pathlib import Path

root = Path(os.environ["APPWORLD_ROOT"])
print("appworld_env_imports_ok", {
    "appworld": metadata.version("appworld"),
    "root": str(root),
    "has_data": (root / "data").exists(),
    "has_tests": (root / ".appworld" / "tests").exists(),
})
PY

conda-pack -p "${ENV_PREFIX}" -o "${PACK_DIR}/appworld.tar.gz" --force
sha256sum "${PACK_DIR}/appworld.tar.gz" > "${PACK_DIR}/appworld.tar.gz.sha256"
printf "%s\n" "${REVISION}" > "${PACK_DIR}/appworld.revision"

echo "APPWORLD_ENV=${ENV_PREFIX}"
echo "APPWORLD_ROOT=${APPWORLD_ROOT}"
echo "APPWORLD_LIB=${APPWORLD_LIB}"
echo "APPWORLD_PACK=${PACK_DIR}/appworld.tar.gz"
echo "APPWORLD_REVISION=${REVISION}"
