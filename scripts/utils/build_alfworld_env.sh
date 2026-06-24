#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MICROMAMBA=${MICROMAMBA:-${MLF_NAS_ROOT}/tools/micromamba/bin/micromamba}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-${MLF_NAS_ROOT}/tools/micromamba/root}
CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-/tmp/mlf-runtime/alfworld/conda-pkgs}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${MLF_NAS_ROOT}/envs/pip-cache}
ENV_PREFIX=${ALFWORLD_ENV_PREFIX:-${MLF_NAS_ROOT}/envs/alfworld}
PACK_DIR=${PACK_DIR:-${MLF_NAS_ROOT}/packs}
REVISION=${ALFWORLD_REVISION:-alfworld-$(date +%Y%m%d)}

export MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS PIP_CACHE_DIR PYTHONNOUSERSITE=1
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M || true

mkdir -p "${PACK_DIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "$(dirname "${ENV_PREFIX}")"

if [ ! -x "${MICROMAMBA}" ]; then
  echo "Missing micromamba: ${MICROMAMBA}" >&2
  exit 1
fi

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  "${MICROMAMBA}" create -y -p "${ENV_PREFIX}" python=3.10 pip -c conda-forge
fi

export PATH="${ENV_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "alfworld>=0.4.2" PyYAML conda-pack

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

conda-pack -p "${ENV_PREFIX}" -o "${PACK_DIR}/alfworld.tar.gz" --force
sha256sum "${PACK_DIR}/alfworld.tar.gz" > "${PACK_DIR}/alfworld.tar.gz.sha256"
printf "%s\n" "${REVISION}" > "${PACK_DIR}/alfworld.revision"

echo "ALFWORLD_PACK=${PACK_DIR}/alfworld.tar.gz"
echo "ALFWORLD_REVISION=${REVISION}"
