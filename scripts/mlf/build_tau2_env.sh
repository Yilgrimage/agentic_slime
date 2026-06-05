#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MICROMAMBA=${MICROMAMBA:-${MLF_NAS_ROOT}/tools/micromamba/bin/micromamba}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-${MLF_NAS_ROOT}/tools/micromamba/root}
CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-/tmp/mlf-runtime/tau2/conda-pkgs}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${MLF_NAS_ROOT}/envs/pip-cache}
ENV_PREFIX=${TAU2_ENV_PREFIX:-${MLF_NAS_ROOT}/envs/tau2}
TAU2_LIB=${TAU2_LIB:-${MLF_NAS_ROOT}/code/tau2-bench}
PACK_DIR=${PACK_DIR:-${MLF_NAS_ROOT}/packs}
REVISION=${TAU2_REVISION:-tau2-$(date +%Y%m%d)}

export MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS PIP_CACHE_DIR PYTHONNOUSERSITE=1
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_CONDA _CE_M || true

mkdir -p "${PACK_DIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "$(dirname "${ENV_PREFIX}")" "$(dirname "${TAU2_LIB}")"

if [ ! -x "${MICROMAMBA}" ]; then
  echo "Missing micromamba: ${MICROMAMBA}" >&2
  exit 1
fi

if [ ! -d "${TAU2_LIB}/.git" ]; then
  git clone --depth 1 https://github.com/sierra-research/tau2-bench.git "${TAU2_LIB}"
else
  git -C "${TAU2_LIB}" fetch --depth 1 origin main
  git -C "${TAU2_LIB}" reset --hard origin/main
fi

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  "${MICROMAMBA}" create -y -p "${ENV_PREFIX}" python=3.12 pip -c conda-forge
fi

export PATH="${ENV_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

python -m pip install --upgrade pip setuptools wheel
python -m pip install conda-pack

# Install the package into the conda env, but keep the source tree outside the
# pack so future source edits on NAS do not affect an unpacked runtime env.
python -m pip install "${TAU2_LIB}[gym]"

python - <<'PY'
import importlib.metadata as metadata

print("tau2_env_imports_ok", {
    "tau2": metadata.version("tau2"),
})
PY

conda-pack -p "${ENV_PREFIX}" -o "${PACK_DIR}/tau2.tar.gz" --force
sha256sum "${PACK_DIR}/tau2.tar.gz" > "${PACK_DIR}/tau2.tar.gz.sha256"
printf "%s\n" "${REVISION}" > "${PACK_DIR}/tau2.revision"

echo "TAU2_ENV=${ENV_PREFIX}"
echo "TAU2_LIB=${TAU2_LIB}"
echo "TAU2_PACK=${PACK_DIR}/tau2.tar.gz"
echo "TAU2_REVISION=${REVISION}"
