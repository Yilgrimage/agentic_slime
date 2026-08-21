#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SERVER_OPS_CONFIG:-${HOME}/.jingyuan/server_ops.env}"
if [ -z "${ROOT_DIR:-}" ] && [ -f "${CONFIG_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
LOCAL_RUNTIME_DIR=${LOCAL_RUNTIME_DIR:-/tmp/server-ops-runtime}
MICROMAMBA=${MICROMAMBA:-${ROOT_DIR}/tools/micromamba/bin/micromamba}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-${ROOT_DIR}/tools/micromamba/root}
CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-${LOCAL_RUNTIME_DIR}/wandb/conda-pkgs}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${ROOT_DIR}/envs/pip-cache}
ENV_PREFIX=${WANDB_ENV_PREFIX:-${ROOT_DIR}/envs/wandb}
PACK_DIR=${PACK_DIR:-${ROOT_DIR}/packs}
PYTHON_VERSION=${WANDB_PYTHON_VERSION:-${PYTHON_VERSION:-3.12}}
WANDB_SPEC=${WANDB_SPEC:-wandb}
WANDB_EXTRA_PIP_PACKAGES=${WANDB_EXTRA_PIP_PACKAGES:-}
WANDB_RECREATE=${WANDB_RECREATE:-0}
REVISION=${WANDB_REVISION:-wandb-$(date -u +%Y%m%d)}

export MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS PIP_CACHE_DIR PYTHONNOUSERSITE=1
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M || true

mkdir -p "${PACK_DIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "$(dirname "${ENV_PREFIX}")"

if [ ! -x "${MICROMAMBA}" ]; then
  echo "Missing micromamba: ${MICROMAMBA}" >&2
  exit 1
fi

if [ "${WANDB_RECREATE}" = "1" ] && [ -d "${ENV_PREFIX}" ]; then
  rm -rf "${ENV_PREFIX}"
fi

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  "${MICROMAMBA}" create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip -c conda-forge
fi

export PATH="${ENV_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y byted-wandb byte-wandb >/dev/null 2>&1 || true
python -m pip install --upgrade "${WANDB_SPEC}" conda-pack
if [ -n "${WANDB_EXTRA_PIP_PACKAGES}" ]; then
  # shellcheck disable=SC2086
  python -m pip install --upgrade ${WANDB_EXTRA_PIP_PACKAGES}
fi

python -m pip check
python - <<'PY'
import importlib.metadata as metadata
import wandb

versions = {"wandb": metadata.version("wandb")}
for package in ("protobuf", "requests", "pydantic", "conda-pack"):
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        pass

print("wandb_env_imports_ok", versions, getattr(wandb, "__file__", "unknown"))
PY

tmp_pack="${PACK_DIR}/wandb.tmp.tar.gz"
tmp_revision="${PACK_DIR}/wandb.revision.tmp"

conda-pack -p "${ENV_PREFIX}" -o "${tmp_pack}" --force
mv "${tmp_pack}" "${PACK_DIR}/wandb.tar.gz"
sha256sum "${PACK_DIR}/wandb.tar.gz" > "${PACK_DIR}/wandb.tar.gz.sha256"
{
  printf "%s\n" "${REVISION}"
  printf "python=%s\n" "${PYTHON_VERSION}"
  printf "wandb_spec=%s\n" "${WANDB_SPEC}"
  printf "extra_pip_packages=%s\n" "${WANDB_EXTRA_PIP_PACKAGES}"
  python -m pip freeze
} > "${tmp_revision}"
mv "${tmp_revision}" "${PACK_DIR}/wandb.revision"
chmod a+r "${PACK_DIR}/wandb.tar.gz" "${PACK_DIR}/wandb.tar.gz.sha256" "${PACK_DIR}/wandb.revision"

echo "WANDB_ENV=${ENV_PREFIX}"
echo "WANDB_PACK=${PACK_DIR}/wandb.tar.gz"
echo "WANDB_REVISION=${REVISION}"
