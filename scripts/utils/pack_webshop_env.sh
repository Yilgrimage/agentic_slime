#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
ENV_PREFIX=${WEBSHOP_ENV_PREFIX:-${MLF_NAS_ROOT}/envs/webshop}
PACK_DIR=${PACK_DIR:-${MLF_NAS_ROOT}/packs}
REVISION=${WEBSHOP_REVISION:-webshop-$(date +%Y%m%d)}

export PYTHONNOUSERSITE=1
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_CONDA _CE_M || true
export PATH="${ENV_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  echo "Missing WebShop env python: ${ENV_PREFIX}/bin/python" >&2
  exit 1
fi

mkdir -p "${PACK_DIR}"

python - <<'PY'
import flask
import numpy
import pydantic
import requests
import urllib3

print("webshop_env_imports_ok", {
    "flask": flask.__version__,
    "numpy": numpy.__version__,
    "pydantic": pydantic.__version__,
    "requests": requests.__version__,
    "urllib3": urllib3.__version__,
})
PY

python -m pip install conda-pack
conda-pack -p "${ENV_PREFIX}" -o "${PACK_DIR}/webshop.tar.gz" --force
sha256sum "${PACK_DIR}/webshop.tar.gz" > "${PACK_DIR}/webshop.tar.gz.sha256"
printf "%s\n" "${REVISION}" > "${PACK_DIR}/webshop.revision"

echo "WEBSHOP_PACK=${PACK_DIR}/webshop.tar.gz"
echo "WEBSHOP_REVISION=${REVISION}"
