#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MICROMAMBA=${MICROMAMBA:-${MLF_NAS_ROOT}/tools/micromamba/bin/micromamba}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-${MLF_NAS_ROOT}/tools/micromamba/root}
CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-/tmp/mlf-runtime/webshop/conda-pkgs}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${MLF_NAS_ROOT}/envs/pip-cache}
ENV_PREFIX=${WEBSHOP_ENV_PREFIX:-${MLF_NAS_ROOT}/envs/webshop-clean}
WEBSHOP_LIB=${WEBSHOP_LIB:-${MLF_NAS_ROOT}/code/WebShop}
WEBSHOP_DATA=${WEBSHOP_DATA:-${MLF_NAS_ROOT}/data/webshop}
WEBSHOP_MODEL_SOURCE_SITE=${WEBSHOP_MODEL_SOURCE_SITE:-${MLF_NAS_ROOT}/envs/webshop/lib/python3.8/site-packages}
PACK_DIR=${PACK_DIR:-${MLF_NAS_ROOT}/packs}
REVISION=${WEBSHOP_REVISION:-webshop-clean-$(date +%Y%m%d)}

export MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS PIP_CACHE_DIR PYTHONNOUSERSITE=1
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M || true

mkdir -p "${PACK_DIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "$(dirname "${ENV_PREFIX}")"

if [ ! -x "${MICROMAMBA}" ]; then
  echo "Missing micromamba: ${MICROMAMBA}" >&2
  exit 1
fi

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  "${MICROMAMBA}" create -y -p "${ENV_PREFIX}" python=3.8 pip openjdk=11 -c conda-forge
fi

export PATH="${ENV_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export JAVA_HOME="${ENV_PREFIX}"
export JVM_PATH="${ENV_PREFIX}/lib/server/libjvm.so"
if [ ! -f "${JVM_PATH}" ] && [ -f "${ENV_PREFIX}/lib/jvm/lib/server/libjvm.so" ]; then
  export JAVA_HOME="${ENV_PREFIX}/lib/jvm"
  export JVM_PATH="${ENV_PREFIX}/lib/jvm/lib/server/libjvm.so"
fi

python -m pip install \
  beautifulsoup4==4.11.1 \
  cleantext==1.1.4 \
  Flask==2.1.2 \
  "Werkzeug<2.3" \
  gym==0.24.0 \
  gdown==5.2.2 \
  numpy==1.24.4 \
  pandas==1.4.2 \
  pyserini==0.17.0 \
  PyYAML==6.0.1 \
  rank_bm25==0.2.2 \
  requests==2.27.1 \
  rich==12.4.4 \
  scikit_learn==1.1.1 \
  spacy==3.3.0 \
  thefuzz==0.19.0 \
  tqdm==4.64.0 \
  conda-pack

python -m pip install --no-deps pydantic==1.10.15

TARGET_SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

if ! python - <<'PY'
import en_core_web_sm  # noqa: F401
PY
then
  if [ ! -d "${WEBSHOP_MODEL_SOURCE_SITE}/en_core_web_sm" ]; then
    echo "Missing en_core_web_sm source package: ${WEBSHOP_MODEL_SOURCE_SITE}/en_core_web_sm" >&2
    exit 1
  fi
  cp -a "${WEBSHOP_MODEL_SOURCE_SITE}/en_core_web_sm" "${TARGET_SITE}/"
  cp -a "${WEBSHOP_MODEL_SOURCE_SITE}"/en_core_web_sm-*.dist-info "${TARGET_SITE}/"
fi

PYTHONPATH="${MLF_NAS_ROOT}/code/slime:${WEBSHOP_LIB}" WEBSHOP_LIB="${WEBSHOP_LIB}" WEBSHOP_DATA="${WEBSHOP_DATA}" python - <<'PY'
import os

from examples.webshop.env_server import _install_text_env_import_stubs, _load_text_env_class

_install_text_env_import_stubs()
cls = _load_text_env_class(os.environ["WEBSHOP_LIB"])
env = cls(observation_mode="text", num_products=1000, human_goals=True)
obs = env.reset(session=0)
print("webshop_clean_env_imports_ok", cls.__name__, type(obs).__name__, len(str(obs)))
PY

conda-pack -p "${ENV_PREFIX}" -o "${PACK_DIR}/webshop.tar.gz" --force
sha256sum "${PACK_DIR}/webshop.tar.gz" > "${PACK_DIR}/webshop.tar.gz.sha256"
printf "%s\n" "${REVISION}" > "${PACK_DIR}/webshop.revision"

echo "WEBSHOP_ENV=${ENV_PREFIX}"
echo "WEBSHOP_PACK=${PACK_DIR}/webshop.tar.gz"
echo "WEBSHOP_REVISION=${REVISION}"
