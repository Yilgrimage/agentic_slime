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
CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-${LOCAL_RUNTIME_DIR}/webshop/conda-pkgs}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${ROOT_DIR}/envs/pip-cache}
ENV_PREFIX=${WEBSHOP_ENV_PREFIX:-${ROOT_DIR}/envs/webshop-clean}
WEBSHOP_LIB=${WEBSHOP_LIB:-${ROOT_DIR}/code/WebShop}
PACK_DIR=${PACK_DIR:-${ROOT_DIR}/packs}
WEBSHOP_REPO_URL=${WEBSHOP_REPO_URL:-https://github.com/princeton-nlp/WebShop.git}
WEBSHOP_COMMIT=${WEBSHOP_COMMIT:-64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd}
WEBSHOP_SPACY_MODEL_URL=${WEBSHOP_SPACY_MODEL_URL:-https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0-py3-none-any.whl}
WEBSHOP_RECREATE=${WEBSHOP_RECREATE:-0}
REVISION=${WEBSHOP_REVISION:-webshop-${WEBSHOP_COMMIT:0:12}}

export MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS PIP_CACHE_DIR PYTHONNOUSERSITE=1
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M || true

mkdir -p "${PACK_DIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "$(dirname "${ENV_PREFIX}")"
runtime_pack_prepare_prefix "${ENV_PREFIX}" "${WEBSHOP_RECREATE}"

if [ ! -x "${MICROMAMBA}" ]; then
  echo "Missing micromamba: ${MICROMAMBA}" >&2
  exit 1
fi

runtime_pack_require_checkout "${WEBSHOP_REPO_URL}" "${WEBSHOP_COMMIT}" "${WEBSHOP_LIB}"

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

if ! python - <<'PY'
import en_core_web_sm  # noqa: F401
PY
then
  python -m pip install "${WEBSHOP_SPACY_MODEL_URL}"
fi

python - <<'PY'
import flask
import pyserini
import spacy

print("webshop_clean_env_imports_ok", flask.__version__, spacy.__version__, pyserini.__file__)
PY

echo "WEBSHOP_LIB=${WEBSHOP_LIB}"
runtime_pack_publish \
  webshop "${ENV_PREFIX}" "${REVISION}" \
  "builder_repo=$(runtime_pack_git_revision "${REPO_DIR}")" \
  "webshop_repo=${WEBSHOP_REPO_URL}" \
  "webshop_commit=${WEBSHOP_COMMIT}" \
  "spacy_model=${WEBSHOP_SPACY_MODEL_URL}"
