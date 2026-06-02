#!/bin/bash

set -ex

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
ENV_DIR=${WEBSHOP_ENV:-${MLF_NAS_ROOT}/envs/webshop}
SRC_DIR=${WEBSHOP_SRC:-${MLF_NAS_ROOT}/code/WebShop}
OUT_DIR=${WEBSHOP_DATA_NAS:-${MLF_NAS_ROOT}/data/webshop}
SLIME_DIR=${SLIME_DIR:-${MLF_NAS_ROOT}/code/slime}

mkdir -p "${OUT_DIR}/data" "${OUT_DIR}/search_engine"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "Missing WebShop env: ${ENV_DIR}"
  exit 1
fi
if [ ! -d "${SRC_DIR}" ]; then
  echo "Missing WebShop source: ${SRC_DIR}"
  exit 1
fi

"${ENV_DIR}/bin/python" -m pip install gdown

download_file() {
  local output=$1
  shift
  if [ ! -s "${output}" ]; then
    for url in "$@"; do
      if [[ "${url}" == https://drive.google.com/* ]]; then
        timeout 45s "${ENV_DIR}/bin/gdown" "${url}" -O "${output}" && return 0
      else
        curl -L "${url}" -o "${output}" && test -s "${output}" && return 0
      fi
    done
    echo "Failed to download ${output}"
    exit 1
  fi
}

download_file "${OUT_DIR}/data/items_shuffle_1000.json" \
  "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib" \
  "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_shuffle_1000.json"
download_file "${OUT_DIR}/data/items_ins_v2_1000.json" \
  "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu" \
  "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_ins_v2_1000.json"
download_file "${OUT_DIR}/data/items_human_ins.json" \
  "https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O" \
  "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_human_ins.json"

mkdir -p "${SRC_DIR}/data"
cp -a "${OUT_DIR}/data/." "${SRC_DIR}/data/"

cd "${SRC_DIR}/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k
PYTHONPATH="${SLIME_DIR}:${SRC_DIR}" JAVA_HOME="${ENV_DIR}/lib/jvm" JVM_PATH="${ENV_DIR}/lib/jvm/lib/server/libjvm.so" \
  "${ENV_DIR}/bin/python" - <<'PY'
import runpy
from examples.webshop.env_server import _install_text_env_import_stubs

_install_text_env_import_stubs()
runpy.run_path("convert_product_file_format.py", run_name="__main__")
PY

JAVA_HOME="${ENV_DIR}/lib/jvm" JVM_PATH="${ENV_DIR}/lib/jvm/lib/server/libjvm.so" \
  "${ENV_DIR}/bin/python" -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources_1k \
  --index indexes_1k \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

rm -rf "${OUT_DIR}/search_engine/indexes_1k"
cp -a indexes_1k "${OUT_DIR}/search_engine/indexes_1k"

echo "Prepared WebShop data under ${OUT_DIR}"
