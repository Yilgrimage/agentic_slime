#!/bin/bash

set -ex

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
ENV_DIR=${WEBSHOP_ENV:-${MLF_NAS_ROOT}/envs/webshop}
SRC_DIR=${WEBSHOP_SRC:-${MLF_NAS_ROOT}/code/WebShop}
OUT_DIR=${WEBSHOP_DATA_NAS:-${MLF_NAS_ROOT}/data/webshop}
SLIME_DIR=${SLIME_DIR:-${MLF_NAS_ROOT}/code/slime}
WEBSHOP_DATA_SIZE=${WEBSHOP_DATA_SIZE:-full}
INDEX_THREADS=${INDEX_THREADS:-4}

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
        timeout 45s "${ENV_DIR}/bin/python" -m gdown "${url}" -O "${output}" && return 0
      else
        curl -L "${url}" -o "${output}" && test -s "${output}" && return 0
      fi
    done
    echo "Failed to download ${output}"
    exit 1
  fi
}

download_file "${OUT_DIR}/data/items_shuffle_1000.json" \
  "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_shuffle_1000.json" \
  "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib"
download_file "${OUT_DIR}/data/items_ins_v2_1000.json" \
  "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_ins_v2_1000.json" \
  "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu"
download_file "${OUT_DIR}/data/items_human_ins.json" \
  "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_human_ins.json" \
  "https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O"

if [ "${WEBSHOP_DATA_SIZE}" = "full" ] || [ "${WEBSHOP_DATA_SIZE}" = "all" ]; then
  download_file "${OUT_DIR}/data/items_shuffle.json" \
    "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_shuffle.json" \
    "https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB"
  download_file "${OUT_DIR}/data/items_ins_v2.json" \
    "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_ins_v2.json" \
    "https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi"
elif [ "${WEBSHOP_DATA_SIZE}" != "small" ]; then
  echo "WEBSHOP_DATA_SIZE must be small or full, got ${WEBSHOP_DATA_SIZE}" >&2
  exit 1
fi

mkdir -p "${SRC_DIR}/data"
cp -a "${OUT_DIR}/data/." "${SRC_DIR}/data/"

cd "${SRC_DIR}/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k
PYTHONPATH="${SLIME_DIR}:${SRC_DIR}" JAVA_HOME="${ENV_DIR}/lib/jvm" JVM_PATH="${ENV_DIR}/lib/jvm/lib/server/libjvm.so" \
  WEBSHOP_DATA_SIZE="${WEBSHOP_DATA_SIZE}" "${ENV_DIR}/bin/python" - <<'PY'
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from examples.agent_env.webshop.server import _install_text_env_import_stubs

_install_text_env_import_stubs()
sys.path.insert(0, "../")

import web_agent_site.engine.engine as engine
from web_agent_site.engine.engine import load_products

if os.environ.get("WEBSHOP_DATA_SIZE") in {"full", "all"}:
    product_path = "../data/items_shuffle.json"
    engine.DEFAULT_ATTR_PATH = "../data/items_ins_v2.json"
else:
    product_path = "../data/items_shuffle_1000.json"
    engine.DEFAULT_ATTR_PATH = "../data/items_ins_v2_1000.json"

all_products, *_ = load_products(filepath=product_path, human_goals=True)
docs = []
for p in tqdm(all_products, total=len(all_products)):
    option_texts = []
    for option_name, option_contents in (p.get("options") or {}).items():
        option_texts.append(f"{option_name}: {', '.join(option_contents)}")
    doc = {
        "id": p["asin"],
        "contents": " ".join(
            [
                p.get("Title") or "",
                p.get("Description") or "",
                (p.get("BulletPoints") or [""])[0],
                ", and ".join(option_texts),
            ]
        ).lower(),
        "product": p,
    }
    docs.append(doc)

outputs = {
    "resources_100/documents.jsonl": docs[:100],
    "resources_1k/documents.jsonl": docs[:1000],
    "resources_100k/documents.jsonl": docs[:100000],
    "resources/documents.jsonl": docs,
}
for path, rows in outputs.items():
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
print("webshop_documents", len(docs), "resources_100k", len(outputs["resources_100k/documents.jsonl"]))
PY

JAVA_HOME="${ENV_DIR}/lib/jvm" JVM_PATH="${ENV_DIR}/lib/jvm/lib/server/libjvm.so" \
  "${ENV_DIR}/bin/python" -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources_1k \
  --index indexes_1k \
  --generator DefaultLuceneDocumentGenerator \
  --threads "${INDEX_THREADS}" \
  --storePositions --storeDocvectors --storeRaw

rm -rf "${OUT_DIR}/search_engine/indexes_1k"
cp -a indexes_1k "${OUT_DIR}/search_engine/indexes_1k"

if [ "${WEBSHOP_DATA_SIZE}" = "full" ] || [ "${WEBSHOP_DATA_SIZE}" = "all" ]; then
  rm -rf indexes_100k
  JAVA_HOME="${ENV_DIR}/lib/jvm" JVM_PATH="${ENV_DIR}/lib/jvm/lib/server/libjvm.so" \
    "${ENV_DIR}/bin/python" -m pyserini.index.lucene \
    --collection JsonCollection \
    --input resources_100k \
    --index indexes_100k \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${INDEX_THREADS}" \
    --storePositions --storeDocvectors --storeRaw

  rm -rf "${OUT_DIR}/search_engine/indexes_100k"
  cp -a indexes_100k "${OUT_DIR}/search_engine/indexes_100k"
fi

echo "Prepared WebShop data under ${OUT_DIR}"
