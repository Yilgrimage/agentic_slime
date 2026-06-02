#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
PACK_DIR=${PACK_DIR:-${MLF_NAS_ROOT}/packs}

materialize_pack() {
  local name=$1
  local pack="${PACK_DIR}/${name}.tar.gz"
  local target="${MLF_LOCAL_ENVS}/${name}"
  if [ ! -f "${pack}" ]; then
    echo "Missing pack: ${pack}" >&2
    exit 1
  fi
  if [ ! -x "${target}/bin/python" ]; then
    mkdir -p "${target}"
    tar -xzf "${pack}" -C "${target}"
    "${target}/bin/conda-unpack"
  fi
  echo "${name}_ENV=${target}"
}

copy_dir_once() {
  local src=$1
  local dst=$2
  if [ -d "${src}" ] && [ ! -d "${dst}" ]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  fi
  echo "${dst}"
}

mkdir -p "${MLF_LOCAL_ENVS}" "${MLF_LOCAL_ROOT}"

materialize_pack slime
materialize_pack alfworld
materialize_pack webshop

copy_dir_once "${MLF_NAS_ROOT}/models/Qwen3-8B" "${MLF_LOCAL_ROOT}/models/Qwen3-8B" >/dev/null
copy_dir_once "${MLF_NAS_ROOT}/models/Qwen3-8B_torch_dist" "${MLF_LOCAL_ROOT}/models/Qwen3-8B_torch_dist" >/dev/null
copy_dir_once "${MLF_NAS_ROOT}/data/alfworld" "${MLF_LOCAL_ROOT}/data/alfworld" >/dev/null
copy_dir_once "${MLF_NAS_ROOT}/data/webshop" "${MLF_LOCAL_ROOT}/data/webshop" >/dev/null
copy_dir_once "${MLF_NAS_ROOT}/code/WebShop" "${MLF_LOCAL_ROOT}/code/WebShop" >/dev/null

if [ -d "${MLF_LOCAL_ROOT}/data/webshop/data" ]; then
  mkdir -p "${MLF_LOCAL_ROOT}/code/WebShop/data"
  cp -a "${MLF_LOCAL_ROOT}/data/webshop/data/." "${MLF_LOCAL_ROOT}/code/WebShop/data/"
fi

if [ -d "${MLF_LOCAL_ROOT}/data/webshop/search_engine/indexes_1k" ]; then
  mkdir -p "${MLF_LOCAL_ROOT}/code/WebShop/search_engine"
  rm -rf "${MLF_LOCAL_ROOT}/code/WebShop/search_engine/indexes_1k"
  cp -a "${MLF_LOCAL_ROOT}/data/webshop/search_engine/indexes_1k" "${MLF_LOCAL_ROOT}/code/WebShop/search_engine/indexes_1k"
fi

echo "MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS}"
echo "MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT}"
echo "QWEN3_8B=${MLF_LOCAL_ROOT}/models/Qwen3-8B"
echo "QWEN3_8B_TORCH_DIST=${MLF_LOCAL_ROOT}/models/Qwen3-8B_torch_dist"
echo "ALFWORLD_DATA=${MLF_LOCAL_ROOT}/data/alfworld"
echo "WEBSHOP_DATA=${MLF_LOCAL_ROOT}/data/webshop"
echo "WEBSHOP_LIB=${MLF_LOCAL_ROOT}/code/WebShop"
