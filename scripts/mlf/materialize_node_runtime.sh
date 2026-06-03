#!/usr/bin/env bash
set -euo pipefail

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
PACK_DIR=${PACK_DIR:-${MLF_NAS_ROOT}/packs}

ENVS=${ENVS:-slime,alfworld,webshop}
DATASETS=${DATASETS:-alfworld,webshop}
MODELS=${MODELS:-qwen3-8b}
SOURCES=${SOURCES:-webshop}
FORCE=0
CHECK_HASH=1
SKIP_ENVS=0
SKIP_DATA=0
SKIP_MODELS=0
SKIP_SOURCES=0

usage() {
  cat <<'EOF'
Usage: materialize_node_runtime.sh [options]

Materialize conda packs, data packs, models, and source mirrors on the current node.

Options:
  --envs LIST        Comma list: slime,alfworld,webshop,none
  --data LIST        Comma list: alfworld,webshop,none
  --models LIST      Comma list: qwen3-8b,none
  --sources LIST     Comma list: webshop,none
  --force            Reinstall/copy even if local stamp matches
  --no-check-hash    Use existence checks only
  --skip-envs
  --skip-data
  --skip-models
  --skip-sources
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --envs) ENVS=$2; shift 2 ;;
    --data) DATASETS=$2; shift 2 ;;
    --models) MODELS=$2; shift 2 ;;
    --sources) SOURCES=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    --no-check-hash) CHECK_HASH=0; shift ;;
    --skip-envs) SKIP_ENVS=1; shift ;;
    --skip-data) SKIP_DATA=1; shift ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --skip-sources) SKIP_SOURCES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "${MLF_LOCAL_ENVS}" "${MLF_LOCAL_ROOT}"

contains_item() {
  local list=$1
  local item=$2
  [ "${list}" != "none" ] || return 1
  case ",${list}," in
    *",${item},"*) return 0 ;;
    *) return 1 ;;
  esac
}

sha_file_value() {
  local sha_file=$1
  awk '{print $1}' "${sha_file}"
}

stamp_matches() {
  local stamp=$1
  local sha_file=$2
  [ "${CHECK_HASH}" -eq 1 ] || return 1
  [ -f "${stamp}" ] || return 1
  [ -f "${sha_file}" ] || return 1
  [ "$(cat "${stamp}")" = "$(sha_file_value "${sha_file}")" ]
}

write_stamp() {
  local stamp=$1
  local sha_file=$2
  if [ "${CHECK_HASH}" -eq 1 ] && [ -f "${sha_file}" ]; then
    sha_file_value "${sha_file}" > "${stamp}"
  fi
}

copy_tree_excluding_parts() {
  local src=$1
  local dst=$2
  mkdir -p "${dst}"
  tar -C "${src}" --exclude='*.parts' --exclude='*.part' -cf - . | tar -C "${dst}" -xf -
}

materialize_pack() {
  local name=$1
  local pack="${PACK_DIR}/${name}.tar.gz"
  local sha_file="${pack}.sha256"
  local target="${MLF_LOCAL_ENVS}/${name}"
  local stamp="${target}/.pack.sha256"
  if [ ! -f "${pack}" ]; then
    echo "Missing pack: ${pack}" >&2
    exit 1
  fi
  if [ "${FORCE}" -eq 0 ] && [ -x "${target}/bin/python" ]; then
    if [ "${CHECK_HASH}" -eq 0 ] || stamp_matches "${stamp}" "${sha_file}" || [ ! -f "${sha_file}" ]; then
      echo "${name}_ENV=${target} (cached)"
      return
    fi
  fi
  echo "Materializing env ${name} -> ${target}"
  rm -rf "${target}"
  mkdir -p "${target}"
  tar -xzf "${pack}" -C "${target}"
  "${target}/bin/conda-unpack"
  write_stamp "${stamp}" "${sha_file}"
  echo "${name}_ENV=${target}"
}

copy_dir_once() {
  local src=$1
  local dst=$2
  if [ -d "${src}" ] && { [ "${FORCE}" -eq 1 ] || [ ! -d "${dst}" ]; }; then
    rm -rf "${dst}"
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  fi
}

materialize_data() {
  local name=$1
  local src_dir="${MLF_NAS_ROOT}/data/${name}"
  local pack="${PACK_DIR}/${name}-data.tar.gz"
  local sha_file="${pack}.sha256"
  local dst="${MLF_LOCAL_ROOT}/data/${name}"
  local stamp="${dst}/.data-pack.sha256"
  mkdir -p "${MLF_LOCAL_ROOT}/data"
  if [ "${FORCE}" -eq 0 ] && [ -d "${dst}" ]; then
    if [ "${CHECK_HASH}" -eq 0 ] || stamp_matches "${stamp}" "${sha_file}" || { [ -d "${src_dir}" ] && [ ! -f "${sha_file}" ]; }; then
      echo "${name}_DATA=${dst} (cached)"
      return
    fi
  fi
  rm -rf "${dst}"
  if [ -f "${pack}" ]; then
    echo "Materializing data pack ${name} -> ${dst}"
    tar -xzf "${pack}" -C "${MLF_LOCAL_ROOT}/data"
    write_stamp "${stamp}" "${sha_file}"
  elif [ -d "${src_dir}" ]; then
    echo "Copying data directory ${name} -> ${dst}"
    copy_tree_excluding_parts "${src_dir}" "${dst}"
  else
    echo "Missing data source for ${name}: ${src_dir} or ${pack}" >&2
    exit 1
  fi
  echo "${name}_DATA=${dst}"
}

mirror_webshop_runtime_data() {
  if [ ! -d "${MLF_LOCAL_ROOT}/data/webshop" ] || [ ! -d "${MLF_LOCAL_ROOT}/code/WebShop" ]; then
    return
  fi
  if [ -d "${MLF_LOCAL_ROOT}/data/webshop/data" ]; then
    copy_tree_excluding_parts "${MLF_LOCAL_ROOT}/data/webshop/data" "${MLF_LOCAL_ROOT}/code/WebShop/data"
  fi
  if [ -d "${MLF_LOCAL_ROOT}/data/webshop/search_engine/indexes_1k" ]; then
    mkdir -p "${MLF_LOCAL_ROOT}/code/WebShop/search_engine"
    rm -rf "${MLF_LOCAL_ROOT}/code/WebShop/search_engine/indexes_1k"
    cp -a "${MLF_LOCAL_ROOT}/data/webshop/search_engine/indexes_1k" "${MLF_LOCAL_ROOT}/code/WebShop/search_engine/indexes_1k"
  fi
  if [ -d "${MLF_LOCAL_ROOT}/data/webshop/search_engine/indexes_100k" ]; then
    mkdir -p "${MLF_LOCAL_ROOT}/code/WebShop/search_engine"
    rm -rf "${MLF_LOCAL_ROOT}/code/WebShop/search_engine/indexes_100k"
    cp -a "${MLF_LOCAL_ROOT}/data/webshop/search_engine/indexes_100k" "${MLF_LOCAL_ROOT}/code/WebShop/search_engine/indexes_100k"
  fi
}

if [ "${SKIP_ENVS}" -eq 0 ]; then
  for env_name in slime alfworld webshop; do
    if contains_item "${ENVS}" "${env_name}"; then
      materialize_pack "${env_name}"
    fi
  done
fi

if [ "${SKIP_SOURCES}" -eq 0 ]; then
  if contains_item "${SOURCES}" "webshop"; then
    copy_dir_once "${MLF_NAS_ROOT}/code/WebShop" "${MLF_LOCAL_ROOT}/code/WebShop"
  fi
fi

if [ "${SKIP_MODELS}" -eq 0 ]; then
  if contains_item "${MODELS}" "qwen3-8b"; then
    copy_dir_once "${MLF_NAS_ROOT}/models/Qwen3-8B" "${MLF_LOCAL_ROOT}/models/Qwen3-8B"
    copy_dir_once "${MLF_NAS_ROOT}/models/Qwen3-8B_torch_dist" "${MLF_LOCAL_ROOT}/models/Qwen3-8B_torch_dist"
  fi
fi

if [ "${SKIP_DATA}" -eq 0 ]; then
  for data_name in alfworld webshop; do
    if contains_item "${DATASETS}" "${data_name}"; then
      materialize_data "${data_name}"
    fi
  done
  if contains_item "${DATASETS}" "webshop"; then
    mirror_webshop_runtime_data
  fi
fi

echo "MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS}"
echo "MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT}"
echo "QWEN3_8B=${MLF_LOCAL_ROOT}/models/Qwen3-8B"
echo "QWEN3_8B_TORCH_DIST=${MLF_LOCAL_ROOT}/models/Qwen3-8B_torch_dist"
echo "ALFWORLD_DATA=${MLF_LOCAL_ROOT}/data/alfworld"
echo "WEBSHOP_DATA=${MLF_LOCAL_ROOT}/data/webshop"
echo "WEBSHOP_LIB=${MLF_LOCAL_ROOT}/code/WebShop"
