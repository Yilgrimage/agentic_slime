#!/bin/bash

set -ex

MLF_NAS_ROOT=${MLF_NAS_ROOT:-/mnt/bn/jixf-nas-lq/mlf}
MLF_LOCAL_ROOT=${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}
MLF_LOCAL_ENVS=${MLF_LOCAL_ENVS:-/tmp/mlf-envs}

NAS_PACK=${WEBSHOP_PACK_PATH:-${MLF_NAS_ROOT}/packs/webshop.tar.gz}
LOCAL_ENV=${LOCAL_WEBSHOP_ENV:-${MLF_LOCAL_ENVS}/webshop}
NAS_SRC=${NAS_WEBSHOP_SRC:-${MLF_NAS_ROOT}/code/WebShop}
LOCAL_SRC=${LOCAL_WEBSHOP_SRC:-${MLF_LOCAL_ROOT}/code/WebShop}
NAS_DATA=${NAS_WEBSHOP_DATA:-${MLF_NAS_ROOT}/data/webshop}
LOCAL_DATA=${LOCAL_WEBSHOP_DATA:-${MLF_LOCAL_ROOT}/data/webshop}

mkdir -p "$(dirname "${LOCAL_ENV}")" "$(dirname "${LOCAL_SRC}")" "$(dirname "${LOCAL_DATA}")"

copy_tree_excluding_parts() {
  local src=$1
  local dst=$2
  mkdir -p "${dst}"
  tar -C "${src}" --exclude='*.parts' --exclude='*.part' -cf - . | tar -C "${dst}" -xf -
}

if [ ! -x "${LOCAL_ENV}/bin/python" ]; then
  mkdir -p "${LOCAL_ENV}"
  tar -xzf "${NAS_PACK}" -C "${LOCAL_ENV}"
  "${LOCAL_ENV}/bin/conda-unpack"
fi

if [ ! -d "${LOCAL_SRC}" ] && [ -d "${NAS_SRC}" ]; then
  cp -a "${NAS_SRC}" "${LOCAL_SRC}"
fi

if [ -d "${NAS_DATA}" ]; then
  copy_tree_excluding_parts "${NAS_DATA}" "${LOCAL_DATA}"
  if [ -d "${NAS_DATA}/data" ]; then
    mkdir -p "${LOCAL_SRC}/data"
    copy_tree_excluding_parts "${NAS_DATA}/data" "${LOCAL_SRC}/data"
  fi
  if [ -d "${NAS_DATA}/search_engine/indexes_1k" ]; then
    mkdir -p "${LOCAL_SRC}/search_engine"
    rm -rf "${LOCAL_SRC}/search_engine/indexes_1k"
    cp -a "${NAS_DATA}/search_engine/indexes_1k" "${LOCAL_SRC}/search_engine/indexes_1k"
  fi
  if [ -d "${NAS_DATA}/search_engine/indexes_100k" ]; then
    mkdir -p "${LOCAL_SRC}/search_engine"
    rm -rf "${LOCAL_SRC}/search_engine/indexes_100k"
    cp -a "${NAS_DATA}/search_engine/indexes_100k" "${LOCAL_SRC}/search_engine/indexes_100k"
  fi
fi

echo "WEBSHOP_ENV=${LOCAL_ENV}"
echo "WEBSHOP_LIB=${LOCAL_SRC}"
echo "WEBSHOP_DATA=${LOCAL_DATA}"
echo "JAVA_HOME=${LOCAL_ENV}/lib/jvm"
echo "JVM_PATH=${LOCAL_ENV}/lib/jvm/lib/server/libjvm.so"
