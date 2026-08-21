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
ARCHIVE_DIR=${ARCHIVE_DIR:-${ROOT_DIR}/envs/archives}
PACK_DIR=${PACK_DIR:-${ROOT_DIR}/packs}
CURRENT_FILE=${CURRENT_FILE:-${ARCHIVE_DIR}/slime-official.current}
SLIME_ENV_PREFIX=${SLIME_ENV_PREFIX:-}
SLIME_BASE_REVISION=${SLIME_BASE_REVISION:-}
SLIME_REFRESH_BASE=${SLIME_REFRESH_BASE:-0}
SLIME_SOURCE=${SLIME_SOURCE:-${REPO_DIR}}
SGLANG_SOURCE=${SGLANG_SOURCE:-${ROOT_DIR}/code/sglang}
MEGATRON_SOURCE=${MEGATRON_SOURCE:-${ROOT_DIR}/code/Megatron-LM}

mkdir -p "${PACK_DIR}" "${ARCHIVE_DIR}"

create_base_archive() {
  local prefix=$1
  local revision=$2
  local archive="${ARCHIVE_DIR}/${revision}.tar.gz"
  local tmp_archive="${archive}.tmp.$$"
  local conda_pack="${prefix}/bin/conda-pack"
  [ -x "${prefix}/bin/python" ] || {
    echo "Missing Slime build env python: ${prefix}/bin/python" >&2
    return 1
  }
  if [ ! -x "${conda_pack}" ]; then
    "${prefix}/bin/python" -m pip install conda-pack
  fi
  "${conda_pack}" -p "${prefix}" -o "${tmp_archive}" --force --ignore-editable-packages
  mv -f "${tmp_archive}" "${archive}"
  (cd "${ARCHIVE_DIR}" && sha256sum "${revision}.tar.gz" > "${revision}.tar.gz.sha256")
  printf '%s\n' "${revision}" > "${CURRENT_FILE}"
}

if [ "${SLIME_REFRESH_BASE}" = "1" ]; then
  if [ -z "${SLIME_ENV_PREFIX}" ]; then
    echo "SLIME_REFRESH_BASE=1 requires SLIME_ENV_PREFIX." >&2
    exit 1
  fi
  if [ -z "${SLIME_BASE_REVISION}" ]; then
    SLIME_BASE_REVISION="slime-$(git -C "${SLIME_SOURCE}" rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d)"
  fi
  create_base_archive "${SLIME_ENV_PREFIX}" "${SLIME_BASE_REVISION}"
elif [ ! -f "${CURRENT_FILE}" ]; then
  if [ -z "${SLIME_ENV_PREFIX}" ]; then
    echo "Missing ${CURRENT_FILE} and SLIME_ENV_PREFIX is unset." >&2
    echo "Build the Slime conda env with build_conda.sh, then rerun with SLIME_ENV_PREFIX=/path/to/env." >&2
    exit 1
  fi
  if [ -z "${SLIME_BASE_REVISION}" ]; then
    SLIME_BASE_REVISION="slime-$(git -C "${SLIME_SOURCE}" rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d)"
  fi
  create_base_archive "${SLIME_ENV_PREFIX}" "${SLIME_BASE_REVISION}"
fi

REVISION=$(cat "${CURRENT_FILE}")
SRC="${ARCHIVE_DIR}/${REVISION}.tar.gz"
if [ ! -f "${SRC}" ]; then
  if [ -z "${SLIME_ENV_PREFIX}" ]; then
    echo "Missing slime archive: ${SRC}" >&2
    echo "Set SLIME_ENV_PREFIX to recreate it without relying on an old archive." >&2
    exit 1
  fi
  create_base_archive "${SLIME_ENV_PREFIX}" "${REVISION}"
fi

for required in \
  "${SLIME_SOURCE}/slime" \
  "${SGLANG_SOURCE}/python/sglang" \
  "${MEGATRON_SOURCE}/megatron"; do
  [ -d "${required}" ] || { echo "Missing required runtime source: ${required}" >&2; exit 1; }
done

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/slime-pack.XXXXXX")
TMP_PACK="${PACK_DIR}/.slime.tar.gz.tmp.$$"
cleanup() {
  rm -rf "${WORK_DIR}"
  rm -f "${TMP_PACK}"
}
trap cleanup EXIT

ENV_DIR="${WORK_DIR}/env"
mkdir -p "${ENV_DIR}/src"
tar -xzf "${SRC}" -C "${ENV_DIR}"

copy_source() {
  local src=$1
  local dst=$2
  rm -rf "${dst}"
  mkdir -p "${dst}"
  tar -C "${src}" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.mypy_cache' \
    --exclude='.pytest_cache' \
    --exclude='.eggs' \
    --exclude='*.egg-info' \
    --exclude='target' \
    --exclude='build' \
    --exclude='dist' \
    -cf - . | tar -C "${dst}" -xf -
}

copy_source "${SLIME_SOURCE}" "${ENV_DIR}/src/slime"
copy_source "${SGLANG_SOURCE}" "${ENV_DIR}/src/sglang"
copy_source "${MEGATRON_SOURCE}" "${ENV_DIR}/src/Megatron-LM"

SITE_PACKAGES=$(find "${ENV_DIR}/lib" -maxdepth 3 -type d -name site-packages -print -quit)
if [ -z "${SITE_PACKAGES}" ]; then
  echo "Cannot locate site-packages in staged Slime env" >&2
  exit 1
fi
cat > "${SITE_PACKAGES}/agentic_slime_runtime_sources.pth" <<'EOF'
import os,sys; sys.path.append(os.path.join(sys.prefix,'src','sglang','python')); sys.path.append(os.path.join(sys.prefix,'src','slime')); sys.path.append(os.path.join(sys.prefix,'src','Megatron-LM'))
EOF

SLIME_REVISION=$(git -C "${SLIME_SOURCE}" rev-parse HEAD 2>/dev/null || echo unknown)
SGLANG_REVISION=$(git -C "${SGLANG_SOURCE}" rev-parse HEAD 2>/dev/null || echo unknown)
MEGATRON_REVISION=$(git -C "${MEGATRON_SOURCE}" rev-parse HEAD 2>/dev/null || echo unknown)
SLIME_DIFF_SHA=$(git -C "${SLIME_SOURCE}" diff --binary HEAD 2>/dev/null | sha256sum | awk '{print $1}')
SGLANG_DIFF_SHA=$(git -C "${SGLANG_SOURCE}" diff --binary HEAD 2>/dev/null | sha256sum | awk '{print $1}')
MEGATRON_DIFF_SHA=$(git -C "${MEGATRON_SOURCE}" diff --binary HEAD 2>/dev/null | sha256sum | awk '{print $1}')
PACK_REVISION="${REVISION}+slime-${SLIME_REVISION:0:12}+sglang-${SGLANG_REVISION:0:12}+megatron-${MEGATRON_REVISION:0:12}"

tar -C "${ENV_DIR}" -czf "${TMP_PACK}" .
mv -f "${TMP_PACK}" "${PACK_DIR}/slime.tar.gz"

if [ -f "${SRC}.sha256" ]; then
  cp -f "${SRC}.sha256" "${PACK_DIR}/slime.tar.gz.base.sha256"
fi
(cd "${PACK_DIR}" && sha256sum slime.tar.gz > slime.tar.gz.sha256)
printf "%s\n" "${PACK_REVISION}" > "${PACK_DIR}/slime.revision"
{
  printf "manifest_version=1\n"
  printf "runtime=slime\n"
  printf "built_at_utc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "pip_index_url=%s\n" "${PIP_INDEX_URL:-unknown}"
  printf "base_revision=%s\n" "${REVISION}"
  printf "slime_source=%s\n" "${SLIME_SOURCE}"
  printf "slime_commit=%s\n" "${SLIME_REVISION}"
  printf "slime_diff_sha256=%s\n" "${SLIME_DIFF_SHA}"
  printf "sglang_source=%s\n" "${SGLANG_SOURCE}"
  printf "sglang_commit=%s\n" "${SGLANG_REVISION}"
  printf "sglang_diff_sha256=%s\n" "${SGLANG_DIFF_SHA}"
  printf "megatron_source=%s\n" "${MEGATRON_SOURCE}"
  printf "megatron_commit=%s\n" "${MEGATRON_REVISION}"
  printf "megatron_diff_sha256=%s\n" "${MEGATRON_DIFF_SHA}"
  printf "pack_revision=%s\n" "${PACK_REVISION}"
  if [ -n "${SLIME_ENV_PREFIX}" ] && [ -x "${SLIME_ENV_PREFIX}/bin/python" ]; then
    printf "\n[pip_freeze]\n"
    "${SLIME_ENV_PREFIX}/bin/python" -m pip freeze
  fi
} > "${PACK_DIR}/slime.manifest.txt"
cp -f "${PACK_DIR}/slime.manifest.txt" "${PACK_DIR}/slime.pack_note"
chmod a+r \
  "${PACK_DIR}/slime.tar.gz" \
  "${PACK_DIR}/slime.tar.gz.sha256" \
  "${PACK_DIR}/slime.revision" \
  "${PACK_DIR}/slime.manifest.txt" \
  "${PACK_DIR}/slime.pack_note"

echo "SLIME_PACK=${PACK_DIR}/slime.tar.gz"
echo "SLIME_REVISION=${PACK_REVISION}"
echo "SLIME_MANIFEST=${PACK_DIR}/slime.manifest.txt"
