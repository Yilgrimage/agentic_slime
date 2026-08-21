#!/usr/bin/env bash

# Shared helpers for reproducible runtime-pack builders. This file is sourced;
# keep it free of top-level side effects.

runtime_pack_require_command() {
  local command_name=$1
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Missing required command: ${command_name}" >&2
    return 1
  }
}

runtime_pack_configure_pip() {
  export PIP_CONFIG_FILE="${PACK_PIP_CONFIG_FILE:-/dev/null}"
  export PIP_INDEX_URL="${PACK_PIP_INDEX_URL:-https://pypi.org/simple}"
  unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
}

runtime_pack_git_revision() {
  local directory=$1
  git -C "${directory}" rev-parse HEAD 2>/dev/null || printf '%s\n' unknown
}

runtime_pack_require_checkout() {
  local url=$1
  local revision=$2
  local directory=$3

  runtime_pack_require_command git
  mkdir -p "$(dirname "${directory}")"
  if [ ! -d "${directory}/.git" ]; then
    git clone --filter=blob:none "${url}" "${directory}"
  fi
  if [ -n "$(git -C "${directory}" status --short)" ]; then
    echo "Refusing to change dirty source checkout: ${directory}" >&2
    return 1
  fi
  if ! git -C "${directory}" cat-file -e "${revision}^{commit}" 2>/dev/null; then
    git -C "${directory}" fetch --depth 1 origin "${revision}"
  fi
  git -C "${directory}" checkout --detach "${revision}"
  if [ "$(git -C "${directory}" rev-parse HEAD)" != "${revision}" ]; then
    echo "Source checkout did not resolve requested revision ${revision}: ${directory}" >&2
    return 1
  fi
}

runtime_pack_prepare_prefix() {
  local prefix=$1
  local recreate=${2:-0}
  if [ "${recreate}" = "1" ] && [ -e "${prefix}" ]; then
    rm -rf "${prefix}"
  fi
  if [ -e "${prefix}" ] && [ ! -x "${prefix}/bin/python" ]; then
    echo "Existing runtime prefix is incomplete: ${prefix}" >&2
    echo "Remove it or rebuild with the runtime-specific RECREATE=1 variable." >&2
    return 1
  fi
}

runtime_pack_publish() {
  local name=$1
  local prefix=$2
  local revision=$3
  shift 3
  local pack_dir=${PACK_DIR:?PACK_DIR must be set}
  local pack="${pack_dir}/${name}.tar.gz"
  local tmp_pack="${pack_dir}/.${name}.tar.gz.tmp.$$"
  local manifest="${pack_dir}/${name}.manifest.txt"
  local tmp_manifest="${pack_dir}/.${name}.manifest.tmp.$$"
  local conda_pack="${prefix}/bin/conda-pack"
  local source_entry

  [ -x "${prefix}/bin/python" ] || {
    echo "Missing runtime python: ${prefix}/bin/python" >&2
    return 1
  }
  [ -x "${conda_pack}" ] || {
    echo "Missing conda-pack executable: ${conda_pack}" >&2
    return 1
  }
  mkdir -p "${pack_dir}"
  rm -f "${tmp_pack}" "${tmp_manifest}"

  "${conda_pack}" -p "${prefix}" -o "${tmp_pack}" --force
  mv -f "${tmp_pack}" "${pack}"
  (
    cd "${pack_dir}"
    sha256sum "${name}.tar.gz" > "${name}.tar.gz.sha256"
  )
  printf '%s\n' "${revision}" > "${pack_dir}/${name}.revision"

  {
    printf 'manifest_version=1\n'
    printf 'runtime=%s\n' "${name}"
    printf 'revision=%s\n' "${revision}"
    printf 'built_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'python=%s\n' "$("${prefix}/bin/python" -c 'import platform; print(platform.python_version())')"
    printf 'platform=%s\n' "$(uname -srm)"
    printf 'pip_index_url=%s\n' "${PIP_INDEX_URL:-unknown}"
    for source_entry in "$@"; do
      printf '%s\n' "${source_entry}"
    done
    printf '\n[pip_freeze]\n'
    "${prefix}/bin/python" -m pip freeze
    if [ -n "${MICROMAMBA:-}" ] && [ -x "${MICROMAMBA}" ]; then
      printf '\n[conda_explicit]\n'
      "${MICROMAMBA}" list -p "${prefix}" --explicit 2>/dev/null || true
    fi
  } > "${tmp_manifest}"
  mv -f "${tmp_manifest}" "${manifest}"
  chmod a+r "${pack}" "${pack}.sha256" "${pack_dir}/${name}.revision" "${manifest}"

  echo "${name^^}_PACK=${pack}"
  echo "${name^^}_REVISION=${revision}"
  echo "${name^^}_MANIFEST=${manifest}"
}
