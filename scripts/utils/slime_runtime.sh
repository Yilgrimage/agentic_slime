#!/usr/bin/env bash

set_slime_runtime_defaults() {
  LOCAL_ENVS_DIR=${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}
  SLIME_RUNTIME=${SLIME_RUNTIME:-auto}
  SLIME_PACK_NAME=${SLIME_PACK_NAME:-slime}
  SLIME_PACK_PATH=${SLIME_PACK_PATH:-${LOCAL_ENVS_DIR}/${SLIME_PACK_NAME}}
  SLIME_IMAGE_ENV=${SLIME_IMAGE_ENV:-/usr}
  SLIME_IMAGE_PYTHON=${SLIME_IMAGE_PYTHON:-${SLIME_IMAGE_ENV}/bin/python}
  MEGATRON_IMAGE_PATH=${MEGATRON_IMAGE_PATH:-/root/Megatron-LM}
  export SLIME_RUNTIME SLIME_PACK_NAME SLIME_PACK_PATH SLIME_IMAGE_ENV SLIME_IMAGE_PYTHON
  export MEGATRON_IMAGE_PATH
}

set_wandb_runtime_defaults() {
  LOCAL_ENVS_DIR=${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}
  WANDB_RUNTIME=${WANDB_RUNTIME:-pack}
  WANDB_PACK_NAME=${WANDB_PACK_NAME:-wandb}
  WANDB_PACK_PATH=${WANDB_PACK_PATH:-${LOCAL_ENVS_DIR}/${WANDB_PACK_NAME}}
  export WANDB_RUNTIME WANDB_PACK_NAME WANDB_PACK_PATH
}

resolve_slime_cuda_home() {
  if [ -z "${SLIME_CUDA_HOME:-}" ]; then
    if [ -n "${CUDA_HOME:-}" ]; then
      SLIME_CUDA_HOME="${CUDA_HOME}"
    elif [ -d /usr/local/cuda ]; then
      SLIME_CUDA_HOME=/usr/local/cuda
    else
      SLIME_CUDA_HOME="${SLIME_ENV}"
    fi
  fi
  export SLIME_CUDA_HOME
}

build_slime_library_path() {
  local tail=${1:-}
  local path="${SLIME_CUDA_HOME}/lib:${SLIME_CUDA_HOME}/lib64"
  local cudnn_dir
  local python_tag

  # Conda-pack Slime runtimes can contain pip-provided CUDNN libraries under
  # site-packages and older CUDNN libraries under ${SLIME_ENV}/lib. Keep each
  # CUDNN family internally consistent by resolving the pip CUDNN directory
  # before the generic env library directory when it exists.
  python_tag="$("${SLIME_PYTHON}" - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  cudnn_dir="${SLIME_ENV}/lib/${python_tag}/site-packages/nvidia/cudnn/lib"
  if [ -d "${cudnn_dir}" ]; then
    path="${path}:${cudnn_dir}"
  fi

  path="${path}:${SLIME_ENV}/lib:${SLIME_ENV}/lib64"
  if [ -n "${tail}" ]; then
    path="${path}:${tail}"
  fi
  printf '%s\n' "${path}"
}

slime_python_imports_slime() {
  local python=$1
  "${python}" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("slime") else 1)
PY
}

python_major_minor() {
  local python=$1
  "${python}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

python_site_packages() {
  local python=$1
  "${python}" - <<'PY'
import site
import sysconfig

paths = []
for path in site.getsitepackages():
    if path:
        paths.append(path)
purelib = sysconfig.get_paths().get("purelib")
if purelib:
    paths.append(purelib)

seen = set()
ordered = []
for path in paths:
    if path not in seen:
        seen.add(path)
        ordered.append(path)
print(":".join(ordered))
PY
}

python_imports_wandb() {
  local python=$1
  "${python}" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("wandb") else 1)
PY
}

slime_runtime_candidate_available() {
  local python=$1
  local backend_path=$2
  [ -x "${python}" ] || return 1
  [ -d "${backend_path}" ] || return 1
  slime_python_imports_slime "${python}"
}

activate_slime_runtime() {
  SLIME_ENV=$1
  SLIME_PYTHON=$2
  MEGATRON_PATH=$3
  export SLIME_RUNTIME SLIME_ENV SLIME_PYTHON
  export SLIME_PACK_NAME SLIME_PACK_PATH SLIME_IMAGE_ENV SLIME_IMAGE_PYTHON
  export MEGATRON_PATH MEGATRON_IMAGE_PATH
}

activate_wandb_runtime() {
  WANDB_RUNTIME_RESOLVED=$1
  WANDB_ENV=$2
  WANDB_PYTHON=$3
  WANDB_PYTHONPATH=$4
  export WANDB_RUNTIME WANDB_RUNTIME_RESOLVED WANDB_ENV WANDB_PYTHON WANDB_PYTHONPATH
}

require_slime_runtime_candidate() {
  local label=$1
  local python=$2
  local backend_path=$3
  if [ ! -x "${python}" ]; then
    echo "SLIME_RUNTIME=${label} but missing python: ${python}" >&2
    return 1
  fi
  if ! slime_python_imports_slime "${python}"; then
    echo "SLIME_RUNTIME=${label} but python cannot import slime: ${python}" >&2
    return 1
  fi
  if [ ! -d "${backend_path}" ]; then
    echo "SLIME_RUNTIME=${label} but missing bundled Megatron-LM source: ${backend_path}" >&2
    return 1
  fi
}

require_wandb_runtime_candidate() {
  local label=$1
  local python=$2
  if [ ! -x "${python}" ]; then
    echo "WANDB_RUNTIME=${label} but missing python: ${python}" >&2
    return 1
  fi
  if ! python_imports_wandb "${python}"; then
    echo "WANDB_RUNTIME=${label} but python cannot import wandb: ${python}" >&2
    return 1
  fi
  if [ "$(python_major_minor "${python}")" != "$(python_major_minor "${SLIME_PYTHON}")" ]; then
    echo "WANDB_RUNTIME=${label} python version $(python_major_minor "${python}") does not match Slime python $(python_major_minor "${SLIME_PYTHON}")" >&2
    return 1
  fi
}

resolve_slime_runtime() {
  set_slime_runtime_defaults

  if [ -n "${SLIME_ENV:-}" ]; then
    SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_ENV}/bin/python}
    MEGATRON_PATH=${MEGATRON_PATH:-${SLIME_ENV}/src/Megatron-LM}
    require_slime_runtime_candidate "explicit" "${SLIME_PYTHON}" "${MEGATRON_PATH}"
    activate_slime_runtime "${SLIME_ENV}" "${SLIME_PYTHON}" "${MEGATRON_PATH}"
  else
    case "${SLIME_RUNTIME}" in
      image)
        MEGATRON_PATH=${MEGATRON_PATH:-${MEGATRON_IMAGE_PATH}}
        require_slime_runtime_candidate "image" "${SLIME_IMAGE_PYTHON}" "${MEGATRON_PATH}"
        activate_slime_runtime "${SLIME_IMAGE_ENV}" "${SLIME_IMAGE_PYTHON}" "${MEGATRON_PATH}"
        ;;
      pack|conda_pack|conda-pack)
        MEGATRON_PATH=${MEGATRON_PATH:-${SLIME_PACK_PATH}/src/Megatron-LM}
        require_slime_runtime_candidate "pack" "${SLIME_PACK_PATH}/bin/python" "${MEGATRON_PATH}"
        activate_slime_runtime "${SLIME_PACK_PATH}" "${SLIME_PACK_PATH}/bin/python" "${MEGATRON_PATH}"
        ;;
      auto)
        if slime_runtime_candidate_available "${SLIME_IMAGE_PYTHON}" "${MEGATRON_IMAGE_PATH}"; then
          activate_slime_runtime "${SLIME_IMAGE_ENV}" "${SLIME_IMAGE_PYTHON}" "${MEGATRON_IMAGE_PATH}"
        elif slime_runtime_candidate_available "${SLIME_PACK_PATH}/bin/python" "${SLIME_PACK_PATH}/src/Megatron-LM"; then
          activate_slime_runtime "${SLIME_PACK_PATH}" "${SLIME_PACK_PATH}/bin/python" "${SLIME_PACK_PATH}/src/Megatron-LM"
        else
          cat >&2 <<EOF
Unable to resolve Slime runtime.
Set one of:
  SLIME_RUNTIME=image with Slime and Megatron-LM bundled in the image
  SLIME_RUNTIME=pack with Slime and Megatron-LM bundled in ${SLIME_PACK_PATH}
  SLIME_ENV=/path/to/slime/env with src/Megatron-LM inside it
EOF
          return 1
        fi
        ;;
      *)
        echo "Unsupported SLIME_RUNTIME=${SLIME_RUNTIME}; expected auto, image, or pack" >&2
        return 1
        ;;
    esac
  fi

  if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Missing Megatron-LM checkout: ${MEGATRON_PATH}" >&2
    return 1
  fi
}

resolve_wandb_runtime() {
  set_wandb_runtime_defaults
  if [ -z "${SLIME_PYTHON:-}" ]; then
    resolve_slime_runtime || return 1
  fi

  unset WANDB_RUNTIME_RESOLVED WANDB_ENV WANDB_PYTHON WANDB_PYTHONPATH

  case "${WANDB_RUNTIME}" in
    pack|conda_pack|conda-pack)
      WANDB_PYTHON="${WANDB_PACK_PATH}/bin/python"
      require_wandb_runtime_candidate "pack" "${WANDB_PYTHON}" || return 1
      activate_wandb_runtime "pack" "${WANDB_PACK_PATH}" "${WANDB_PYTHON}" "$(python_site_packages "${WANDB_PYTHON}")"
      ;;
    *)
      echo "Unsupported WANDB_RUNTIME=${WANDB_RUNTIME}; expected pack. Build and materialize ${WANDB_PACK_NAME}.tar.gz, or disable W&B." >&2
      return 1
      ;;
  esac
}

resolve_megatron_path() {
  if [ -z "${MEGATRON_PATH:-}" ] || [ -z "${SLIME_ENV:-}" ] || [ -z "${SLIME_PYTHON:-}" ]; then
    resolve_slime_runtime
    return
  fi
  if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Missing Megatron-LM checkout: ${MEGATRON_PATH}" >&2
    return 1
  fi
  export MEGATRON_PATH
}
