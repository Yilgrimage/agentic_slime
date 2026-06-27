#!/usr/bin/env bash

set_slime_runtime_defaults() {
  LOCAL_ENVS_DIR=${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}
  SLIME_RUNTIME=${SLIME_RUNTIME:-auto}
  SLIME_PACK_NAME=${SLIME_PACK_NAME:-slime-official}
  SLIME_PACK_PATH=${SLIME_PACK_PATH:-${LOCAL_ENVS_DIR}/${SLIME_PACK_NAME}}
  SLIME_IMAGE_ENV=${SLIME_IMAGE_ENV:-/usr}
  SLIME_IMAGE_PYTHON=${SLIME_IMAGE_PYTHON:-${SLIME_IMAGE_ENV}/bin/python}
  export SLIME_RUNTIME SLIME_PACK_NAME SLIME_PACK_PATH SLIME_IMAGE_ENV SLIME_IMAGE_PYTHON
}

set_megatron_defaults() {
  MEGATRON_IMAGE_PATH=${MEGATRON_IMAGE_PATH:-/root/Megatron-LM}
  export MEGATRON_IMAGE_PATH
}

resolve_slime_runtime() {
  set_slime_runtime_defaults

  if [ -n "${SLIME_ENV:-}" ]; then
    SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_ENV}/bin/python}
  else
    case "${SLIME_RUNTIME}" in
      image)
        if [ ! -x "${SLIME_IMAGE_PYTHON}" ]; then
          echo "SLIME_RUNTIME=image but missing image python: ${SLIME_IMAGE_PYTHON}" >&2
          return 1
        fi
        SLIME_ENV="${SLIME_IMAGE_ENV}"
        SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_IMAGE_PYTHON}}
        ;;
      pack|conda_pack|conda-pack)
        if [ ! -x "${SLIME_PACK_PATH}/bin/python" ]; then
          echo "SLIME_RUNTIME=pack but missing packed slime env: ${SLIME_PACK_PATH}/bin/python" >&2
          return 1
        fi
        SLIME_ENV="${SLIME_PACK_PATH}"
        SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_ENV}/bin/python}
        ;;
      auto)
        if [ -x "${SLIME_PACK_PATH}/bin/python" ]; then
          SLIME_ENV="${SLIME_PACK_PATH}"
          SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_ENV}/bin/python}
        elif [ "${SLIME_PACK_NAME}" != "slime" ] && [ -x "${LOCAL_ENVS_DIR}/slime/bin/python" ]; then
          SLIME_ENV="${LOCAL_ENVS_DIR}/slime"
          SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_ENV}/bin/python}
        elif [ -x "${SLIME_IMAGE_PYTHON}" ]; then
          SLIME_ENV="${SLIME_IMAGE_ENV}"
          SLIME_PYTHON=${SLIME_PYTHON:-${SLIME_IMAGE_PYTHON}}
        else
          cat >&2 <<EOF
Unable to resolve Slime runtime.
Set one of:
  SLIME_RUNTIME=image with SLIME_IMAGE_PYTHON=${SLIME_IMAGE_PYTHON}
  SLIME_RUNTIME=pack with SLIME_PACK_PATH=${SLIME_PACK_PATH}
  SLIME_ENV=/path/to/slime/env
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

  if [ ! -x "${SLIME_PYTHON}" ]; then
    echo "Missing slime python: ${SLIME_PYTHON}" >&2
    return 1
  fi

  export SLIME_RUNTIME SLIME_ENV SLIME_PYTHON
  export SLIME_PACK_NAME SLIME_PACK_PATH SLIME_IMAGE_ENV SLIME_IMAGE_PYTHON
}

resolve_megatron_path() {
  set_megatron_defaults
  if [ -z "${MEGATRON_PATH:-}" ]; then
    if [ -d "${MEGATRON_IMAGE_PATH}" ]; then
      MEGATRON_PATH="${MEGATRON_IMAGE_PATH}"
    else
      cat >&2 <<EOF
Missing MEGATRON_PATH.
Use an image that provides ${MEGATRON_IMAGE_PATH}, or set MEGATRON_PATH explicitly.
EOF
      return 1
    fi
  fi
  if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Missing Megatron-LM checkout: ${MEGATRON_PATH}" >&2
    return 1
  fi
  export MEGATRON_PATH MEGATRON_IMAGE_PATH
}
