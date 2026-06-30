#!/usr/bin/env bash

# Shared dataset preparation and validation helpers for server-ops scripts.
# This file is sourced by public entrypoints; keep side effects out of it.

ops_bool_enabled() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

ops_list_items() {
  local list=${1:-none}
  [ "${list}" != "none" ] || return 0
  printf '%s\n' "${list}" | tr ', ' '\n' | awk 'NF {print $1}'
}

copy_tree_excluding_parts() {
  local src=$1
  local dst=$2
  mkdir -p "${dst}"
  tar -C "${src}" --exclude='*.parts' --exclude='*.part' -cf - . | tar -C "${dst}" -xf -
}

dir_has_file() {
  local dir=$1
  [ -d "${dir}" ] || return 1
  [ -n "$(find "${dir}" -type f -print -quit 2>/dev/null)" ]
}

require_file() {
  local file=$1
  if [ ! -f "${file}" ]; then
    echo "Missing required file: ${file}" >&2
    return 1
  fi
}

require_dir_with_file() {
  local dir=$1
  if ! dir_has_file "${dir}"; then
    echo "Missing required non-empty directory: ${dir}" >&2
    return 1
  fi
}

require_no_stale_path() {
  local path=$1
  local stale=${2:-/tmp/mlf-runtime}
  [ -e "${path}" ] || return 0
  if grep -Rsl --exclude='._*' -- "${stale}" "${path}" >/dev/null 2>&1; then
    echo "Stale local path ${stale} found under ${path}" >&2
    return 1
  fi
}

alfworld_has_paired_game() {
  local split_dir=$1
  local game_file
  [ -d "${split_dir}" ] || return 1
  game_file=$(
    while IFS= read -r candidate; do
      if [ -f "$(dirname "${candidate}")/traj_data.json" ]; then
        printf '%s\n' "${candidate}"
        break
      fi
    done < <(find "${split_dir}" -type f -name game.tw-pddl -print 2>/dev/null)
  )
  [ -n "${game_file}" ]
}

validate_alfworld_data() {
  local data_dir=$1
  local missing=0
  local split
  for file in \
    "${data_dir}/logic/alfred.pddl" \
    "${data_dir}/logic/alfred.twl2"; do
    require_file "${file}" || missing=1
  done
  for split in train valid_seen valid_unseen; do
    if ! alfworld_has_paired_game "${data_dir}/json_2.1.1/${split}"; then
      echo "ALFWorld split ${split} has no task containing both game.tw-pddl and traj_data.json under ${data_dir}/json_2.1.1/${split}" >&2
      missing=1
    fi
  done
  [ "${missing}" -eq 0 ]
}

smoke_alfworld_data_load() {
  local data_dir=$1
  local python="${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}/alfworld/bin/python"
  [ -x "${python}" ] || {
    echo "Cannot run ALFWorld data load smoke: missing ${python}. Materialize the alfworld env first or set LOCAL_ENVS_DIR." >&2
    return 1
  }
  ALFWORLD_DATA="${data_dir}" "${python}" - "${data_dir}" <<'PY'
import os
import sys

data_dir = sys.argv[1]
alfworld_lib = os.path.join(data_dir, "pythonlibs", "alfworld_text")
if os.path.isdir(alfworld_lib) and alfworld_lib not in sys.path:
    sys.path.insert(0, alfworld_lib)

from alfworld.agents.environment import get_environment

config = {
    "dataset": {
        "data_path": f"{data_dir}/json_2.1.1/train",
        "eval_id_data_path": f"{data_dir}/json_2.1.1/valid_seen",
        "eval_ood_data_path": f"{data_dir}/json_2.1.1/valid_unseen",
        "num_train_games": -1,
        "num_eval_games": -1,
    },
    "env": {
        "type": "AlfredTWEnv",
        "domain_randomization": False,
        "task_types": [1, 2, 3, 4, 5, 6],
        "expert_type": "handcoded",
        "goal_desc_human_anns_prob": 0.0,
    },
    "general": {"training_method": "dqn"},
    "rl": {"training": {"max_nb_steps_per_episode": 1}},
    "dagger": {"training": {"max_nb_steps_per_episode": 1}},
    "logic": {
        "domain": f"{data_dir}/logic/alfred.pddl",
        "grammar": f"{data_dir}/logic/alfred.twl2",
    },
}

env_cls = get_environment("AlfredTWEnv")
for name, train_eval in {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}.items():
    wrapper = env_cls(config, train_eval=train_eval)
    game_files = list(getattr(wrapper, "game_files", None) or [])
    if not game_files:
        raise RuntimeError(f"ALFWorld split {name} loaded no game files")
print("ALFWorld data load smoke passed")
PY
}

validate_webshop_data() {
  local data_dir=$1
  local missing=0
  for file in \
    "${data_dir}/data/items_shuffle.json" \
    "${data_dir}/data/items_ins_v2.json" \
    "${data_dir}/data/items_human_ins.json"; do
    require_file "${file}" || missing=1
  done
  require_dir_with_file "${data_dir}/search_engine/indexes_100k" || missing=1
  [ "${missing}" -eq 0 ]
}

validate_tau2_data() {
  local data_dir=$1
  local missing=0
  for file in \
    "${data_dir}/data/tau2/domains/airline/tasks.json" \
    "${data_dir}/data/tau2/domains/airline/db.json" \
    "${data_dir}/data/tau2/domains/retail/tasks.json" \
    "${data_dir}/data/tau2/domains/retail/db.json" \
    "${data_dir}/data/tau2/domains/telecom/tasks.json" \
    "${data_dir}/data/tau2/domains/telecom/db.toml" \
    "${data_dir}/areal_synthetic/tau2_rl_train.jsonl"; do
    require_file "${file}" || missing=1
  done
  require_dir_with_file "${data_dir}/areal_synthetic/tau2_rl_database" || missing=1
  require_dir_with_file "${data_dir}/areal_synthetic/tasks" || missing=1
  require_file "${data_dir}/areal_synthetic_prompt/train_all.jsonl" || missing=1
  require_no_stale_path "${data_dir}/areal_synthetic/tasks" || missing=1
  require_no_stale_path "${data_dir}/areal_synthetic_prompt" || missing=1
  [ "${missing}" -eq 0 ]
}

validate_mcp_server_data() {
  local data_dir=$1
  local required_file
  require_dir_with_file "${data_dir}" || return 1
  if ! find "${data_dir}" -type f \( -name '*.jsonl' -o -name '*.json' \) -print -quit 2>/dev/null | grep -q .; then
    echo "MCP server data has no JSON/JSONL task or teacher files under ${data_dir}" >&2
    return 1
  fi
  for required_file in $(ops_list_items "${MCP_SERVER_REQUIRED_FILES:-none}"); do
    require_file "${data_dir}/${required_file}" || return 1
  done
  require_no_stale_path "${data_dir}" "${MCP_SERVER_STALE_PATH:-/tmp/mlf-runtime}" || return 1
}

validate_dataset() {
  local name=$1
  local data_dir=$2
  local validate_data=${VALIDATE_DATA:-1}
  local validate_data_load=${VALIDATE_DATA_LOAD:-0}
  ops_bool_enabled "${validate_data}" || return 0
  if [ ! -d "${data_dir}" ]; then
    echo "Missing data directory for ${name}: ${data_dir}" >&2
    return 1
  fi
  case "${name}" in
    alfworld)
      validate_alfworld_data "${data_dir}" || return 1
      ops_bool_enabled "${validate_data_load}" || return 0
      smoke_alfworld_data_load "${data_dir}" || return 1
      ;;
    webshop)
      validate_webshop_data "${data_dir}" || return 1
      ;;
    tau2)
      validate_tau2_data "${data_dir}" || return 1
      ;;
    mcp_server)
      validate_mcp_server_data "${data_dir}" || return 1
      ;;
    *)
      echo "No dataset-specific validator for ${name}; checked directory existence only."
      ;;
  esac
}

find_alfworld_downloader() {
  local candidate="${LOCAL_ENVS_DIR:-/tmp/server-ops-envs}/alfworld/bin/alfworld-download"
  if [ -x "${candidate}" ]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  command -v alfworld-download 2>/dev/null || return 1
}

with_dataset_lock() {
  local name=$1
  shift
  local lock_dir="${ROOT_DIR}/data/.locks/${name}.lock"
  mkdir -p "${ROOT_DIR}/data/.locks"
  while ! mkdir "${lock_dir}" 2>/dev/null; do
    echo "Waiting for dataset prepare lock: ${lock_dir}"
    sleep 10
  done
  trap 'rmdir "'"${lock_dir}"'" 2>/dev/null || true' RETURN
  "$@"
  trap - RETURN
  rmdir "${lock_dir}" 2>/dev/null || true
}

prepare_alfworld_data_locked() {
  local dst="${ROOT_DIR}/data/alfworld"
  local downloader
  if [ "${FORCE:-0}" -eq 0 ] && validate_alfworld_data "${dst}" >/dev/null 2>&1; then
    echo "alfworld_DATA=${dst} (prepared)"
    return 0
  fi
  downloader=$(find_alfworld_downloader) || {
    echo "Cannot prepare ALFWorld data: alfworld-download not found. Materialize the alfworld env first or provide ${dst}." >&2
    return 1
  }
  echo "Preparing ALFWorld data -> ${dst}"
  mkdir -p "${dst}"
  "${downloader}" --data-dir "${dst}"
  validate_dataset alfworld "${dst}"
  echo "alfworld_DATA=${dst}"
}

prepare_alfworld_data() {
  with_dataset_lock alfworld prepare_alfworld_data_locked
}

copy_webshop_from_source() {
  local src=$1
  local dst=$2
  local tmp="${dst}.tmp.$$"
  rm -rf "${tmp}"
  mkdir -p "${tmp}/data" "${tmp}/search_engine"
  for file in items_shuffle.json items_ins_v2.json items_human_ins.json; do
    cp -a "${src}/data/${file}" "${tmp}/data/${file}"
  done
  for index_dir in indexes_100k indexes_1k; do
    if [ -d "${src}/search_engine/${index_dir}" ]; then
      copy_tree_excluding_parts "${src}/search_engine/${index_dir}" "${tmp}/search_engine/${index_dir}"
    fi
  done
  validate_webshop_data "${tmp}"
  rm -rf "${dst}"
  mv "${tmp}" "${dst}"
}

prepare_webshop_data() {
  local dst="${ROOT_DIR}/data/webshop"
  local src="${WEBSHOP_DATA_SOURCE_DIR:-}"
  if [ "${FORCE:-0}" -eq 0 ] && validate_webshop_data "${dst}" >/dev/null 2>&1; then
    echo "webshop_DATA=${dst} (prepared)"
    return 0
  fi
  if [ -n "${src}" ]; then
    :
  elif [ -d "${ROOT_DIR}/code/WebShop/data" ] && [ -d "${ROOT_DIR}/code/WebShop/search_engine" ]; then
    src="${ROOT_DIR}/code/WebShop"
  else
    echo "Cannot prepare WebShop data: provide ${dst} or WEBSHOP_DATA_SOURCE_DIR with data/ and search_engine/." >&2
    return 1
  fi
  echo "Preparing WebShop data from ${src} -> ${dst}"
  copy_webshop_from_source "${src}" "${dst}"
  validate_dataset webshop "${dst}"
  echo "webshop_DATA=${dst}"
}

normalize_tau2_official_source() {
  local src=$1
  if [ -d "${src}/tau2/domains" ]; then
    printf '%s\n' "${src}"
    return 0
  fi
  if [ -d "${src}/data/tau2/domains" ]; then
    printf '%s\n' "${src}/data"
    return 0
  fi
  return 1
}

normalize_tau2_areal_source() {
  local src=$1
  if [ -f "${src}/tau2_rl_train.jsonl" ] && [ -d "${src}/tau2_rl_database" ]; then
    printf '%s\n' "${src}"
    return 0
  fi
  if [ -f "${src}/areal_synthetic/tau2_rl_train.jsonl" ] && [ -d "${src}/areal_synthetic/tau2_rl_database" ]; then
    printf '%s\n' "${src}/areal_synthetic"
    return 0
  fi
  return 1
}

download_tau2_areal() {
  local dst=$1
  local python=${PYTHON:-python}
  local script="${ROOT_DIR}/code/slime/examples/agent_env/tau2/scripts/download_areal_tau2_data.py"
  [ -f "${script}" ] || {
    echo "Cannot download tau2 AReaL data: missing ${script}" >&2
    return 1
  }
  "${python}" "${script}" --output-dir "${dst}"
}

generate_tau2_areal_artifacts() {
  local data_dir=$1
  local python=${PYTHON:-python3}
  local script="${ROOT_DIR}/code/slime/examples/agent_env/tau2/prompt_data.py"
  local task_ref_root='${AGENT_ENV_DATA_DIR}/areal_synthetic'
  [ -f "${script}" ] || {
    echo "Cannot generate tau2 AReaL prompt data: missing ${script}" >&2
    return 1
  }
  require_file "${data_dir}/areal_synthetic/tau2_rl_train.jsonl" || return 1
  require_dir_with_file "${data_dir}/areal_synthetic/tau2_rl_database" || return 1
  rm -rf "${data_dir}/areal_synthetic/tasks" "${data_dir}/areal_synthetic_prompt"
  mkdir -p "${data_dir}/areal_synthetic_prompt"
  "${python}" "${script}" \
    --source areal_synthetic \
    --output "${data_dir}/areal_synthetic_prompt/train_all.jsonl" \
    --data-dir "${data_dir}/data" \
    --areal-root "${data_dir}/areal_synthetic" \
    --areal-input tau2_rl_train.jsonl \
    --task-file-dir tasks \
    --task-ref-root "${task_ref_root}" \
    --domains retail,airline,telecom \
    --split train \
    --seed 42
}

prepare_tau2_data() {
  local dst="${ROOT_DIR}/data/tau2"
  local tmp="${dst}.tmp.$$"
  local official_src="${TAU2_DATA_SOURCE_DIR:-${ROOT_DIR}/code/tau2-bench/data}"
  local areal_src="${TAU2_AREAL_SOURCE_DIR:-}"
  local normalized
  if [ "${FORCE:-0}" -eq 0 ] && validate_tau2_data "${dst}" >/dev/null 2>&1; then
    echo "tau2_DATA=${dst} (prepared)"
    return 0
  fi
  if ! normalized=$(normalize_tau2_official_source "${official_src}"); then
    if [ -d "${dst}/data/tau2/domains" ]; then
      normalized="${dst}/data"
    else
      echo "Cannot prepare tau2 official data: set TAU2_DATA_SOURCE_DIR or provide ${dst}/data/tau2/domains." >&2
      return 1
    fi
  fi
  rm -rf "${tmp}"
  mkdir -p "${tmp}/data"
  copy_tree_excluding_parts "${normalized}" "${tmp}/data"
  if [ -n "${areal_src}" ]; then
    normalized=$(normalize_tau2_areal_source "${areal_src}") || {
      echo "Invalid TAU2_AREAL_SOURCE_DIR: ${areal_src}" >&2
      return 1
    }
    copy_tree_excluding_parts "${normalized}" "${tmp}/areal_synthetic"
  elif [ -d "${dst}/areal_synthetic" ]; then
    copy_tree_excluding_parts "${dst}/areal_synthetic" "${tmp}/areal_synthetic"
  elif ops_bool_enabled "${TAU2_DOWNLOAD_AREAL:-0}"; then
    download_tau2_areal "${tmp}/areal_synthetic"
  else
    rm -rf "${tmp}"
    echo "Cannot prepare tau2 AReaL synthetic data: provide TAU2_AREAL_SOURCE_DIR or set TAU2_DOWNLOAD_AREAL=1." >&2
    return 1
  fi
  generate_tau2_areal_artifacts "${tmp}"
  validate_tau2_data "${tmp}"
  rm -rf "${dst}"
  mv "${tmp}" "${dst}"
  validate_dataset tau2 "${dst}"
  echo "tau2_DATA=${dst}"
}

copy_optional_file() {
  local src=$1
  local dst=$2
  [ -n "${src}" ] || return 0
  require_file "${src}" || return 1
  mkdir -p "$(dirname "${dst}")"
  cp -a "${src}" "${dst}"
}

prepare_mcp_server_data() {
  local dst="${ROOT_DIR}/data/mcp_server"
  local tmp="${dst}.tmp.$$"
  local src="${MCP_SERVER_DATA_SOURCE_DIR:-}"
  if [ "${FORCE:-0}" -eq 0 ] && validate_mcp_server_data "${dst}" >/dev/null 2>&1; then
    echo "mcp_server_DATA=${dst} (prepared)"
    return 0
  fi

  rm -rf "${tmp}"
  mkdir -p "${tmp}"

  if [ -n "${src}" ]; then
    [ -d "${src}" ] || {
      echo "Invalid MCP_SERVER_DATA_SOURCE_DIR: ${src}" >&2
      rm -rf "${tmp}"
      return 1
    }
    copy_tree_excluding_parts "${src}" "${tmp}"
  fi

  copy_optional_file \
    "${MCP_SERVER_IPR_PRODUCT_CHECK_SOURCE_FILE:-}" \
    "${tmp}/ipr_product_check/${MCP_SERVER_IPR_PRODUCT_CHECK_TARGET_FILE:-standardized_data_fix_shuf_tail1800.jsonl}"
  copy_optional_file \
    "${MCP_SERVER_ROPD_SMOKE_SOURCE_FILE:-}" \
    "${tmp}/ropd_smoke/${MCP_SERVER_ROPD_SMOKE_TARGET_FILE:-ipr_norm_brand_teacher.jsonl}"

  if [ -d "${ROOT_DIR}/data/valleydance/ipr_product_check" ] && [ ! -d "${tmp}/ipr_product_check" ]; then
    copy_tree_excluding_parts "${ROOT_DIR}/data/valleydance/ipr_product_check" "${tmp}/ipr_product_check"
  fi
  if [ -d "${ROOT_DIR}/data/agent_env/ropd_smoke" ] && [ ! -d "${tmp}/ropd_smoke" ]; then
    copy_tree_excluding_parts "${ROOT_DIR}/data/agent_env/ropd_smoke" "${tmp}/ropd_smoke"
  fi

  validate_mcp_server_data "${tmp}"
  rm -rf "${dst}"
  mv "${tmp}" "${dst}"
  validate_dataset mcp_server "${dst}"
  echo "mcp_server_DATA=${dst}"
}

prepare_dataset() {
  local name=$1
  case "${name}" in
    alfworld) prepare_alfworld_data ;;
    webshop) prepare_webshop_data ;;
    tau2) prepare_tau2_data ;;
    mcp_server) prepare_mcp_server_data ;;
    *) echo "No prepare implementation for dataset: ${name}" >&2; return 1 ;;
  esac
}
