#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
ROOT_DIR=${ROOT_DIR:-$(cd "${REPO_DIR}/../.." && pwd)}
SLIME_PYTHON=${SLIME_PYTHON:-python3}

RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT_DIR}/runs/teacher_rollouts/tau2}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_STAMP}}
PROMPT_DATA=${PROMPT_DATA:-}

AGENT_ENV_SERVER_URL=${AGENT_ENV_SERVER_URL:-${TAU2_ENV_SERVER_URL:-}}
POLICY_BASE_URL=${POLICY_BASE_URL:-}
POLICY_CHAT_PATH=${POLICY_CHAT_PATH:-/v1/chat/completions}
POLICY_MODEL=${POLICY_MODEL:-${OPENAI_MODEL:-}}
POLICY_API_KEY=${POLICY_API_KEY:-${OPENAI_API_KEY:-}}
POLICY_API_KEY_PATH=${POLICY_API_KEY_PATH:-${OPENAI_API_KEY_PATH:-}}
POLICY_PARALLEL_TOOL_CALLS=${POLICY_PARALLEL_TOOL_CALLS:-}

SPLIT=${SPLIT:-train}
SEED=${SEED:-42}
TAU2_DATA_SOURCE=${TAU2_DATA_SOURCE:-areal_synthetic}
TAU2_DOMAINS=${TAU2_DOMAINS:-retail,airline,telecom}
TAU2_TASK_SETS=${TAU2_TASK_SETS:-}
TAU2_DOMAIN_WEIGHTS=${TAU2_DOMAIN_WEIGHTS:-}
PROMPT_NUM_TASKS=${PROMPT_NUM_TASKS:-}
MAX_TASKS_PER_DOMAIN=${MAX_TASKS_PER_DOMAIN:-}

COLLECT_LIMIT=${COLLECT_LIMIT:-}
COLLECT_OFFSET=${COLLECT_OFFSET:-0}
COLLECT_CONCURRENCY=${COLLECT_CONCURRENCY:-8}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-1200}
POLICY_TIMEOUT_S=${POLICY_TIMEOUT_S:-300}
MAX_TURNS=${MAX_TURNS:-40}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
TEACHER_SUCCESS_ONLY=${TEACHER_SUCCESS_ONLY:-1}
TEACHER_MAX_CHARS=${TEACHER_MAX_CHARS:-0}

if [ -z "${AGENT_ENV_SERVER_URL}" ]; then
  echo "Set AGENT_ENV_SERVER_URL or TAU2_ENV_SERVER_URL to an existing tau2 env server/router." >&2
  exit 2
fi
if [ -z "${POLICY_BASE_URL}" ]; then
  echo "Set POLICY_BASE_URL to an OpenAI-compatible teacher policy endpoint." >&2
  exit 2
fi
if [ -z "${POLICY_MODEL}" ]; then
  echo "Set POLICY_MODEL or OPENAI_MODEL for the teacher policy endpoint." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

if [ -z "${PROMPT_DATA}" ]; then
  prompt_suffix=${PROMPT_NUM_TASKS:-all}
  PROMPT_DATA="${OUTPUT_DIR}/prompt_data/tau2_${SPLIT}_${prompt_suffix}.jsonl"
  prompt_args=(
    --output "${PROMPT_DATA}"
    --source "${TAU2_DATA_SOURCE}"
    --domains "${TAU2_DOMAINS}"
    --split "${SPLIT}"
    --seed "${SEED}"
  )
  if [ -n "${TAU2_TASK_SETS}" ]; then
    prompt_args+=(--task-sets "${TAU2_TASK_SETS}")
  fi
  if [ -n "${TAU2_DOMAIN_WEIGHTS}" ]; then
    prompt_args+=(--domain-weights "${TAU2_DOMAIN_WEIGHTS}")
  fi
  if [ -n "${PROMPT_NUM_TASKS}" ] && [ "${PROMPT_NUM_TASKS}" != "all" ]; then
    prompt_args+=(--num-tasks "${PROMPT_NUM_TASKS}")
  fi
  if [ -n "${MAX_TASKS_PER_DOMAIN}" ]; then
    prompt_args+=(--max-tasks-per-domain "${MAX_TASKS_PER_DOMAIN}")
  fi
  if [ -n "${TAU2_AREAL_ROOT:-}" ]; then
    prompt_args+=(--areal-root "${TAU2_AREAL_ROOT}")
  fi
  if [ -n "${TAU2_AREAL_INPUT:-}" ]; then
    prompt_args+=(--areal-input "${TAU2_AREAL_INPUT}")
  fi
  if [ -n "${TAU2_TASK_FILE_DIR:-}" ]; then
    prompt_args+=(--task-file-dir "${TAU2_TASK_FILE_DIR}")
  fi
  if [ -n "${TAU2_TASK_REF_ROOT:-}" ]; then
    prompt_args+=(--task-ref-root "${TAU2_TASK_REF_ROOT}")
  fi
  if [ -n "${TAU2_DATA_DIR:-}" ]; then
    prompt_args+=(--data-dir "${TAU2_DATA_DIR}")
  fi
  "${SLIME_PYTHON}" "${REPO_DIR}/examples/agent_env/tau2/prompt_data.py" "${prompt_args[@]}"
fi

collect_args=(
  --env-server-url "${AGENT_ENV_SERVER_URL}"
  --prompt-data "${PROMPT_DATA}"
  --output-jsonl "${OUTPUT_DIR}/rollouts.jsonl"
  --teacher-jsonl "${OUTPUT_DIR}/teacher.jsonl"
  --policy-base-url "${POLICY_BASE_URL}"
  --policy-chat-path "${POLICY_CHAT_PATH}"
  --policy-model "${POLICY_MODEL}"
  --policy-parallel-tool-calls "${POLICY_PARALLEL_TOOL_CALLS}"
  --split "${SPLIT}"
  --offset "${COLLECT_OFFSET}"
  --concurrency "${COLLECT_CONCURRENCY}"
  --request-timeout-s "${REQUEST_TIMEOUT_S}"
  --policy-timeout-s "${POLICY_TIMEOUT_S}"
  --max-turns "${MAX_TURNS}"
  --max-response-tokens "${MAX_RESPONSE_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --teacher-max-chars "${TEACHER_MAX_CHARS}"
  --include-trace
)
if [ -n "${POLICY_API_KEY}" ]; then
  collect_args+=(--policy-api-key "${POLICY_API_KEY}")
fi
if [ -n "${POLICY_API_KEY_PATH}" ]; then
  collect_args+=(--policy-api-key-path "${POLICY_API_KEY_PATH}")
fi
if [ -n "${COLLECT_LIMIT}" ]; then
  collect_args+=(--limit "${COLLECT_LIMIT}")
fi
if [ "${TEACHER_SUCCESS_ONLY}" = "1" ]; then
  collect_args+=(--teacher-success-only)
fi

"${SLIME_PYTHON}" "${SCRIPT_DIR}/collect_env_rollouts.py" "${collect_args[@]}"

export AGENT_ENV_SERVER_URL POLICY_BASE_URL POLICY_CHAT_PATH POLICY_MODEL SPLIT SEED
export POLICY_PARALLEL_TOOL_CALLS
export TAU2_DATA_SOURCE TAU2_DOMAINS TAU2_TASK_SETS TAU2_DOMAIN_WEIGHTS PROMPT_NUM_TASKS MAX_TASKS_PER_DOMAIN
export COLLECT_LIMIT COLLECT_OFFSET COLLECT_CONCURRENCY REQUEST_TIMEOUT_S POLICY_TIMEOUT_S
export MAX_TURNS MAX_RESPONSE_TOKENS TEMPERATURE TOP_P TEACHER_SUCCESS_ONLY TEACHER_MAX_CHARS PROMPT_DATA

"${SLIME_PYTHON}" - "${OUTPUT_DIR}/manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
keys = [
    "AGENT_ENV_SERVER_URL",
    "POLICY_BASE_URL",
    "POLICY_CHAT_PATH",
    "POLICY_MODEL",
    "POLICY_PARALLEL_TOOL_CALLS",
    "SPLIT",
    "SEED",
    "TAU2_DATA_SOURCE",
    "TAU2_DOMAINS",
    "TAU2_TASK_SETS",
    "TAU2_DOMAIN_WEIGHTS",
    "PROMPT_NUM_TASKS",
    "MAX_TASKS_PER_DOMAIN",
    "COLLECT_LIMIT",
    "COLLECT_OFFSET",
    "COLLECT_CONCURRENCY",
    "REQUEST_TIMEOUT_S",
    "POLICY_TIMEOUT_S",
    "MAX_TURNS",
    "MAX_RESPONSE_TOKENS",
    "TEMPERATURE",
    "TOP_P",
    "TEACHER_SUCCESS_ONLY",
    "TEACHER_MAX_CHARS",
]
manifest = {key: os.environ.get(key, "") for key in keys}
manifest["prompt_data"] = os.environ.get("PROMPT_DATA", "")
manifest["outputs"] = {
    "rollouts_jsonl": str(output.parent / "rollouts.jsonl"),
    "teacher_jsonl": str(output.parent / "teacher.jsonl"),
}
output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"manifest": str(output)}, ensure_ascii=False))
PY
