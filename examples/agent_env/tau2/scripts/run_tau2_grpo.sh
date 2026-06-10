#!/bin/bash
set -euo pipefail

export ENV_NAME=tau2
export LITELLM_LOCAL_MODEL_COST_MAP=${LITELLM_LOCAL_MODEL_COST_MAP:-True}
export CUSTOM_GENERATE_FUNCTION_PATH=examples.agent_env.tau2.rollout.generate
export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-${MLF_LOCAL_ROOT:-/tmp/mlf-runtime}/configs/tau2_launch.yaml}
export ENV_SERVER_URL_VAR=TAU2_ENV_SERVER_URL
export PROMPT_DATA_PYTHON=${PROMPT_DATA_PYTHON:-${TAU2_ENV:-${MLF_LOCAL_ENVS:-/tmp/mlf-envs}/tau2}/bin/python}
export PROMPT_USE_SERVER_NUM_TASKS=0

eval "$("${PROMPT_DATA_PYTHON}" - <<'PY'
import os
import shlex
from pathlib import Path

import yaml

config_path = Path(os.environ["CUSTOM_CONFIG_PATH"])
cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
prompt = (cfg or {}).get("agent_prompt_data") or {}
local_root = os.environ.get("MLF_LOCAL_ROOT", "/tmp/mlf-runtime")
repo_dir = os.environ.get("REPO_DIR", "/mnt/bn/jixf-nas-lq/mlf/code/slime")

def emit(name, value):
    if value is None or value == "":
        return
    if isinstance(value, bool):
        value = "1" if value else "0"
    elif isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    else:
        value = str(value)
    print(f"export {name}={shlex.quote(value)}")

source = prompt.get("source") or "official"
data_dir = prompt.get("data_dir") or f"{local_root}/data/tau2/data"
domains = prompt.get("domains") or ["retail"]
if isinstance(domains, str):
    domains_csv = domains
else:
    domains_csv = ",".join(str(item) for item in domains)
num_tasks = prompt.get("num_tasks", "all")
if num_tasks is None:
    num_tasks = "all"
seed = prompt.get("seed", os.environ.get("SEED", "42"))
output_dir = prompt.get("output_dir") or f"{local_root}/data/tau2/{source}_prompt"

args = [
    "--source", source,
    "--data-dir", data_dir,
    "--domains", domains_csv,
    "--split", str(prompt.get("split") or "train"),
    "--seed", str(seed),
]
if source == "areal_synthetic":
    args.extend(["--areal-root", str(prompt.get("areal_root") or f"{local_root}/data/tau2/areal_synthetic")])
    if prompt.get("areal_input"):
        args.extend(["--areal-input", str(prompt["areal_input"])])
    if prompt.get("task_file_dir"):
        args.extend(["--task-file-dir", str(prompt["task_file_dir"])])
if prompt.get("domain_weights"):
    args.extend(["--domain-weights", str(prompt["domain_weights"])])
if prompt.get("task_sets"):
    args.extend(["--task-sets", str(prompt["task_sets"])])

emit("TAU2_DATA_SOURCE", source)
emit("TAU2_DATA_DIR", data_dir)
emit("PROMPT_DATA_SCRIPT", prompt.get("script") or f"{repo_dir}/examples/agent_env/tau2/prompt_data.py")
emit("PROMPT_DATA_PYTHON", prompt.get("python") or os.environ.get("PROMPT_DATA_PYTHON"))
emit("PROMPT_NUM_TASKS", num_tasks)
emit("DATA_DIR", output_dir)
emit("PROMPT_DATA_EXTRA_ARGS", " ".join(shlex.quote(item) for item in args))
PY
)"
export DYNAMIC_SAMPLING_FILTER_PATH=${DYNAMIC_SAMPLING_FILTER_PATH:-}

exec bash "${REPO_DIR:-/mnt/bn/jixf-nas-lq/mlf/code/slime}/examples/agent_env/scripts/run_qwen3_8b_agent_env_grpo.sh"
