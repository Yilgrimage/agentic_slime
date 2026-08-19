import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = REPO_ROOT / "examples/agent_env/scripts/render_eval_config.py"


def render_config(tmp_path, env_name, template_name, splits):
    datasets = []
    command = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--template",
        str(REPO_ROOT / f"examples/agent_env/{template_name}/eval_config.yaml"),
        "--output",
        str(tmp_path / "resolved.yaml"),
        "--env-name",
        env_name,
    ]
    for split in splits:
        path = tmp_path / f"{split}.jsonl"
        path.write_text('{}\n', encoding="utf-8")
        command.extend(["--dataset", f"{split}={path}"])
        datasets.append(path)
    subprocess.run(command, check=True)
    return yaml.safe_load((tmp_path / "resolved.yaml").read_text(encoding="utf-8")), datasets


def test_generic_eval_template_registers_every_requested_split(tmp_path):
    config, paths = render_config(
        tmp_path,
        "appworld",
        "appworld",
        ["test_normal", "test_challenge"],
    )

    datasets = config["eval"]["datasets"]
    assert [dataset["name"] for dataset in datasets] == [
        "appworld-test_normal",
        "appworld-test_challenge",
    ]
    assert [dataset["path"] for dataset in datasets] == [str(path) for path in paths]
    assert [dataset["metadata_overrides"]["split"] for dataset in datasets] == [
        "test_normal",
        "test_challenge",
    ]


def test_split_specific_templates_can_render_a_subset(tmp_path):
    config, paths = render_config(tmp_path, "alfworld", "alfworld", ["valid_unseen"])

    datasets = config["eval"]["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["name"] == "alfworld-valid-unseen"
    assert datasets[0]["path"] == str(paths[0])
    assert datasets[0]["metadata_overrides"]["split"] == "valid_unseen"
