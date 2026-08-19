#!/usr/bin/env python3
"""Resolve generated native-eval datasets into one auditable YAML config."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def parse_dataset(value: str) -> tuple[str, Path]:
    split, separator, raw_path = value.partition("=")
    if not separator or not split or not raw_path:
        raise argparse.ArgumentTypeError(
            f"invalid dataset {value!r}; expected SPLIT=/absolute/path.jsonl"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"eval dataset path must be absolute: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"eval dataset does not exist: {path}")
    return split, path


def template_for_split(datasets: list[dict], split: str) -> dict:
    matching = [
        dataset
        for dataset in datasets
        if (dataset.get("metadata_overrides") or {}).get("split") == split
    ]
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        raise SystemExit(f"eval config has multiple dataset templates for split={split}")
    if len(datasets) == 1:
        return datasets[0]
    raise SystemExit(
        f"eval config has no dataset template for split={split}; "
        "add metadata_overrides.split to disambiguate its dataset entries"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("eval"), dict):
        raise SystemExit(f"eval config must contain an eval mapping: {args.template}")
    templates = config["eval"].get("datasets")
    if not isinstance(templates, list) or not templates:
        raise SystemExit(f"eval config must contain at least one dataset: {args.template}")
    if any(not isinstance(dataset, dict) for dataset in templates):
        raise SystemExit(f"eval datasets must be mappings: {args.template}")

    requested_splits = [split for split, _ in args.dataset]
    if len(requested_splits) != len(set(requested_splits)):
        raise SystemExit(f"duplicate eval splits requested: {requested_splits}")

    resolved_datasets = []
    for split, path in args.dataset:
        dataset = copy.deepcopy(template_for_split(templates, split))
        template_split = (dataset.get("metadata_overrides") or {}).get("split")
        template_name = dataset.get("name")
        if template_split != split or not isinstance(template_name, str) or "${" in template_name:
            dataset["name"] = f"{args.env_name}-{split}"
        dataset["path"] = str(path)
        metadata_overrides = dataset.get("metadata_overrides") or {}
        if not isinstance(metadata_overrides, dict):
            raise SystemExit(
                f"metadata_overrides must be a mapping for split={split}: {args.template}"
            )
        metadata_overrides["split"] = split
        dataset["metadata_overrides"] = metadata_overrides
        resolved_datasets.append(dataset)

    config["eval"]["datasets"] = resolved_datasets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(
        f"Resolved {len(resolved_datasets)} native eval dataset(s) to {args.output}: "
        + ", ".join(dataset["name"] for dataset in resolved_datasets)
    )


if __name__ == "__main__":
    main()
