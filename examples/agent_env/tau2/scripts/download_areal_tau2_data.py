from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AReaL synthetic tau2 RL data into a separated data root.")
    parser.add_argument("--output-dir", required=True, help="Target directory, e.g. /tmp/mlf-runtime/data/tau2/areal_synthetic.")
    parser.add_argument("--repo-id", default="inclusionAI/AReaL-tau2-data")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            cache_dir=args.cache_dir,
            allow_patterns=["tau2_rl_train.jsonl", "tau2_rl_database/**", "README.md"],
        )
    )

    copied = []
    for rel in ("tau2_rl_train.jsonl", "README.md"):
        src = snapshot / rel
        if src.exists():
            dst = output_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)

    src_db = snapshot / "tau2_rl_database"
    dst_db = output_dir / "tau2_rl_database"
    if src_db.exists():
        if dst_db.exists():
            shutil.rmtree(dst_db)
        shutil.copytree(src_db, dst_db)
        copied.extend(str(path.relative_to(output_dir)) for path in dst_db.rglob("*") if path.is_file())

    manifest = {
        "data_source": "areal_synthetic",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot": str(snapshot),
        "output_dir": str(output_dir),
        "files": copied,
    }
    (output_dir / "SOURCE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "num_files": len(copied)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
