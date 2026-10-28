#!/usr/bin/env python3
"""Record current HF commit IDs after uploads, using the saved CLI login.

Requires huggingface_hub, as installed with the hf CLI. Makes read-only remote
requests; only the local config/artifacts.json is updated.
"""
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "config/artifacts.json"
    config = json.loads(path.read_text())
    api = HfApi()
    visibility = []
    for kind in ("model", "dataset"):
        item = config["huggingface"][kind]
        repo_id = item.get("repo_id")
        if not repo_id:
            raise SystemExit(f"Set the {kind} repo_id in config/artifacts.json first.")
        info = api.repo_info(repo_id=repo_id, repo_type=kind)
        item["revision"] = info.sha
        visibility.append(info.private)
        print(f"{kind}: {repo_id} @ {info.sha}")
    status = "uploaded_private" if all(visibility) else "uploaded_public" if not any(visibility) else "uploaded_mixed"
    config.setdefault("destinations", {})["publication_status"] = status
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(path)
    print("Local Hugging Face revisions updated.")


if __name__ == "__main__":
    main()
