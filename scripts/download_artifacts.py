#!/usr/bin/env python3
"""Download pinned Hugging Face artifacts, or check the existing local copies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("all", "dataset", "model"), default="all")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "config/artifacts.json").read_text())
    for item in manifest["files"]:
        if args.kind != "all" and item["kind"] != args.kind:
            continue
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT / "artifacts"):
            raise ValueError("Artifact path must be inside artifacts/")
        if path.is_file() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]:
            print(f"Verified {item['path']}")
            continue
        if args.check_only:
            raise SystemExit(f"Missing or changed artifact: {item['path']}")
        remote = manifest["huggingface"][item["kind"]]
        repo_id, revision = remote["repo_id"], remote["revision"]
        if not repo_id or not revision:
            raise SystemExit("Set the actual Hugging Face repo_id and commit revision in config/artifacts.json first.")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SystemExit("Pin each Hugging Face revision to its 40-character commit hash.")
        prefix = "datasets/" if item["kind"] == "dataset" else ""
        url = f"https://huggingface.co/{prefix}{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/{quote(item['filename'], safe='')}"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        headers = {}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
                raise ValueError(f"Downloaded artifact failed integrity verification: {item['filename']}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {item['path']}")


if __name__ == "__main__":
    main()
