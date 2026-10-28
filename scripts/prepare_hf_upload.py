#!/usr/bin/env python3
"""Assemble local upload directories from verified artifacts and release cards."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/download_artifacts.py"), "--check-only"], check=True)
    manifest = json.loads((ROOT / "config/artifacts.json").read_text())
    for kind in ("dataset", "model"):
        destination = ROOT / "hf_upload" / kind
        destination.mkdir(parents=True, exist_ok=True)
        for item in manifest["files"]:
            if item["kind"] == kind:
                shutil.copy2(ROOT / item["path"], destination / item["filename"])
        shutil.copy2(ROOT / "release" / f"hf_{kind}" / "README.md", destination / "README.md")
        shutil.copy2(ROOT / "NOTICE.md", destination / "NOTICE.md")
        shutil.copy2(ROOT / "LICENSE", destination / "LICENSE")
        if kind == "dataset":
            shutil.copy2(ROOT / "docs/dataset.md", destination / "DATA_DICTIONARY.md")
        else:
            for name in ("paper_ssm.py", "paper_bundle.py"):
                shutil.copy2(ROOT / "src" / name, destination / name)
            shutil.copy2(ROOT / "requirements.txt", destination / "requirements.txt")
        print(f"Prepared {destination}")


if __name__ == "__main__":
    main()
