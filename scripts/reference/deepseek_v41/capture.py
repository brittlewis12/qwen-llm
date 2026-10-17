"""Offline verification; explicit recapture: uv run --no-project python scripts/reference/deepseek_v41/capture.py --fetch

Recapture checks the reviewed manifest, never regenerates trust anchors. No weights.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
FILES = (
    "config.json",
    "inference/model.py",
    "inference/kernel.py",
    "inference/engram.py",
    "LICENSE",
)


def verify(fetch=False):
    manifest = json.loads((ROOT / "sources.json").read_text())
    if (
        manifest["repository"] != REPO
        or manifest["revision"] != REVISION
        or set(manifest["sha256"]) != set(FILES)
    ):
        raise ValueError("source manifest does not match the pinned contract")
    contents = {}
    for name in FILES:
        path = ROOT / "fixtures" / name
        data = (
            urllib.request.urlopen(
                f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=60
            ).read()
            if fetch
            else path.read_bytes()
        )
        if hashlib.sha256(data).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"SHA256 mismatch: {name}")
        contents[name] = data
    if fetch:
        for name, data in contents.items():
            path = ROOT / "fixtures" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return contents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    verify(parser.parse_args().fetch)
    print("Verified five pinned source fixtures")
