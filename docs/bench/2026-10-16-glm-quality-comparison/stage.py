# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Expose publisher-selected shards under canonical GGUF split filenames."""

import argparse
import json
import os
from pathlib import Path


def stage(download, manifest):
    download = download.resolve()
    target = download / "glm5-next"
    entries = manifest["shards"]
    if target.is_symlink() or not entries:
        raise ValueError("invalid staging directory or empty manifest")
    planned = []
    for entry in entries:
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError("invalid shard filename")
        source = (download / entry["source_path"]).resolve()
        if not source.is_relative_to(download):
            raise ValueError("source escapes download directory")
        if not source.is_file() or source.stat().st_size != entry["size"]:
            raise ValueError("missing or incomplete downloaded shard")
        dest = target / name
        if os.path.lexists(dest) and (
            not dest.is_symlink() or dest.resolve() != source
        ):
            raise ValueError("refusing to replace an existing entrypoint")
        planned.append((source, dest))
    target.mkdir(exist_ok=True)
    for source, dest in planned:
        if not os.path.lexists(dest):
            dest.symlink_to(os.path.relpath(source, target))
    return planned[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(stage(args.download, json.loads(args.manifest.read_text())))
