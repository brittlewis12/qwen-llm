#!/usr/bin/env python3
"""Freeze the four natural-source prompts used by the artifact-choice probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prior-code", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metal_sources = (Path("kernels/moe.metal"), Path("kernels/mat_vec.metal"))
    metal = "".join(
        f"// SOURCE: {path.as_posix()}\n"
        f"{(args.repo_root / path).read_text(encoding='utf-8').rstrip()}\n"
        for path in metal_sources
    )
    items = [
        ("rust-agent-corpus", "code", args.prior_code.read_text(encoding="utf-8")),
        ("metal-routed-kernels", "code", metal),
        (
            "python-dflash-profile",
            "code",
            (args.repo_root / "scripts/profile/dflash_k0s.py").read_text(encoding="utf-8"),
        ),
        (
            "performance-log",
            "prose",
            (args.repo_root / "docs/PERF-LOG.md").read_text(encoding="utf-8"),
        ),
    ]
    manifest = {
        "schema_version": 1,
        "cohort": "artifact-choice-v1",
        "depth": 49152,
        "targets_per_prompt": 1024,
        "construction": "full natural-source text, tokenized without special tokens; no cycling or target substitution",
        "prompts": [],
    }
    for prompt_id, stratum, text in items:
        output = args.out_dir / f"{prompt_id}.txt"
        output.write_text(text, encoding="utf-8")
        manifest["prompts"].append(
            {
                "id": prompt_id,
                "stratum": stratum,
                "file": output.name,
                "source_utf8_sha256": sha256(text.encode("utf-8")),
                "bytes": len(text.encode("utf-8")),
            }
        )
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
