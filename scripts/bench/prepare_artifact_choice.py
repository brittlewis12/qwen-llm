#!/usr/bin/env python3
"""Freeze the four natural-source prompts used by the artifact-choice probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tracked_paths(root: Path) -> list[Path]:
    output = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
    return [root / item.decode("utf-8") for item in output.split(b"\0") if item]


def natural_source(entry: dict) -> tuple[str, str]:
    if "text_file" in entry:
        path = Path(entry["text_file"])
        return path.read_text(encoding="utf-8"), entry.get("source", str(path))

    extensions = {value.lower() for value in entry["extensions"]}
    roots = [Path(value) for value in entry["roots"]]
    target_bytes = int(entry.get("target_bytes", 400_000))
    pieces: list[str] = []
    byte_count = 0
    files: list[Path] = []
    for root in roots:
        files.extend(
            path for path in tracked_paths(root)
            if path.suffix.lower() in extensions
        )
    seen: set[Path] = set()
    for path in sorted(files, key=lambda value: str(value)):
        if path in seen:
            continue
        seen.add(path)
        try:
            size = path.stat().st_size
            if size < 40 or size > 1_500_000:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if "\0" in text:
            continue
        relative = next(
            (path.relative_to(root).as_posix() for root in roots if path.is_relative_to(root)),
            path.name,
        )
        pieces.append(f"// SOURCE: {relative}\n{text.rstrip()}\n")
        byte_count += len(text.encode("utf-8"))
        if byte_count >= target_bytes:
            break
    if byte_count < target_bytes:
        raise ValueError(
            f"{entry['id']}: selected only {byte_count} source bytes, wanted {target_bytes}"
        )
    return "\n".join(pieces), entry.get("source", ", ".join(str(root) for root in roots))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--prior-code", type=Path)
    parser.add_argument("--spec-json", type=Path, help="wide cohort prompt specification")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.spec_json:
        spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
        manifest = {
            "schema_version": 1,
            "cohort": spec.get("cohort", "artifact-choice-wide"),
            "depth": spec.get("depth", 49152),
            "targets_per_prompt": spec.get("targets_per_prompt", 1024),
            "construction": "full tracked natural source files, tokenized without special tokens; no cycling or target substitution",
            "prompts": [],
        }
        for entry in spec["prompts"]:
            text, source = natural_source(entry)
            output = args.out_dir / f"{entry['id']}.txt"
            output.write_text(text, encoding="utf-8")
            data = text.encode("utf-8")
            manifest["prompts"].append(
                {
                    "id": entry["id"],
                    "stratum": entry.get("stratum", "code"),
                    "language": entry.get("language"),
                    "source": source,
                    "file": output.name,
                    "source_utf8_sha256": sha256(data),
                    "bytes": len(data),
                }
            )
        manifest_path = args.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(manifest_path)
        return

    if not args.repo_root or not args.prior_code:
        parser.error("--repo-root and --prior-code are required unless --spec-json is set")

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
