#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compile directly into a fresh frozen archive; invoke under with-metal-lease."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1]).resolve()
if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
    raise SystemExit("build requires a clean commit-pinned owned checkout")
out.mkdir(mode=0o700, exist_ok=False)


def run(*args):
    print(json.dumps({"command": args}), flush=True)
    subprocess.run(args, cwd=root, check=True)


head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True
).strip()
run("git", "archive", "--format=tar", f"--output={out / 'source.tar'}", head)
src = root / "scripts/bench/register-operands"
run(
    "xcrun",
    "clang++",
    "-std=c++20",
    "-O2",
    str(src / "test_contracts.cpp"),
    "-o",
    str(out / "test_contracts"),
)
run(str(out / "test_contracts"))
run(
    "xcrun",
    "--sdk",
    "macosx",
    "metal",
    "-std=metal4.0",
    "-mmacosx-version-min=26.0",
    "-O3",
    "-ffast-math",
    "-c",
    str(root / "kernels/tensor/moe_grouped_slots_tensor.metal"),
    "-o",
    str(out / "baseline26.air"),
)
run(
    "xcrun",
    "--sdk",
    "macosx",
    "metallib",
    str(out / "baseline26.air"),
    "-o",
    str(out / "baseline26.metallib"),
)
run(
    "xcrun",
    "--sdk",
    "macosx",
    "metal",
    "-std=metal4.1",
    "-mmacosx-version-min=27.0",
    "-O3",
    "-ffast-math",
    "-c",
    str(src / "probe.metal"),
    "-o",
    str(out / "probe.air"),
)
run(
    "xcrun",
    "--sdk",
    "macosx",
    "metallib",
    str(out / "probe.air"),
    "-o",
    str(out / "private.metallib"),
)
run(
    "xcrun",
    "--sdk",
    "macosx",
    "clang++",
    "-std=c++20",
    "-O3",
    "-fobjc-arc",
    "-mmacosx-version-min=27.0",
    "-framework",
    "Foundation",
    "-framework",
    "Metal",
    str(src / "probe.mm"),
    "-o",
    str(out / "probe"),
)
identity = {
    "head": head,
    "source_root": str(root),
    "target": "standalone metal4.1@27.0; separate incumbent metal4.0@26.0; product unchanged",
    "metal": subprocess.check_output(["xcrun", "metal", "--version"], text=True),
    "sdk": subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-version"], text=True
    ),
    "sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file()
    },
}
(out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
print(json.dumps(identity), flush=True)
