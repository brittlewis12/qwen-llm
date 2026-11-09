#!/usr/bin/env python3
"""Prepare the cache probe in a clean isolated worktree."""
from pathlib import Path
import shutil
import subprocess
import sys

here = Path(__file__).resolve().parent
root = here.parents[3]
subprocess.run([sys.executable, str(here.parent / "layer/prepare.py")], check=True)
metal = root / "crates/qwen-llm/src/metal"
with (metal / "layer_executor.rs").open("a") as f:
    f.write('\n#[path = "cache_probe.rs"]\nmod cache_probe;\npub use cache_probe::*;\n')
shutil.copyfile(here / "logic.rs", metal / "cache_probe.rs")
with (metal / "mod.rs").open("a") as f:
    f.write('\npub use dispatch_profile::CachePacket;\n')
shutil.copyfile(here / "probe.rs", root / "crates/qwen-llm/examples/fn_cache_probe.rs")
