#!/usr/bin/env python3
"""Prepare the temporary library/example in a clean, isolated worktree."""
from pathlib import Path
import shutil
import subprocess


here = Path(__file__).resolve().parent
root = here.parents[3]
subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=root, check=True)
patch = here.parent / "capture.patch"
subprocess.run(["git", "apply", "--unidiff-zero", "--check", str(patch)], cwd=root, check=True)
subprocess.run(["git", "apply", "--unidiff-zero", str(patch)], cwd=root, check=True)
metal = root / "crates/qwen-llm/src/metal"
shutil.copyfile(here / "executor.rs", metal / "layer_executor.rs")
with (metal / "dispatch_profile.rs").open("a") as f:
    f.write('\n#[path = "layer_executor.rs"]\nmod layer_executor;\npub use layer_executor::*;\n')
shutil.copyfile(here / "probe.rs", root / "crates/qwen-llm/examples/fn_layer_prototype.rs")
