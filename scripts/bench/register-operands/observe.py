#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One bounded owned child via the established observer; no retries or floors."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import Policy, Session

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--deadline", required=True, type=int, choices=(240, 360))
parser.add_argument("--timing", action="store_true")
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
env = os.environ.copy()
if args.timing:
    for key in list(env):
        if key.startswith(
            (
                "MTL_DEBUG_LAYER",
                "MTL_SHADER_VALIDATION",
                "SHADER_VALIDATION",
                "METAL_DEVICE_WRAPPER",
                "MTL_CAPTURE",
                "METAL_CAPTURE",
            )
        ) or key in ("DYLD_INSERT_LIBRARIES", "DYLD_FRAMEWORK_PATH"):
            del env[key]
else:
    env["MTL_DEBUG_LAYER"] = "1"
env["QWEN_METAL_LEASE_WAIT"] = "1"
session = Session(Policy())
result = session.run_command(
    command,
    stdout_path=args.output.with_suffix(".stdout.jsonl"),
    stderr_path=args.output.with_suffix(".stderr.log"),
    events_path=args.output.with_suffix(".events.jsonl"),
    timeout_s=args.deadline,
    env=env,
)
print(json.dumps(result.__dict__))
raise SystemExit(0 if result.reason == "completed" else 1)
