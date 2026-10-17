# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Bound one owned inference process without touching other sessions."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.seconds <= 0 or not args.command:
        parser.error("a positive deadline and command are required")
    try:
        return subprocess.run(
            args.command, timeout=args.seconds, check=False
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"Owned inference process exceeded {args.seconds}s", file=sys.stderr)
        return 124


if __name__ == "__main__":
    sys.exit(main())
