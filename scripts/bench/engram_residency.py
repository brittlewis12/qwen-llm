#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Sample page-cache residency of a file and host VM counters.

Uses `mincore(2)` on a read-only shared mapping, so it observes residency
without faulting the file. Writes one JSON line per interval. Measurement-only.
"""

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PROT_READ = 1
MAP_SHARED = 1


def vm_stat():
    text = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    values = {}
    for name, count in re.findall(r"^(.*?):\s+(\d+)\.", text, re.MULTILINE):
        values[name.strip()] = int(count)
    return page, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    libc.mincore.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    size = args.file.stat().st_size
    page = os.sysconf("SC_PAGE_SIZE")
    length = (size + page - 1) // page * page
    fd = os.open(args.file, os.O_RDONLY)
    address = libc.mmap(None, length, PROT_READ, MAP_SHARED, fd, 0)
    if address in (None, ctypes.c_void_p(-1).value):
        raise SystemExit(f"mmap failed: errno {ctypes.get_errno()}")
    pages = length // page
    vector = ctypes.create_string_buffer(pages)
    running = True
    if args.once:
        limit = 1
    else:
        limit = None

    def stop(*_):
        nonlocal running
        running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    with args.out.open("a") as out:
        samples = 0
        while running and (limit is None or samples < limit):
            started = time.time()
            rc = libc.mincore(address, length, vector)
            errno = ctypes.get_errno()
            resident = sum(1 for byte in vector.raw if byte & 1) if rc == 0 else None
            page_size, counters = vm_stat()
            record = {
                "time": started,
                "file": str(args.file),
                "bytes": size,
                "page_bytes": page_size,
                "mincore_rc": rc,
                "mincore_errno": errno if rc else 0,
                "resident_pages": resident,
                "resident_bytes": resident * page_size
                if resident is not None
                else None,
                "resident_share": resident * page_size / size
                if resident is not None
                else None,
                "vm": {
                    "free": counters.get("Pages free", 0),
                    "inactive": counters.get("Pages inactive", 0),
                    "speculative": counters.get("Pages speculative", 0),
                    "wired": counters.get("Pages wired down", 0),
                    "file_backed": counters.get("File-backed pages", 0),
                    "purgeable": counters.get("Pages purgeable", 0),
                    "compressor": counters.get("Pages occupied by compressor", 0),
                },
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            samples += 1
            if limit is None:
                time.sleep(max(0, args.interval - (time.time() - started)))
    libc.munmap(address, length)
    os.close(fd)


if __name__ == "__main__":
    main()
