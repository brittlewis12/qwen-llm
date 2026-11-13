#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Target-evict one file's unified page-cache pages with msync(MS_INVALIDATE).

Mirrors qwen_llm::cache_probe::invalidate_file_cache for bench scripts. Leaves
other files alone. Reports mincore residency before and after.
"""

import argparse
import ctypes
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--out")
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
    libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.msync.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    size = os.path.getsize(args.file)
    page = os.sysconf("SC_PAGE_SIZE")
    length = (size + page - 1) // page * page
    fd = os.open(args.file, os.O_RDONLY)
    address = libc.mmap(None, length, 1, 1, fd, 0)
    if address in (None, ctypes.c_void_p(-1).value):
        raise SystemExit(f"mmap failed: errno {ctypes.get_errno()}")
    vector = ctypes.create_string_buffer(length // page)

    def resident():
        rc = libc.mincore(address, length, vector)
        return sum(1 for byte in vector.raw if byte & 1) if rc == 0 else None

    before = resident()
    started = time.time()
    rc = libc.msync(address, length, 0x0010 | 0x0002)
    elapsed = time.time() - started
    after = resident()
    record = {
        "file": args.file,
        "bytes": size,
        "resident_pages_before": before,
        "resident_pages_after": after,
        "msync_rc": rc,
        "msync_errno": ctypes.get_errno(),
        "seconds": elapsed,
    }
    libc.munmap(address, length)
    os.close(fd)
    text = json.dumps(record) + "\n"
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
