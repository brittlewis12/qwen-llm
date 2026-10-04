# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""List GLMCAP01 capture records (step, layer, occurrence, name, type, shape,
L2 norm), rejecting truncated or inconsistent records."""

import struct
import sys

import numpy as np

DTYPES = {0: np.float32, 1: np.float16, 26: np.int32}


def read_exact(f, n, what):
    data = f.read(n)
    if len(data) != n:
        raise SystemExit(f"truncated {what}: wanted {n} bytes, got {len(data)}")
    return data


def records(path):
    with open(path, "rb") as f:
        if f.read(8) != b"GLMCAP01":
            raise SystemExit(f"{path}: not a GLMCAP01 file")
        while header := f.read(16):
            if len(header) != 16:
                raise SystemExit("truncated record header")
            step, layer, occurrence, name_len = struct.unpack("<IIII", header)
            name = read_exact(f, name_len, "name").decode()
            (ggml_type,) = struct.unpack("<I", read_exact(f, 4, "type"))
            ne = struct.unpack("<4q", read_exact(f, 32, "shape"))
            (nbytes,) = struct.unpack("<q", read_exact(f, 8, "byte length"))
            if ggml_type not in DTYPES:
                raise SystemExit(f"{name}: unsupported ggml type {ggml_type}")
            if any(n <= 0 for n in ne):
                raise SystemExit(f"{name}: non-positive dimension {ne}")
            expected = int(np.prod(ne)) * np.dtype(DTYPES[ggml_type]).itemsize
            if nbytes != expected:
                raise SystemExit(
                    f"{name}: {nbytes} bytes for shape {ne}, expected {expected}"
                )
            data = np.frombuffer(read_exact(f, nbytes, name), dtype=DTYPES[ggml_type])
            yield (
                step,
                (-1 if layer == 0xFFFFFFFF else layer),
                occurrence,
                name,
                ggml_type,
                ne,
                data,
            )


def main():
    for step, layer, occurrence, name, ggml_type, ne, data in records(sys.argv[1]):
        norm = float(np.linalg.norm(data.astype(np.float64)))
        shape = [n for n in ne if n != 1] or [1]
        print(
            f"step={step:4d} layer={layer:3d} occ={occurrence} {name:20s} type={ggml_type:2d} ne={shape} l2={norm:.6g}"
        )


if __name__ == "__main__":
    main()
