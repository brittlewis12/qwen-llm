# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""List GLMCAP01 capture records: step, layer, name, type, shape and L2 norm."""

import struct
import sys

import numpy as np

DTYPES = {0: np.float32, 1: np.float16, 26: np.int32}


def records(path):
    with open(path, "rb") as f:
        if f.read(8) != b"GLMCAP01":
            raise SystemExit(f"{path}: not a GLMCAP01 file")
        while header := f.read(12):
            step, layer, name_len = struct.unpack("<III", header)
            name = f.read(name_len).decode()
            (ggml_type,) = struct.unpack("<I", f.read(4))
            ne = struct.unpack("<4q", f.read(32))
            (nbytes,) = struct.unpack("<q", f.read(8))
            data = np.frombuffer(f.read(nbytes), dtype=DTYPES[ggml_type])
            yield (
                step,
                (-1 if layer == 0xFFFFFFFF else layer),
                name,
                ggml_type,
                ne,
                data,
            )


def main():
    for step, layer, name, ggml_type, ne, data in records(sys.argv[1]):
        norm = float(np.linalg.norm(data.astype(np.float64)))
        shape = [n for n in ne if n != 1] or [1]
        print(
            f"step={step:4d} layer={layer:3d} {name:20s} type={ggml_type:2d} ne={shape} l2={norm:.6g}"
        )


if __name__ == "__main__":
    main()
